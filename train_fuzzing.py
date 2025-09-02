"""Training script for hierarchical fuzzing reinforcement learning.

This script trains the fuzzing-adapted HSD algorithm using:
- Protocol dataset loading and preprocessing
- Mamba feature integration
- Three-tier reward system
- 90/10 field selection strategy
"""

import json
import os
import random
import sys
import time
import logging

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

from alg_fuzzing_hsd import AlgFuzzingHSD
from env.fuzzing_environment import FuzzingEnvironment
import replay_buffer


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_fuzzing_config(base_config):
    """Create fuzzing-specific configuration from base config."""
    fuzzing_config = base_config.copy()
    
    # Add fuzzing-specific parameters
    fuzzing_config['fuzzing'] = {
        'protocol_fields': ['header', 'source_addr', 'dest_addr', 'payload', 'checksum', 'flags'],
        'mutation_actions': ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy'],
        'field_selection_reward_ratio': 0.9,  # 90% reward-based, 10% random
        'max_mutation_size': 1024,
        'mutation_history_length': 10,
        'max_episode_steps': 1000,
        'mamba_feature_dim': 256,
        'field_context_dim': 32,
        'coverage_feature_dim': 64
    }
    
    # Add reward system configuration
    fuzzing_config['rewards'] = {
        'base_reward_weight': 1.0,
        'medium_reward_weight': 5.0,
        'high_reward_weight': 100.0,
        'coverage_weight': 1.0,
        'novelty_weight': 0.5,
        'uniqueness_weight': 0.3,
        'crash_weight': 1.0,
        'timeout_weight': 0.8,
        'anomaly_weight': 0.6,
        'vulnerability_weight': 1.0
    }
    
    # Update network configuration for fuzzing
    fuzzing_config['nn_fuzzing_hsd'] = {
        'n_h_decoder': 128,
        'n_h1_low': 64,
        'n_h2_low': 64,
        'n_h1': 128,
        'n_h2': 128,
        'n_h_mixer': 64
    }
    
    return fuzzing_config


def train_fuzzing_function(config):
    """Main training function for fuzzing HSD."""
    
    # Extract configuration sections
    config_env = config.get('env', {})
    config_main = config['main']
    config_alg = config['alg']
    config_h = config['h_params']
    config_fuzzing = config['fuzzing']
    config_rewards = config['rewards']
    
    # Setup
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']
    
    # Create results directory
    results_dir = f'../results/{dir_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("Starting fuzzing HSD training")
    
    # Save configuration
    with open(f'{results_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Training parameters
    N_train = config_alg['N_train']
    N_eval = config_alg['N_eval']
    period = config_alg['period']
    buffer_size = config_alg['buffer_size']
    batch_size = config_alg['batch_size']
    pretrain_episodes = config_alg['pretrain_episodes']
    steps_per_train = config_alg['steps_per_train']
    
    # Exploration parameters
    epsilon_start = config_alg['epsilon_start']
    epsilon_end = config_alg['epsilon_end']
    epsilon_div = config_alg['epsilon_div']
    epsilon_step = (epsilon_start - epsilon_end) / float(epsilon_div)
    epsilon = epsilon_start
    
    # Fuzzing parameters
    n_fields = len(config_fuzzing['protocol_fields'])
    n_mutation_actions = len(config_fuzzing['mutation_actions'])
    field_selection_steps = config_h['steps_per_assign']
    
    # Curriculum learning for field selection
    N_fields_current = min(n_fields, config_h.get('N_roles_start', n_fields))
    curriculum_threshold = config_h['curriculum_threshold']
    
    # Reward weighting
    alpha = config_h['alpha_start']
    alpha_end = config_h['alpha_end']
    alpha_step = config_h['alpha_step']
    alpha_threshold = config_h['alpha_threshold']
    
    # Create fuzzing environment
    env_config = {**config_env, **config_fuzzing, **config_rewards}
    env = FuzzingEnvironment(env_config)
    
    # Get environment dimensions
    l_state = env.state_dim
    l_obs = env.obs_dim
    l_mutation_actions = env.action_dim
    
    logger.info(f"Environment dimensions - State: {l_state}, Obs: {l_obs}, Actions: {l_mutation_actions}")
    logger.info(f"Protocol fields: {n_fields}, Current active: {N_fields_current}")
    
    # Create fuzzing HSD algorithm
    alg = AlgFuzzingHSD(
        config_alg=config_alg,
        config_h=config_h,
        config_fuzzing=config_fuzzing,
        n_fields=n_fields,
        l_state=l_state,
        l_obs=l_obs,
        l_mutation_actions=l_mutation_actions,
        nn=config['nn_fuzzing_hsd']
    )
    
    # TensorFlow session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    
    # TensorBoard writer
    if summarize:
        writer = tf.summary.FileWriter(results_dir, sess.graph)
    
    # Model saver
    saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])
    
    # Replay buffers
    buf_high = replay_buffer.Replay_Buffer(size=buffer_size)  # Field selection
    buf_low = replay_buffer.Replay_Buffer(size=buffer_size)   # Mutation actions
    
    # Dataset for decoder training (field selection history)
    decoder_dataset = []
    
    # Logging setup
    header = "Episode,Step,Step_train,N_fields,alpha,Decoder_prob,R_avg,R_eval,Steps_per_eps,Coverage,Crashes,Vulns,T_env,T_alg\n"
    with open(f"{results_dir}/log.csv", 'w') as f:
        f.write(header)
    
    # Training metrics
    t_env = 0
    t_alg = 0
    reward_period = 0
    decoder_prob = 0
    step = 0
    step_train = 0
    step_h = 0
    
    # Episode metrics
    total_coverage = 0
    total_crashes = 0
    total_vulnerabilities = 0
    
    logger.info("Starting training loop")
    
    for idx_episode in range(1, N_train + 1):
        
        # Reset environment
        state, _, list_obs, _, done = env.reset()
        
        # High-level state tracking
        state_h, list_obs_h = state.copy(), [obs.copy() for obs in list_obs]
        reward_h = 0
        
        # Field selection (high-level action)
        field_indices = np.random.randint(0, N_fields_current, n_fields)
        field_selections = np.zeros([n_fields, n_fields])
        field_selections[np.arange(n_fields), field_indices] = 1
        
        # Mutation history tracking
        mutation_trajectories = [[] for _ in range(n_fields)]
        
        # Episode tracking
        reward_episode = 0
        step_episode = 0
        summarized = 0
        summarized_h = 0
        
        while not done:
            
            # Intrinsic reward calculation
            intrinsic_rewards = np.zeros(n_fields)
            
            # High-level field selection every N steps
            if step_episode % field_selection_steps == 0:
                if step_episode != 0:
                    # Store high-level transition
                    r_discounted = reward_h * (config_alg['gamma'] ** field_selection_steps)
                    buf_high.add(np.array([
                        state_h, np.array(list_obs_h), field_indices, 
                        r_discounted, state, np.array(list_obs), done
                    ]))
                    
                    # Store mutation trajectories for decoder
                    for field_idx in range(n_fields):
                        if len(mutation_trajectories[field_idx]) >= field_selection_steps:
                            decoder_dataset.append(np.array([
                                np.array(mutation_trajectories[field_idx][-field_selection_steps:]),
                                field_selections[field_idx]
                            ]))
                    
                    # Compute intrinsic rewards
                    intrinsic_rewards = alg.compute_intrinsic_reward(
                        sess, mutation_trajectories, field_selections)
                
                step_h += 1
                
                # Select fields (high-level action)
                if idx_episode < pretrain_episodes:
                    field_indices = np.random.randint(0, N_fields_current, n_fields)
                else:
                    t_alg_start = time.time()
                    field_indices = alg.assign_fields(
                        list_obs, epsilon, sess, env.field_importance_scores)
                    t_alg += time.time() - t_alg_start
                
                # Convert to one-hot
                field_selections = np.zeros([n_fields, n_fields])
                field_selections[np.arange(n_fields), field_indices] = 1
                
                # Train high-level policy
                if (idx_episode >= pretrain_episodes) and (step_h % steps_per_train == 0):
                    batch = buf_high.sample_batch(batch_size)
                    t_alg_start = time.time()
                    if summarize and idx_episode % period == 0 and not summarized_h:
                        alg.train_policy_high(sess, batch, step_train, summarize=True, writer=writer)
                        summarized_h = True
                    else:
                        alg.train_policy_high(sess, batch, step_train, summarize=False, writer=None)
                    step_train += 1
                    t_alg += time.time() - t_alg_start
                
                # Update high-level state
                state_h, list_obs_h = state.copy(), [obs.copy() for obs in list_obs]
                reward_h = 0
            
            # Low-level mutation action selection
            if idx_episode < pretrain_episodes:
                mutation_actions = env.random_actions()
            else:
                t_alg_start = time.time()
                mutation_actions = alg.run_mutation_actor(list_obs, field_selections, epsilon, sess)
                t_alg += time.time() - t_alg_start
            
            # Execute actions in environment
            t_env_start = time.time()
            state_next, _, list_obs_next, _, reward, local_rewards, done, info = env.step(
                mutation_actions, field_indices)
            t_env += time.time() - t_env_start
            
            # Update episode metrics
            execution_result = info.get('execution_result', {})
            total_coverage += execution_result.get('coverage_increase', 0)
            if execution_result.get('crash_detected', False):
                total_crashes += 1
            if execution_result.get('vulnerability_found', False):
                total_vulnerabilities += 1
            
            # Combine global and intrinsic rewards
            combined_rewards = alpha * np.array(local_rewards) + (1 - alpha) * intrinsic_rewards
            
            # Store mutation trajectories
            for field_idx in range(n_fields):
                mutation_trajectories[field_idx].append(list_obs[field_idx])
                # Keep limited history
                if len(mutation_trajectories[field_idx]) > field_selection_steps:
                    mutation_trajectories[field_idx] = mutation_trajectories[field_idx][-field_selection_steps:]
            
            # Store low-level transitions
            step += 1
            step_episode += 1
            
            buf_low.add(np.array([
                np.array(list_obs), mutation_actions, combined_rewards,
                np.array(list_obs_next), field_selections, done
            ], dtype=object))
            
            # Train low-level policy
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                batch = buf_low.sample_batch(batch_size)
                t_alg_start = time.time()
                if summarize and idx_episode % period == 0 and not summarized:
                    alg.train_policy_low(sess, batch, step_train, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_policy_low(sess, batch, step_train, summarize=False, writer=None)
                step_train += 1
                t_alg += time.time() - t_alg_start
            
            # Update state
            state = state_next
            list_obs = list_obs_next
            reward_episode += reward
            reward_h += reward
            
            # Handle episode termination
            if done:
                # Final high-level transition
                r_discounted = reward_h * config_alg['gamma'] ** (step_episode % field_selection_steps)
                buf_high.add(np.array([
                    state_h, np.array(list_obs_h), field_indices,
                    r_discounted, state, np.array(list_obs), done
                ]))
                
                # Final decoder data
                if step_episode >= field_selection_steps:
                    for field_idx in range(n_fields):
                        if len(mutation_trajectories[field_idx]) >= field_selection_steps:
                            decoder_dataset.append(np.array([
                                np.array(mutation_trajectories[field_idx][-field_selection_steps:]),
                                field_selections[field_idx]
                            ]))
        
        # Train decoder if enough data
        N_batch_decoder = config_h.get('N_batch_hsd', 1000)
        if len(decoder_dataset) >= N_batch_decoder:
            t_alg_start = time.time()
            if summarize:
                decoder_prob = alg.train_decoder(sess, decoder_dataset[:N_batch_decoder], 
                                               step_train, summarize=True, writer=writer)
            else:
                decoder_prob = alg.train_decoder(sess, decoder_dataset[:N_batch_decoder], 
                                               step_train, summarize=False, writer=None)
            step_train += 1
            t_alg += time.time() - t_alg_start
            
            # Curriculum: increase number of active fields
            if decoder_prob >= curriculum_threshold:
                N_fields_current = min(int(1.5 * N_fields_current + 1), n_fields)
                logger.info(f"Curriculum update: N_fields_current = {N_fields_current}")
            
            # Clear dataset
            decoder_dataset = []
        
        # Update exploration
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
        
        reward_period += reward_episode
        
        # Periodic evaluation and logging
        if idx_episode == 1 or idx_episode % (5 * period) == 0:
            print(f"{'Episode':<10}{'Step':<10}{'Step_train':<12}{'N_fields':<10}{'Alpha':<8}{'Decoder':<10}{'R_avg':<8}{'Coverage':<10}{'Crashes':<8}{'Vulns':<8}")
        
        if idx_episode % period == 0:
            # Evaluation episodes (simplified for fuzzing)
            eval_reward = reward_period / period
            
            # Adjust alpha based on performance
            if eval_reward >= alpha_threshold:
                alpha = max(alpha_end, alpha - alpha_step)
            
            # Log results
            coverage_rate = total_coverage / max(step, 1)
            crash_rate = total_crashes / max(idx_episode, 1)
            vuln_rate = total_vulnerabilities / max(idx_episode, 1)
            
            log_entry = f'{idx_episode},{step},{step_train},{N_fields_current},{alpha:.2f},{decoder_prob:.3e},{eval_reward:.2f},{eval_reward:.2f},{step_episode},{coverage_rate:.5f},{crash_rate:.3f},{vuln_rate:.3f},{t_env:.5e},{t_alg:.5e}\n'
            
            with open(f'{results_dir}/log.csv', 'a') as f:
                f.write(log_entry)
            
            print(f'{idx_episode:<10}{step:<10}{step_train:<12}{N_fields_current:<10}{alpha:<8.2f}{decoder_prob:<10.3e}{eval_reward:<8.2f}{coverage_rate:<10.5f}{crash_rate:<8.3f}{vuln_rate:<8.3f}')
            
            reward_period = 0
            
            # Save good models
            if eval_reward >= config_main.get('save_threshold', 0.7):
                saver.save(sess, f'{results_dir}/model_good.ckpt-{idx_episode}')
        
        # Periodic model saving
        if idx_episode % save_period == 0:
            saver.save(sess, f'{results_dir}/model.ckpt-{idx_episode}')
    
    # Final model save
    saver.save(sess, f'{results_dir}/{model_name}')
    
    # Save timing information
    with open(f'{results_dir}/time.txt', 'w') as f:
        f.write('t_env_total,t_env_per_step,t_alg_total,t_alg_per_step\n')
        f.write(f'{t_env:.5e},{t_env/step:.5e},{t_alg:.5e},{t_alg/step:.5e}')
    
    logger.info("Training completed successfully")
    
    # Print final statistics
    logger.info(f"Final Statistics:")
    logger.info(f"  Total episodes: {N_train}")
    logger.info(f"  Total steps: {step}")
    logger.info(f"  Total coverage gained: {total_coverage:.4f}")
    logger.info(f"  Total crashes found: {total_crashes}")
    logger.info(f"  Total vulnerabilities found: {total_vulnerabilities}")
    logger.info(f"  Environment time: {t_env:.2f}s")
    logger.info(f"  Algorithm time: {t_alg:.2f}s")


if __name__ == '__main__':
    
    # Load base configuration
    with open('config.json', 'r') as f:
        base_config = json.load(f)
    
    # Create fuzzing-specific configuration
    config = create_fuzzing_config(base_config)
    
    # Override with fuzzing-specific settings
    config['main']['alg_name'] = 'fuzzing_hsd'
    config['main']['dir_name'] = 'fuzzing_hsd_run'
    
    # Run training
    train_fuzzing_function(config)