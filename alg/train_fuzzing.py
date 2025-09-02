"""Training script for Fuzzing Hierarchical Skill Discovery.

This script trains the hierarchical reinforcement learning agent for intelligent
protocol fuzzing, replacing the sports simulation with fuzzing environment.
"""

import json
import os
import random
import sys
import time

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import alg_fuzzing_hsd
from fuzzing_environment import FuzzingEnvironment
import replay_buffer


def train_fuzzing_function(config):
    """Main training function for fuzzing HSD.
    
    Args:
        config: Configuration dictionary
    """
    # Extract configuration sections
    config_env = config['env']
    config_main = config['main']
    config_alg = config['alg']
    config_h = config['h_params']
    config_fuzzing = config['fuzzing_params']
    config_reward = config['reward_config']
    
    # Set random seeds
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    # Setup directories and logging
    alg_name = config_main['alg_name']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']
    
    os.makedirs('../results/%s' % dir_name, exist_ok=True)
    with open('../results/%s/%s' % (dir_name, 'config.json'), 'w') as f:
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
    
    # Hierarchical parameters
    field_selection_steps = config_h['field_selection_steps']
    N_fields = len(config_fuzzing['protocol_fields'])
    N_actions = len(config_fuzzing['mutation_actions'])
    N_batch_decoder = config_h['N_batch_decoder']
    
    # Curriculum learning parameters
    N_fields_current = config_h['N_fields_start']
    curriculum_threshold = config_h['curriculum_threshold']
    
    # Reward mixing parameters
    alpha = config_h['alpha_start']
    alpha_end = config_h['alpha_end']
    alpha_step = config_h['alpha_step']
    alpha_threshold = config_h['alpha_threshold']
    
    # Create fuzzing environment
    env = FuzzingEnvironment(config_env, config_main)
    
    # Environment dimensions
    l_state = env.state_dim
    l_obs = env.obs_dim
    l_mamba = config_fuzzing['mamba_feature_dim']
    
    # Create algorithm
    alg = alg_fuzzing_hsd.FuzzingHSDAlgorithm(
        config_alg, config_h, config_fuzzing, l_state, l_obs, l_mamba,
        N_fields, N_actions, config['nn_fuzzing'])
    
    # TensorFlow session setup
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    
    if summarize:
        writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])
    
    # Replay buffers
    buf_field = replay_buffer.Replay_Buffer(size=buffer_size)  # Field selection buffer
    buf_mutation = replay_buffer.Replay_Buffer(size=buffer_size)  # Mutation action buffer
    
    # Dataset for decoder training
    decoder_dataset = []
    
    # Initialize logging
    header = "Episode,Step,Step_train,N_fields,alpha,Exp_prob,R_avg,Coverage,Crashes,Vulns,T_env,T_alg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    
    # Training state variables
    t_env = 0
    t_alg = 0
    reward_period = 0
    expected_prob = 0
    step = 0
    step_train = 0
    step_h = 0
    
    print("Starting fuzzing training with %d fields and %d mutation actions" % (N_fields, N_actions))
    
    for idx_episode in range(1, N_train + 1):
        
        # Reset environment
        dataset_path = config_fuzzing.get('dataset_path')
        state, _, list_obs, _, done = env.reset(dataset_path)
        
        # Initialize hierarchical state
        state_h = state
        list_obs_h = list_obs
        reward_h = 0
        
        # Field importance tracking
        field_importance = np.ones(N_fields_current) / N_fields_current
        
        # Initialize field selection
        field_idx = np.random.randint(0, N_fields_current)
        field_selection_oh = np.zeros(N_fields)
        field_selection_oh[field_idx] = 1
        
        # Trajectory tracking for decoder
        field_trajectories = []
        mamba_features_history = []
        
        reward_episode = 0
        summarized = 0
        summarized_h = 0
        step_episode = 0
        
        while not done:
            
            # High-level: Field selection every field_selection_steps
            if step_episode % field_selection_steps == 0:
                if step_episode != 0:
                    # Store high-level transition
                    r_discounted = reward_h * (config_alg['gamma'] ** field_selection_steps)
                    buf_field.add(np.array([
                        state_h, list_obs_h[0], field_selection_oh, r_discounted,
                        state, list_obs[0], done
                    ]))
                    
                    # Store trajectory for decoder training
                    if len(field_trajectories) >= field_selection_steps:
                        # Get Mamba features for current packet
                        current_mamba = env.current_features if env.use_mamba_features else np.zeros(l_mamba)
                        decoder_dataset.append(np.array([
                            np.array(field_trajectories[-field_selection_steps:]),
                            current_mamba,
                            field_selection_oh
                        ]))
                
                step_h += 1
                
                # Select field (high-level action)
                if idx_episode < pretrain_episodes:
                    field_idx = np.random.randint(0, N_fields_current)
                else:
                    t_alg_start = time.time()
                    field_idx = alg.select_field(
                        np.array([list_obs[0]]), 
                        np.array([field_importance[:N_fields_current]]), 
                        epsilon, sess)
                    t_alg += time.time() - t_alg_start
                    
                # Update field selection one-hot
                field_selection_oh = np.zeros(N_fields)
                if field_idx < N_fields_current:
                    field_selection_oh[field_idx] = 1
                
                # Train high-level policy
                if (idx_episode >= pretrain_episodes) and (step_h % steps_per_train == 0) and len(buf_field.buffer) >= batch_size:
                    batch = buf_field.sample_batch(batch_size)
                    t_alg_start = time.time()
                    if summarize and idx_episode % period == 0 and not summarized_h:
                        alg.train_field_selector(sess, batch, step_train, summarize=True, writer=writer)
                        summarized_h = True
                    else:
                        alg.train_field_selector(sess, batch, step_train, summarize=False, writer=None)
                    step_train += 1
                    t_alg += time.time() - t_alg_start
                
                # Update hierarchical state
                state_h = state
                list_obs_h = list_obs
                reward_h = 0
            
            # Low-level: Mutation action selection
            if idx_episode < pretrain_episodes:
                mutation_idx = np.random.randint(0, N_actions)
            else:
                # Create mutation context
                packet_info = {
                    'field_length': len(env.current_packet.get(
                        env.protocol_fields[field_idx]['name'], b'')),
                    'field_entropy': np.random.random()  # Placeholder
                }
                mutation_history = {'success_rate': 0.1, 'avg_reward': 0.0}
                mutation_context = alg.create_mutation_context(field_idx, packet_info, mutation_history)
                
                t_alg_start = time.time()
                mutation_idx = alg.select_mutation_action(
                    np.array([list_obs[0]]),
                    np.array([mutation_context]),
                    np.array([field_selection_oh]),
                    epsilon, sess)
                t_alg += time.time() - t_alg_start
            
            # Execute fuzzing step
            t_env_start = time.time()
            state_next, _, list_obs_next, _, reward, local_rewards, done, info = env.step(
                field_idx, mutation_idx)
            t_env += time.time() - t_env_start
            
            # Store trajectory information
            field_trajectories.append(list_obs[0])
            if env.use_mamba_features:
                mamba_features_history.append(env.current_features)
            
            # Create mutation action one-hot
            mutation_action_oh = np.zeros(N_actions)
            mutation_action_oh[mutation_idx] = 1
            
            # Store low-level transition
            mutation_context = alg.create_mutation_context(field_idx, {}, {})
            buf_mutation.add(np.array([
                list_obs[0], mutation_action_oh, reward, list_obs_next[0],
                mutation_context, field_selection_oh, done
            ]))
            
            # Train low-level policy
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0) and len(buf_mutation.buffer) >= batch_size:
                batch = buf_mutation.sample_batch(batch_size)
                t_alg_start = time.time()
                if summarize and idx_episode % period == 0 and not summarized:
                    alg.train_mutation_policy(sess, batch, step_train, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_mutation_policy(sess, batch, step_train, summarize=False, writer=None)
                step_train += 1
                t_alg += time.time() - t_alg_start
            
            # Update field importance based on reward
            importance_update = alg.get_field_importance_update(field_idx, reward)
            if field_idx < len(field_importance):
                field_importance[field_idx] = importance_update
                # Renormalize
                field_importance = field_importance / np.sum(field_importance)
            
            step += 1
            step_episode += 1
            state = state_next
            list_obs = list_obs_next
            reward_episode += reward
            reward_h += reward
            
            if done:
                # Store final high-level transition
                r_discounted = reward_h * (config_alg['gamma'] ** (step_episode % field_selection_steps))
                buf_field.add(np.array([
                    state_h, list_obs_h[0], field_selection_oh, r_discounted,
                    state, list_obs[0], done
                ]))
        
        # Train decoder if enough data collected
        if len(decoder_dataset) >= N_batch_decoder:
            t_alg_start = time.time()
            if summarize:
                expected_prob = alg.train_field_decoder(sess, decoder_dataset[:N_batch_decoder], 
                                                      step_train, summarize=True, writer=writer)
            else:
                expected_prob = alg.train_field_decoder(sess, decoder_dataset[:N_batch_decoder], 
                                                      step_train, summarize=False, writer=None)
            step_train += 1
            t_alg += time.time() - t_alg_start
            
            # Curriculum learning: increase fields if decoder is confident
            if expected_prob >= curriculum_threshold:
                N_fields_current = min(int(1.5 * N_fields_current + 1), N_fields)
                field_importance = np.ones(N_fields_current) / N_fields_current
            
            # Clear dataset
            decoder_dataset = []
        
        # Update exploration
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
        
        reward_period += reward_episode
        
        # Logging and evaluation
        if idx_episode == 1 or idx_episode % (5 * period) == 0:
            print('{:>10s}{:>10s}{:>12s}{:>10s}{:>8s}{:>10s}{:>8s}{:>10s}{:>8s}{:>8s}{:>12s}{:>12s}'.format(
                'Episode', 'Step', 'Step_train', 'N_fields', 'alpha', 'Exp_prob',
                'R_avg', 'Coverage', 'Crashes', 'Vulns', 'T_env', 'T_alg'))
        
        if idx_episode % period == 0:
            # Get environment statistics
            stats = env.get_statistics()
            coverage_rate = stats.get('coverage_diversity', 0) / max(1, stats.get('total_mutations', 1))
            crash_rate = stats.get('crashes_found', 0) / max(1, stats.get('total_mutations', 1))
            vuln_rate = stats.get('vulnerabilities_found', 0) / max(1, stats.get('total_mutations', 1))
            
            # Adjust alpha based on performance
            if crash_rate >= alpha_threshold:
                alpha = max(alpha_end, alpha - alpha_step)
            
            # Save model if performance is good
            if crash_rate >= config_main['save_threshold']:
                saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model_good.ckpt", idx_episode))
            
            # Log results
            s = '%d,%d,%d,%d,%.2f,%.3e,%.2f,%.4f,%.4f,%.4f,%.5e,%.5e\n' % (
                idx_episode, step, step_train, N_fields_current, alpha, expected_prob,
                reward_period / float(period), coverage_rate, crash_rate, vuln_rate, t_env, t_alg)
            
            with open('../results/%s/log.csv' % dir_name, 'a') as f:
                f.write(s)
            
            print('{:10d}{:10d}{:12d}{:10d}{:8.2f}{:10.3e}{:8.2f}{:10.4f}{:8.4f}{:8.4f}{:12.5e}{:12.5e}'.format(
                idx_episode, step, step_train, N_fields_current, alpha, expected_prob,
                reward_period / float(period), coverage_rate, crash_rate, vuln_rate, t_env, t_alg))
            
            reward_period = 0
        
        # Periodic model saving
        if idx_episode % save_period == 0:
            saver.save(sess, '../results/%s/%s-%d' % (dir_name, "model.ckpt", idx_episode))
    
    # Save final model
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    
    # Save timing information
    with open('../results/%s/time.txt' % dir_name, 'w') as f:
        f.write('t_env_total,t_env_per_step,t_alg_total,t_alg_per_step\n')
        f.write('%.5e,%.5e,%.5e,%.5e' % (t_env, t_env/step, t_alg, t_alg/step))
    
    print("Training completed. Results saved to ../results/%s/" % dir_name)


def evaluate_fuzzing_performance(alg, env, sess, N_eval=10):
    """Evaluate the trained fuzzing agent.
    
    Args:
        alg: Trained algorithm instance
        env: Fuzzing environment
        sess: TensorFlow session
        N_eval: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_coverage = 0
    total_crashes = 0
    total_vulnerabilities = 0
    total_rewards = 0
    total_steps = 0
    
    for episode in range(N_eval):
        state, _, list_obs, _, done = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while not done and episode_steps < 500:  # Limit evaluation episode length
            # Simple greedy policy for evaluation
            field_importance = np.ones(env.n_fields) / env.n_fields
            field_idx = alg.select_field(
                np.array([list_obs[0]]), 
                np.array([field_importance]), 
                epsilon=0.0, sess=sess)  # No exploration
            
            mutation_context = alg.create_mutation_context(field_idx, {}, {})
            field_selection_oh = np.zeros(env.n_fields)
            field_selection_oh[field_idx] = 1
            
            mutation_idx = alg.select_mutation_action(
                np.array([list_obs[0]]),
                np.array([mutation_context]),
                np.array([field_selection_oh]),
                epsilon=0.0, sess=sess)  # No exploration
            
            state, _, list_obs, _, reward, _, done, info = env.step(field_idx, mutation_idx)
            episode_reward += reward
            episode_steps += 1
        
        # Collect statistics
        stats = env.get_statistics()
        total_coverage += stats.get('coverage_diversity', 0)
        total_crashes += stats.get('crashes_found', 0)
        total_vulnerabilities += stats.get('vulnerabilities_found', 0)
        total_rewards += episode_reward
        total_steps += episode_steps
    
    # Calculate averages
    eval_metrics = {
        'avg_coverage': total_coverage / N_eval,
        'avg_crashes': total_crashes / N_eval,
        'avg_vulnerabilities': total_vulnerabilities / N_eval,
        'avg_reward': total_rewards / N_eval,
        'avg_steps': total_steps / N_eval,
        'crash_rate': total_crashes / max(1, total_steps),
        'vulnerability_rate': total_vulnerabilities / max(1, total_steps)
    }
    
    return eval_metrics


if __name__ == '__main__':
    # Load configuration
    config_file = 'config_fuzzing.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("Starting fuzzing HSD training with configuration:", config_file)
    train_fuzzing_function(config)