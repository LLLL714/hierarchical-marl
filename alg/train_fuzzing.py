"""Training script for Fuzzing Hierarchical Skill Discovery (HSD).

This script trains the fuzzing adaptation of HSD for protocol testing.
Supports hierarchical field selection and mutation action learning.
"""

import json
import os
import random
import sys
import time
import numpy as np

# Add environment path
sys.path.append('../env/')

try:
    import tensorflow as tf
except ImportError:
    print("Warning: TensorFlow not available. Using dummy placeholders.")
    # Create dummy tf module for testing
    class DummyTF:
        def __init__(self):
            pass
        def set_random_seed(self, seed):
            pass
        def Session(self, config=None):
            return DummySession()
        def global_variables_initializer(self):
            return None
        def train(self):
            return DummyTrain()
        def ConfigProto(self):
            return DummyConfig()
        def summary(self):
            return DummySummary()
    
    class DummySession:
        def run(self, *args, **kwargs):
            return None
    
    class DummyTrain:
        def Saver(self, **kwargs):
            return DummySaver()
    
    class DummySaver:
        def save(self, *args, **kwargs):
            pass
    
    class DummyConfig:
        def __init__(self):
            self.gpu_options = DummyGPU()
    
    class DummyGPU:
        def __init__(self):
            self.allow_growth = True
    
    class DummySummary:
        def FileWriter(self, *args, **kwargs):
            return DummyWriter()
    
    class DummyWriter:
        def add_summary(self, *args, **kwargs):
            pass
    
    tf = DummyTF()

import alg_fuzzing_hsd
import fuzzing_environment
import replay_buffer


def train_fuzzing_hsd(config):
    """Main training function for fuzzing HSD.
    
    Args:
        config: Configuration dictionary
    """
    config_env = config['env']
    config_main = config['main']
    config_alg = config['alg']
    config_fuzzing = config['fuzzing_params']
    
    # Set random seeds
    seed = config_main['seed']
    np.random.seed(seed)
    random.seed(seed)
    try:
        tf.set_random_seed(seed)
    except:
        pass
    
    # Extract configuration
    alg_name = config_main['alg_name']
    dir_name = config_main['dir_name']
    model_name = config_main['model_name']
    summarize = config_main['summarize']
    save_period = config_main['save_period']
    
    # Create results directory
    os.makedirs('../results/%s' % dir_name, exist_ok=True)
    with open('../results/%s/config.json' % dir_name, 'w') as f:
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
    field_selection_steps = config_fuzzing['field_selection_steps']
    exploit_ratio = config_fuzzing['exploit_ratio']
    
    # Curriculum learning
    n_fields_current = min(config_fuzzing.get('field_curriculum_start', 3), n_fields)
    field_curriculum_threshold = config_fuzzing.get('field_curriculum_threshold', 0.85)
    
    # Initialize fuzzing environment
    env = fuzzing_environment.FuzzingEnvironment(config_fuzzing)
    
    # Environment dimensions
    l_state = env.state_dim
    l_obs = env.state_dim  # Same as state for simplicity
    l_mutation_actions = len(config_fuzzing['mutation_actions'])
    
    print(f"Environment initialized:")
    print(f"  State dimension: {l_state}")
    print(f"  Observation dimension: {l_obs}")
    print(f"  Number of fields: {n_fields}")
    print(f"  Number of mutation actions: {l_mutation_actions}")
    
    # Initialize algorithm
    try:
        alg = alg_fuzzing_hsd.FuzzingHSDAlg(
            config_alg, config_fuzzing, n_fields, l_state, l_obs, l_mutation_actions, config['nn_fuzzing_hsd'])
    except Exception as e:
        print(f"Warning: Could not initialize TensorFlow algorithm: {e}")
        print("Running in dummy mode for testing...")
        alg = DummyAlgorithm()
    
    # TensorFlow session (if available)
    try:
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        sess = tf.Session(config=config_proto)
        sess.run(tf.global_variables_initializer())
        sess.run(alg.list_initialize_target_ops)
        tf_available = True
    except:
        print("TensorFlow session not available, using dummy session")
        sess = DummySession()
        tf_available = False
    
    # Initialize summaries and saver
    if tf_available and summarize:
        try:
            writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
        except:
            writer = None
    else:
        writer = None
    
    try:
        saver = tf.train.Saver(max_to_keep=config_main['max_to_keep'])
    except:
        saver = DummySaver()
    
    # Replay buffers
    buf_high = replay_buffer.Replay_Buffer(size=buffer_size)  # High-level field selection
    buf_low = replay_buffer.Replay_Buffer(size=buffer_size)   # Low-level mutation actions
    
    # Dataset for decoder training
    decoder_dataset = []
    
    # Initialize tracking variables
    field_importance_history = np.ones(n_fields) / n_fields
    coverage_tracker = set()
    
    # Logging
    header = "Episode,Step,Step_train,N_fields,Epsilon,Coverage,Crashes,Vulnerabilities,R_avg,R_eval,T_env,T_alg\n"
    with open("../results/%s/log.csv" % dir_name, 'w') as f:
        f.write(header)
    
    # Training loop variables
    t_env = 0
    t_alg = 0
    step = 0
    step_train = 0
    step_h = 0
    
    episode_rewards = []
    coverage_increases = []
    crashes_found = []
    vulnerabilities_found = []
    
    print(f"Starting training for {N_train} episodes...")
    
    for idx_episode in range(1, N_train + 1):
        # Reset environment with new protocol data
        state, info = env.reset()
        
        # Episode tracking
        episode_reward = 0
        episode_coverage = 0
        episode_crashes = 0
        episode_vulnerabilities = 0
        episode_steps = 0
        
        # High-level variables
        state_h = state.copy()
        field_selection_period = 0
        current_field_idx = 0
        
        # Trajectory for decoder
        field_trajectory = []
        
        done = False
        summarized = False
        summarized_h = False
        
        while not done and episode_steps < config_env['max_steps']:
            t_env_start = time.time()
            
            # High-level action: field selection
            if field_selection_period % field_selection_steps == 0:
                if field_selection_period > 0:
                    # Store high-level transition
                    field_action_1hot = np.zeros(n_fields)
                    field_action_1hot[current_field_idx] = 1
                    
                    # Dummy Mamba features for now
                    mamba_features = np.random.normal(0, 0.1, config_fuzzing['mamba_feature_dim'])
                    
                    if tf_available:
                        high_level_transition = [
                            state_h, state_h, mamba_features, field_action_1hot,
                            episode_reward, state, state, mamba_features, done
                        ]
                    else:
                        high_level_transition = [state_h, current_field_idx, episode_reward, state, done]
                    
                    buf_high.add(np.array(high_level_transition, dtype=object))
                    
                    # Add to decoder dataset
                    if len(field_trajectory) >= field_selection_steps:
                        field_onehot = np.zeros(n_fields)
                        field_onehot[current_field_idx] = 1
                        decoder_dataset.append([
                            np.array(field_trajectory[-field_selection_steps:]),
                            field_onehot
                        ])
                
                step_h += 1
                
                # Select new field
                if idx_episode < pretrain_episodes:
                    current_field_idx = np.random.randint(0, n_fields_current)
                else:
                    # Use 90/10 strategy from environment
                    field_probs = env.get_field_selection_probabilities()
                    current_field_idx = np.random.choice(n_fields, p=field_probs)
                    current_field_idx = min(current_field_idx, n_fields_current - 1)
                
                # Update field importance
                env.field_importance_history = field_importance_history
                
                # Reset for new field selection period
                state_h = state.copy()
                field_selection_period = 0
                episode_reward = 0
            
            # Low-level action: mutation selection
            field_features = env.get_mutation_action_features(current_field_idx)
            mutation_history = env._get_mutation_history_for_field_type('unknown')
            
            if idx_episode < pretrain_episodes:
                mutation_action = np.random.randint(0, l_mutation_actions)
            else:
                if tf_available:
                    try:
                        # Use algorithm to select mutation action
                        obs_batch = np.array([state])
                        mamba_batch = np.array([np.random.normal(0, 0.1, config_fuzzing['mamba_feature_dim'])])
                        field_batch = np.array([field_features])
                        history_batch = np.array([mutation_history])
                        
                        mutation_action = alg.select_mutation_action(
                            obs_batch, mamba_batch, field_batch, history_batch, epsilon, sess)
                    except Exception as e:
                        print(f"Warning: Algorithm selection failed: {e}")
                        mutation_action = np.random.randint(0, l_mutation_actions)
                else:
                    # Simple heuristic selection when TF not available
                    mutation_action = np.argmax(mutation_history + np.random.normal(0, 0.1, l_mutation_actions))
                    mutation_action = max(0, min(mutation_action, l_mutation_actions - 1))
            
            # Execute environment step
            next_state, reward, done, step_info = env.step(current_field_idx, mutation_action)
            t_env += time.time() - t_env_start
            
            # Track episode metrics
            episode_reward += reward
            episode_coverage += step_info.get('coverage_increase', 0)
            if step_info.get('crash_detected', False):
                episode_crashes += 1
            if step_info.get('vulnerability_found', False):
                episode_vulnerabilities += 1
            
            # Update field importance
            if current_field_idx < len(field_importance_history):
                alpha = 0.1
                field_importance_history[current_field_idx] = (
                    (1 - alpha) * field_importance_history[current_field_idx] + 
                    alpha * max(0, reward))
            
            # Store low-level transition
            mutation_action_1hot = np.zeros(l_mutation_actions)
            mutation_action_1hot[mutation_action] = 1
            
            if tf_available:
                low_level_transition = [
                    state, np.random.normal(0, 0.1, config_fuzzing['mamba_feature_dim']),
                    field_features, mutation_action_1hot, reward,
                    next_state, np.random.normal(0, 0.1, config_fuzzing['mamba_feature_dim']),
                    field_features, mutation_history, done
                ]
            else:
                low_level_transition = [state, mutation_action, reward, next_state, done]
            
            buf_low.add(np.array(low_level_transition, dtype=object))
            
            # Add to field trajectory
            field_trajectory.append(state.copy())
            if len(field_trajectory) > field_selection_steps * 2:
                field_trajectory = field_trajectory[-field_selection_steps:]
            
            # Training
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                if buf_high.size() >= batch_size:
                    batch_high = buf_high.sample_batch(batch_size)
                    t_alg_start = time.time()
                    try:
                        if tf_available and summarize and idx_episode % period == 0 and not summarized_h:
                            alg.train_field_selection(sess, batch_high, step_train, summarize=True, writer=writer)
                            summarized_h = True
                        elif tf_available:
                            alg.train_field_selection(sess, batch_high, step_train, summarize=False, writer=None)
                    except Exception as e:
                        print(f"Warning: High-level training failed: {e}")
                    t_alg += time.time() - t_alg_start
                
                if buf_low.size() >= batch_size:
                    batch_low = buf_low.sample_batch(batch_size)
                    t_alg_start = time.time()
                    try:
                        if tf_available and summarize and idx_episode % period == 0 and not summarized:
                            alg.train_mutation_actions(sess, batch_low, step_train, summarize=True, writer=writer)
                            summarized = True
                        elif tf_available:
                            alg.train_mutation_actions(sess, batch_low, step_train, summarize=False, writer=None)
                    except Exception as e:
                        print(f"Warning: Low-level training failed: {e}")
                    t_alg += time.time() - t_alg_start
                
                step_train += 1
            
            # Update state
            state = next_state
            step += 1
            episode_steps += 1
            field_selection_period += 1
        
        # End of episode processing
        if done and field_selection_period > 0:
            # Final high-level transition
            field_action_1hot = np.zeros(n_fields)
            field_action_1hot[current_field_idx] = 1
            mamba_features = np.random.normal(0, 0.1, config_fuzzing['mamba_feature_dim'])
            
            if tf_available:
                final_transition = [
                    state_h, state_h, mamba_features, field_action_1hot,
                    episode_reward, state, state, mamba_features, done
                ]
            else:
                final_transition = [state_h, current_field_idx, episode_reward, state, done]
            
            buf_high.add(np.array(final_transition, dtype=object))
        
        # Decoder training
        if len(decoder_dataset) >= config_fuzzing.get('decoder_batch_size', 100):
            t_alg_start = time.time()
            try:
                if tf_available:
                    expected_prob = alg.train_decoder(sess, decoder_dataset[:100], step_train, 
                                                    summarize=(summarize and idx_episode % period == 0), 
                                                    writer=writer)
                    # Check curriculum advancement
                    if expected_prob >= field_curriculum_threshold:
                        n_fields_current = min(n_fields_current + 1, n_fields)
                else:
                    expected_prob = 0.5  # Dummy value
            except Exception as e:
                print(f"Warning: Decoder training failed: {e}")
                expected_prob = 0.5
            
            t_alg += time.time() - t_alg_start
            decoder_dataset = []  # Reset dataset
            step_train += 1
        else:
            expected_prob = 0.0
        
        # Update exploration
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        coverage_increases.append(episode_coverage)
        crashes_found.append(episode_crashes)
        vulnerabilities_found.append(episode_vulnerabilities)
        
        # Periodic evaluation and logging
        if idx_episode == 1 or idx_episode % (5 * period) == 0:
            print(f"{'Episode':>10}{'Step':>10}{'Train':>10}{'Fields':>8}{'Epsilon':>10}{'Coverage':>12}{'Crashes':>10}{'Vulns':>8}{'R_avg':>10}{'T_env':>12}")
        
        if idx_episode % period == 0:
            # Calculate averages
            recent_rewards = episode_rewards[-period:] if len(episode_rewards) >= period else episode_rewards
            recent_coverage = coverage_increases[-period:] if len(coverage_increases) >= period else coverage_increases
            recent_crashes = crashes_found[-period:] if len(crashes_found) >= period else crashes_found
            recent_vulns = vulnerabilities_found[-period:] if len(vulnerabilities_found) >= period else vulnerabilities_found
            
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            total_coverage = np.sum(recent_coverage) if recent_coverage else 0
            total_crashes = np.sum(recent_crashes) if recent_crashes else 0
            total_vulns = np.sum(recent_vulns) if recent_vulns else 0
            
            # Evaluation (simplified)
            eval_reward = avg_reward  # In real implementation, run separate evaluation episodes
            
            # Save model if performance threshold met
            if total_vulns > 0 or total_crashes >= 5:
                try:
                    saver.save(sess, '../results/%s/model_good.ckpt-%d' % (dir_name, idx_episode))
                except:
                    pass
            
            # Log results
            log_entry = f"{idx_episode},{step},{step_train},{n_fields_current},{epsilon:.3f},{total_coverage:.2f},{total_crashes},{total_vulns},{avg_reward:.3f},{eval_reward:.3f},{t_env:.3e},{t_alg:.3e}\n"
            with open('../results/%s/log.csv' % dir_name, 'a') as f:
                f.write(log_entry)
            
            print(f"{idx_episode:10d}{step:10d}{step_train:10d}{n_fields_current:8d}{epsilon:10.3f}{total_coverage:12.2f}{total_crashes:10d}{total_vulns:8d}{avg_reward:10.3f}{t_env:12.3e}")
        
        # Periodic model saving
        if idx_episode % save_period == 0:
            try:
                saver.save(sess, '../results/%s/model.ckpt-%d' % (dir_name, idx_episode))
            except:
                pass
    
    # Final model save
    try:
        saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    except:
        pass
    
    # Save timing information
    with open('../results/%s/time.txt' % dir_name, 'w') as f:
        f.write('t_env_total,t_env_per_step,t_alg_total,t_alg_per_step\n')
        f.write('%.5e,%.5e,%.5e,%.5e' % (t_env, t_env/max(step, 1), t_alg, t_alg/max(step_train, 1)))
    
    print(f"Training completed!")
    print(f"Total episodes: {N_train}")
    print(f"Total steps: {step}")
    print(f"Total training steps: {step_train}")
    print(f"Environment time: {t_env:.3f}s")
    print(f"Algorithm time: {t_alg:.3f}s")


class DummyAlgorithm:
    """Dummy algorithm for testing when TensorFlow is not available."""
    
    def __init__(self):
        self.list_initialize_target_ops = []
    
    def select_field(self, *args, **kwargs):
        return 0
    
    def select_mutation_action(self, *args, **kwargs):
        return 0
    
    def train_field_selection(self, *args, **kwargs):
        pass
    
    def train_mutation_actions(self, *args, **kwargs):
        pass
    
    def train_decoder(self, *args, **kwargs):
        return 0.5


if __name__ == '__main__':
    # Load configuration
    config_file = 'config_fuzzing.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found!")
        print("Using default configuration...")
        config = {
            "main": {"seed": 12345, "dir_name": "fuzzing_test", "alg_name": "fuzzing_hsd", 
                    "model_name": "model.ckpt", "summarize": False, "save_period": 1000},
            "alg": {"N_train": 1000, "N_eval": 10, "period": 100, "epsilon_start": 0.5, 
                   "epsilon_end": 0.05, "epsilon_div": 500, "buffer_size": 10000, 
                   "batch_size": 64, "pretrain_episodes": 10, "steps_per_train": 5},
            "fuzzing_params": {"protocol_fields": [{"name": "test", "type": "header"}] * 4,
                              "mutation_actions": ["insert", "delete", "flip", "replace"],
                              "field_selection_steps": 10, "exploit_ratio": 0.9,
                              "mamba_feature_dim": 64, "use_mamba_features": False},
            "env": {"max_steps": 100},
            "nn_fuzzing_hsd": {"n_h_decoder": 64, "n_h1_low": 32, "n_h2_low": 32,
                              "n_h1": 64, "n_h2": 64, "n_h_mixer": 32}
        }
    
    print("Starting Fuzzing HSD Training...")
    print(f"Configuration: {config_file}")
    
    train_fuzzing_hsd(config)