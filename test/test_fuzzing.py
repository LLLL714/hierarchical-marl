"""Test script for fuzzing hierarchical reinforcement learning.

This script tests the fuzzing HSD implementation by:
- Loading trained models
- Running fuzzing episodes
- Evaluating coverage, crash detection, and vulnerability discovery
- Analyzing field selection and mutation strategies
"""

import json
import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append('../env/')
sys.path.append('../alg/')

from alg_fuzzing_hsd import AlgFuzzingHSD
from env.fuzzing_environment import FuzzingEnvironment


def setup_test_logging(log_dir):
    """Setup logging for testing."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'testing.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class FuzzingTestAnalyzer:
    """Analyzer for fuzzing test results."""
    
    def __init__(self):
        self.episode_rewards = []
        self.coverage_history = []
        self.crash_counts = []
        self.vulnerability_counts = []
        self.field_selection_counts = defaultdict(int)
        self.mutation_action_counts = defaultdict(int)
        self.execution_times = []
        
    def record_episode(self, episode_data):
        """Record data from a test episode."""
        self.episode_rewards.append(episode_data['total_reward'])
        self.coverage_history.append(episode_data['total_coverage'])
        self.crash_counts.append(episode_data['crashes_found'])
        self.vulnerability_counts.append(episode_data['vulnerabilities_found'])
        self.execution_times.append(episode_data['execution_time'])
        
        # Record field selections
        for field_idx in episode_data['field_selections']:
            self.field_selection_counts[field_idx] += 1
            
        # Record mutation actions
        for action in episode_data['mutation_actions']:
            self.mutation_action_counts[action] += 1
    
    def generate_report(self, output_dir):
        """Generate comprehensive test report."""
        report = {
            'summary': self._generate_summary(),
            'performance': self._generate_performance_metrics(),
            'strategy_analysis': self._generate_strategy_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(os.path.join(output_dir, 'test_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate plots
        self._generate_plots(output_dir)
        
        return report
    
    def _generate_summary(self):
        """Generate summary statistics."""
        if not self.episode_rewards:
            return {"error": "No episodes recorded"}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'total_coverage': np.sum(self.coverage_history),
            'total_crashes': np.sum(self.crash_counts),
            'total_vulnerabilities': np.sum(self.vulnerability_counts),
            'average_execution_time': np.mean(self.execution_times)
        }
    
    def _generate_performance_metrics(self):
        """Generate performance metrics."""
        return {
            'reward_statistics': {
                'mean': np.mean(self.episode_rewards),
                'std': np.std(self.episode_rewards),
                'min': np.min(self.episode_rewards),
                'max': np.max(self.episode_rewards)
            },
            'coverage_rate': np.mean(self.coverage_history),
            'crash_discovery_rate': np.mean([1 if c > 0 else 0 for c in self.crash_counts]),
            'vulnerability_discovery_rate': np.mean([1 if v > 0 else 0 for v in self.vulnerability_counts]),
            'efficiency_metrics': {
                'crashes_per_episode': np.mean(self.crash_counts),
                'vulnerabilities_per_episode': np.mean(self.vulnerability_counts),
                'coverage_per_second': np.sum(self.coverage_history) / np.sum(self.execution_times) if self.execution_times else 0
            }
        }
    
    def _generate_strategy_analysis(self):
        """Analyze fuzzing strategies."""
        total_selections = sum(self.field_selection_counts.values())
        total_actions = sum(self.mutation_action_counts.values())
        
        field_distribution = {
            str(k): v / total_selections if total_selections > 0 else 0 
            for k, v in self.field_selection_counts.items()
        }
        
        action_distribution = {
            str(k): v / total_actions if total_actions > 0 else 0 
            for k, v in self.mutation_action_counts.items()
        }
        
        return {
            'field_selection_distribution': field_distribution,
            'mutation_action_distribution': action_distribution,
            'strategy_diversity': {
                'unique_fields_used': len(self.field_selection_counts),
                'unique_actions_used': len(self.mutation_action_counts),
                'field_entropy': self._calculate_entropy(list(field_distribution.values())),
                'action_entropy': self._calculate_entropy(list(action_distribution.values()))
            }
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        avg_reward = np.mean(self.episode_rewards)
        if avg_reward < 1.0:
            recommendations.append("Consider adjusting reward weights to improve learning signal")
        
        # Coverage recommendations
        coverage_rate = np.mean(self.coverage_history)
        if coverage_rate < 0.1:
            recommendations.append("Low coverage rate detected - consider improving exploration strategy")
        
        # Strategy recommendations
        field_entropy = self._calculate_entropy(list(self.field_selection_counts.values()))
        if field_entropy < 1.0:
            recommendations.append("Low field selection diversity - consider increasing exploration")
        
        return recommendations
    
    def _calculate_entropy(self, probabilities):
        """Calculate Shannon entropy."""
        probabilities = [p for p in probabilities if p > 0]
        if not probabilities:
            return 0.0
        return -sum(p * np.log2(p) for p in probabilities)
    
    def _generate_plots(self, output_dir):
        """Generate visualization plots."""
        try:
            # Reward progression plot
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            # Coverage progression
            plt.subplot(2, 2, 2)
            plt.plot(np.cumsum(self.coverage_history))
            plt.title('Cumulative Coverage')
            plt.xlabel('Episode')
            plt.ylabel('Coverage')
            
            # Crash/vulnerability discovery
            plt.subplot(2, 2, 3)
            plt.plot(np.cumsum(self.crash_counts), label='Crashes')
            plt.plot(np.cumsum(self.vulnerability_counts), label='Vulnerabilities')
            plt.title('Security Issues Discovered')
            plt.xlabel('Episode')
            plt.ylabel('Count')
            plt.legend()
            
            # Field selection distribution
            plt.subplot(2, 2, 4)
            if self.field_selection_counts:
                fields, counts = zip(*self.field_selection_counts.items())
                plt.bar(range(len(fields)), counts)
                plt.title('Field Selection Distribution')
                plt.xlabel('Field Index')
                plt.ylabel('Selection Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'test_analysis.png'))
            plt.close()
            
        except Exception as e:
            logging.warning(f"Could not generate plots: {e}")


def load_fuzzing_config(config_path):
    """Load and prepare configuration for testing."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add fuzzing configuration if not present
    if 'fuzzing' not in config:
        config['fuzzing'] = {
            'protocol_fields': ['header', 'source_addr', 'dest_addr', 'payload', 'checksum', 'flags'],
            'mutation_actions': ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy'],
            'field_selection_reward_ratio': 0.9,
            'max_mutation_size': 1024,
            'mutation_history_length': 10,
            'max_episode_steps': 1000,
            'mamba_feature_dim': 256,
            'field_context_dim': 32,
            'coverage_feature_dim': 64
        }
    
    if 'rewards' not in config:
        config['rewards'] = {
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
    
    if 'nn_fuzzing_hsd' not in config:
        config['nn_fuzzing_hsd'] = {
            'n_h_decoder': 128,
            'n_h1_low': 64,
            'n_h2_low': 64,
            'n_h1': 128,
            'n_h2': 128,
            'n_h_mixer': 64
        }
    
    return config


def test_fuzzing_hsd(config, model_path, n_test_episodes=100):
    """Test the fuzzing HSD algorithm."""
    
    # Extract configuration
    config_main = config['main']
    config_alg = config['alg']
    config_h = config['h_params']
    config_fuzzing = config['fuzzing']
    config_rewards = config['rewards']
    
    # Setup test directory
    test_dir = f"../results/{config_main['dir_name']}/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_test_logging(test_dir)
    logger.info(f"Starting fuzzing HSD test with {n_test_episodes} episodes")
    
    # Environment setup
    env_config = {**config.get('env', {}), **config_fuzzing, **config_rewards}
    env = FuzzingEnvironment(env_config)
    
    # Get dimensions
    n_fields = len(config_fuzzing['protocol_fields'])
    l_state = env.state_dim
    l_obs = env.obs_dim
    l_mutation_actions = env.action_dim
    
    # Create algorithm
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
    
    # Load model
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Initialize with random weights for testing
        sess.run(tf.global_variables_initializer())
        logger.warning("Using randomly initialized model")
    
    # Test analyzer
    analyzer = FuzzingTestAnalyzer()
    
    # Test parameters
    render = config_main.get('render', False)
    epsilon = 0.05  # Small exploration for testing
    field_selection_steps = config_h['steps_per_assign']
    
    logger.info("Starting test episodes")
    
    for episode in range(n_test_episodes):
        
        # Episode tracking
        episode_start_time = time.time()
        episode_reward = 0
        episode_coverage = 0
        episode_crashes = 0
        episode_vulnerabilities = 0
        episode_field_selections = []
        episode_mutation_actions = []
        
        # Reset environment
        state, _, list_obs, _, done = env.reset()
        step_episode = 0
        
        # Initial field selection
        field_indices = alg.assign_fields(list_obs, epsilon, sess, env.field_importance_scores)
        field_selections = np.zeros([n_fields, n_fields])
        field_selections[np.arange(n_fields), field_indices] = 1
        
        episode_field_selections.extend(field_indices)
        
        while not done:
            
            # Field selection every N steps
            if step_episode % field_selection_steps == 0 and step_episode > 0:
                field_indices = alg.assign_fields(list_obs, epsilon, sess, env.field_importance_scores)
                field_selections = np.zeros([n_fields, n_fields])
                field_selections[np.arange(n_fields), field_indices] = 1
                episode_field_selections.extend(field_indices)
            
            # Mutation action selection
            mutation_actions = alg.run_mutation_actor(list_obs, field_selections, epsilon, sess)
            episode_mutation_actions.extend(mutation_actions)
            
            # Execute actions
            state_next, _, list_obs_next, _, reward, local_rewards, done, info = env.step(
                mutation_actions, field_indices)
            
            # Update episode metrics
            episode_reward += reward
            execution_result = info.get('execution_result', {})
            episode_coverage += execution_result.get('coverage_increase', 0)
            
            if execution_result.get('crash_detected', False):
                episode_crashes += 1
            if execution_result.get('vulnerability_found', False):
                episode_vulnerabilities += 1
            
            # Update state
            state = state_next
            list_obs = list_obs_next
            step_episode += 1
            
            # Safety limit
            if step_episode >= 1000:
                break
        
        # Record episode data
        episode_data = {
            'episode': episode,
            'total_reward': episode_reward,
            'total_coverage': episode_coverage,
            'crashes_found': episode_crashes,
            'vulnerabilities_found': episode_vulnerabilities,
            'field_selections': episode_field_selections,
            'mutation_actions': episode_mutation_actions,
            'execution_time': time.time() - episode_start_time,
            'steps': step_episode
        }
        
        analyzer.record_episode(episode_data)
        
        # Progress logging
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{n_test_episodes} - "
                       f"Reward: {episode_reward:.2f}, Coverage: {episode_coverage:.4f}, "
                       f"Crashes: {episode_crashes}, Vulnerabilities: {episode_vulnerabilities}")
    
    # Generate comprehensive report
    logger.info("Generating test report")
    report = analyzer.generate_report(test_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("FUZZING HSD TEST SUMMARY")
    print("="*60)
    
    summary = report['summary']
    performance = report['performance']
    
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"Average Reward: {summary['average_reward']:.4f}")
    print(f"Total Coverage: {summary['total_coverage']:.4f}")
    print(f"Total Crashes: {summary['total_crashes']}")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"Average Execution Time: {summary['average_execution_time']:.2f}s")
    print()
    
    print("PERFORMANCE METRICS:")
    print(f"Crash Discovery Rate: {performance['crash_discovery_rate']:.2%}")
    print(f"Vulnerability Discovery Rate: {performance['vulnerability_discovery_rate']:.2%}")
    print(f"Coverage per Second: {performance['efficiency_metrics']['coverage_per_second']:.6f}")
    print()
    
    strategy = report['strategy_analysis']
    print("STRATEGY ANALYSIS:")
    print(f"Field Selection Entropy: {strategy['strategy_diversity']['field_entropy']:.3f}")
    print(f"Action Selection Entropy: {strategy['strategy_diversity']['action_entropy']:.3f}")
    print()
    
    if report['recommendations']:
        print("RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    print("="*60)
    
    logger.info(f"Test completed. Report saved to {test_dir}")
    
    return report


if __name__ == '__main__':
    
    # Configuration
    config_path = '../alg/config.json'
    
    # Check if config exists, if not use default
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using default configuration")
        config = {
            'main': {
                'dir_name': 'fuzzing_hsd_test',
                'model_name': 'model.ckpt',
                'render': False
            },
            'alg': {
                'lr_Q': 1e-4,
                'lr_actor': 1e-4,
                'lr_decoder': 1e-4,
                'gamma': 0.99,
                'tau': 0.01
            },
            'h_params': {
                'steps_per_assign': 10,
                'traj_skip': 2,
                'obs_truncate_length': 10,
                'use_state_difference': True
            }
        }
    else:
        config = load_fuzzing_config(config_path)
    
    # Model path
    model_dir = f"../results/{config['main']['dir_name']}"
    model_path = f"{model_dir}/{config['main']['model_name']}"
    
    # If model doesn't exist, create a test model path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Testing with randomly initialized weights")
        model_path = None
    
    # Number of test episodes
    n_test_episodes = 50
    
    # Run test
    test_report = test_fuzzing_hsd(config, model_path, n_test_episodes)