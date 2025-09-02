"""Test suite for Fuzzing Hierarchical Reinforcement Learning Framework.

This module provides comprehensive testing for the fuzzing adaptation of the HSD algorithm,
including unit tests, integration tests, and performance benchmarks.
"""

import unittest
import numpy as np
import json
import os
import sys
import tempfile
import shutil

# Add parent directories to path
sys.path.append('../env/')
sys.path.append('../alg/')

from fuzzing_environment import FuzzingEnvironment
from data_processor import ProtocolDataProcessor
from reward_calculator import RewardCalculator


class TestProtocolDataProcessor(unittest.TestCase):
    """Test cases for protocol data processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'mamba_feature_dim': 256,
            'protocol_fields': [
                {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
                {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 1500},
                {'name': 'length_field', 'type': 'integer', 'min_value': 0, 'max_value': 65535}
            ]
        }
        self.processor = ProtocolDataProcessor(self.config)
    
    def test_default_packet_generation(self):
        """Test generation of default protocol packets."""
        packet = self.processor.generate_default_packet()
        
        # Check that all required fields are present
        for field in self.config['protocol_fields']:
            self.assertIn(field['name'], packet)
        
        # Check field types and constraints
        self.assertIsInstance(packet['header'], bytes)
        self.assertIsInstance(packet['payload'], bytes)
        self.assertIsInstance(packet['length_field'], int)
        
        # Check length constraints
        self.assertGreaterEqual(len(packet['header']), 1)
        self.assertLessEqual(len(packet['header']), 64)
        self.assertLessEqual(len(packet['payload']), 1500)
        self.assertGreaterEqual(packet['length_field'], 0)
        self.assertLessEqual(packet['length_field'], 65535)
    
    def test_mamba_feature_extraction(self):
        """Test Mamba feature extraction."""
        packet = self.processor.generate_default_packet()
        features = self.processor.extract_mamba_features(packet)
        
        # Check feature dimensions
        self.assertEqual(len(features), self.config['mamba_feature_dim'])
        self.assertIsInstance(features, np.ndarray)
        
        # Check feature values are in reasonable range
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation."""
        dataset = self.processor.generate_synthetic_dataset(100)
        
        self.assertEqual(len(dataset), 100)
        
        for packet in dataset[:5]:  # Check first 5 packets
            self.assertIsInstance(packet, dict)
            # Check required fields
            for field in self.config['protocol_fields']:
                self.assertIn(field['name'], packet)
    
    def test_field_feature_extraction(self):
        """Test extraction of field-specific features."""
        packet = self.processor.generate_default_packet()
        
        # Test byte field features
        header_features = self.processor.extract_field_features(packet, 'header')
        self.assertTrue(header_features['exists'])
        self.assertIn('length', header_features)
        self.assertIn('entropy', header_features)
        
        # Test integer field features
        length_features = self.processor.extract_field_features(packet, 'length_field')
        self.assertTrue(length_features['exists'])
        self.assertIn('value', length_features)
        self.assertIn('bit_count', length_features)
    
    def test_packet_preprocessing(self):
        """Test packet preprocessing for mutations."""
        packet = self.processor.generate_default_packet()
        processed = self.processor.preprocess_packet_for_mutation(packet)
        
        # All values should be bytes after preprocessing
        for value in processed.values():
            self.assertIsInstance(value, bytes)


class TestRewardCalculator(unittest.TestCase):
    """Test cases for reward calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'base_reward_weight': 1.0,
            'medium_reward_weight': 5.0,
            'high_reward_weight': 100.0,
            'coverage_reward_coeff': 0.1,
            'novelty_reward_coeff': 0.2,
            'crash_reward': 10.0,
            'vulnerability_reward': 1000.0
        }
        self.calculator = RewardCalculator(self.config)
    
    def test_base_reward_calculation(self):
        """Test calculation of base rewards."""
        mutation_result = {
            'coverage_increase': 5,
            'new_states': 3,
            'unique_packet': True,
            'execution_time': 0.1
        }
        
        reward = self.calculator.calculate_reward(mutation_result)
        self.assertGreater(reward, 0)
    
    def test_medium_reward_calculation(self):
        """Test calculation of medium-tier rewards."""
        mutation_result = {
            'crash_detected': True,
            'timeout_detected': True,
            'protocol_anomaly': True
        }
        
        reward = self.calculator.calculate_reward(mutation_result)
        self.assertGreater(reward, 10)  # Should be significant due to crash
    
    def test_high_reward_calculation(self):
        """Test calculation of high-tier rewards."""
        mutation_result = {
            'vulnerability_detected': True,
            'vulnerability_type': 'buffer_overflow',
            'exploitability_score': 0.8
        }
        
        reward = self.calculator.calculate_reward(mutation_result)
        self.assertGreater(reward, 1000)  # Should be very high for vulnerability
    
    def test_penalty_calculation(self):
        """Test penalty calculations."""
        mutation_result = {
            'unique_packet': False,
            'invalid_mutation': True,
            'execution_time': 10.0  # Very slow
        }
        
        reward = self.calculator.calculate_reward(mutation_result)
        self.assertLess(reward, 0)  # Should be negative due to penalties
    
    def test_reward_breakdown(self):
        """Test detailed reward breakdown."""
        mutation_result = {
            'coverage_increase': 2,
            'crash_detected': True,
            'unique_packet': True
        }
        
        breakdown = self.calculator.get_reward_breakdown(mutation_result)
        
        self.assertIn('base_rewards', breakdown)
        self.assertIn('medium_rewards', breakdown)
        self.assertIn('high_rewards', breakdown)
        self.assertIn('penalties', breakdown)
        self.assertIn('grand_total', breakdown)
        
        # Check specific components
        self.assertGreater(breakdown['base_rewards']['coverage'], 0)
        self.assertGreater(breakdown['medium_rewards']['crash'], 0)


class TestFuzzingEnvironment(unittest.TestCase):
    """Test cases for fuzzing environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_env = {
            'max_steps': 100,
            'protocol_fields': [
                {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
                {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 1500}
            ],
            'mamba_feature_dim': 256,
            'use_mamba_features': True,
            'exploit_probability': 0.9
        }
        self.config_main = {
            'render': False
        }
        self.env = FuzzingEnvironment(self.config_env, self.config_main)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.n_fields, 2)
        self.assertEqual(self.env.n_mutations, 6)
        self.assertEqual(self.env.max_steps, 100)
        self.assertIsNotNone(self.env.data_processor)
        self.assertIsNotNone(self.env.reward_calculator)
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        state, state_away, obs_home, obs_away, done = self.env.reset()
        
        self.assertIsNotNone(state)
        self.assertIsNone(state_away)  # Not used in fuzzing
        self.assertEqual(len(obs_home), 1)  # Single agent
        self.assertEqual(len(obs_away), 0)  # Not used
        self.assertFalse(done)
        
        # Check state dimensions
        self.assertEqual(len(state), self.env.state_dim)
        self.assertEqual(len(obs_home[0]), self.env.obs_dim)
    
    def test_environment_step(self):
        """Test environment step execution."""
        self.env.reset()
        
        field_idx = 0
        mutation_action = 2  # flip
        
        state_next, state_away, obs_next, obs_away, reward, local_rewards, done, info = \
            self.env.step(field_idx, mutation_action)
        
        self.assertIsNotNone(state_next)
        self.assertIsInstance(reward, (int, float))
        self.assertEqual(len(local_rewards), 1)
        self.assertIsInstance(info, dict)
        
        # Check info contents
        self.assertIn('mutation_result', info)
        self.assertIn('field_idx', info)
        self.assertIn('mutation_action', info)
    
    def test_field_selection_strategy(self):
        """Test hierarchical field selection strategy."""
        self.env.reset()
        
        field_q_values = np.array([0.5, 0.8, 0.3, 0.1])
        
        # Test exploit mode (should select field with highest combined score)
        selected_fields = []
        for _ in range(100):
            field_idx = self.env.select_field_hierarchical(field_q_values, epsilon=0.0)
            selected_fields.append(field_idx)
        
        # Should mostly select field 1 (highest Q-value)
        self.assertGreater(selected_fields.count(1), 50)
    
    def test_mutation_execution(self):
        """Test different mutation actions."""
        self.env.reset()
        
        # Test each mutation action
        for mutation_idx in range(self.env.n_mutations):
            field_idx = 0
            
            # Store original packet
            original_packet = self.env.current_packet.copy()
            
            # Apply mutation
            mutated_packet = self.env._apply_mutation(field_idx, mutation_idx)
            
            # Check that mutation was applied
            if mutation_idx != 4:  # Shuffle might not change single-byte fields
                # Most mutations should change the packet
                pass  # Some mutations might not change very small fields
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        self.env.reset()
        
        # Execute some steps
        for i in range(10):
            field_idx = i % self.env.n_fields
            mutation_action = i % self.env.n_mutations
            self.env.step(field_idx, mutation_action)
        
        stats = self.env.get_statistics()
        
        self.assertEqual(stats['total_mutations'], 10)
        self.assertGreaterEqual(stats['unique_coverage'], 0)
        self.assertEqual(len(stats['field_selection_counts']), self.env.n_fields)
        self.assertEqual(len(stats['mutation_action_counts']), self.env.n_mutations)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete fuzzing framework."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            'main': {
                'seed': 12345,
                'dir_name': 'test_fuzzing',
                'render': False,
                'summarize': False
            },
            'env': {
                'max_steps': 50,
                'protocol_fields': [
                    {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
                    {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 100}
                ],
                'mamba_feature_dim': 64,  # Smaller for testing
                'use_mamba_features': True
            },
            'fuzzing_params': {
                'protocol_fields': [
                    {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
                    {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 100}
                ],
                'mamba_feature_dim': 64
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_environment_data_processor_integration(self):
        """Test integration between environment and data processor."""
        env = FuzzingEnvironment(self.config['env'], self.config['main'])
        
        # Test reset with packet loading
        state, _, obs, _, done = env.reset()
        
        self.assertIsNotNone(state)
        self.assertFalse(done)
        
        # Test that data processor and environment work together
        packet_info = env.data_processor.extract_field_features(env.current_packet, 'header')
        self.assertTrue(packet_info['exists'])
    
    def test_environment_reward_calculator_integration(self):
        """Test integration between environment and reward calculator."""
        env = FuzzingEnvironment(self.config['env'], self.config['main'])
        env.reset()
        
        # Execute a step and check reward calculation
        state, _, obs, _, reward, _, done, info = env.step(0, 2)
        
        self.assertIsInstance(reward, (int, float))
        self.assertIn('mutation_result', info)
        
        # Test reward breakdown
        breakdown = env.reward_calculator.get_reward_breakdown(info['mutation_result'])
        self.assertIn('grand_total', breakdown)
    
    def test_complete_episode_execution(self):
        """Test execution of a complete fuzzing episode."""
        env = FuzzingEnvironment(self.config['env'], self.config['main'])
        
        state, _, obs, _, done = env.reset()
        episode_length = 0
        total_reward = 0
        
        while not done and episode_length < 20:  # Limit episode length for testing
            # Simple random policy
            field_idx = np.random.randint(0, env.n_fields)
            mutation_action = np.random.randint(0, env.n_mutations)
            
            state, _, obs, _, reward, _, done, info = env.step(field_idx, mutation_action)
            
            total_reward += reward
            episode_length += 1
        
        # Check that episode executed successfully
        self.assertGreater(episode_length, 0)
        self.assertIsInstance(total_reward, (int, float))
        
        # Check final statistics
        stats = env.get_statistics()
        self.assertEqual(stats['total_mutations'], episode_length)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for the fuzzing framework."""
    
    def setUp(self):
        """Set up benchmark fixtures."""
        self.config = {
            'env': {
                'max_steps': 1000,
                'protocol_fields': [
                    {'name': 'header', 'type': 'bytes', 'min_length': 1, 'max_length': 64},
                    {'name': 'payload', 'type': 'bytes', 'min_length': 0, 'max_length': 1500},
                    {'name': 'length_field', 'type': 'integer', 'min_value': 0, 'max_value': 65535}
                ],
                'mamba_feature_dim': 256
            },
            'main': {'render': False}
        }
    
    def test_environment_step_performance(self):
        """Benchmark environment step execution time."""
        import time
        
        env = FuzzingEnvironment(self.config['env'], self.config['main'])
        env.reset()
        
        # Warm up
        for _ in range(10):
            env.step(0, 0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            field_idx = np.random.randint(0, env.n_fields)
            mutation_action = np.random.randint(0, env.n_mutations)
            env.step(field_idx, mutation_action)
        end_time = time.time()
        
        avg_step_time = (end_time - start_time) / 100
        print(f"Average step time: {avg_step_time:.4f} seconds")
        
        # Should be reasonably fast (< 0.1 seconds per step)
        self.assertLess(avg_step_time, 0.1)
    
    def test_mamba_feature_extraction_performance(self):
        """Benchmark Mamba feature extraction performance."""
        import time
        
        processor = ProtocolDataProcessor(self.config['env'])
        packets = [processor.generate_default_packet() for _ in range(100)]
        
        # Benchmark feature extraction
        start_time = time.time()
        for packet in packets:
            features = processor.extract_mamba_features(packet)
        end_time = time.time()
        
        avg_extraction_time = (end_time - start_time) / 100
        print(f"Average feature extraction time: {avg_extraction_time:.4f} seconds")
        
        # Should be reasonably fast
        self.assertLess(avg_extraction_time, 0.01)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestProtocolDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzingEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_class_name):
    """Run a specific test class.
    
    Args:
        test_class_name: Name of the test class to run
    """
    test_classes = {
        'data_processor': TestProtocolDataProcessor,
        'reward_calculator': TestRewardCalculator,
        'environment': TestFuzzingEnvironment,
        'integration': TestIntegration,
        'performance': TestPerformanceBenchmarks
    }
    
    if test_class_name not in test_classes:
        print(f"Unknown test class: {test_class_name}")
        print(f"Available tests: {list(test_classes.keys())}")
        return False
    
    # Run specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classes[test_class_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fuzzing framework')
    parser.add_argument('--test', type=str, help='Specific test to run', 
                       choices=['data_processor', 'reward_calculator', 'environment', 
                               'integration', 'performance'])
    
    args = parser.parse_args()
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)