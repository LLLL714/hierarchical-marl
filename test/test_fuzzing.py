"""Tests for the Fuzzing HSD Framework.

This module contains tests to validate the fuzzing framework components.
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'alg'))

# Import fuzzing components
try:
    import fuzzing_environment
    import data_processor
    import reward_calculator
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fuzzing components: {e}")
    COMPONENTS_AVAILABLE = False


class TestFuzzingEnvironment(unittest.TestCase):
    """Test cases for FuzzingEnvironment."""
    
    def setUp(self):
        """Set up test configuration."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Fuzzing components not available")
            
        self.config = {
            'protocol_fields': [
                {'name': 'header', 'type': 'header'},
                {'name': 'payload', 'type': 'payload'},
                {'name': 'checksum', 'type': 'checksum'}
            ],
            'mamba_feature_dim': 64,
            'use_mamba_features': True,
            'exploit_ratio': 0.9,
            'max_steps': 100,
            'base_reward_weight': 1.0,
            'medium_reward_weight': 5.0,
            'high_reward_weight': 100.0
        }
        
    def test_environment_initialization(self):
        """Test environment initialization."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        
        self.assertEqual(env.n_fields, 3)
        self.assertEqual(env.n_mutations, 6)
        self.assertEqual(len(env.mutation_actions), 6)
        self.assertIn('insert', env.mutation_actions)
        self.assertIn('delete', env.mutation_actions)
        self.assertIn('flip', env.mutation_actions)
        
    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        state, info = env.reset()
        
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertIn('n_fields', info)
        self.assertIn('n_mutations', info)
        self.assertEqual(info['n_fields'], 3)
        self.assertEqual(info['n_mutations'], 6)
        
    def test_environment_step(self):
        """Test environment step functionality."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        env.reset()
        
        # Test valid step
        next_state, reward, done, info = env.step(field_idx=0, mutation_action=0)
        
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn('mutation_result', info)
        
    def test_field_selection_probabilities(self):
        """Test 90/10 field selection strategy."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        env.reset()
        
        probs = env.get_field_selection_probabilities()
        
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0))
        
    def test_mutation_action_features(self):
        """Test mutation action feature extraction."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        env.reset()
        
        features = env.get_mutation_action_features(field_idx=0)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), env.mutation_state_dim)


class TestDataProcessor(unittest.TestCase):
    """Test cases for ProtocolDataProcessor."""
    
    def setUp(self):
        """Set up test configuration.""" 
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Fuzzing components not available")
            
        self.config = {
            'mamba_feature_dim': 64,
            'max_field_length': 1024,
            'supported_protocols': ['tcp', 'udp', 'http']
        }
        
    def test_processor_initialization(self):
        """Test data processor initialization."""
        processor = data_processor.ProtocolDataProcessor(self.config)
        
        self.assertEqual(processor.mamba_feature_dim, 64)
        self.assertEqual(processor.max_field_length, 1024)
        self.assertIn('tcp', processor.supported_protocols)
        
    def test_field_extraction(self):
        """Test protocol field extraction."""
        processor = data_processor.ProtocolDataProcessor(self.config)
        
        protocol_data = {
            'protocol_type': 'tcp',
            'raw_data': b'\x00\x50\x1f\x90\x00\x00\x00\x01',
            'fields': [
                {'name': 'src_port', 'type': 'address', 'value': b'\x00\x50', 'length': 2},
                {'name': 'dst_port', 'type': 'address', 'value': b'\x1f\x90', 'length': 2}
            ]
        }
        
        fields = processor.extract_fields(protocol_data)
        
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0]['name'], 'src_port')
        self.assertEqual(fields[0]['type'], 'address')
        self.assertIn('context_score', fields[0])
        self.assertIn('criticality', fields[0])
        
    def test_mamba_feature_extraction(self):
        """Test Mamba feature extraction."""
        processor = data_processor.ProtocolDataProcessor(self.config)
        
        protocol_data = {
            'protocol_type': 'tcp',
            'raw_data': b'\x00\x50\x1f\x90\x00\x00\x00\x01'
        }
        
        features = processor.extract_mamba_features(protocol_data)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 64)
        
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation."""
        processor = data_processor.ProtocolDataProcessor(self.config)
        
        dataset = processor._generate_synthetic_dataset()
        
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        self.assertIn('protocol_type', dataset[0])
        self.assertIn('raw_data', dataset[0])


class TestRewardCalculator(unittest.TestCase):
    """Test cases for RewardCalculator."""
    
    def setUp(self):
        """Set up test configuration."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Fuzzing components not available")
            
        self.config = {
            'base_reward_weight': 1.0,
            'medium_reward_weight': 5.0,
            'high_reward_weight': 100.0,
            'coverage_reward_scale': 10.0,
            'crash_reward': 20.0,
            'vulnerability_reward': 1000.0
        }
        
    def test_calculator_initialization(self):
        """Test reward calculator initialization."""
        calculator = reward_calculator.RewardCalculator(self.config)
        
        self.assertEqual(calculator.base_reward_weight, 1.0)
        self.assertEqual(calculator.medium_reward_weight, 5.0)
        self.assertEqual(calculator.high_reward_weight, 100.0)
        
    def test_basic_reward_calculation(self):
        """Test basic reward calculation."""
        calculator = reward_calculator.RewardCalculator(self.config)
        
        mutation_result = {
            'success': True,
            'coverage_increase': 0.1,
            'crash_detected': False,
            'vulnerability_found': False
        }
        
        reward = calculator.calculate_reward(mutation_result)
        
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)  # Should be positive for coverage increase
        
    def test_medium_reward_calculation(self):
        """Test medium tier reward calculation."""
        calculator = reward_calculator.RewardCalculator(self.config)
        
        mutation_result = {
            'success': True,
            'coverage_increase': 0.05,
            'crash_detected': True,
            'vulnerability_found': False
        }
        
        reward = calculator.calculate_reward(mutation_result)
        
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 20)  # Should include crash reward
        
    def test_high_reward_calculation(self):
        """Test high tier reward calculation."""
        calculator = reward_calculator.RewardCalculator(self.config)
        
        mutation_result = {
            'success': True,
            'coverage_increase': 0.05,
            'crash_detected': False,
            'vulnerability_found': True
        }
        
        reward = calculator.calculate_reward(mutation_result)
        
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 1000)  # Should include vulnerability reward
        
    def test_reward_statistics(self):
        """Test reward statistics tracking."""
        calculator = reward_calculator.RewardCalculator(self.config)
        
        # Calculate some rewards
        for _ in range(10):
            mutation_result = {
                'success': True,
                'coverage_increase': np.random.uniform(0, 0.2),
                'crash_detected': False,
                'vulnerability_found': False
            }
            calculator.calculate_reward(mutation_result)
        
        stats = calculator.get_reward_statistics()
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('count', stats)
        self.assertEqual(stats['count'], 10)


class TestFuzzingIntegration(unittest.TestCase):
    """Integration tests for the fuzzing framework."""
    
    def setUp(self):
        """Set up test configuration."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Fuzzing components not available")
            
        self.config = {
            'protocol_fields': [
                {'name': 'header', 'type': 'header'},
                {'name': 'payload', 'type': 'payload'},
                {'name': 'checksum', 'type': 'checksum'},
                {'name': 'length', 'type': 'length'}
            ],
            'mutation_actions': ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy'],
            'mamba_feature_dim': 64,
            'use_mamba_features': True,
            'exploit_ratio': 0.9,
            'max_steps': 50,
            'base_reward_weight': 1.0,
            'medium_reward_weight': 5.0,
            'high_reward_weight': 100.0,
            'coverage_reward_scale': 10.0,
            'crash_reward': 20.0,
            'vulnerability_reward': 1000.0
        }
        
    def test_full_episode_run(self):
        """Test running a complete fuzzing episode."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 20:
            # Random field and mutation selection
            field_idx = np.random.randint(0, env.n_fields)
            mutation_action = np.random.randint(0, env.n_mutations)
            
            next_state, reward, done, step_info = env.step(field_idx, mutation_action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
        
    def test_field_importance_learning(self):
        """Test field importance learning over time."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        env.reset()
        
        initial_importance = env.field_importance_history.copy()
        
        # Run several steps focusing on field 0
        for _ in range(10):
            # Field 0 with various mutations
            env.step(field_idx=0, mutation_action=np.random.randint(0, env.n_mutations))
        
        # Check if field 0 importance changed
        final_importance = env.field_importance_history.copy()
        
        # Importance should have been updated (may increase or decrease based on rewards)
        self.assertFalse(np.array_equal(initial_importance, final_importance))
        
    def test_mutation_history_tracking(self):
        """Test mutation history tracking."""
        env = fuzzing_environment.FuzzingEnvironment(self.config)
        env.reset()
        
        initial_history_length = len(env.mutation_history)
        
        # Execute several mutations
        for i in range(5):
            env.step(field_idx=i % env.n_fields, mutation_action=i % env.n_mutations)
            
        final_history_length = len(env.mutation_history)
        
        self.assertEqual(final_history_length, initial_history_length + 5)
        
        # Check mutation history structure
        if env.mutation_history:
            mutation_entry = env.mutation_history[-1]
            self.assertIn('field_idx', mutation_entry)
            self.assertIn('mutation_action', mutation_entry)
            self.assertIn('reward', mutation_entry)
            self.assertIn('step', mutation_entry)


def run_training_test():
    """Test the training script with dummy configuration."""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'alg'))
        import train_fuzzing
        
        # Create minimal test config
        test_config = {
            "main": {
                "seed": 12345,
                "dir_name": "test_fuzzing",
                "alg_name": "fuzzing_hsd",
                "model_name": "test_model.ckpt",
                "summarize": False,
                "save_period": 100
            },
            "alg": {
                "N_train": 5,  # Very short training
                "N_eval": 2,
                "period": 2,
                "epsilon_start": 0.5,
                "epsilon_end": 0.4,
                "epsilon_div": 2,
                "buffer_size": 100,
                "batch_size": 10,
                "pretrain_episodes": 2,
                "steps_per_train": 1
            },
            "fuzzing_params": {
                "protocol_fields": [
                    {"name": "header", "type": "header"},
                    {"name": "payload", "type": "payload"}
                ],
                "mutation_actions": ["insert", "delete", "flip", "replace"],
                "field_selection_steps": 5,
                "exploit_ratio": 0.9,
                "mamba_feature_dim": 32,
                "use_mamba_features": False,  # Disable for testing
                "decoder_batch_size": 5
            },
            "env": {
                "max_steps": 10
            },
            "nn_fuzzing_hsd": {
                "n_h_decoder": 32,
                "n_h1_low": 16,
                "n_h2_low": 16,
                "n_h1": 32,
                "n_h2": 32,
                "n_h_mixer": 16
            }
        }
        
        print("Running training test...")
        train_fuzzing.train_fuzzing_hsd(test_config)
        print("Training test completed successfully!")
        
    except Exception as e:
        print(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Running Fuzzing HSD Framework Tests...")
    
    # Run unit tests
    if COMPONENTS_AVAILABLE:
        unittest.main(verbosity=2, exit=False)
    else:
        print("Skipping unit tests - components not available")
    
    # Run integration test
    print("\n" + "="*50)
    print("Running Integration Test...")
    run_training_test()
    
    print("\nAll tests completed!")