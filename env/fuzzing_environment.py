"""Fuzzing Environment for Hierarchical RL Framework.

This module implements the fuzzing environment that replaces the original 
sports environment, adapting the HSD framework for protocol fuzzing.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from data_processor import ProtocolDataProcessor
from reward_calculator import RewardCalculator


class FuzzingEnvironment:
    """
    Fuzzing environment that supports hierarchical RL for protocol testing.
    
    High-level policy selects protocol fields (90% exploit, 10% explore)
    Low-level policy selects mutation actions (insert, delete, flip, replace, shuffle, copy)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fuzzing environment.
        
        Args:
            config: Configuration dictionary containing fuzzing parameters
        """
        self.config = config
        
        # Protocol configuration
        self.protocol_fields = config.get('protocol_fields', [])
        self.n_fields = len(self.protocol_fields)
        
        # Mutation actions: insert, delete, flip, replace, shuffle, copy
        self.mutation_actions = ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy']
        self.n_mutations = len(self.mutation_actions)
        
        # Mamba feature extraction
        self.mamba_feature_dim = config.get('mamba_feature_dim', 256)
        self.use_mamba_features = config.get('use_mamba_features', True)
        
        # Field selection strategy (90/10)
        self.exploit_ratio = config.get('exploit_ratio', 0.9)
        self.explore_ratio = 1.0 - self.exploit_ratio
        
        # Environment dimensions
        self.field_state_dim = self.n_fields * 4  # field type, length, importance, context
        self.mutation_state_dim = 16  # current state features for mutation selection
        if self.use_mamba_features:
            self.state_dim = self.field_state_dim + self.mamba_feature_dim + self.mutation_state_dim
        else:
            self.state_dim = self.field_state_dim + self.mutation_state_dim
            
        self.field_action_dim = self.n_fields  # one action per field
        self.mutation_action_dim = self.n_mutations  # one action per mutation type
        
        # Initialize components
        self.data_processor = ProtocolDataProcessor(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Environment state
        self.current_protocol_data = None
        self.current_fields = []
        self.current_mamba_features = None
        self.field_importance_history = np.ones(self.n_fields)  # Historical importance scores
        self.step_count = 0
        self.max_steps = config.get('max_steps', 1000)
        
        # Mutation history for intrinsic rewards
        self.mutation_history = []
        self.coverage_tracker = set()
        
    def reset(self, protocol_data: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with new protocol data.
        
        Args:
            protocol_data: Preprocessed protocol packet data
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        if protocol_data is not None:
            self.current_protocol_data = protocol_data
        else:
            # Use default/random protocol data if none provided
            self.current_protocol_data = self._generate_default_protocol_data()
            
        # Process protocol data and extract features
        self.current_fields = self.data_processor.extract_fields(self.current_protocol_data)
        
        if self.use_mamba_features:
            self.current_mamba_features = self.data_processor.extract_mamba_features(
                self.current_protocol_data
            )
        else:
            self.current_mamba_features = np.zeros(self.mamba_feature_dim)
            
        # Reset environment state
        self.step_count = 0
        self.mutation_history = []
        self.coverage_tracker = set()
        
        # Get initial state
        initial_state = self._get_state()
        
        info = {
            'n_fields': self.n_fields,
            'n_mutations': self.n_mutations,
            'protocol_type': self.current_protocol_data.get('protocol_type', 'unknown')
        }
        
        return initial_state, info
    
    def step(self, field_idx: int, mutation_action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.
        
        Args:
            field_idx: Index of selected protocol field
            mutation_action: Index of selected mutation action
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Validate actions
        field_idx = max(0, min(field_idx, self.n_fields - 1))
        mutation_action = max(0, min(mutation_action, self.n_mutations - 1))
        
        # Execute mutation
        mutation_result = self._execute_mutation(field_idx, mutation_action)
        
        # Calculate reward using 3-tier system
        reward = self.reward_calculator.calculate_reward(mutation_result)
        
        # Update field importance based on reward
        self._update_field_importance(field_idx, reward)
        
        # Update mutation history
        self.mutation_history.append({
            'field_idx': field_idx,
            'mutation_action': mutation_action,
            'reward': reward,
            'step': self.step_count
        })
        
        # Check if episode is done
        done = self._is_done(mutation_result)
        
        # Get next state
        next_state = self._get_state()
        
        # Prepare info dictionary
        info = {
            'mutation_result': mutation_result,
            'field_idx': field_idx,
            'mutation_action': self.mutation_actions[mutation_action],
            'coverage_increase': mutation_result.get('coverage_increase', 0),
            'crash_detected': mutation_result.get('crash_detected', False),
            'vulnerability_found': mutation_result.get('vulnerability_found', False)
        }
        
        return next_state, reward, done, info
    
    def get_field_selection_probabilities(self) -> np.ndarray:
        """Get field selection probabilities for 90/10 strategy.
        
        Returns:
            Probability distribution over fields
        """
        # Normalize importance scores
        importance_probs = self.field_importance_history / np.sum(self.field_importance_history)
        
        # Apply 90/10 strategy
        exploit_probs = self.exploit_ratio * importance_probs
        explore_probs = self.explore_ratio * np.ones(self.n_fields) / self.n_fields
        
        total_probs = exploit_probs + explore_probs
        return total_probs / np.sum(total_probs)  # Normalize
    
    def get_mutation_action_features(self, field_idx: int) -> np.ndarray:
        """Get features for mutation action selection.
        
        Args:
            field_idx: Index of selected field
            
        Returns:
            Feature vector for mutation action selection
        """
        if field_idx >= len(self.current_fields):
            return np.zeros(self.mutation_state_dim)
            
        field = self.current_fields[field_idx]
        
        # Extract field characteristics
        field_type = self._encode_field_type(field.get('type', 'unknown'))
        field_length = min(field.get('length', 0) / 100.0, 1.0)  # Normalize
        field_position = field_idx / max(1, self.n_fields - 1)  # Relative position
        
        # Historical performance of mutations on this field type
        mutation_history = self._get_mutation_history_for_field_type(field.get('type', 'unknown'))
        
        # Current state context
        context_features = np.array([
            self.step_count / self.max_steps,  # Progress in episode
            len(self.coverage_tracker) / 1000.0,  # Normalized coverage
            min(len(self.mutation_history) / 100.0, 1.0),  # Mutation density
            field_position
        ])
        
        # Combine all features
        features = np.concatenate([
            field_type,
            [field_length],
            mutation_history,
            context_features
        ])
        
        # Pad or truncate to exact dimension
        if len(features) > self.mutation_state_dim:
            features = features[:self.mutation_state_dim]
        elif len(features) < self.mutation_state_dim:
            features = np.pad(features, (0, self.mutation_state_dim - len(features)))
            
        return features
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state.
        
        Returns:
            State vector containing field features, Mamba features, and context
        """
        # Field-level features
        field_features = self._get_field_features()
        
        # Mamba features
        if self.use_mamba_features and self.current_mamba_features is not None:
            mamba_features = self.current_mamba_features
        else:
            mamba_features = np.zeros(self.mamba_feature_dim)
            
        # Current context features  
        context_features = np.array([
            self.step_count / self.max_steps,
            len(self.coverage_tracker) / 1000.0,
            min(len(self.mutation_history) / 100.0, 1.0),
            np.mean(self.field_importance_history),
            np.std(self.field_importance_history),
            len([h for h in self.mutation_history[-10:] if h['reward'] > 0]) / 10.0,
            0.0, 0.0, 0.0, 0.0  # Reserved for future context features
        ])
        
        # Pad context to exact dimension
        if len(context_features) > self.mutation_state_dim:
            context_features = context_features[:self.mutation_state_dim]
        elif len(context_features) < self.mutation_state_dim:
            context_features = np.pad(context_features, 
                                    (0, self.mutation_state_dim - len(context_features)))
        
        # Combine all features
        if self.use_mamba_features:
            state = np.concatenate([field_features, mamba_features, context_features])
        else:
            state = np.concatenate([field_features, context_features])
            
        return state.astype(np.float32)
    
    def _get_field_features(self) -> np.ndarray:
        """Extract features for all protocol fields.
        
        Returns:
            Feature vector representing all fields
        """
        features = []
        for i in range(self.n_fields):
            if i < len(self.current_fields):
                field = self.current_fields[i]
                field_type = self._encode_field_type(field.get('type', 'unknown'))
                field_length = min(field.get('length', 0) / 100.0, 1.0)
                field_importance = self.field_importance_history[i]
                field_context = field.get('context_score', 0.5)
                
                field_features = np.concatenate([
                    field_type,
                    [field_length, field_importance, field_context]
                ])
            else:
                # Padding for fields that don't exist
                field_features = np.zeros(4)
                
            features.append(field_features)
            
        return np.concatenate(features)
    
    def _encode_field_type(self, field_type: str) -> np.ndarray:
        """Encode field type as one-hot vector.
        
        Args:
            field_type: String field type
            
        Returns:
            One-hot encoded field type
        """
        types = ['header', 'payload', 'checksum', 'length', 'address', 'flag', 'data', 'unknown']
        if field_type in types:
            idx = types.index(field_type)
        else:
            idx = types.index('unknown')
            
        one_hot = np.zeros(len(types))
        one_hot[idx] = 1.0
        return one_hot[:1]  # Return only first element to save space, can expand if needed
    
    def _execute_mutation(self, field_idx: int, mutation_action: int) -> Dict:
        """Execute mutation on selected field.
        
        Args:
            field_idx: Index of field to mutate
            mutation_action: Index of mutation action
            
        Returns:
            Dictionary containing mutation results
        """
        if field_idx >= len(self.current_fields):
            return {'success': False, 'reason': 'invalid_field_index'}
            
        field = self.current_fields[field_idx]
        mutation_type = self.mutation_actions[mutation_action]
        
        # Simulate mutation execution (in real implementation, this would 
        # apply actual mutations to protocol data)
        result = {
            'success': True,
            'field_idx': field_idx,
            'mutation_type': mutation_type,
            'field_type': field.get('type', 'unknown'),
            'original_value': field.get('value', ''),
            'mutated_value': self._simulate_mutation(field, mutation_type),
            'coverage_increase': 0,
            'crash_detected': False,
            'vulnerability_found': False,
            'timeout_detected': False,
            'protocol_error': False
        }
        
        # Simulate execution results based on mutation characteristics
        result.update(self._simulate_execution_results(result))
        
        return result
    
    def _simulate_mutation(self, field: Dict, mutation_type: str) -> str:
        """Simulate mutation application.
        
        Args:
            field: Field dictionary
            mutation_type: Type of mutation to apply
            
        Returns:
            Simulated mutated value
        """
        original = field.get('value', '')
        
        if mutation_type == 'insert':
            pos = random.randint(0, len(original))
            return original[:pos] + 'XX' + original[pos:]
        elif mutation_type == 'delete':
            if len(original) > 2:
                pos = random.randint(0, len(original) - 2)
                return original[:pos] + original[pos+2:]
            return original
        elif mutation_type == 'flip':
            if original:
                data = list(original)
                pos = random.randint(0, len(data) - 1)
                data[pos] = chr(ord(data[pos]) ^ 0xFF)
                return ''.join(data)
            return original
        elif mutation_type == 'replace':
            return 'MUT' + original[3:] if len(original) > 3 else 'MUT'
        elif mutation_type == 'shuffle':
            data = list(original)
            random.shuffle(data)
            return ''.join(data)
        elif mutation_type == 'copy':
            return original + original
        
        return original
    
    def _simulate_execution_results(self, mutation_result: Dict) -> Dict:
        """Simulate execution results for testing purposes.
        
        Args:
            mutation_result: Current mutation result
            
        Returns:
            Additional execution results
        """
        # Simulate based on field type and mutation type
        field_type = mutation_result['field_type']
        mutation_type = mutation_result['mutation_type']
        
        results = {}
        
        # Coverage simulation (higher for new mutation combinations)
        mutation_key = f"{field_type}_{mutation_type}"
        if mutation_key not in self.coverage_tracker:
            self.coverage_tracker.add(mutation_key)
            results['coverage_increase'] = random.uniform(0.1, 0.5)
        else:
            results['coverage_increase'] = random.uniform(0.0, 0.1)
        
        # Crash simulation (rare, more likely with certain mutations)
        crash_probability = 0.001
        if mutation_type in ['flip', 'replace'] and field_type in ['header', 'checksum']:
            crash_probability = 0.01
        results['crash_detected'] = random.random() < crash_probability
        
        # Vulnerability simulation (very rare, highest reward)
        vuln_probability = 0.0001
        if mutation_type == 'copy' and field_type == 'payload':
            vuln_probability = 0.001
        results['vulnerability_found'] = random.random() < vuln_probability
        
        # Timeout simulation
        timeout_probability = 0.002
        if mutation_type in ['copy', 'insert']:
            timeout_probability = 0.005
        results['timeout_detected'] = random.random() < timeout_probability
        
        # Protocol error simulation
        error_probability = 0.01
        if field_type in ['header', 'checksum', 'length']:
            error_probability = 0.05
        results['protocol_error'] = random.random() < error_probability
        
        return results
    
    def _update_field_importance(self, field_idx: int, reward: float):
        """Update field importance based on received reward.
        
        Args:
            field_idx: Index of field that was mutated
            reward: Reward received from mutation
        """
        if field_idx < len(self.field_importance_history):
            # Exponential moving average update
            alpha = 0.1
            self.field_importance_history[field_idx] = (
                (1 - alpha) * self.field_importance_history[field_idx] + alpha * max(0, reward)
            )
    
    def _get_mutation_history_for_field_type(self, field_type: str) -> np.ndarray:
        """Get historical performance of mutations on field type.
        
        Args:
            field_type: Type of field
            
        Returns:
            Array of historical rewards for each mutation type
        """
        history = np.zeros(self.n_mutations)
        counts = np.zeros(self.n_mutations)
        
        for mutation in self.mutation_history[-50:]:  # Last 50 mutations
            if mutation.get('field_type') == field_type:
                action_idx = mutation.get('mutation_action', 0)
                if action_idx < self.n_mutations:
                    history[action_idx] += mutation.get('reward', 0)
                    counts[action_idx] += 1
        
        # Average rewards, with small smoothing
        for i in range(self.n_mutations):
            if counts[i] > 0:
                history[i] = history[i] / counts[i]
            else:
                history[i] = 0.1  # Small default value for untried mutations
                
        return history
    
    def _is_done(self, mutation_result: Dict) -> bool:
        """Check if episode should terminate.
        
        Args:
            mutation_result: Result from latest mutation
            
        Returns:
            True if episode is done
        """
        # Terminate if max steps reached
        if self.step_count >= self.max_steps:
            return True
            
        # Terminate if vulnerability found (success condition)
        if mutation_result.get('vulnerability_found', False):
            return True
            
        # Terminate if too many crashes (failure condition)
        recent_crashes = sum(1 for h in self.mutation_history[-10:] 
                           if h.get('crash_detected', False))
        if recent_crashes >= 5:
            return True
            
        return False
    
    def _generate_default_protocol_data(self) -> Dict:
        """Generate default protocol data for testing.
        
        Returns:
            Default protocol data dictionary
        """
        return {
            'protocol_type': 'test_protocol',
            'raw_data': b'\x01\x02\x03\x04\x05\x06\x07\x08',
            'fields': [
                {'name': 'header', 'type': 'header', 'value': '\x01\x02', 'length': 2},
                {'name': 'length', 'type': 'length', 'value': '\x03\x04', 'length': 2},
                {'name': 'payload', 'type': 'payload', 'value': '\x05\x06', 'length': 2},
                {'name': 'checksum', 'type': 'checksum', 'value': '\x07\x08', 'length': 2}
            ]
        }

    @property
    def observation_space_dim(self) -> int:
        """Get observation space dimension."""
        return self.state_dim
    
    @property
    def field_action_space_dim(self) -> int:
        """Get field action space dimension."""
        return self.field_action_dim
    
    @property  
    def mutation_action_space_dim(self) -> int:
        """Get mutation action space dimension."""
        return self.mutation_action_dim