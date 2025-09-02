"""Fuzzing environment for hierarchical reinforcement learning.

This environment adapts the HSD framework for protocol fuzzing by:
- Receiving preprocessed protocol packet datasets
- Supporting Mamba feature extraction
- Implementing field-based mutation operations
- Providing coverage and vulnerability detection feedback
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import logging

from .data_processor import ProtocolDataProcessor
from .reward_calculator import RewardCalculator


class FuzzingEnvironment:
    """Main fuzzing environment that interfaces with the HSD algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fuzzing environment.
        
        Args:
            config: Configuration dictionary containing fuzzing parameters
        """
        self.config = config
        
        # Protocol and mutation configuration
        self.protocol_fields = config.get('protocol_fields', [])
        self.mutation_actions = ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy']
        self.max_mutation_size = config.get('max_mutation_size', 1024)
        
        # Field selection strategy (90% reward-based, 10% random)
        self.field_selection_reward_ratio = config.get('field_selection_reward_ratio', 0.9)
        
        # Environment dimensions
        self.n_fields = len(self.protocol_fields)
        self.n_mutation_actions = len(self.mutation_actions)
        
        # Initialize processors
        self.data_processor = ProtocolDataProcessor(config)
        self.reward_calculator = RewardCalculator(config)
        
        # Current state
        self.current_packet = None
        self.current_features = None
        self.field_importance_scores = np.zeros(self.n_fields)
        self.mutation_history = []
        self.step_count = 0
        
        # Coverage and vulnerability tracking
        self.coverage_tracker = CoverageTracker(config)
        self.vulnerability_detector = VulnerabilityDetector(config)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    @property
    def state_dim(self) -> int:
        """Dimension of global state."""
        # State includes: field features, mutation history, coverage info
        mamba_dim = self.config.get('mamba_feature_dim', 256)
        history_dim = self.config.get('mutation_history_length', 10) * self.n_mutation_actions
        coverage_dim = self.config.get('coverage_feature_dim', 64)
        return mamba_dim + history_dim + coverage_dim + self.n_fields
        
    @property
    def obs_dim(self) -> int:
        """Dimension of agent observation."""
        # Each agent observes: field features, field importance, current context
        mamba_dim = self.config.get('mamba_feature_dim', 256)
        field_context_dim = self.config.get('field_context_dim', 32)
        return mamba_dim + field_context_dim + 1  # +1 for field importance score
        
    @property
    def action_dim(self) -> int:
        """Dimension of action space (mutation actions)."""
        return self.n_mutation_actions
        
    def reset(self, protocol_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], bool]:
        """Reset environment with new protocol data.
        
        Args:
            protocol_data: Optional protocol packet data. If None, loads from dataset.
            
        Returns:
            Tuple of (state_home, state_away, list_obs_home, list_obs_away, done)
            where state_away and list_obs_away are dummy values for compatibility
        """
        # Load new protocol data
        if protocol_data is None:
            protocol_data = self.data_processor.sample_protocol_data()
            
        self.current_packet = protocol_data
        self.current_features = self.data_processor.extract_mamba_features(protocol_data)
        
        # Reset state
        self.step_count = 0
        self.mutation_history = []
        self.field_importance_scores = np.zeros(self.n_fields)
        
        # Reset tracking
        self.coverage_tracker.reset()
        self.vulnerability_detector.reset()
        
        # Generate initial state and observations
        state_home = self._generate_global_state()
        list_obs_home = self._generate_agent_observations()
        
        # Dummy values for compatibility with HSD interface
        state_away = np.zeros_like(state_home)
        list_obs_away = [np.zeros_like(obs) for obs in list_obs_home]
        done = False
        
        return state_home, state_away, list_obs_home, list_obs_away, done
        
    def step(self, actions: List[int], field_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float, List[float], bool, Dict]:
        """Execute mutation actions on selected fields.
        
        Args:
            actions: List of mutation action indices for each agent
            field_indices: Optional list of field indices. If None, uses high-level policy selection.
            
        Returns:
            Tuple of (state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done, info)
        """
        self.step_count += 1
        
        # If field indices not provided, select based on importance scores and exploration
        if field_indices is None:
            field_indices = self._select_fields(len(actions))
            
        # Execute mutations
        mutated_packet, mutation_results = self._execute_mutations(actions, field_indices)
        
        # Update mutation history
        self._update_mutation_history(actions, field_indices)
        
        # Execute fuzzing and collect feedback
        execution_result = self._execute_fuzzing(mutated_packet)
        
        # Calculate rewards
        reward, local_rewards = self.reward_calculator.calculate_reward(
            execution_result, mutation_results, self.step_count
        )
        
        # Update field importance scores based on reward
        self._update_field_importance(field_indices, reward)
        
        # Generate new state and observations
        state_home = self._generate_global_state()
        list_obs_home = self._generate_agent_observations()
        
        # Check termination conditions
        done = self._check_done(execution_result)
        
        # Dummy values for compatibility
        state_away = np.zeros_like(state_home)
        list_obs_away = [np.zeros_like(obs) for obs in list_obs_home]
        
        # Info dict
        info = {
            'execution_result': execution_result,
            'mutation_results': mutation_results,
            'field_indices': field_indices,
            'step_count': self.step_count
        }
        
        return state_home, state_away, list_obs_home, list_obs_away, reward, local_rewards, done, info
        
    def random_actions(self) -> List[int]:
        """Generate random mutation actions."""
        n_agents = len(self.protocol_fields)
        return [random.randint(0, self.n_mutation_actions - 1) for _ in range(n_agents)]
        
    def _select_fields(self, n_agents: int) -> List[int]:
        """Select fields using 90% reward-based, 10% random strategy."""
        field_indices = []
        
        for _ in range(n_agents):
            if random.random() < self.field_selection_reward_ratio:
                # Reward-based selection (weighted by importance scores)
                if np.sum(self.field_importance_scores) > 0:
                    probabilities = self.field_importance_scores / np.sum(self.field_importance_scores)
                    field_idx = np.random.choice(self.n_fields, p=probabilities)
                else:
                    field_idx = random.randint(0, self.n_fields - 1)
            else:
                # Random selection for exploration
                field_idx = random.randint(0, self.n_fields - 1)
                
            field_indices.append(field_idx)
            
        return field_indices
        
    def _execute_mutations(self, actions: List[int], field_indices: List[int]) -> Tuple[Dict, List[Dict]]:
        """Execute mutation actions on specified fields."""
        mutated_packet = self.current_packet.copy()
        mutation_results = []
        
        for action_idx, field_idx in zip(actions, field_indices):
            action_name = self.mutation_actions[action_idx]
            field_name = self.protocol_fields[field_idx]
            
            # Apply mutation
            result = self._apply_mutation(mutated_packet, field_name, action_name)
            mutation_results.append(result)
            
        return mutated_packet, mutation_results
        
    def _apply_mutation(self, packet: Dict, field_name: str, action_name: str) -> Dict:
        """Apply specific mutation to a field."""
        result = {
            'field': field_name,
            'action': action_name,
            'success': True,
            'original_value': packet.get(field_name),
            'mutated_value': None
        }
        
        try:
            field_value = packet.get(field_name, b'')
            
            if action_name == 'insert':
                # Insert random bytes
                insert_data = np.random.bytes(random.randint(1, 16))
                pos = random.randint(0, len(field_value))
                mutated_value = field_value[:pos] + insert_data + field_value[pos:]
                
            elif action_name == 'delete':
                # Delete random segment
                if len(field_value) > 1:
                    start = random.randint(0, len(field_value) - 1)
                    end = random.randint(start + 1, len(field_value))
                    mutated_value = field_value[:start] + field_value[end:]
                else:
                    mutated_value = field_value
                    
            elif action_name == 'flip':
                # Flip random bits
                if len(field_value) > 0:
                    mutated_bytes = bytearray(field_value)
                    byte_idx = random.randint(0, len(mutated_bytes) - 1)
                    bit_idx = random.randint(0, 7)
                    mutated_bytes[byte_idx] ^= (1 << bit_idx)
                    mutated_value = bytes(mutated_bytes)
                else:
                    mutated_value = field_value
                    
            elif action_name == 'replace':
                # Replace with random data
                if len(field_value) > 0:
                    mutated_value = np.random.bytes(len(field_value))
                else:
                    mutated_value = np.random.bytes(random.randint(1, 16))
                    
            elif action_name == 'shuffle':
                # Shuffle bytes
                if len(field_value) > 1:
                    mutated_bytes = list(field_value)
                    random.shuffle(mutated_bytes)
                    mutated_value = bytes(mutated_bytes)
                else:
                    mutated_value = field_value
                    
            elif action_name == 'copy':
                # Duplicate the field
                mutated_value = field_value + field_value
                
            else:
                mutated_value = field_value
                result['success'] = False
                
            # Apply size limits
            if len(mutated_value) > self.max_mutation_size:
                mutated_value = mutated_value[:self.max_mutation_size]
                
            packet[field_name] = mutated_value
            result['mutated_value'] = mutated_value
            
        except Exception as e:
            self.logger.warning(f"Mutation failed for field {field_name}: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
        
    def _execute_fuzzing(self, mutated_packet: Dict) -> Dict:
        """Execute fuzzing with mutated packet and collect results."""
        # This would interface with actual fuzzing target
        # For now, simulate fuzzing results
        execution_result = {
            'coverage_increase': random.uniform(0, 0.1),  # Simulated coverage increase
            'crash_detected': random.random() < 0.01,    # 1% chance of crash
            'timeout_detected': random.random() < 0.02,  # 2% chance of timeout
            'vulnerability_found': random.random() < 0.001,  # 0.1% chance of vulnerability
            'execution_time': random.uniform(0.1, 1.0),
            'unique_packet': True  # Assume packet is unique for now
        }
        
        # Update trackers
        self.coverage_tracker.update(execution_result)
        self.vulnerability_detector.update(execution_result)
        
        return execution_result
        
    def _update_mutation_history(self, actions: List[int], field_indices: List[int]):
        """Update mutation history for state representation."""
        history_entry = {
            'actions': actions,
            'field_indices': field_indices,
            'step': self.step_count
        }
        
        self.mutation_history.append(history_entry)
        
        # Keep only recent history
        max_history = self.config.get('mutation_history_length', 10)
        if len(self.mutation_history) > max_history:
            self.mutation_history = self.mutation_history[-max_history:]
            
    def _update_field_importance(self, field_indices: List[int], reward: float):
        """Update field importance scores based on reward feedback."""
        # Simple update rule: increase importance for fields that led to higher rewards
        for field_idx in field_indices:
            self.field_importance_scores[field_idx] = (
                0.9 * self.field_importance_scores[field_idx] + 0.1 * max(0, reward)
            )
            
    def _generate_global_state(self) -> np.ndarray:
        """Generate global state representation."""
        # Combine Mamba features, mutation history, coverage info
        state_parts = []
        
        # Mamba features
        if self.current_features is not None:
            state_parts.append(self.current_features.flatten())
        else:
            mamba_dim = self.config.get('mamba_feature_dim', 256)
            state_parts.append(np.zeros(mamba_dim))
            
        # Mutation history encoding
        history_encoding = self._encode_mutation_history()
        state_parts.append(history_encoding)
        
        # Coverage information
        coverage_info = self.coverage_tracker.get_state_features()
        state_parts.append(coverage_info)
        
        # Field importance scores
        state_parts.append(self.field_importance_scores)
        
        return np.concatenate(state_parts)
        
    def _generate_agent_observations(self) -> List[np.ndarray]:
        """Generate observations for each agent (field)."""
        observations = []
        
        for field_idx in range(self.n_fields):
            obs_parts = []
            
            # Mamba features for this field
            if self.current_features is not None:
                field_features = self.current_features.flatten()  # Simplified
            else:
                mamba_dim = self.config.get('mamba_feature_dim', 256)
                field_features = np.zeros(mamba_dim)
            obs_parts.append(field_features)
            
            # Field context (simplified as field index embedding)
            field_context_dim = self.config.get('field_context_dim', 32)
            field_context = np.zeros(field_context_dim)
            if field_idx < field_context_dim:
                field_context[field_idx] = 1.0
            obs_parts.append(field_context)
            
            # Field importance score
            obs_parts.append(np.array([self.field_importance_scores[field_idx]]))
            
            observations.append(np.concatenate(obs_parts))
            
        return observations
        
    def _encode_mutation_history(self) -> np.ndarray:
        """Encode mutation history into fixed-size vector."""
        max_history = self.config.get('mutation_history_length', 10)
        encoding_dim = max_history * self.n_mutation_actions
        encoding = np.zeros(encoding_dim)
        
        for i, entry in enumerate(self.mutation_history[-max_history:]):
            base_idx = i * self.n_mutation_actions
            for action in entry['actions']:
                if base_idx + action < encoding_dim:
                    encoding[base_idx + action] += 1
                    
        return encoding
        
    def _check_done(self, execution_result: Dict) -> bool:
        """Check if episode should terminate."""
        # Terminate on vulnerability discovery or after max steps
        max_steps = self.config.get('max_episode_steps', 1000)
        
        return (
            execution_result.get('vulnerability_found', False) or
            self.step_count >= max_steps
        )


class CoverageTracker:
    """Tracks code coverage during fuzzing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coverage_data = set()
        self.total_coverage = 0.0
        
    def reset(self):
        """Reset coverage tracking."""
        self.coverage_data.clear()
        self.total_coverage = 0.0
        
    def update(self, execution_result: Dict):
        """Update coverage based on execution result."""
        coverage_increase = execution_result.get('coverage_increase', 0)
        self.total_coverage += coverage_increase
        
    def get_state_features(self) -> np.ndarray:
        """Get coverage features for state representation."""
        feature_dim = self.config.get('coverage_feature_dim', 64)
        features = np.zeros(feature_dim)
        
        # Simple encoding: total coverage in first element
        features[0] = min(self.total_coverage, 1.0)  # Normalize
        
        return features


class VulnerabilityDetector:
    """Detects vulnerabilities during fuzzing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vulnerabilities_found = []
        
    def reset(self):
        """Reset vulnerability detection."""
        self.vulnerabilities_found.clear()
        
    def update(self, execution_result: Dict):
        """Update vulnerability detection based on execution result."""
        if execution_result.get('vulnerability_found', False):
            self.vulnerabilities_found.append({
                'timestamp': execution_result.get('timestamp'),
                'type': execution_result.get('vulnerability_type', 'unknown')
            })