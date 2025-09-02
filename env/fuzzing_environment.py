"""Fuzzing Environment for Hierarchical Reinforcement Learning.

This environment replaces the STS2 sports simulator with a protocol fuzzing environment
that supports hierarchical decision making for intelligent fuzzing campaigns.

High-level policy: Protocol field selection
Low-level policy: Mutation action selection
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from .data_processor import ProtocolDataProcessor
from .reward_calculator import RewardCalculator


class FuzzingEnvironment:
    """Protocol fuzzing environment with hierarchical RL interface."""
    
    def __init__(self, config_env: Dict, config_main: Dict, test: bool = False):
        """Initialize fuzzing environment.
        
        Args:
            config_env: Environment configuration parameters
            config_main: Main experiment configuration  
            test: Whether this is a test environment
        """
        self.config_env = config_env
        self.config_main = config_main
        self.test = test
        
        # Protocol and mutation configuration
        self.protocol_fields = config_env.get('protocol_fields', [])
        self.n_fields = len(self.protocol_fields)
        self.mutation_actions = ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy']
        self.n_mutations = len(self.mutation_actions)
        
        # Fuzzing campaign parameters
        self.max_steps = config_env.get('max_steps', 1000)
        self.current_step = 0
        
        # Data processing and reward calculation
        self.data_processor = ProtocolDataProcessor(config_env)
        self.reward_calculator = RewardCalculator(config_env)
        
        # Current fuzzing state
        self.current_packet = None
        self.current_features = None
        self.coverage_history = set()
        self.crash_history = []
        self.vulnerability_history = []
        
        # Field selection strategy (90/10 split)
        self.exploit_probability = config_env.get('exploit_probability', 0.9)
        self.field_importance_scores = np.ones(self.n_fields) / self.n_fields
        
        # Mamba feature integration
        self.mamba_feature_dim = config_env.get('mamba_feature_dim', 256)
        self.use_mamba_features = config_env.get('use_mamba_features', True)
        
        # Environment dimensions for RL interface
        self.state_dim = self._calculate_state_dim()
        self.obs_dim = self._calculate_obs_dim()
        self.action_dim_high = self.n_fields  # Field selection actions
        self.action_dim_low = self.n_mutations  # Mutation actions
        
        # Statistics tracking
        self.stats = {
            'total_mutations': 0,
            'crashes_found': 0,
            'vulnerabilities_found': 0,
            'unique_coverage': 0,
            'field_selection_counts': np.zeros(self.n_fields),
            'mutation_action_counts': np.zeros(self.n_mutations)
        }
        
    def _calculate_state_dim(self) -> int:
        """Calculate global state dimension for hierarchical RL."""
        base_dim = (
            self.mamba_feature_dim +  # Mamba features
            self.n_fields +           # Field importance scores
            self.n_mutations +        # Action availability
            4                         # Coverage, crashes, vulnerabilities, step progress
        )
        return base_dim
        
    def _calculate_obs_dim(self) -> int:
        """Calculate observation dimension for individual agents."""
        # In fuzzing context, we have a single agent that makes hierarchical decisions
        return self.state_dim
        
    def reset(self, protocol_dataset_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], bool]:
        """Reset environment for new fuzzing campaign.
        
        Args:
            protocol_dataset_path: Path to preprocessed protocol dataset
            
        Returns:
            state_home: Global state representation
            state_away: Not used in fuzzing (None)
            list_obs_home: List of agent observations
            list_obs_away: Not used in fuzzing (empty list)
            done: Episode termination flag
        """
        self.current_step = 0
        self.coverage_history.clear()
        self.crash_history.clear()
        self.vulnerability_history.clear()
        
        # Reset statistics
        self.stats = {
            'total_mutations': 0,
            'crashes_found': 0,
            'vulnerabilities_found': 0,
            'unique_coverage': 0,
            'field_selection_counts': np.zeros(self.n_fields),
            'mutation_action_counts': np.zeros(self.n_mutations)
        }
        
        # Load new protocol data
        if protocol_dataset_path:
            self.current_packet = self.data_processor.load_random_packet(protocol_dataset_path)
        else:
            self.current_packet = self.data_processor.generate_default_packet()
            
        # Extract Mamba features
        if self.use_mamba_features:
            self.current_features = self.data_processor.extract_mamba_features(self.current_packet)
        else:
            self.current_features = np.zeros(self.mamba_feature_dim)
            
        # Get initial state
        state = self._get_state()
        obs = [state]  # Single agent in fuzzing context
        
        return state, None, obs, [], False
        
    def step(self, field_idx: int, mutation_action: int) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float, np.ndarray, bool, Dict]:
        """Execute one fuzzing step with hierarchical actions.
        
        Args:
            field_idx: Selected protocol field index (high-level action)
            mutation_action: Selected mutation action index (low-level action)
            
        Returns:
            state_home: Next global state
            state_away: Not used (None)
            list_obs_home: Next observations
            list_obs_away: Not used (empty list)
            reward: Environment reward
            local_rewards: Local rewards for agents
            done: Episode termination
            info: Additional information
        """
        self.current_step += 1
        
        # Validate actions
        if field_idx >= self.n_fields or mutation_action >= self.n_mutations:
            raise ValueError(f"Invalid action: field_idx={field_idx}, mutation_action={mutation_action}")
            
        # Update statistics
        self.stats['total_mutations'] += 1
        self.stats['field_selection_counts'][field_idx] += 1
        self.stats['mutation_action_counts'][mutation_action] += 1
        
        # Execute mutation
        mutated_packet = self._apply_mutation(field_idx, mutation_action)
        
        # Execute fuzzing and collect feedback
        mutation_result = self._execute_fuzzing(mutated_packet)
        
        # Calculate reward based on fuzzing results
        reward = self.reward_calculator.calculate_reward(mutation_result)
        
        # Update field importance scores based on reward
        self._update_field_importance(field_idx, reward)
        
        # Update environment state
        self.current_packet = mutated_packet
        if self.use_mamba_features:
            self.current_features = self.data_processor.extract_mamba_features(mutated_packet)
            
        # Check termination conditions
        done = self._is_done()
        
        # Prepare return values
        state = self._get_state()
        obs = [state]
        local_rewards = np.array([reward])  # Single agent
        
        info = {
            'mutation_result': mutation_result,
            'field_idx': field_idx,
            'mutation_action': mutation_action,
            'stats': self.stats.copy()
        }
        
        return state, None, obs, [], reward, local_rewards, done, info
        
    def _apply_mutation(self, field_idx: int, mutation_action: int) -> Dict:
        """Apply mutation to selected protocol field.
        
        Args:
            field_idx: Index of field to mutate
            mutation_action: Type of mutation to apply
            
        Returns:
            Mutated packet dictionary
        """
        mutated_packet = self.current_packet.copy()
        field_name = self.protocol_fields[field_idx]['name']
        current_value = mutated_packet.get(field_name, b'')
        
        mutation_name = self.mutation_actions[mutation_action]
        
        if mutation_name == 'insert':
            # Insert random bytes
            insert_data = np.random.randint(0, 256, size=random.randint(1, 8), dtype=np.uint8).tobytes()
            mutated_packet[field_name] = current_value + insert_data
            
        elif mutation_name == 'delete':
            # Delete random portion
            if len(current_value) > 1:
                start = random.randint(0, len(current_value) - 1)
                end = random.randint(start + 1, len(current_value))
                mutated_packet[field_name] = current_value[:start] + current_value[end:]
                
        elif mutation_name == 'flip':
            # Flip random bits
            if current_value:
                value_array = bytearray(current_value)
                for _ in range(random.randint(1, min(8, len(value_array)))):
                    byte_idx = random.randint(0, len(value_array) - 1)
                    bit_idx = random.randint(0, 7)
                    value_array[byte_idx] ^= (1 << bit_idx)
                mutated_packet[field_name] = bytes(value_array)
                
        elif mutation_name == 'replace':
            # Replace with random data
            if current_value:
                new_length = random.randint(1, max(1, len(current_value) * 2))
                mutated_packet[field_name] = np.random.randint(0, 256, size=new_length, dtype=np.uint8).tobytes()
                
        elif mutation_name == 'shuffle':
            # Shuffle bytes
            if len(current_value) > 1:
                value_list = list(current_value)
                random.shuffle(value_list)
                mutated_packet[field_name] = bytes(value_list)
                
        elif mutation_name == 'copy':
            # Duplicate field content
            mutated_packet[field_name] = current_value + current_value
            
        return mutated_packet
        
    def _execute_fuzzing(self, packet: Dict) -> Dict:
        """Execute fuzzing with mutated packet and collect results.
        
        Args:
            packet: Mutated protocol packet
            
        Returns:
            Dictionary with fuzzing results
        """
        # Simulate fuzzing execution
        # In real implementation, this would interface with actual fuzzing tools
        
        result = {
            'coverage_increase': 0,
            'new_states': 0,
            'crash_detected': False,
            'timeout_detected': False,
            'vulnerability_detected': False,
            'execution_time': random.uniform(0.001, 0.1),
            'unique_packet': True
        }
        
        # Simulate coverage feedback
        packet_hash = hash(str(packet))
        if packet_hash not in self.coverage_history:
            result['coverage_increase'] = random.randint(1, 10)
            result['new_states'] = random.randint(0, 5)
            self.coverage_history.add(packet_hash)
            self.stats['unique_coverage'] += result['coverage_increase']
        else:
            result['unique_packet'] = False
            
        # Simulate crash detection (rare event)
        if random.random() < 0.001:  # 0.1% chance
            result['crash_detected'] = True
            self.crash_history.append(packet.copy())
            self.stats['crashes_found'] += 1
            
        # Simulate timeout detection
        if random.random() < 0.005:  # 0.5% chance
            result['timeout_detected'] = True
            
        # Simulate vulnerability detection (very rare)
        if random.random() < 0.0001:  # 0.01% chance
            result['vulnerability_detected'] = True
            self.vulnerability_history.append(packet.copy())
            self.stats['vulnerabilities_found'] += 1
            
        return result
        
    def _update_field_importance(self, field_idx: int, reward: float):
        """Update field importance scores based on reward feedback.
        
        Args:
            field_idx: Index of field that was mutated
            reward: Reward received for the mutation
        """
        # Use exponential moving average to update importance scores
        learning_rate = 0.1
        
        # Normalize reward to [0, 1] range for importance scoring
        normalized_reward = max(0, min(1, reward / 100.0))
        
        # Update importance score for the selected field
        self.field_importance_scores[field_idx] = (
            (1 - learning_rate) * self.field_importance_scores[field_idx] +
            learning_rate * normalized_reward
        )
        
        # Ensure importance scores stay normalized
        total_importance = np.sum(self.field_importance_scores)
        if total_importance > 0:
            self.field_importance_scores /= total_importance
        else:
            self.field_importance_scores = np.ones(self.n_fields) / self.n_fields
            
    def _get_state(self) -> np.ndarray:
        """Get current environment state representation.
        
        Returns:
            State vector for RL agent
        """
        state_components = []
        
        # Mamba features
        state_components.append(self.current_features)
        
        # Field importance scores
        state_components.append(self.field_importance_scores)
        
        # Mutation action availability (all actions always available)
        state_components.append(np.ones(self.n_mutations))
        
        # Environment statistics
        progress = self.current_step / self.max_steps
        coverage_rate = len(self.coverage_history) / max(1, self.stats['total_mutations'])
        crash_rate = self.stats['crashes_found'] / max(1, self.stats['total_mutations'])
        vuln_rate = self.stats['vulnerabilities_found'] / max(1, self.stats['total_mutations'])
        
        state_components.append(np.array([progress, coverage_rate, crash_rate, vuln_rate]))
        
        return np.concatenate(state_components)
        
    def _is_done(self) -> bool:
        """Check if fuzzing episode should terminate.
        
        Returns:
            True if episode should end
        """
        # Terminate if max steps reached
        if self.current_step >= self.max_steps:
            return True
            
        # Terminate if vulnerability found (high-level goal achieved)
        if self.stats['vulnerabilities_found'] > 0:
            return True
            
        return False
        
    def select_field_hierarchical(self, field_q_values: np.ndarray, epsilon: float = 0.1) -> int:
        """Select protocol field using hierarchical strategy (90/10 exploit/explore).
        
        Args:
            field_q_values: Q-values for each field from high-level policy
            epsilon: Additional exploration parameter
            
        Returns:
            Selected field index
        """
        if random.random() < epsilon:
            # Pure random exploration
            return random.randint(0, self.n_fields - 1)
            
        # 90/10 strategy: 90% based on reward history, 10% random
        if random.random() < self.exploit_probability:
            # Exploit: Select based on combined Q-values and importance scores
            combined_scores = field_q_values * self.field_importance_scores
            return np.argmax(combined_scores)
        else:
            # Explore: Random selection
            return random.randint(0, self.n_fields - 1)
            
    def get_mutation_context(self, field_idx: int) -> Dict:
        """Get context information for mutation action selection.
        
        Args:
            field_idx: Selected field index
            
        Returns:
            Context dictionary for low-level policy
        """
        field_info = self.protocol_fields[field_idx]
        current_value = self.current_packet.get(field_info['name'], b'')
        
        context = {
            'field_type': field_info.get('type', 'bytes'),
            'field_length': len(current_value),
            'field_importance': self.field_importance_scores[field_idx],
            'previous_mutations': self.stats['field_selection_counts'][field_idx],
            'current_step': self.current_step,
            'total_mutations': self.stats['total_mutations']
        }
        
        return context
        
    def get_statistics(self) -> Dict:
        """Get current fuzzing campaign statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        stats.update({
            'coverage_diversity': len(self.coverage_history),
            'mutation_efficiency': stats['crashes_found'] / max(1, stats['total_mutations']),
            'field_importance_scores': self.field_importance_scores.copy(),
            'steps_remaining': self.max_steps - self.current_step
        })
        return stats