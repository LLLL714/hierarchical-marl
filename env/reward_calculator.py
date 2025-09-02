"""Reward Calculator for Fuzzing Framework.

This module implements the three-tier reward system for the hierarchical fuzzing 
reinforcement learning framework:
- Basic rewards: Coverage, novelty, retention
- Medium rewards: Crashes, timeouts, protocol anomalies  
- High rewards: Vulnerability discovery
"""

import numpy as np
from typing import Dict, List, Any, Optional
import time


class RewardCalculator:
    """Calculates hierarchical rewards for fuzzing activities."""
    
    def __init__(self, config: Dict):
        """Initialize reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.config = config
        
        # Reward tier weights
        self.base_weight = config.get('base_reward_weight', 1.0)
        self.medium_weight = config.get('medium_reward_weight', 5.0) 
        self.high_weight = config.get('high_reward_weight', 100.0)
        
        # Specific reward coefficients
        self.coverage_coefficient = config.get('coverage_reward_coeff', 0.1)
        self.novelty_coefficient = config.get('novelty_reward_coeff', 0.2)
        self.retention_coefficient = config.get('retention_reward_coeff', 0.05)
        
        self.crash_reward = config.get('crash_reward', 10.0)
        self.timeout_reward = config.get('timeout_reward', 5.0)
        self.anomaly_reward = config.get('anomaly_reward', 3.0)
        
        self.vulnerability_reward = config.get('vulnerability_reward', 1000.0)
        
        # Reward history for adaptive scaling
        self.reward_history = []
        self.max_history_length = config.get('reward_history_length', 1000)
        
        # Coverage tracking
        self.coverage_baseline = 0
        self.novelty_threshold = config.get('novelty_threshold', 0.8)
        
        # Reward penalties
        self.duplicate_penalty = config.get('duplicate_penalty', -0.1)
        self.invalid_mutation_penalty = config.get('invalid_mutation_penalty', -1.0)
        
    def calculate_reward(self, mutation_result: Dict) -> float:
        """Calculate total reward for a fuzzing mutation result.
        
        Args:
            mutation_result: Dictionary containing fuzzing results
            
        Returns:
            Total reward value
        """
        total_reward = 0.0
        
        # Calculate base rewards
        base_reward = self._calculate_base_rewards(mutation_result)
        total_reward += self.base_weight * base_reward
        
        # Calculate medium-tier rewards
        medium_reward = self._calculate_medium_rewards(mutation_result)
        total_reward += self.medium_weight * medium_reward
        
        # Calculate high-tier rewards
        high_reward = self._calculate_high_rewards(mutation_result)
        total_reward += self.high_weight * high_reward
        
        # Apply penalties
        penalty = self._calculate_penalties(mutation_result)
        total_reward += penalty
        
        # Apply adaptive scaling
        scaled_reward = self._apply_adaptive_scaling(total_reward)
        
        # Update reward history
        self._update_reward_history(scaled_reward)
        
        return scaled_reward
        
    def _calculate_base_rewards(self, mutation_result: Dict) -> float:
        """Calculate basic-tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Base reward value
        """
        base_reward = 0.0
        
        # Coverage increase reward
        coverage_increase = mutation_result.get('coverage_increase', 0)
        if coverage_increase > 0:
            # Logarithmic scaling to prevent explosive rewards
            coverage_reward = self.coverage_coefficient * np.log(1 + coverage_increase)
            base_reward += coverage_reward
            
        # State novelty reward
        new_states = mutation_result.get('new_states', 0)
        if new_states > 0:
            novelty_reward = self.novelty_coefficient * np.sqrt(new_states)
            base_reward += novelty_reward
            
        # Unique packet retention reward
        if mutation_result.get('unique_packet', False):
            base_reward += self.retention_coefficient
            
        # Execution stability reward (inverse of execution time variance)
        exec_time = mutation_result.get('execution_time', 0.1)
        if 0.001 <= exec_time <= 1.0:  # Normal execution time range
            stability_reward = self.retention_coefficient * 0.5
            base_reward += stability_reward
            
        return base_reward
        
    def _calculate_medium_rewards(self, mutation_result: Dict) -> float:
        """Calculate medium-tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Medium reward value
        """
        medium_reward = 0.0
        
        # Crash detection reward
        if mutation_result.get('crash_detected', False):
            crash_type = mutation_result.get('crash_type', 'generic')
            if crash_type == 'segfault':
                medium_reward += self.crash_reward * 1.5
            elif crash_type == 'heap_corruption':
                medium_reward += self.crash_reward * 2.0
            else:
                medium_reward += self.crash_reward
                
        # Timeout/deadlock detection reward
        if mutation_result.get('timeout_detected', False):
            timeout_duration = mutation_result.get('timeout_duration', 1.0)
            # Longer timeouts may indicate more interesting conditions
            timeout_multiplier = min(2.0, 1.0 + np.log(timeout_duration))
            medium_reward += self.timeout_reward * timeout_multiplier
            
        # Protocol state anomaly reward
        if mutation_result.get('protocol_anomaly', False):
            anomaly_severity = mutation_result.get('anomaly_severity', 1.0)
            medium_reward += self.anomaly_reward * anomaly_severity
            
        # Memory usage anomaly reward
        memory_anomaly = mutation_result.get('memory_anomaly', False)
        if memory_anomaly:
            medium_reward += self.anomaly_reward * 0.7
            
        # Resource exhaustion reward
        if mutation_result.get('resource_exhaustion', False):
            medium_reward += self.anomaly_reward * 0.8
            
        return medium_reward
        
    def _calculate_high_rewards(self, mutation_result: Dict) -> float:
        """Calculate high-tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            High reward value
        """
        high_reward = 0.0
        
        # Vulnerability detection reward (highest priority)
        if mutation_result.get('vulnerability_detected', False):
            vuln_type = mutation_result.get('vulnerability_type', 'unknown')
            
            # Different vulnerability types have different rewards
            vuln_multipliers = {
                'buffer_overflow': 1.5,
                'use_after_free': 1.8,
                'double_free': 1.6,
                'null_pointer_dereference': 1.2,
                'format_string': 1.4,
                'integer_overflow': 1.3,
                'race_condition': 2.0,
                'unknown': 1.0
            }
            
            multiplier = vuln_multipliers.get(vuln_type, 1.0)
            
            # Additional reward for exploitability assessment
            exploitability = mutation_result.get('exploitability_score', 0.5)
            exploitability_bonus = exploitability * self.vulnerability_reward * 0.5
            
            vulnerability_reward = self.vulnerability_reward * multiplier + exploitability_bonus
            high_reward += vulnerability_reward
            
        # Security boundary violation reward
        if mutation_result.get('security_violation', False):
            violation_severity = mutation_result.get('violation_severity', 0.5)
            high_reward += self.vulnerability_reward * 0.3 * violation_severity
            
        # Privilege escalation indicators
        if mutation_result.get('privilege_escalation', False):
            high_reward += self.vulnerability_reward * 0.4
            
        return high_reward
        
    def _calculate_penalties(self, mutation_result: Dict) -> float:
        """Calculate penalty rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Penalty value (negative)
        """
        penalty = 0.0
        
        # Duplicate packet penalty
        if not mutation_result.get('unique_packet', True):
            penalty += self.duplicate_penalty
            
        # Invalid mutation penalty
        if mutation_result.get('invalid_mutation', False):
            penalty += self.invalid_mutation_penalty
            
        # Excessive execution time penalty
        exec_time = mutation_result.get('execution_time', 0.1)
        if exec_time > 5.0:  # Very slow execution
            penalty += -0.5 * np.log(exec_time)
            
        # Malformed packet penalty
        if mutation_result.get('malformed_packet', False):
            penalty += -0.2
            
        # No-op mutation penalty (mutation that doesn't change anything)
        if mutation_result.get('no_change', False):
            penalty += -0.1
            
        return penalty
        
    def _apply_adaptive_scaling(self, reward: float) -> float:
        """Apply adaptive scaling based on reward history.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Scaled reward value
        """
        if len(self.reward_history) < 10:
            return reward
            
        # Calculate recent reward statistics
        recent_rewards = self.reward_history[-100:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        if std_reward < 1e-6:  # Avoid division by zero
            return reward
            
        # Z-score normalization with bounds
        normalized_reward = (reward - mean_reward) / std_reward
        
        # Apply sigmoid scaling to prevent extreme values
        scaled_reward = 2.0 / (1.0 + np.exp(-normalized_reward)) - 1.0
        
        # Scale back to reasonable range
        scaled_reward *= 10.0
        
        return scaled_reward
        
    def _update_reward_history(self, reward: float):
        """Update reward history for adaptive scaling.
        
        Args:
            reward: New reward value to add
        """
        self.reward_history.append(reward)
        
        # Trim history if it gets too long
        if len(self.reward_history) > self.max_history_length:
            self.reward_history = self.reward_history[-self.max_history_length//2:]
            
    def calculate_field_importance_reward(self, field_idx: int, mutation_result: Dict, 
                                        historical_performance: Dict) -> float:
        """Calculate field-specific importance reward for updating field selection.
        
        Args:
            field_idx: Index of mutated field
            mutation_result: Results of the mutation
            historical_performance: Historical performance for this field
            
        Returns:
            Field importance reward
        """
        base_reward = self.calculate_reward(mutation_result)
        
        # Adjust based on field's historical performance
        field_avg_reward = historical_performance.get('avg_reward', 0.0)
        field_success_rate = historical_performance.get('success_rate', 0.0)
        field_mutation_count = historical_performance.get('mutation_count', 1)
        
        # Bonus for fields that historically perform well
        history_bonus = 0.0
        if field_mutation_count > 10:  # Only apply after sufficient data
            if field_avg_reward > 0:
                history_bonus = min(2.0, field_avg_reward * 0.1)
            
            if field_success_rate > 0.1:  # 10% success rate threshold
                history_bonus += field_success_rate * 0.5
                
        # Exploration bonus for under-explored fields
        exploration_bonus = 0.0
        if field_mutation_count < 5:
            exploration_bonus = 0.5  # Encourage exploration of new fields
            
        total_field_reward = base_reward + history_bonus + exploration_bonus
        
        return total_field_reward
        
    def calculate_mutation_action_reward(self, action_idx: int, mutation_result: Dict,
                                       context: Dict) -> float:
        """Calculate mutation action-specific reward.
        
        Args:
            action_idx: Index of mutation action used
            mutation_result: Results of the mutation
            context: Context information (field type, size, etc.)
            
        Returns:
            Action-specific reward
        """
        base_reward = self.calculate_reward(mutation_result)
        
        # Action effectiveness bonuses based on context
        action_bonus = 0.0
        
        field_type = context.get('field_type', 'bytes')
        field_length = context.get('field_length', 0)
        
        # Action-specific bonuses
        action_names = ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy']
        action_name = action_names[action_idx] if action_idx < len(action_names) else 'unknown'
        
        if action_name == 'flip' and field_type == 'bitfield':
            action_bonus += 0.2  # Bit flipping good for bitfields
        elif action_name == 'insert' and field_length < 10:
            action_bonus += 0.1  # Insertion good for small fields
        elif action_name == 'delete' and field_length > 100:
            action_bonus += 0.1  # Deletion good for large fields
        elif action_name == 'replace' and field_type == 'integer':
            action_bonus += 0.15  # Replacement good for integer fields
        elif action_name == 'shuffle' and field_length > 4:
            action_bonus += 0.1  # Shuffling good for multi-byte fields
            
        return base_reward + action_bonus
        
    def get_reward_statistics(self) -> Dict:
        """Get statistics about reward distribution.
        
        Returns:
            Dictionary with reward statistics
        """
        if not self.reward_history:
            return {}
            
        rewards = np.array(self.reward_history)
        
        stats = {
            'total_rewards': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            'positive_rewards': int(np.sum(rewards > 0)),
            'negative_rewards': int(np.sum(rewards < 0)),
            'zero_rewards': int(np.sum(rewards == 0))
        }
        
        # Percentiles
        percentiles = [10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}_reward'] = float(np.percentile(rewards, p))
            
        return stats
        
    def adjust_reward_parameters(self, performance_metrics: Dict):
        """Dynamically adjust reward parameters based on performance.
        
        Args:
            performance_metrics: Current performance metrics
        """
        # Adjust based on discovery rates
        crash_rate = performance_metrics.get('crash_rate', 0.0)
        vuln_rate = performance_metrics.get('vulnerability_rate', 0.0)
        coverage_rate = performance_metrics.get('coverage_growth_rate', 0.0)
        
        # If too many crashes, reduce crash reward to encourage diversity
        if crash_rate > 0.1:  # More than 10% crash rate
            self.crash_reward *= 0.9
            
        # If vulnerabilities are rare, increase reward
        if vuln_rate < 0.001:  # Less than 0.1% vulnerability rate
            self.vulnerability_reward *= 1.1
            
        # If coverage growth is slow, increase coverage rewards
        if coverage_rate < 0.01:  # Less than 1% coverage growth
            self.coverage_coefficient *= 1.05
            
        # Bound the parameters to reasonable ranges
        self.crash_reward = max(1.0, min(50.0, self.crash_reward))
        self.vulnerability_reward = max(100.0, min(10000.0, self.vulnerability_reward))
        self.coverage_coefficient = max(0.01, min(1.0, self.coverage_coefficient))
        
    def reset_reward_history(self):
        """Reset reward history (e.g., between episodes)."""
        self.reward_history.clear()
        
    def get_reward_breakdown(self, mutation_result: Dict) -> Dict:
        """Get detailed breakdown of reward components.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Dictionary with reward component breakdown
        """
        breakdown = {
            'base_rewards': {
                'coverage': 0.0,
                'novelty': 0.0, 
                'retention': 0.0,
                'stability': 0.0
            },
            'medium_rewards': {
                'crash': 0.0,
                'timeout': 0.0,
                'anomaly': 0.0,
                'memory': 0.0
            },
            'high_rewards': {
                'vulnerability': 0.0,
                'security_violation': 0.0,
                'privilege_escalation': 0.0
            },
            'penalties': {
                'duplicate': 0.0,
                'invalid': 0.0,
                'slow_execution': 0.0,
                'malformed': 0.0
            }
        }
        
        # Calculate each component separately for detailed analysis
        coverage_increase = mutation_result.get('coverage_increase', 0)
        if coverage_increase > 0:
            breakdown['base_rewards']['coverage'] = self.coverage_coefficient * np.log(1 + coverage_increase)
            
        new_states = mutation_result.get('new_states', 0)
        if new_states > 0:
            breakdown['base_rewards']['novelty'] = self.novelty_coefficient * np.sqrt(new_states)
            
        if mutation_result.get('unique_packet', False):
            breakdown['base_rewards']['retention'] = self.retention_coefficient
            
        if mutation_result.get('crash_detected', False):
            breakdown['medium_rewards']['crash'] = self.crash_reward
            
        if mutation_result.get('vulnerability_detected', False):
            breakdown['high_rewards']['vulnerability'] = self.vulnerability_reward
            
        if not mutation_result.get('unique_packet', True):
            breakdown['penalties']['duplicate'] = self.duplicate_penalty
            
        # Calculate totals
        breakdown['total_base'] = sum(breakdown['base_rewards'].values())
        breakdown['total_medium'] = sum(breakdown['medium_rewards'].values())
        breakdown['total_high'] = sum(breakdown['high_rewards'].values())
        breakdown['total_penalties'] = sum(breakdown['penalties'].values())
        
        breakdown['grand_total'] = (
            self.base_weight * breakdown['total_base'] +
            self.medium_weight * breakdown['total_medium'] +
            self.high_weight * breakdown['total_high'] +
            breakdown['total_penalties']
        )
        
        return breakdown