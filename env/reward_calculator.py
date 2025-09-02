"""Reward Calculator for Fuzzing Framework.

This module implements the three-tier reward system for fuzzing:
- Basic rewards: code coverage, novelty, packet retention
- Medium rewards: crashes, timeouts, protocol errors
- High rewards: vulnerability discovery
"""

import numpy as np
from typing import Dict, List, Optional, Any


class RewardCalculator:
    """
    Implements three-tier reward system for fuzzing framework.
    
    Tier 1 (Basic): Coverage increase, state novelty, unique packet retention
    Tier 2 (Medium): Crashes, timeouts, protocol state anomalies  
    Tier 3 (High): Vulnerability discovery (highest reward)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.config = config
        
        # Reward tier weights
        self.base_reward_weight = config.get('base_reward_weight', 1.0)
        self.medium_reward_weight = config.get('medium_reward_weight', 5.0)
        self.high_reward_weight = config.get('high_reward_weight', 100.0)
        
        # Basic reward parameters
        self.coverage_reward_scale = config.get('coverage_reward_scale', 10.0)
        self.novelty_reward_scale = config.get('novelty_reward_scale', 5.0)
        self.retention_reward_scale = config.get('retention_reward_scale', 2.0)
        
        # Medium reward parameters
        self.crash_reward = config.get('crash_reward', 20.0)
        self.timeout_reward = config.get('timeout_reward', 15.0)
        self.protocol_error_reward = config.get('protocol_error_reward', 10.0)
        
        # High reward parameters
        self.vulnerability_reward = config.get('vulnerability_reward', 1000.0)
        
        # State tracking for novelty computation
        self.seen_states = set()
        self.coverage_history = []
        self.recent_rewards = []
        
        # Reward normalization
        self.reward_normalization = config.get('reward_normalization', True)
        self.max_reward_magnitude = config.get('max_reward_magnitude', 100.0)
        
    def calculate_reward(self, mutation_result: Dict) -> float:
        """Calculate total reward using three-tier system.
        
        Args:
            mutation_result: Dictionary containing mutation execution results
            
        Returns:
            Total reward value
        """
        # Calculate rewards for each tier
        basic_reward = self._calculate_basic_rewards(mutation_result)
        medium_reward = self._calculate_medium_rewards(mutation_result)
        high_reward = self._calculate_high_rewards(mutation_result)
        
        # Apply tier weights
        weighted_basic = self.base_reward_weight * basic_reward
        weighted_medium = self.medium_reward_weight * medium_reward
        weighted_high = self.high_reward_weight * high_reward
        
        # Combine rewards
        total_reward = weighted_basic + weighted_medium + weighted_high
        
        # Apply normalization if enabled
        if self.reward_normalization:
            total_reward = self._normalize_reward(total_reward)
        
        # Track for future calculations
        self.recent_rewards.append(total_reward)
        if len(self.recent_rewards) > 1000:  # Keep last 1000 rewards
            self.recent_rewards = self.recent_rewards[-1000:]
            
        return float(total_reward)
    
    def _calculate_basic_rewards(self, mutation_result: Dict) -> float:
        """Calculate basic tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Basic reward value
        """
        reward = 0.0
        
        # Coverage increase reward
        coverage_increase = mutation_result.get('coverage_increase', 0.0)
        if coverage_increase > 0:
            # Logarithmic scaling for diminishing returns
            coverage_reward = self.coverage_reward_scale * np.log(1 + coverage_increase)
            reward += coverage_reward
            
            # Track coverage history
            self.coverage_history.append(coverage_increase)
            if len(self.coverage_history) > 500:
                self.coverage_history = self.coverage_history[-500:]
        
        # State novelty reward
        novelty_reward = self._calculate_novelty_reward(mutation_result)
        reward += novelty_reward
        
        # Unique packet retention reward
        retention_reward = self._calculate_retention_reward(mutation_result)
        reward += retention_reward
        
        return reward
    
    def _calculate_medium_rewards(self, mutation_result: Dict) -> float:
        """Calculate medium tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Medium reward value
        """
        reward = 0.0
        
        # Crash detection reward
        if mutation_result.get('crash_detected', False):
            reward += self.crash_reward
            
            # Bonus for crash in critical components
            if self._is_critical_crash(mutation_result):
                reward += self.crash_reward * 0.5
        
        # Timeout/deadlock reward
        if mutation_result.get('timeout_detected', False):
            reward += self.timeout_reward
            
            # Bonus for deterministic timeouts
            if self._is_deterministic_timeout(mutation_result):
                reward += self.timeout_reward * 0.3
        
        # Protocol state anomaly reward
        if mutation_result.get('protocol_error', False):
            reward += self.protocol_error_reward
            
            # Bonus for state corruption
            if self._is_state_corruption(mutation_result):
                reward += self.protocol_error_reward * 0.4
        
        return reward
    
    def _calculate_high_rewards(self, mutation_result: Dict) -> float:
        """Calculate high tier rewards.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            High reward value
        """
        reward = 0.0
        
        # Vulnerability discovery reward (highest priority)
        if mutation_result.get('vulnerability_found', False):
            reward += self.vulnerability_reward
            
            # Bonus based on vulnerability severity
            severity = mutation_result.get('vulnerability_severity', 'medium')
            severity_multiplier = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'critical': 2.0
            }
            reward *= severity_multiplier.get(severity, 1.0)
            
            # Bonus for exploit development potential
            if mutation_result.get('exploitable', False):
                reward += self.vulnerability_reward * 0.5
        
        return reward
    
    def _calculate_novelty_reward(self, mutation_result: Dict) -> float:
        """Calculate reward for state novelty.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Novelty reward value
        """
        # Create state signature from mutation characteristics
        state_signature = self._create_state_signature(mutation_result)
        
        if state_signature not in self.seen_states:
            self.seen_states.add(state_signature)
            
            # Novel state reward
            novelty_reward = self.novelty_reward_scale
            
            # Bonus for rare mutation combinations
            if self._is_rare_combination(mutation_result):
                novelty_reward *= 1.5
                
            return novelty_reward
        else:
            # Small penalty for repeated states
            return -0.1
    
    def _calculate_retention_reward(self, mutation_result: Dict) -> float:
        """Calculate reward for unique packet retention.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Retention reward value
        """
        # Reward for mutations that create interesting but valid packets
        if (mutation_result.get('success', False) and 
            not mutation_result.get('crash_detected', False) and
            mutation_result.get('coverage_increase', 0) > 0):
            
            # Base retention reward
            retention_reward = self.retention_reward_scale
            
            # Bonus for complex mutations
            complexity = self._estimate_mutation_complexity(mutation_result)
            retention_reward *= (1.0 + complexity * 0.5)
            
            return retention_reward
        
        return 0.0
    
    def _create_state_signature(self, mutation_result: Dict) -> str:
        """Create unique signature for mutation state.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            State signature string
        """
        components = [
            str(mutation_result.get('field_idx', 0)),
            mutation_result.get('mutation_type', 'unknown'),
            mutation_result.get('field_type', 'unknown'),
            str(len(mutation_result.get('mutated_value', ''))),
            str(mutation_result.get('coverage_increase', 0.0) > 0),
            str(mutation_result.get('protocol_error', False))
        ]
        
        return '_'.join(components)
    
    def _is_rare_combination(self, mutation_result: Dict) -> bool:
        """Check if mutation represents a rare combination.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            True if combination is rare
        """
        mutation_type = mutation_result.get('mutation_type', 'unknown')
        field_type = mutation_result.get('field_type', 'unknown')
        
        # Define rare combinations
        rare_combinations = {
            ('copy', 'checksum'),
            ('shuffle', 'length'),
            ('insert', 'header'),
            ('replace', 'address')
        }
        
        return (mutation_type, field_type) in rare_combinations
    
    def _estimate_mutation_complexity(self, mutation_result: Dict) -> float:
        """Estimate complexity of mutation.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            Complexity score between 0 and 1
        """
        complexity = 0.0
        
        # Mutation type complexity
        mutation_type = mutation_result.get('mutation_type', 'unknown')
        type_complexity = {
            'flip': 0.2,
            'replace': 0.3,
            'delete': 0.4,
            'insert': 0.6,
            'shuffle': 0.7,
            'copy': 0.8
        }
        complexity += type_complexity.get(mutation_type, 0.1)
        
        # Field type complexity
        field_type = mutation_result.get('field_type', 'unknown')
        if field_type in ['checksum', 'length']:
            complexity += 0.3  # More complex to mutate correctly
        elif field_type == 'header':
            complexity += 0.2
        
        # Value size complexity
        mutated_value = mutation_result.get('mutated_value', '')
        size_complexity = min(len(str(mutated_value)) / 100.0, 0.3)
        complexity += size_complexity
        
        return min(complexity, 1.0)
    
    def _is_critical_crash(self, mutation_result: Dict) -> bool:
        """Check if crash occurred in critical component.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            True if crash is in critical component
        """
        field_type = mutation_result.get('field_type', 'unknown')
        mutation_type = mutation_result.get('mutation_type', 'unknown')
        
        # Critical crashes: header/checksum mutations that cause crashes
        if field_type in ['header', 'checksum'] and mutation_type in ['flip', 'replace']:
            return True
            
        # Crashes from length field mutations
        if field_type == 'length' and mutation_type in ['replace', 'flip']:
            return True
            
        return False
    
    def _is_deterministic_timeout(self, mutation_result: Dict) -> bool:
        """Check if timeout is deterministic/reproducible.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            True if timeout appears deterministic
        """
        # Timeouts from copy/insert mutations are often deterministic
        mutation_type = mutation_result.get('mutation_type', 'unknown')
        if mutation_type in ['copy', 'insert']:
            return True
            
        # Large payload mutations causing timeouts
        mutated_value = mutation_result.get('mutated_value', '')
        if len(str(mutated_value)) > 1000:
            return True
            
        return False
    
    def _is_state_corruption(self, mutation_result: Dict) -> bool:
        """Check if protocol error indicates state corruption.
        
        Args:
            mutation_result: Mutation execution results
            
        Returns:
            True if state corruption is likely
        """
        field_type = mutation_result.get('field_type', 'unknown')
        
        # State corruption more likely with certain field types
        if field_type in ['header', 'length', 'flag']:
            return True
            
        # Shuffle mutations often cause state corruption
        mutation_type = mutation_result.get('mutation_type', 'unknown')
        if mutation_type == 'shuffle':
            return True
            
        return False
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to prevent extreme values.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward
        """
        # Clip to maximum magnitude
        clipped_reward = np.clip(reward, -self.max_reward_magnitude, self.max_reward_magnitude)
        
        # Apply sigmoid-like normalization for very large rewards
        if abs(clipped_reward) > self.max_reward_magnitude * 0.8:
            sign = np.sign(clipped_reward)
            magnitude = abs(clipped_reward)
            # Soft clipping using tanh
            normalized_magnitude = self.max_reward_magnitude * np.tanh(magnitude / self.max_reward_magnitude)
            clipped_reward = sign * normalized_magnitude
        
        return clipped_reward
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about recent rewards.
        
        Returns:
            Dictionary with reward statistics
        """
        if not self.recent_rewards:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        rewards = np.array(self.recent_rewards)
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'count': len(rewards)
        }
    
    def get_coverage_trend(self) -> Dict[str, float]:
        """Get coverage increase trend.
        
        Returns:
            Dictionary with coverage trend information
        """
        if len(self.coverage_history) < 2:
            return {
                'trend': 0.0,
                'recent_average': 0.0,
                'total_coverage': 0.0
            }
        
        recent = self.coverage_history[-20:]  # Last 20 coverage increases
        older = self.coverage_history[-40:-20] if len(self.coverage_history) >= 40 else []
        
        recent_avg = np.mean(recent) if recent else 0.0
        older_avg = np.mean(older) if older else recent_avg
        
        trend = recent_avg - older_avg  # Positive = improving coverage
        
        return {
            'trend': float(trend),
            'recent_average': float(recent_avg),
            'total_coverage': float(np.sum(self.coverage_history))
        }
    
    def reset_state(self):
        """Reset internal state for new episode."""
        # Don't reset seen_states completely as we want to maintain novelty across episodes
        # But limit size to prevent memory issues
        if len(self.seen_states) > 10000:
            # Keep only recent states (this is a simplification)
            self.seen_states = set(list(self.seen_states)[-5000:])
        
        # Reset episode-specific tracking
        self.coverage_history = []
        self.recent_rewards = []
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update reward calculator configuration.
        
        Args:
            new_config: New configuration parameters
        """
        # Update weights
        self.base_reward_weight = new_config.get('base_reward_weight', self.base_reward_weight)
        self.medium_reward_weight = new_config.get('medium_reward_weight', self.medium_reward_weight)
        self.high_reward_weight = new_config.get('high_reward_weight', self.high_reward_weight)
        
        # Update scales
        self.coverage_reward_scale = new_config.get('coverage_reward_scale', self.coverage_reward_scale)
        self.novelty_reward_scale = new_config.get('novelty_reward_scale', self.novelty_reward_scale)
        self.retention_reward_scale = new_config.get('retention_reward_scale', self.retention_reward_scale)
        
        # Update specific rewards
        self.crash_reward = new_config.get('crash_reward', self.crash_reward)
        self.timeout_reward = new_config.get('timeout_reward', self.timeout_reward)
        self.protocol_error_reward = new_config.get('protocol_error_reward', self.protocol_error_reward)
        self.vulnerability_reward = new_config.get('vulnerability_reward', self.vulnerability_reward)
        
        # Merge config
        self.config.update(new_config)