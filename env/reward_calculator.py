"""Three-tier reward calculator for fuzzing reinforcement learning.

Implements the reward structure:
- Basic rewards: Coverage increase, state novelty, unique packet retention
- Medium rewards: Crash signals, timeouts/deadlocks, protocol state anomalies  
- High rewards: Actual vulnerability discovery (highest reward)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import time


class RewardCalculator:
    """Calculates hierarchical rewards for fuzzing actions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        self.config = config
        
        # Reward weights for three tiers
        self.base_reward_weight = config.get('base_reward_weight', 1.0)
        self.medium_reward_weight = config.get('medium_reward_weight', 5.0)
        self.high_reward_weight = config.get('high_reward_weight', 100.0)
        
        # Individual reward component weights
        self.coverage_weight = config.get('coverage_weight', 1.0)
        self.novelty_weight = config.get('novelty_weight', 0.5)
        self.uniqueness_weight = config.get('uniqueness_weight', 0.3)
        self.crash_weight = config.get('crash_weight', 1.0)
        self.timeout_weight = config.get('timeout_weight', 0.8)
        self.anomaly_weight = config.get('anomaly_weight', 0.6)
        self.vulnerability_weight = config.get('vulnerability_weight', 1.0)
        
        # State tracking for novelty calculation
        self.seen_states = set()
        self.coverage_history = []
        self.max_coverage = 0.0
        
        # Unique packet tracking
        self.unique_packets = set()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_reward(self, execution_result: Dict, mutation_results: List[Dict], step_count: int) -> Tuple[float, List[float]]:
        """Calculate total reward and individual agent rewards.
        
        Args:
            execution_result: Results from fuzzing execution
            mutation_results: Results from mutation operations
            step_count: Current step in episode
            
        Returns:
            Tuple of (total_reward, list_of_individual_rewards)
        """
        # Calculate three-tier rewards
        basic_reward = self._calculate_basic_rewards(execution_result, mutation_results)
        medium_reward = self._calculate_medium_rewards(execution_result)
        high_reward = self._calculate_high_rewards(execution_result)
        
        # Weighted combination
        total_reward = (
            self.base_reward_weight * basic_reward +
            self.medium_reward_weight * medium_reward +
            self.high_reward_weight * high_reward
        )
        
        # Calculate individual agent rewards (one per mutation action)
        individual_rewards = self._calculate_individual_rewards(
            execution_result, mutation_results, total_reward
        )
        
        # Log rewards for debugging
        self._log_reward_breakdown(basic_reward, medium_reward, high_reward, total_reward)
        
        return total_reward, individual_rewards
        
    def _calculate_basic_rewards(self, execution_result: Dict, mutation_results: List[Dict]) -> float:
        """Calculate basic tier rewards.
        
        Components:
        - Coverage increase
        - State novelty  
        - Unique packet retention
        """
        reward = 0.0
        
        # Coverage increase reward
        coverage_reward = self._calculate_coverage_reward(execution_result)
        reward += self.coverage_weight * coverage_reward
        
        # State novelty reward
        novelty_reward = self._calculate_novelty_reward(execution_result)
        reward += self.novelty_weight * novelty_reward
        
        # Unique packet retention reward
        uniqueness_reward = self._calculate_uniqueness_reward(execution_result)
        reward += self.uniqueness_weight * uniqueness_reward
        
        return reward
        
    def _calculate_medium_rewards(self, execution_result: Dict) -> float:
        """Calculate medium tier rewards.
        
        Components:
        - Crash signals
        - Timeout/deadlock detection
        - Protocol state anomalies
        """
        reward = 0.0
        
        # Crash detection reward
        if execution_result.get('crash_detected', False):
            crash_reward = self._calculate_crash_reward(execution_result)
            reward += self.crash_weight * crash_reward
            
        # Timeout detection reward
        if execution_result.get('timeout_detected', False):
            timeout_reward = self._calculate_timeout_reward(execution_result)
            reward += self.timeout_weight * timeout_reward
            
        # Protocol anomaly reward
        anomaly_reward = self._calculate_anomaly_reward(execution_result)
        reward += self.anomaly_weight * anomaly_reward
        
        return reward
        
    def _calculate_high_rewards(self, execution_result: Dict) -> float:
        """Calculate high tier rewards.
        
        Components:
        - Vulnerability discovery (highest priority)
        """
        reward = 0.0
        
        if execution_result.get('vulnerability_found', False):
            vulnerability_reward = self._calculate_vulnerability_reward(execution_result)
            reward += self.vulnerability_weight * vulnerability_reward
            
        return reward
        
    def _calculate_coverage_reward(self, execution_result: Dict) -> float:
        """Calculate reward based on coverage increase."""
        coverage_increase = execution_result.get('coverage_increase', 0.0)
        
        # Store coverage history
        current_coverage = getattr(self, 'current_coverage', 0.0) + coverage_increase
        self.current_coverage = current_coverage
        self.coverage_history.append(current_coverage)
        
        # Reward is proportional to coverage increase
        # Bonus for reaching new maximum coverage
        if current_coverage > self.max_coverage:
            bonus = 0.1  # Bonus for new maximum
            self.max_coverage = current_coverage
        else:
            bonus = 0.0
            
        # Diminishing returns for small increases
        if coverage_increase > 0:
            reward = np.log(1 + coverage_increase * 100) + bonus
        else:
            reward = 0.0
            
        return reward
        
    def _calculate_novelty_reward(self, execution_result: Dict) -> float:
        """Calculate reward based on state novelty."""
        # Create state signature from execution result
        state_signature = self._create_state_signature(execution_result)
        
        if state_signature not in self.seen_states:
            self.seen_states.add(state_signature)
            # Reward for discovering new state
            reward = 1.0 / (1 + len(self.seen_states) * 0.01)  # Diminishing returns
        else:
            reward = 0.0
            
        return reward
        
    def _calculate_uniqueness_reward(self, execution_result: Dict) -> float:
        """Calculate reward for generating unique packets."""
        # This would typically hash the packet content
        # For now, use a simple uniqueness check
        if execution_result.get('unique_packet', True):
            # Small reward for unique packets
            reward = 0.1
        else:
            reward = 0.0
            
        return reward
        
    def _calculate_crash_reward(self, execution_result: Dict) -> float:
        """Calculate reward for crash detection."""
        crash_type = execution_result.get('crash_type', 'unknown')
        
        # Different rewards for different crash types
        crash_rewards = {
            'segmentation_fault': 1.0,
            'access_violation': 1.0,
            'assertion_failure': 0.8,
            'buffer_overflow': 1.2,
            'use_after_free': 1.1,
            'double_free': 1.1,
            'unknown': 0.5
        }
        
        base_reward = crash_rewards.get(crash_type, 0.5)
        
        # Bonus for reproducible crashes
        if execution_result.get('reproducible', False):
            base_reward *= 1.5
            
        return base_reward
        
    def _calculate_timeout_reward(self, execution_result: Dict) -> float:
        """Calculate reward for timeout/deadlock detection."""
        timeout_duration = execution_result.get('timeout_duration', 0)
        
        # Longer timeouts might indicate more serious issues
        if timeout_duration > 10.0:  # seconds
            reward = 0.8  # High timeout reward
        elif timeout_duration > 5.0:
            reward = 0.6  # Medium timeout reward
        else:
            reward = 0.4  # Low timeout reward
            
        return reward
        
    def _calculate_anomaly_reward(self, execution_result: Dict) -> float:
        """Calculate reward for protocol state anomalies."""
        anomalies = execution_result.get('protocol_anomalies', [])
        
        if not anomalies:
            return 0.0
            
        reward = 0.0
        anomaly_rewards = {
            'invalid_state_transition': 0.3,
            'malformed_response': 0.4,
            'unexpected_behavior': 0.5,
            'resource_exhaustion': 0.6,
            'logic_error': 0.7
        }
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', 'unknown')
            reward += anomaly_rewards.get(anomaly_type, 0.2)
            
        return min(reward, 1.0)  # Cap at 1.0
        
    def _calculate_vulnerability_reward(self, execution_result: Dict) -> float:
        """Calculate reward for vulnerability discovery."""
        vulnerability_type = execution_result.get('vulnerability_type', 'unknown')
        
        # Different rewards for different vulnerability types
        vuln_rewards = {
            'buffer_overflow': 1.0,
            'sql_injection': 0.9,
            'xss': 0.7,
            'command_injection': 0.95,
            'authentication_bypass': 0.85,
            'privilege_escalation': 0.9,
            'information_disclosure': 0.6,
            'denial_of_service': 0.5,
            'unknown': 0.4
        }
        
        base_reward = vuln_rewards.get(vulnerability_type, 0.4)
        
        # Bonus for critical vulnerabilities
        severity = execution_result.get('vulnerability_severity', 'medium')
        severity_multipliers = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.0,
            'low': 0.7
        }
        
        multiplier = severity_multipliers.get(severity, 1.0)
        
        # Additional bonus for exploitable vulnerabilities
        if execution_result.get('exploitable', False):
            multiplier *= 1.3
            
        return base_reward * multiplier
        
    def _calculate_individual_rewards(self, execution_result: Dict, mutation_results: List[Dict], total_reward: float) -> List[float]:
        """Calculate individual rewards for each mutation action.
        
        Distributes total reward among agents based on mutation success and contribution.
        """
        if not mutation_results:
            return [0.0]
            
        individual_rewards = []
        
        for mutation_result in mutation_results:
            # Base individual reward is fraction of total reward
            base_individual = total_reward / len(mutation_results)
            
            # Adjust based on mutation success
            if mutation_result.get('success', True):
                success_multiplier = 1.0
            else:
                success_multiplier = 0.1  # Small penalty for failed mutations
                
            # Adjust based on mutation impact (if available)
            impact_score = mutation_result.get('impact_score', 1.0)
            
            individual_reward = base_individual * success_multiplier * impact_score
            individual_rewards.append(individual_reward)
            
        return individual_rewards
        
    def _create_state_signature(self, execution_result: Dict) -> str:
        """Create a signature for the current execution state."""
        # Create hash-like signature from key execution properties
        signature_parts = [
            str(execution_result.get('coverage_increase', 0)),
            str(execution_result.get('crash_detected', False)),
            str(execution_result.get('timeout_detected', False)),
            str(execution_result.get('vulnerability_found', False)),
            str(execution_result.get('execution_time', 0))[:4]  # Truncate time
        ]
        
        return '|'.join(signature_parts)
        
    def _log_reward_breakdown(self, basic_reward: float, medium_reward: float, high_reward: float, total_reward: float):
        """Log reward breakdown for debugging."""
        if total_reward > 0.1:  # Only log significant rewards
            self.logger.debug(
                f"Reward breakdown - Basic: {basic_reward:.3f}, "
                f"Medium: {medium_reward:.3f}, High: {high_reward:.3f}, "
                f"Total: {total_reward:.3f}"
            )
            
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about rewards over time."""
        return {
            'total_states_seen': len(self.seen_states),
            'max_coverage_reached': self.max_coverage,
            'coverage_history_length': len(self.coverage_history),
            'current_coverage': getattr(self, 'current_coverage', 0.0)
        }
        
    def reset_episode(self):
        """Reset episode-specific tracking."""
        # Keep long-term state but reset episode counters
        self.current_coverage = 0.0