"""Fuzzing-specific Neural Networks for Hierarchical RL Framework.

This module contains neural network architectures adapted for fuzzing:
- FieldSelectorNetwork: High-level field selection with 90/10 strategy
- MutationActionNetwork: Low-level mutation action selection
- FeatureProcessor: Handles Mamba feature integration
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def get_variable(name, shape):
    """Helper function to create TensorFlow variables."""
    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0, 0.01))


class FieldSelectorNetwork:
    """High-level network for protocol field selection with 90/10 strategy."""
    
    def __init__(self, obs_dim, n_fields, n_h1=128, n_h2=128, scope_name="FieldSelector"):
        """Initialize field selector network.
        
        Args:
            obs_dim: Observation dimension
            n_fields: Number of protocol fields
            n_h1: First hidden layer size
            n_h2: Second hidden layer size
            scope_name: Variable scope name
        """
        self.obs_dim = obs_dim
        self.n_fields = n_fields
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope_name = scope_name
        
    def build_network(self, obs_input, field_importance, is_training=True):
        """Build field selector network.
        
        Args:
            obs_input: Observation input tensor [batch, obs_dim]
            field_importance: Field importance scores [batch, n_fields]
            is_training: Whether in training mode
            
        Returns:
            field_logits: Logits for field selection [batch, n_fields]
            field_probs: Probabilities for field selection [batch, n_fields]
        """
        with tf.variable_scope(self.scope_name):
            # Process observations
            obs_features = self._process_observations(obs_input, is_training)
            
            # Process field importance
            importance_features = self._process_field_importance(field_importance)
            
            # Combine features
            combined_features = tf.concat([obs_features, importance_features], axis=1)
            
            # Field selection head
            field_logits = self._build_field_head(combined_features)
            
            # Apply 90/10 strategy
            field_probs = self._apply_exploit_explore_strategy(field_logits, field_importance)
            
            return field_logits, field_probs
    
    def _process_observations(self, obs_input, is_training):
        """Process observation input through neural network."""
        h1 = tf.layers.dense(inputs=obs_input, units=self.n_h1, activation=tf.nn.relu,
                             use_bias=True, name='obs_h1')
        if is_training:
            h1 = tf.layers.dropout(h1, rate=0.1, name='obs_dropout1')
            
        h2 = tf.layers.dense(inputs=h1, units=self.n_h2, activation=tf.nn.relu,
                             use_bias=True, name='obs_h2')
        if is_training:
            h2 = tf.layers.dropout(h2, rate=0.1, name='obs_dropout2')
            
        return h2
    
    def _process_field_importance(self, field_importance):
        """Process field importance scores."""
        # Normalize importance scores
        normalized_importance = tf.nn.softmax(field_importance, name='normalized_importance')
        
        # Transform through small network
        importance_h = tf.layers.dense(inputs=normalized_importance, units=64, 
                                     activation=tf.nn.relu, use_bias=True, name='importance_h')
        
        return importance_h
    
    def _build_field_head(self, features):
        """Build field selection head."""
        field_h1 = tf.layers.dense(inputs=features, units=128, activation=tf.nn.relu,
                                  use_bias=True, name='field_h1')
        field_h2 = tf.layers.dense(inputs=field_h1, units=64, activation=tf.nn.relu,
                                  use_bias=True, name='field_h2')
        field_logits = tf.layers.dense(inputs=field_h2, units=self.n_fields, 
                                     activation=None, use_bias=True, name='field_logits')
        
        return field_logits
    
    def _apply_exploit_explore_strategy(self, field_logits, field_importance, exploit_ratio=0.9):
        """Apply 90/10 exploit/explore strategy to field selection."""
        # Exploit component: use learned logits weighted by importance
        exploit_logits = field_logits + tf.log(field_importance + 1e-8)
        exploit_probs = tf.nn.softmax(exploit_logits, name='exploit_probs')
        
        # Explore component: uniform distribution
        explore_probs = tf.ones_like(field_logits) / tf.cast(self.n_fields, tf.float32)
        
        # Combine with 90/10 ratio
        combined_probs = (exploit_ratio * exploit_probs + 
                         (1.0 - exploit_ratio) * explore_probs)
        
        return combined_probs


class MutationActionNetwork:
    """Low-level network for mutation action selection."""
    
    def __init__(self, obs_dim, field_features_dim, n_mutations=6, n_h1=64, n_h2=64, scope_name="MutationAction"):
        """Initialize mutation action network.
        
        Args:
            obs_dim: Observation dimension
            field_features_dim: Field-specific feature dimension
            n_mutations: Number of mutation actions
            n_h1: First hidden layer size
            n_h2: Second hidden layer size
            scope_name: Variable scope name
        """
        self.obs_dim = obs_dim
        self.field_features_dim = field_features_dim
        self.n_mutations = n_mutations
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.scope_name = scope_name
        
    def build_network(self, obs_input, field_features, mutation_history, is_training=True):
        """Build mutation action network.
        
        Args:
            obs_input: Global observation input [batch, obs_dim]
            field_features: Field-specific features [batch, field_features_dim]
            mutation_history: Historical mutation performance [batch, n_mutations]
            is_training: Whether in training mode
            
        Returns:
            mutation_logits: Logits for mutation actions [batch, n_mutations]
            mutation_probs: Probabilities for mutation actions [batch, n_mutations]
        """
        with tf.variable_scope(self.scope_name):
            # Process global observations
            obs_features = self._process_global_obs(obs_input, is_training)
            
            # Process field-specific features
            field_processed = self._process_field_features(field_features, is_training)
            
            # Process mutation history
            history_features = self._process_mutation_history(mutation_history)
            
            # Combine all features
            combined_features = tf.concat([obs_features, field_processed, history_features], axis=1)
            
            # Mutation action head with multi-criteria decision making
            mutation_logits = self._build_mutation_head(combined_features)
            
            # Apply exploration strategy (epsilon-greedy built into network)
            mutation_probs = tf.nn.softmax(mutation_logits, name='mutation_probs')
            
            return mutation_logits, mutation_probs
    
    def _process_global_obs(self, obs_input, is_training):
        """Process global observation features."""
        h1 = tf.layers.dense(inputs=obs_input, units=self.n_h1, activation=tf.nn.relu,
                             use_bias=True, name='global_h1')
        if is_training:
            h1 = tf.layers.dropout(h1, rate=0.15, name='global_dropout1')
            
        h2 = tf.layers.dense(inputs=h1, units=self.n_h2, activation=tf.nn.relu,
                             use_bias=True, name='global_h2')
        
        return h2
    
    def _process_field_features(self, field_features, is_training):
        """Process field-specific features."""
        field_h1 = tf.layers.dense(inputs=field_features, units=32, activation=tf.nn.relu,
                                  use_bias=True, name='field_h1')
        if is_training:
            field_h1 = tf.layers.dropout(field_h1, rate=0.1, name='field_dropout')
            
        field_h2 = tf.layers.dense(inputs=field_h1, units=16, activation=tf.nn.relu,
                                  use_bias=True, name='field_h2')
        
        return field_h2
    
    def _process_mutation_history(self, mutation_history):
        """Process historical mutation performance."""
        # Normalize history
        normalized_history = tf.nn.softmax(mutation_history + 1e-8, name='normalized_history')
        
        # Transform through small network
        history_h = tf.layers.dense(inputs=normalized_history, units=16, activation=tf.nn.relu,
                                   use_bias=True, name='history_h')
        
        return history_h
    
    def _build_mutation_head(self, features):
        """Build mutation action selection head with multi-criteria reasoning."""
        # Four decision criteria branches
        
        # 1. Historical performance branch
        history_branch = tf.layers.dense(inputs=features, units=32, activation=tf.nn.relu,
                                       use_bias=True, name='history_branch')
        history_logits = tf.layers.dense(inputs=history_branch, units=self.n_mutations,
                                       activation=None, use_bias=True, name='history_logits')
        
        # 2. Field characteristics branch
        field_branch = tf.layers.dense(inputs=features, units=32, activation=tf.nn.relu,
                                     use_bias=True, name='field_branch')
        field_logits = tf.layers.dense(inputs=field_branch, units=self.n_mutations,
                                     activation=None, use_bias=True, name='field_logits')
        
        # 3. Current state branch
        state_branch = tf.layers.dense(inputs=features, units=32, activation=tf.nn.relu,
                                     use_bias=True, name='state_branch')
        state_logits = tf.layers.dense(inputs=state_branch, units=self.n_mutations,
                                     activation=None, use_bias=True, name='state_logits')
        
        # 4. Exploration branch
        explore_branch = tf.layers.dense(inputs=features, units=16, activation=tf.nn.relu,
                                       use_bias=True, name='explore_branch')
        explore_logits = tf.layers.dense(inputs=explore_branch, units=self.n_mutations,
                                       activation=None, use_bias=True, name='explore_logits')
        
        # Combine branches with learned weights
        branch_weights = tf.layers.dense(inputs=features, units=4, activation=tf.nn.softmax,
                                       use_bias=True, name='branch_weights')
        
        # Weighted combination
        combined_logits = (branch_weights[:, 0:1] * history_logits +
                          branch_weights[:, 1:2] * field_logits +
                          branch_weights[:, 2:3] * state_logits +
                          branch_weights[:, 3:4] * explore_logits)
        
        return combined_logits


class FeatureProcessor:
    """Processes and integrates Mamba features with other inputs."""
    
    def __init__(self, mamba_dim=256, output_dim=128, scope_name="FeatureProcessor"):
        """Initialize feature processor.
        
        Args:
            mamba_dim: Mamba feature dimension
            output_dim: Output feature dimension
            scope_name: Variable scope name
        """
        self.mamba_dim = mamba_dim
        self.output_dim = output_dim
        self.scope_name = scope_name
        
    def build_network(self, mamba_features, protocol_features, is_training=True):
        """Build feature processing network.
        
        Args:
            mamba_features: Mamba extracted features [batch, mamba_dim]
            protocol_features: Protocol-specific features [batch, protocol_dim]
            is_training: Whether in training mode
            
        Returns:
            processed_features: Combined and processed features [batch, output_dim]
        """
        with tf.variable_scope(self.scope_name):
            # Process Mamba features
            mamba_processed = self._process_mamba_features(mamba_features, is_training)
            
            # Process protocol features
            protocol_processed = self._process_protocol_features(protocol_features, is_training)
            
            # Attention-based combination
            combined_features = self._attention_combine(mamba_processed, protocol_processed)
            
            # Final processing
            output_features = self._final_processing(combined_features, is_training)
            
            return output_features
    
    def _process_mamba_features(self, mamba_features, is_training):
        """Process Mamba features through neural network."""
        # Mamba features are high-dimensional and complex
        h1 = tf.layers.dense(inputs=mamba_features, units=128, activation=tf.nn.relu,
                             use_bias=True, name='mamba_h1')
        if is_training:
            h1 = tf.layers.dropout(h1, rate=0.2, name='mamba_dropout1')
            
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu,
                             use_bias=True, name='mamba_h2')
        if is_training:
            h2 = tf.layers.dropout(h2, rate=0.1, name='mamba_dropout2')
            
        return h2
    
    def _process_protocol_features(self, protocol_features, is_training):
        """Process protocol-specific features."""
        h1 = tf.layers.dense(inputs=protocol_features, units=64, activation=tf.nn.relu,
                             use_bias=True, name='protocol_h1')
        if is_training:
            h1 = tf.layers.dropout(h1, rate=0.1, name='protocol_dropout')
            
        h2 = tf.layers.dense(inputs=h1, units=32, activation=tf.nn.relu,
                             use_bias=True, name='protocol_h2')
        
        return h2
    
    def _attention_combine(self, mamba_features, protocol_features):
        """Combine features using attention mechanism."""
        # Simple attention between Mamba and protocol features
        mamba_dim = int(mamba_features.shape[-1])
        protocol_dim = int(protocol_features.shape[-1])
        
        # Attention weights
        attention_input = tf.concat([mamba_features, protocol_features], axis=1)
        attention_weights = tf.layers.dense(inputs=attention_input, units=2, activation=tf.nn.softmax,
                                          use_bias=True, name='attention_weights')
        
        # Weighted combination
        weighted_mamba = attention_weights[:, 0:1] * mamba_features
        weighted_protocol = attention_weights[:, 1:2] * protocol_features
        
        # Project to same dimension and combine
        projected_mamba = tf.layers.dense(inputs=weighted_mamba, units=64, activation=None,
                                        use_bias=True, name='project_mamba')
        projected_protocol = tf.layers.dense(inputs=weighted_protocol, units=64, activation=None,
                                           use_bias=True, name='project_protocol')
        
        combined = projected_mamba + projected_protocol
        combined = tf.nn.relu(combined, name='combined_relu')
        
        return combined
    
    def _final_processing(self, combined_features, is_training):
        """Final processing of combined features."""
        h1 = tf.layers.dense(inputs=combined_features, units=self.output_dim, activation=tf.nn.relu,
                             use_bias=True, name='final_h1')
        if is_training:
            h1 = tf.layers.dropout(h1, rate=0.1, name='final_dropout')
            
        # Layer normalization for stability
        h1_norm = tf.layers.batch_normalization(h1, training=is_training, name='final_norm')
        
        output = tf.layers.dense(inputs=h1_norm, units=self.output_dim, activation=None,
                               use_bias=True, name='final_output')
        
        return output


def fuzzing_decoder(trajs, timesteps, n_h=128, n_logits=8, scope_name="FuzzingDecoder"):
    """Decoder for skill discovery in fuzzing context.
    
    Args:
        trajs: Trajectory observations [batch, timesteps, obs_dim]
        timesteps: Number of timesteps
        n_h: Hidden units in LSTM
        n_logits: Number of output field types
        scope_name: Variable scope name
        
    Returns:
        out: Raw logits [batch, n_logits]
        probs: Softmax probabilities [batch, n_logits]
    """
    with tf.variable_scope(scope_name):
        # Unstack trajectories for RNN processing
        trajs_unstacked = tf.unstack(trajs, timesteps, 1)
        
        # Bidirectional LSTM for sequence modeling
        lstm_forward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
        lstm_backward_cell = rnn.LSTMCell(num_units=n_h, forget_bias=1.0)
        
        rnn_out, state_forward, state_backward = rnn.static_bidirectional_rnn(
            lstm_forward_cell, lstm_backward_cell, trajs_unstacked, dtype=tf.float32)
        
        # Attention over sequence
        attention_weights = tf.layers.dense(inputs=tf.stack(rnn_out, axis=1), units=1,
                                          activation=tf.nn.softmax, use_bias=True, name='attention')
        attention_weights = tf.squeeze(attention_weights, axis=2)  # [batch, timesteps]
        
        # Weighted average
        rnn_stack = tf.stack(rnn_out, axis=1)  # [batch, timesteps, 2*n_h]
        attended_features = tf.reduce_sum(
            rnn_stack * tf.expand_dims(attention_weights, axis=2), axis=1)
        
        # Final classification layer
        h1 = tf.layers.dense(inputs=attended_features, units=n_h, activation=tf.nn.relu,
                             use_bias=True, name='decoder_h1')
        out = tf.layers.dense(inputs=h1, units=n_logits, activation=None,
                            use_bias=True, name='decoder_out')
        
        probs = tf.nn.softmax(out, name='decoder_probs')
        
        return out, probs


def fuzzing_qmix_single(obs, n_h1, n_h2, n_actions, scope_name="FuzzingQmixSingle"):
    """Single agent Q-network for fuzzing field selection.
    
    Args:
        obs: Observation input [batch, obs_dim]
        n_h1: First hidden layer size
        n_h2: Second hidden layer size  
        n_actions: Number of field actions
        scope_name: Variable scope name
        
    Returns:
        Q-values for field selection [batch, n_actions]
    """
    with tf.variable_scope(scope_name):
        h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu,
                             use_bias=True, name='h1')
        
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                             use_bias=True, name='h2')
        
        # Field selection specific features
        field_h = tf.layers.dense(inputs=h2, units=64, activation=tf.nn.relu,
                                use_bias=True, name='field_h')
        
        out = tf.layers.dense(inputs=field_h, units=n_actions, activation=None,
                            use_bias=True, name='out')
        
        return out


def fuzzing_mutation_q(obs, field_features, n_h1, n_h2, n_mutations, scope_name="FuzzingMutationQ"):
    """Q-network for mutation action selection.
    
    Args:
        obs: Global observation [batch, obs_dim]
        field_features: Field-specific features [batch, field_dim]
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        n_mutations: Number of mutation actions
        scope_name: Variable scope name
        
    Returns:
        Q-values for mutation actions [batch, n_mutations]
    """
    with tf.variable_scope(scope_name):
        # Concatenate observations and field features
        combined_input = tf.concat([obs, field_features], axis=1)
        
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                             use_bias=True, name='h1')
        
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                             use_bias=True, name='h2')
        
        # Mutation-specific processing
        mutation_h = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu,
                                   use_bias=True, name='mutation_h')
        
        out = tf.layers.dense(inputs=mutation_h, units=n_mutations, activation=None,
                            use_bias=True, name='out')
        
        return out


def fuzzing_qmix_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer, scope_name="FuzzingQmixMixer"):
    """Mixing network adapted for fuzzing coordination.
    
    Args:
        agent_qs: Individual agent Q-values [batch, n_agents]
        state: Global state [batch, state_dim]
        state_dim: State dimension
        n_agents: Number of agents (fields in fuzzing context)
        n_h_mixer: Hidden units in mixer
        scope_name: Variable scope name
        
    Returns:
        Mixed Q-total [batch, 1]
    """
    with tf.variable_scope(scope_name):
        agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])
        
        # Hypernetworks for weight generation
        hyper_w_1 = get_variable('hyper_w_1', [state_dim, n_h_mixer * n_agents])
        hyper_w_final = get_variable('hyper_w_final', [state_dim, n_h_mixer])
        hyper_b_1 = tf.get_variable('hyper_b_1', [state_dim, n_h_mixer])
        
        # Bias hypernetwork
        hyper_b_final_l1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=tf.nn.relu,
                                           use_bias=False, name='hyper_b_final_l1')
        hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                      use_bias=False, name='hyper_b_final')
        
        # First layer
        w1 = tf.abs(tf.matmul(state, hyper_w_1))  # Ensure positive weights
        b1 = tf.matmul(state, hyper_b_1)
        w1_reshaped = tf.reshape(w1, [-1, n_agents, n_h_mixer])
        b1_reshaped = tf.reshape(b1, [-1, 1, n_h_mixer])
        
        hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
        
        # Second layer
        w_final = tf.abs(tf.matmul(state, hyper_w_final))
        w_final_reshaped = tf.reshape(w_final, [-1, n_h_mixer, 1])
        b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])
        
        y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped
        q_tot = tf.reshape(y, [-1, 1])
        
        return q_tot