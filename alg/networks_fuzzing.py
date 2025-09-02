"""Neural Networks for Fuzzing Hierarchical RL Framework.

This module defines specialized neural networks for the fuzzing adaptation of the HSD framework:
- FieldSelectorNetwork: High-level field selection policy
- MutationActionNetwork: Low-level mutation action selection
- FeatureProcessor: Mamba feature integration network
- FuzzingDecoder: Field importance learning network
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def get_variable(name, shape):
    """Create TensorFlow variable with truncated normal initialization."""
    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0, 0.01))


class FeatureProcessor:
    """Processes Mamba features for fuzzing context."""
    
    @staticmethod
    def process_mamba_features(mamba_features, n_h1=128, n_h2=64, output_dim=32):
        """Process raw Mamba features into fuzzing-relevant representations.
        
        Args:
            mamba_features: TF placeholder for Mamba features [batch, mamba_dim]
            n_h1: First hidden layer size
            n_h2: Second hidden layer size  
            output_dim: Output feature dimension
            
        Returns:
            Processed feature tensor
        """
        with tf.variable_scope("MambaProcessor"):
            h1 = tf.layers.dense(inputs=mamba_features, units=n_h1, activation=tf.nn.relu, 
                               use_bias=True, name='mamba_h1')
            h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                               use_bias=True, name='mamba_h2')
            processed = tf.layers.dense(inputs=h2, units=output_dim, activation=tf.nn.tanh,
                                      use_bias=True, name='mamba_out')
        return processed


def field_selector_network(obs, field_importance, n_h1, n_h2, n_fields):
    """High-level network for protocol field selection.
    
    Args:
        obs: TF placeholder for observations
        field_importance: TF placeholder for field importance scores
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        n_fields: Number of protocol fields
        
    Returns:
        Q-values for field selection
    """
    # Combine observation and field importance information
    combined_input = tf.concat([obs, field_importance], axis=1)
    
    with tf.variable_scope("FieldSelector"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='field_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='field_h2')
        
        # Output Q-values for each field
        q_values = tf.layers.dense(inputs=h2, units=n_fields, activation=None,
                                 use_bias=True, name='field_out')
        
        # Apply attention mechanism based on field importance
        attention_weights = tf.nn.softmax(field_importance, name='field_attention')
        attention_weights = tf.expand_dims(attention_weights, axis=0)
        attention_weights = tf.tile(attention_weights, [tf.shape(q_values)[0], 1])
        
        # Weighted Q-values
        weighted_q_values = q_values * attention_weights
        
    return weighted_q_values


def mutation_action_network(obs, field_context, selected_field, n_h1, n_h2, n_actions):
    """Low-level network for mutation action selection.
    
    Args:
        obs: TF placeholder for observations
        field_context: TF placeholder for field context features
        selected_field: TF placeholder for selected field (one-hot)
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        n_actions: Number of mutation actions
        
    Returns:
        Q-values for mutation actions
    """
    # Combine all inputs for context-aware action selection
    combined_input = tf.concat([obs, field_context, selected_field], axis=1)
    
    with tf.variable_scope("MutationAction"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='mutation_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='mutation_h2')
        
        # Action-specific features
        action_features = tf.layers.dense(inputs=h2, units=n_h2//2, activation=tf.nn.relu,
                                        use_bias=True, name='action_features')
        
        # Output Q-values for each mutation action
        q_values = tf.layers.dense(inputs=action_features, units=n_actions, activation=None,
                                 use_bias=True, name='mutation_out')
        
    return q_values


def fuzzing_decoder(field_trajs, mamba_features, timesteps, n_h_lstm=128, n_h_dense=64, n_fields=8):
    """Decoder network for learning field importance from trajectories.
    
    Args:
        field_trajs: Trajectories of field mutations [batch, timesteps, field_features]
        mamba_features: Mamba feature representations [batch, mamba_dim]
        timesteps: Number of timesteps in trajectories
        n_h_lstm: LSTM hidden units
        n_h_dense: Dense layer hidden units
        n_fields: Number of protocol fields
        
    Returns:
        Tuple of (logits, probabilities) for field importance
    """
    # Process Mamba features
    processed_mamba = FeatureProcessor.process_mamba_features(mamba_features, output_dim=32)
    
    # Unstack trajectories for LSTM processing
    field_trajs_unstacked = tf.unstack(field_trajs, timesteps, 1)
    
    with tf.variable_scope("FuzzingDecoder"):
        # Bidirectional LSTM for trajectory analysis
        lstm_forward_cell = rnn.LSTMCell(num_units=n_h_lstm, forget_bias=1.0)
        lstm_backward_cell = rnn.LSTMCell(num_units=n_h_lstm, forget_bias=1.0)
        
        rnn_out, state_forward, state_backward = rnn.static_bidirectional_rnn(
            lstm_forward_cell, lstm_backward_cell, field_trajs_unstacked, dtype=tf.float32)
        
        # Mean-pool over time
        rnn_mean = tf.reduce_mean(rnn_out, axis=0)
        
        # Combine LSTM output with Mamba features
        combined_features = tf.concat([rnn_mean, processed_mamba], axis=1)
        
        # Dense layers for field importance prediction
        dense1 = tf.layers.dense(inputs=combined_features, units=n_h_dense, activation=tf.nn.relu,
                               use_bias=True, name='decoder_dense1')
        
        # Attention mechanism for field importance
        attention = tf.layers.dense(inputs=dense1, units=n_fields, activation=tf.nn.tanh,
                                  use_bias=True, name='field_attention')
        
        # Final field importance logits
        logits = tf.layers.dense(inputs=dense1, units=n_fields, activation=None,
                               use_bias=True, name='decoder_out')
        
        # Apply attention weighting
        weighted_logits = logits + attention
        
        # Softmax probabilities
        probs = tf.nn.softmax(weighted_logits, name='field_importance_probs')
        
    return weighted_logits, probs


def coverage_prediction_network(obs, field_selection, mutation_action, n_h1=64, n_h2=32):
    """Network to predict coverage increase for reward shaping.
    
    Args:
        obs: Current observations
        field_selection: Selected field (one-hot)
        mutation_action: Selected mutation action (one-hot)
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        
    Returns:
        Predicted coverage increase
    """
    combined_input = tf.concat([obs, field_selection, mutation_action], axis=1)
    
    with tf.variable_scope("CoveragePrediction"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='cov_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='cov_h2')
        
        # Predict coverage increase (positive value)
        coverage_pred = tf.layers.dense(inputs=h2, units=1, activation=tf.nn.softplus,
                                      use_bias=True, name='coverage_out')
        
    return coverage_pred


def vulnerability_detection_network(obs, mutation_result_features, n_h1=64, n_h2=32):
    """Network to predict vulnerability likelihood for reward shaping.
    
    Args:
        obs: Current observations
        mutation_result_features: Features extracted from mutation results
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        
    Returns:
        Vulnerability probability prediction
    """
    combined_input = tf.concat([obs, mutation_result_features], axis=1)
    
    with tf.variable_scope("VulnerabilityDetection"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='vuln_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='vuln_h2')
        
        # Predict vulnerability probability
        vuln_prob = tf.layers.dense(inputs=h2, units=1, activation=tf.nn.sigmoid,
                                  use_bias=True, name='vuln_out')
        
    return vuln_prob


def fuzzing_critic(obs, field_selection, n_h1, n_h2):
    """Value function critic for fuzzing state evaluation.
    
    Args:
        obs: Observation tensor
        field_selection: Selected field one-hot vector
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        
    Returns:
        State value estimate
    """
    combined_input = tf.concat([obs, field_selection], axis=1)
    
    with tf.variable_scope("FuzzingCritic"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='critic_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='critic_h2')
        value = tf.layers.dense(inputs=h2, units=1, activation=None,
                              use_bias=True, name='critic_out')
    
    return value


def hierarchical_mixer(field_q_values, mutation_q_values, state, state_dim, n_fields, n_actions, n_h_mixer):
    """Modified QMIX mixer for hierarchical fuzzing decisions.
    
    Args:
        field_q_values: Q-values for field selection [batch, n_fields]
        mutation_q_values: Q-values for mutation actions [batch, n_actions]
        state: Global state [batch, state_dim]
        state_dim: Dimension of state
        n_fields: Number of protocol fields
        n_actions: Number of mutation actions
        n_h_mixer: Mixer hidden layer size
        
    Returns:
        Mixed Q-value for hierarchical action
    """
    # Combine field and mutation Q-values
    combined_q_values = tf.concat([field_q_values, mutation_q_values], axis=1)
    total_actions = n_fields + n_actions
    
    # Reshape for mixer processing
    combined_q_reshaped = tf.reshape(combined_q_values, [-1, 1, total_actions])
    
    with tf.variable_scope("HierarchicalMixer"):
        # Hypernetworks for mixing weights
        hyper_w_1 = get_variable('hyper_w_1', [state_dim, n_h_mixer * total_actions])
        hyper_w_final = get_variable('hyper_w_final', [state_dim, n_h_mixer])
        hyper_b_1 = tf.get_variable('hyper_b_1', [state_dim, n_h_mixer])
        
        # Bias hypernetwork
        hyper_b_final_l1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=tf.nn.relu,
                                         use_bias=False, name='hyper_b_final_l1')
        hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                      use_bias=False, name='hyper_b_final')
        
        # First mixing layer
        w1 = tf.abs(tf.matmul(state, hyper_w_1))
        b1 = tf.matmul(state, hyper_b_1)
        w1_reshaped = tf.reshape(w1, [-1, total_actions, n_h_mixer])
        b1_reshaped = tf.reshape(b1, [-1, 1, n_h_mixer])
        
        # Apply mixing
        hidden = tf.nn.elu(tf.matmul(combined_q_reshaped, w1_reshaped) + b1_reshaped)
        
        # Second mixing layer
        w_final = tf.abs(tf.matmul(state, hyper_w_final))
        w_final_reshaped = tf.reshape(w_final, [-1, n_h_mixer, 1])
        b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])
        
        # Final Q-value
        q_tot = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped
        q_tot = tf.reshape(q_tot, [-1, 1])
        
    return q_tot


def adaptive_exploration_network(obs, field_importance, exploration_history, n_h=64):
    """Network for adaptive exploration strategy in fuzzing.
    
    Args:
        obs: Current observations
        field_importance: Field importance scores
        exploration_history: History of exploration actions
        n_h: Hidden layer size
        
    Returns:
        Exploration probability for each field
    """
    combined_input = tf.concat([obs, field_importance, exploration_history], axis=1)
    
    with tf.variable_scope("AdaptiveExploration"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h, activation=tf.nn.relu,
                           use_bias=True, name='explore_h1')
        
        # Output exploration probabilities
        exploration_probs = tf.layers.dense(inputs=h1, units=tf.shape(field_importance)[1], 
                                          activation=tf.nn.sigmoid, use_bias=True, name='explore_out')
        
    return exploration_probs


def mutation_effect_predictor(field_features, mutation_type, historical_results, n_h1=64, n_h2=32):
    """Predict the likely effect of a mutation for better action selection.
    
    Args:
        field_features: Features of the target field
        mutation_type: Type of mutation (one-hot)
        historical_results: Historical mutation results for this field
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        
    Returns:
        Predicted mutation effects (coverage, crash probability, etc.)
    """
    combined_input = tf.concat([field_features, mutation_type, historical_results], axis=1)
    
    with tf.variable_scope("MutationEffectPredictor"):
        h1 = tf.layers.dense(inputs=combined_input, units=n_h1, activation=tf.nn.relu,
                           use_bias=True, name='effect_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='effect_h2')
        
        # Predict multiple effect types
        coverage_effect = tf.layers.dense(inputs=h2, units=1, activation=tf.nn.softplus,
                                        use_bias=True, name='coverage_effect')
        crash_prob = tf.layers.dense(inputs=h2, units=1, activation=tf.nn.sigmoid,
                                   use_bias=True, name='crash_prob')
        novelty_score = tf.layers.dense(inputs=h2, units=1, activation=tf.nn.sigmoid,
                                      use_bias=True, name='novelty_score')
        
        effects = tf.concat([coverage_effect, crash_prob, novelty_score], axis=1)
        
    return effects


def field_relationship_network(field_features_matrix, n_fields, n_h=64):
    """Network to model relationships between protocol fields.
    
    Args:
        field_features_matrix: Matrix of features for all fields [batch, n_fields, field_dim]
        n_fields: Number of protocol fields
        n_h: Hidden layer size
        
    Returns:
        Field relationship matrix
    """
    with tf.variable_scope("FieldRelationship"):
        # Process each field
        field_embeddings = []
        for i in range(n_fields):
            field_i = field_features_matrix[:, i, :]
            embedding = tf.layers.dense(inputs=field_i, units=n_h, activation=tf.nn.relu,
                                      use_bias=True, name=f'field_embed_{i}')
            field_embeddings.append(embedding)
        
        # Stack embeddings
        embeddings_matrix = tf.stack(field_embeddings, axis=1)  # [batch, n_fields, n_h]
        
        # Compute pairwise relationships
        # Expand dimensions for broadcasting
        emb_i = tf.expand_dims(embeddings_matrix, axis=2)  # [batch, n_fields, 1, n_h]
        emb_j = tf.expand_dims(embeddings_matrix, axis=1)  # [batch, 1, n_fields, n_h]
        
        # Compute relationship scores using dot product
        relationship_scores = tf.reduce_sum(emb_i * emb_j, axis=3)  # [batch, n_fields, n_fields]
        
        # Apply softmax to get relationship probabilities
        relationship_matrix = tf.nn.softmax(relationship_scores, axis=2)
        
    return relationship_matrix


# Helper functions for network initialization and training

def create_fuzzing_networks(config, n_fields, n_actions, state_dim, obs_dim, mamba_dim):
    """Create all fuzzing-specific networks with shared configuration.
    
    Args:
        config: Network configuration dictionary
        n_fields: Number of protocol fields
        n_actions: Number of mutation actions
        state_dim: State space dimension
        obs_dim: Observation space dimension
        mamba_dim: Mamba feature dimension
        
    Returns:
        Dictionary of network functions and placeholders
    """
    networks = {}
    
    # Placeholders
    networks['obs'] = tf.placeholder(tf.float32, [None, obs_dim], 'obs')
    networks['state'] = tf.placeholder(tf.float32, [None, state_dim], 'state')
    networks['mamba_features'] = tf.placeholder(tf.float32, [None, mamba_dim], 'mamba_features')
    networks['field_importance'] = tf.placeholder(tf.float32, [None, n_fields], 'field_importance')
    networks['field_context'] = tf.placeholder(tf.float32, [None, 16], 'field_context')  # Context features
    networks['selected_field'] = tf.placeholder(tf.float32, [None, n_fields], 'selected_field')
    
    # Network sizes from config
    n_h1 = config.get('n_h1', 128)
    n_h2 = config.get('n_h2', 64)
    n_h_mixer = config.get('n_h_mixer', 64)
    
    # Create networks
    with tf.variable_scope("FuzzingNetworks"):
        # High-level field selection
        networks['field_q_values'] = field_selector_network(
            networks['obs'], networks['field_importance'], n_h1, n_h2, n_fields)
        
        # Low-level mutation action selection
        networks['mutation_q_values'] = mutation_action_network(
            networks['obs'], networks['field_context'], networks['selected_field'], 
            n_h1, n_h2, n_actions)
        
        # Hierarchical Q-value mixing
        networks['mixed_q_value'] = hierarchical_mixer(
            networks['field_q_values'], networks['mutation_q_values'], 
            networks['state'], state_dim, n_fields, n_actions, n_h_mixer)
        
        # Auxiliary networks
        networks['coverage_pred'] = coverage_prediction_network(
            networks['obs'], networks['selected_field'], 
            tf.placeholder(tf.float32, [None, n_actions], 'mutation_action_oh'))
        
        networks['vuln_pred'] = vulnerability_detection_network(
            networks['obs'], tf.placeholder(tf.float32, [None, 32], 'mutation_result_features'))
    
    return networks