"""Neural networks for fuzzing hierarchical reinforcement learning.

This module defines networks adapted for protocol fuzzing:
- Field selector networks (high-level): Choose which protocol fields to mutate
- Mutation action networks (low-level): Choose mutation operations for selected fields  
- Field selection decoder: Predict field selection from mutation history
- Feature processing: Handle Mamba features and protocol-specific data
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def get_variable(name, shape):
    """Create TensorFlow variable with truncated normal initialization."""
    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0, 0.01))


def field_selector_network(obs, n_h1, n_h2, n_fields):
    """Network for selecting protocol fields to mutate.
    
    Args:
        obs: TF placeholder for observations
        n_h1: Hidden layer 1 size
        n_h2: Hidden layer 2 size  
        n_fields: Number of protocol fields
        
    Returns:
        Q-values for field selection
    """
    h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu, 
                        use_bias=True, name='field_selector_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                        use_bias=True, name='field_selector_h2')
    out = tf.layers.dense(inputs=h2, units=n_fields, activation=None,
                         use_bias=True, name='field_selector_out')
    return out


def mutation_actor(obs, field_selection, n_h1, n_h2, n_mutation_actions):
    """Actor network for selecting mutation actions.
    
    Args:
        obs: TF placeholder for field observations
        field_selection: TF placeholder for current field selection (1-hot)
        n_h1: Hidden layer 1 size
        n_h2: Hidden layer 2 size
        n_mutation_actions: Number of mutation action types
        
    Returns:
        Action probabilities
    """
    # Concatenate observation with field selection context
    combined = tf.concat([obs, field_selection], axis=1)
    
    h1 = tf.layers.dense(inputs=combined, units=n_h1, activation=tf.nn.relu,
                        use_bias=True, name='mutation_actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                        use_bias=True, name='mutation_actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_mutation_actions, activation=None,
                         use_bias=True, name='mutation_actor_out')
    probs = tf.nn.softmax(out, name='mutation_actor_softmax')
    
    return probs


def mutation_critic(obs, field_selection, n_h1, n_h2):
    """Critic network for mutation policy (value function).
    
    Args:
        obs: TF placeholder for field observations
        field_selection: TF placeholder for current field selection (1-hot)
        n_h1: Hidden layer 1 size
        n_h2: Hidden layer 2 size
        
    Returns:
        State value estimate
    """
    combined = tf.concat([obs, field_selection], axis=1)
    
    h1 = tf.layers.dense(inputs=combined, units=n_h1, activation=tf.nn.relu,
                        use_bias=True, name='mutation_critic_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                        use_bias=True, name='mutation_critic_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                         use_bias=True, name='mutation_critic_out')
    
    return out


def mutation_Q_network(obs, field_selection, n_h1, n_h2, n_mutation_actions):
    """Q-network for mutation actions.
    
    Args:
        obs: TF placeholder for field observations
        field_selection: TF placeholder for current field selection (1-hot)
        n_h1: Hidden layer 1 size
        n_h2: Hidden layer 2 size
        n_mutation_actions: Number of mutation action types
        
    Returns:
        Q-values for each mutation action
    """
    combined = tf.concat([obs, field_selection], axis=1)
    
    h1 = tf.layers.dense(inputs=combined, units=n_h1, activation=tf.nn.relu,
                        use_bias=True, name='mutation_Q_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                        use_bias=True, name='mutation_Q_h2')
    out = tf.layers.dense(inputs=h2, units=n_mutation_actions, activation=None,
                         use_bias=True, name='mutation_Q_out')
    
    return out


def field_selection_decoder(trajs, timesteps, n_h=128, n_fields=8):
    """Bidirectional LSTM decoder to predict field selection from mutation history.
    
    Args:
        trajs: Mutation trajectory tensor (batch_size, timesteps, obs_dim)
        timesteps: Number of timesteps in trajectory
        n_h: Hidden layer size for LSTM
        n_fields: Number of protocol fields to predict
        
    Returns:
        Tuple of (logits, probabilities) for field selection
    """
    # Bidirectional LSTM
    with tf.variable_scope("BiLSTM"):
        # Define LSTM cells
        lstm_fw_cell = rnn.BasicLSTMCell(n_h, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(n_h, forget_bias=1.0)
        
        # Run bidirectional LSTM
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell, lstm_bw_cell, trajs,
            dtype=tf.float32, scope='field_decoder_lstm')
        
        # Concatenate forward and backward outputs
        output_fw, output_bw = outputs
        # Use final outputs
        final_output = tf.concat([output_fw[:, -1, :], output_bw[:, 0, :]], axis=1)
    
    # Dense layers for field prediction
    with tf.variable_scope("FieldPrediction"):
        h1 = tf.layers.dense(inputs=final_output, units=n_h, activation=tf.nn.relu,
                           use_bias=True, name='decoder_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h//2, activation=tf.nn.relu,
                           use_bias=True, name='decoder_h2')
        logits = tf.layers.dense(inputs=h2, units=n_fields, activation=None,
                               use_bias=True, name='decoder_logits')
        probs = tf.nn.softmax(logits, name='decoder_probs')
    
    return logits, probs


def field_selection_mixer(field_q_values, state, state_dim, n_fields, n_h_mixer):
    """Mixing network for field selection (QMIX-style).
    
    Combines individual field Q-values into a joint Q-value that preserves
    the Individual-Global-Max (IGM) property.
    
    Args:
        field_q_values: Individual Q-values for each field [batch, n_fields]
        state: Global state tensor [batch, state_dim]
        state_dim: Dimension of global state
        n_fields: Number of protocol fields
        n_h_mixer: Hidden layer size for mixer
        
    Returns:
        Mixed Q-value [batch, 1]
    """
    batch_size = tf.shape(field_q_values)[0]
    
    # Generate mixing weights from state
    with tf.variable_scope("MixerWeights"):
        # Weights must be positive to maintain IGM
        w1 = tf.layers.dense(inputs=state, units=n_fields * n_h_mixer, activation=None,
                           use_bias=False, name='mixer_w1')
        w1 = tf.abs(w1)  # Ensure positive weights
        w1 = tf.reshape(w1, [batch_size, n_fields, n_h_mixer])
        
        w2 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=None,
                           use_bias=False, name='mixer_w2')
        w2 = tf.abs(w2)  # Ensure positive weights
        w2 = tf.reshape(w2, [batch_size, n_h_mixer, 1])
    
    # Generate biases from state
    with tf.variable_scope("MixerBiases"):
        b1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=None,
                           use_bias=False, name='mixer_b1')
        b1 = tf.reshape(b1, [batch_size, 1, n_h_mixer])
        
        b2 = tf.layers.dense(inputs=state, units=1, activation=None,
                           use_bias=False, name='mixer_b2')
        b2 = tf.reshape(b2, [batch_size, 1, 1])
    
    # Forward pass through mixer
    # First layer: [batch, n_fields] @ [batch, n_fields, n_h_mixer] + [batch, 1, n_h_mixer]
    field_q_expanded = tf.expand_dims(field_q_values, axis=1)  # [batch, 1, n_fields]
    hidden = tf.matmul(field_q_expanded, w1) + b1  # [batch, 1, n_h_mixer]
    hidden = tf.nn.elu(hidden)  # Non-linear activation
    
    # Second layer: [batch, 1, n_h_mixer] @ [batch, n_h_mixer, 1] + [batch, 1, 1]
    output = tf.matmul(hidden, w2) + b2  # [batch, 1, 1]
    output = tf.squeeze(output, axis=[1, 2])  # [batch]
    
    return output


def mamba_feature_processor(mamba_features, output_dim=128):
    """Process Mamba features for fuzzing context.
    
    Args:
        mamba_features: Raw Mamba feature tensor [batch, mamba_dim]
        output_dim: Desired output dimension
        
    Returns:
        Processed features [batch, output_dim]
    """
    with tf.variable_scope("MambaProcessor"):
        # Layer normalization
        normalized = tf.layers.batch_normalization(mamba_features, training=True,
                                                  name='mamba_norm')
        
        # Dense processing layers
        h1 = tf.layers.dense(inputs=normalized, units=output_dim*2, activation=tf.nn.relu,
                           use_bias=True, name='mamba_proc_h1')
        
        # Dropout for regularization
        h1_drop = tf.layers.dropout(h1, rate=0.1, training=True, name='mamba_dropout')
        
        # Output layer
        output = tf.layers.dense(inputs=h1_drop, units=output_dim, activation=tf.nn.tanh,
                               use_bias=True, name='mamba_proc_out')
    
    return output


def protocol_field_encoder(field_data, field_type, embedding_dim=32):
    """Encode protocol field data into embeddings.
    
    Args:
        field_data: Protocol field data tensor
        field_type: Type of field (header, payload, etc.)
        embedding_dim: Dimension of field embeddings
        
    Returns:
        Field embedding vector
    """
    with tf.variable_scope(f"FieldEncoder_{field_type}"):
        # For now, use simple dense encoding
        # In practice, this could be more sophisticated based on field type
        if len(field_data.shape) == 1:
            field_data = tf.expand_dims(field_data, 0)
            
        # Adaptive pooling for variable-length fields
        if field_data.shape[1] > embedding_dim:
            # Use 1D convolution for downsampling
            conv1 = tf.layers.conv1d(tf.expand_dims(field_data, -1), filters=16, 
                                   kernel_size=3, strides=1, padding='same',
                                   activation=tf.nn.relu, name='field_conv1')
            pooled = tf.layers.average_pooling1d(conv1, pool_size=2, strides=2,
                                               name='field_pool1')
            flattened = tf.layers.flatten(pooled)
        else:
            flattened = tf.layers.flatten(field_data)
            
        # Dense encoding
        encoded = tf.layers.dense(inputs=flattened, units=embedding_dim,
                                activation=tf.nn.relu, use_bias=True,
                                name='field_encoding')
    
    return encoded


def attention_field_selector(field_embeddings, query_state, n_fields, attention_dim=64):
    """Attention-based field selection mechanism.
    
    Args:
        field_embeddings: Embeddings for each field [batch, n_fields, embed_dim]
        query_state: Query state for attention [batch, state_dim]
        n_fields: Number of protocol fields
        attention_dim: Attention mechanism dimension
        
    Returns:
        Attention weights for field selection [batch, n_fields]
    """
    with tf.variable_scope("AttentionFieldSelector"):
        batch_size = tf.shape(field_embeddings)[0]
        embed_dim = field_embeddings.shape[-1]
        
        # Transform query state
        query = tf.layers.dense(inputs=query_state, units=attention_dim,
                              activation=tf.nn.tanh, use_bias=True,
                              name='attention_query')
        query = tf.expand_dims(query, axis=1)  # [batch, 1, attention_dim]
        
        # Transform field embeddings
        keys = tf.layers.dense(inputs=field_embeddings, units=attention_dim,
                             activation=tf.nn.tanh, use_bias=True,
                             name='attention_keys')  # [batch, n_fields, attention_dim]
        
        # Compute attention scores
        scores = tf.matmul(query, keys, transpose_b=True)  # [batch, 1, n_fields]
        scores = tf.squeeze(scores, axis=1)  # [batch, n_fields]
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, name='attention_weights')
    
    return attention_weights


def hierarchical_feature_fusion(mamba_features, field_features, mutation_history, 
                               coverage_features, output_dim=256):
    """Fuse multiple feature types for hierarchical decision making.
    
    Args:
        mamba_features: Mamba-extracted features [batch, mamba_dim]
        field_features: Protocol field features [batch, field_dim]
        mutation_history: Encoded mutation history [batch, history_dim]
        coverage_features: Coverage tracking features [batch, coverage_dim]
        output_dim: Output feature dimension
        
    Returns:
        Fused feature representation [batch, output_dim]
    """
    with tf.variable_scope("HierarchicalFusion"):
        # Process each feature type
        mamba_proc = tf.layers.dense(inputs=mamba_features, units=output_dim//4,
                                   activation=tf.nn.relu, use_bias=True,
                                   name='mamba_projection')
        
        field_proc = tf.layers.dense(inputs=field_features, units=output_dim//4,
                                   activation=tf.nn.relu, use_bias=True,
                                   name='field_projection')
        
        history_proc = tf.layers.dense(inputs=mutation_history, units=output_dim//4,
                                     activation=tf.nn.relu, use_bias=True,
                                     name='history_projection')
        
        coverage_proc = tf.layers.dense(inputs=coverage_features, units=output_dim//4,
                                      activation=tf.nn.relu, use_bias=True,
                                      name='coverage_projection')
        
        # Concatenate processed features
        concatenated = tf.concat([mamba_proc, field_proc, history_proc, coverage_proc], axis=1)
        
        # Final fusion layer
        fused = tf.layers.dense(inputs=concatenated, units=output_dim,
                              activation=tf.nn.tanh, use_bias=True,
                              name='fusion_output')
        
        # Residual connection if dimensions match
        if mamba_features.shape[-1] == output_dim:
            fused = fused + mamba_features
    
    return fused


def vulnerability_detector_network(execution_features, n_h1=128, n_h2=64):
    """Network to predict vulnerability likelihood from execution features.
    
    Args:
        execution_features: Features from fuzzing execution [batch, feature_dim]
        n_h1: First hidden layer size
        n_h2: Second hidden layer size
        
    Returns:
        Vulnerability probability [batch, 1]
    """
    with tf.variable_scope("VulnerabilityDetector"):
        h1 = tf.layers.dense(inputs=execution_features, units=n_h1,
                           activation=tf.nn.relu, use_bias=True,
                           name='vuln_detector_h1')
        
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu,
                           use_bias=True, name='vuln_detector_h2')
        
        # Output probability of vulnerability
        logits = tf.layers.dense(inputs=h2, units=1, activation=None,
                               use_bias=True, name='vuln_detector_logits')
        
        prob = tf.nn.sigmoid(logits, name='vuln_probability')
    
    return prob, logits


# Legacy compatibility functions for original HSD interface
def Qmix_single(obs, n_h1, n_h2, n_actions):
    """Single agent Q-network (compatibility with original QMIX)."""
    return field_selector_network(obs, n_h1, n_h2, n_actions)


def Qmix_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer):
    """QMIX mixer network (compatibility with original)."""
    return field_selection_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer)


def Q_low(obs, z, n_h1, n_h2, n_actions):
    """Low-level Q-network (compatibility with original)."""
    return mutation_Q_network(obs, z, n_h1, n_h2, n_actions)


def actor(obs, z, n_h1, n_h2, n_actions):
    """Actor network (compatibility with original)."""
    return mutation_actor(obs, z, n_h1, n_h2, n_actions)


def critic(obs, z, n_h1, n_h2):
    """Critic network (compatibility with original)."""
    return mutation_critic(obs, z, n_h1, n_h2)


def decoder(trajs, timesteps, n_h=128, n_logits=8):
    """Decoder network (compatibility with original)."""
    return field_selection_decoder(trajs, timesteps, n_h, n_logits)