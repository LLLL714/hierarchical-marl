"""Fuzzing adaptation of Hierarchical Skill Discovery (HSD) algorithm.

This module implements the hierarchical reinforcement learning algorithm adapted for
protocol fuzzing. The algorithm maintains the hierarchical structure but replaces:
- Agent role assignment → Protocol field selection
- Action execution → Mutation action selection  
- Multi-agent coordination → Single-agent hierarchical decision making
"""

import tensorflow as tf
import numpy as np
import sys
import networks_fuzzing as networks


class FuzzingHSDAlgorithm(object):
    """Fuzzing adaptation of the HSD algorithm."""

    def __init__(self, config_alg, config_h, config_fuzzing, l_state, l_obs, l_mamba, 
                 n_fields, n_actions, nn):
        """Initialize fuzzing HSD algorithm.
        
        Args:
            config_alg: General RL algorithm parameters
            config_h: Hierarchical algorithm parameters
            config_fuzzing: Fuzzing-specific parameters
            l_state: State dimension
            l_obs: Observation dimension  
            l_mamba: Mamba feature dimension
            n_fields: Number of protocol fields
            n_actions: Number of mutation actions
            nn: Neural network configuration
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_mamba = l_mamba
        self.n_fields = n_fields
        self.n_actions = n_actions
        self.nn = nn
        
        # RL hyperparameters
        self.tau = config_alg['tau']
        self.lr_field = config_alg.get('lr_field', config_alg['lr_Q'])
        self.lr_mutation = config_alg.get('lr_mutation', config_alg['lr_actor'])
        self.lr_decoder = config_alg['lr_decoder']
        self.gamma = config_alg['gamma']
        
        # Hierarchical parameters
        self.field_selection_steps = config_h.get('field_selection_steps', 
                                                config_h['steps_per_assign'])
        self.traj_skip = config_h['traj_skip'] 
        self.traj_length_downsampled = int(np.ceil(self.field_selection_steps / self.traj_skip))
        self.use_mamba_features = config_h.get('use_mamba_features', True)
        
        # Fuzzing-specific parameters
        self.exploit_probability = config_fuzzing.get('exploit_probability', 0.9)
        self.field_importance_learning_rate = config_fuzzing.get('field_importance_lr', 0.1)
        self.mutation_context_dim = config_fuzzing.get('mutation_context_dim', 16)
        
        # Create computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops()
        self.create_train_ops()
        
        # TensorFlow summaries
        self.create_summary()
        
    def create_networks(self):
        """Create all networks for fuzzing hierarchical RL."""
        
        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')
        self.mamba_features = tf.placeholder(tf.float32, [None, self.l_mamba], 'mamba_features')
        self.field_importance = tf.placeholder(tf.float32, [None, self.n_fields], 'field_importance')
        
        # Field selection network (high-level)
        with tf.variable_scope("FieldSelector_main"):
            self.field_q_values = networks.field_selector_network(
                self.obs, self.field_importance, 
                self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        with tf.variable_scope("FieldSelector_target"):
            self.field_q_values_target = networks.field_selector_network(
                self.obs, self.field_importance,
                self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        
        self.argmax_field = tf.argmax(self.field_q_values, axis=1)
        self.argmax_field_target = tf.argmax(self.field_q_values_target, axis=1)
        
        # Mutation action network (low-level)
        self.mutation_context = tf.placeholder(tf.float32, [None, self.mutation_context_dim], 
                                             'mutation_context')
        self.selected_field_oh = tf.placeholder(tf.float32, [None, self.n_fields], 'selected_field_oh')
        
        with tf.variable_scope("MutationAction_main"):
            self.mutation_q_values = networks.mutation_action_network(
                self.obs, self.mutation_context, self.selected_field_oh,
                self.nn['n_h1_low'], self.nn['n_h2_low'], self.n_actions)
        with tf.variable_scope("MutationAction_target"):
            self.mutation_q_values_target = networks.mutation_action_network(
                self.obs, self.mutation_context, self.selected_field_oh,
                self.nn['n_h1_low'], self.nn['n_h2_low'], self.n_actions)
        
        self.argmax_mutation = tf.argmax(self.mutation_q_values, axis=1)
        self.argmax_mutation_target = tf.argmax(self.mutation_q_values_target, axis=1)
        
        # Decoder for field importance learning
        self.field_trajectory = tf.placeholder(tf.float32, 
                                             [None, self.traj_length_downsampled, self.l_obs], 
                                             'field_trajectory')
        with tf.variable_scope("FieldDecoder"):
            self.decoder_out, self.decoder_probs = networks.fuzzing_decoder(
                self.field_trajectory, self.mamba_features, self.traj_length_downsampled,
                self.nn['n_h_decoder'], self.nn['n_h_decoder']//2, self.n_fields)
        
        # Hierarchical mixer for combined Q-values
        with tf.variable_scope("HierarchicalMixer_main"):
            self.field_q_selected = tf.reduce_sum(
                tf.multiply(self.field_q_values, self.selected_field_oh), axis=1)
            self.mutation_actions_oh = tf.placeholder(tf.float32, [None, self.n_actions], 
                                                    'mutation_actions_oh')
            self.mutation_q_selected = tf.reduce_sum(
                tf.multiply(self.mutation_q_values, self.mutation_actions_oh), axis=1)
            
            # Combine field and mutation Q-values for mixer input
            combined_q_input = tf.stack([self.field_q_selected, self.mutation_q_selected], axis=1)
            self.mixer = networks.hierarchical_mixer(
                tf.expand_dims(self.field_q_selected, axis=1),
                tf.expand_dims(self.mutation_q_selected, axis=1),
                self.state, self.l_state, 1, 1, self.nn['n_h_mixer'])
                
        with tf.variable_scope("HierarchicalMixer_target"):
            self.field_q_target_selected = tf.reduce_sum(
                tf.multiply(self.field_q_values_target, self.selected_field_oh), axis=1)
            self.mutation_q_target_selected = tf.reduce_sum(
                tf.multiply(self.mutation_q_values_target, self.mutation_actions_oh), axis=1)
            
            self.mixer_target = networks.hierarchical_mixer(
                tf.expand_dims(self.field_q_target_selected, axis=1),
                tf.expand_dims(self.mutation_q_target_selected, axis=1),
                self.state, self.l_state, 1, 1, self.nn['n_h_mixer'])
    
    def get_assign_target_ops(self):
        """Create operations for target network updates."""
        list_initial_ops = []
        list_update_ops = []
        
        # Field selector target updates
        list_field_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_main')
        map_name_field_main = {v.name.split('main')[1]: v for v in list_field_main}
        list_field_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_target')
        map_name_field_target = {v.name.split('target')[1]: v for v in list_field_target}
        
        for name, var in map_name_field_main.items():
            list_initial_ops.append(map_name_field_target[name].assign(var))
            list_update_ops.append(map_name_field_target[name].assign(
                self.tau * var + (1 - self.tau) * map_name_field_target[name]))
        
        # Mutation action target updates
        list_mutation_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationAction_main')
        map_name_mutation_main = {v.name.split('main')[1]: v for v in list_mutation_main}
        list_mutation_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationAction_target')
        map_name_mutation_target = {v.name.split('target')[1]: v for v in list_mutation_target}
        
        for name, var in map_name_mutation_main.items():
            list_initial_ops.append(map_name_mutation_target[name].assign(var))
            list_update_ops.append(map_name_mutation_target[name].assign(
                self.tau * var + (1 - self.tau) * map_name_mutation_target[name]))
        
        # Mixer target updates
        list_mixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HierarchicalMixer_main')
        map_name_mixer_main = {v.name.split('main')[1]: v for v in list_mixer_main}
        list_mixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'HierarchicalMixer_target')
        map_name_mixer_target = {v.name.split('target')[1]: v for v in list_mixer_target}
        
        for name, var in map_name_mixer_main.items():
            list_initial_ops.append(map_name_mixer_target[name].assign(var))
            list_update_ops.append(map_name_mixer_target[name].assign(
                self.tau * var + (1 - self.tau) * map_name_mixer_target[name]))
        
        return list_initial_ops, list_update_ops
    
    def create_train_ops(self):
        """Create training operations for all networks."""
        
        # Field selection training (high-level)
        self.field_td_target = tf.placeholder(tf.float32, [None], 'field_td_target')
        self.field_loss = tf.reduce_mean(tf.square(
            self.field_td_target - tf.squeeze(self.mixer)))
        self.field_opt = tf.train.AdamOptimizer(self.lr_field)
        self.field_train_op = self.field_opt.minimize(self.field_loss)
        
        # Mutation action training (low-level)
        self.mutation_td_target = tf.placeholder(tf.float32, [None], 'mutation_td_target')
        self.mutation_loss = tf.reduce_mean(tf.square(
            self.mutation_td_target - self.mutation_q_selected))
        self.mutation_opt = tf.train.AdamOptimizer(self.lr_mutation)
        self.mutation_train_op = self.mutation_opt.minimize(self.mutation_loss)
        
        # Decoder training
        self.field_importance_target = tf.placeholder(tf.float32, [None, self.n_fields], 
                                                    'field_importance_target')
        self.decoder_loss = tf.losses.softmax_cross_entropy(
            self.field_importance_target, self.decoder_out)
        self.decoder_opt = tf.train.AdamOptimizer(self.lr_decoder)
        self.decoder_train_op = self.decoder_opt.minimize(self.decoder_loss)
    
    def create_summary(self):
        """Create TensorFlow summaries for monitoring."""
        # Field selection summaries
        field_summaries = [tf.summary.scalar('field_loss', self.field_loss)]
        field_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_main')
        for v in field_variables:
            field_summaries.append(tf.summary.histogram(v.op.name, v))
        self.field_summary_op = tf.summary.merge(field_summaries)
        
        # Mutation action summaries
        mutation_summaries = [tf.summary.scalar('mutation_loss', self.mutation_loss)]
        mutation_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationAction_main')
        for v in mutation_variables:
            mutation_summaries.append(tf.summary.histogram(v.op.name, v))
        self.mutation_summary_op = tf.summary.merge(mutation_summaries)
        
        # Decoder summaries
        decoder_summaries = [tf.summary.scalar('decoder_loss', self.decoder_loss)]
        decoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldDecoder')
        for v in decoder_variables:
            decoder_summaries.append(tf.summary.histogram(v.op.name, v))
        self.decoder_summary_op = tf.summary.merge(decoder_summaries)
    
    def select_field(self, obs, field_importance, epsilon, sess):
        """Select protocol field using hierarchical policy.
        
        Args:
            obs: Current observation
            field_importance: Current field importance scores
            epsilon: Exploration parameter
            sess: TensorFlow session
            
        Returns:
            Selected field index
        """
        feed = {self.obs: obs, self.field_importance: field_importance}
        field_q_values = sess.run(self.field_q_values, feed_dict=feed)
        
        # Apply 90/10 exploit/explore strategy
        if np.random.rand() < epsilon:
            # Pure random exploration
            return np.random.randint(0, self.n_fields)
        
        if np.random.rand() < self.exploit_probability:
            # Exploit: Select based on Q-values weighted by importance
            combined_scores = field_q_values[0] * field_importance[0]
            return np.argmax(combined_scores)
        else:
            # Explore: Random selection
            return np.random.randint(0, self.n_fields)
    
    def select_mutation_action(self, obs, mutation_context, selected_field_oh, epsilon, sess):
        """Select mutation action using low-level policy.
        
        Args:
            obs: Current observation
            mutation_context: Context features for mutation decision
            selected_field_oh: Selected field as one-hot vector
            epsilon: Exploration parameter
            sess: TensorFlow session
            
        Returns:
            Selected mutation action index
        """
        feed = {
            self.obs: obs,
            self.mutation_context: mutation_context,
            self.selected_field_oh: selected_field_oh
        }
        
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        mutation_q_values = sess.run(self.mutation_q_values, feed_dict=feed)
        return np.argmax(mutation_q_values[0])
    
    def train_field_selector(self, sess, batch, step_train, summarize=False, writer=None):
        """Train the high-level field selection policy.
        
        Args:
            sess: TensorFlow session
            batch: Training batch
            step_train: Training step number
            summarize: Whether to create summaries
            writer: TensorFlow summary writer
        """
        # Process batch for field selection training
        states, obs, field_selections, rewards, states_next, obs_next, done = self.process_field_batch(batch)
        
        # Get target Q-values
        feed = {self.obs: obs_next, self.field_importance: field_selections}
        field_q_target = sess.run(self.field_q_values_target, feed_dict=feed)
        
        # Calculate TD targets
        done_multiplier = -(done - 1)
        targets = rewards + self.gamma * np.max(field_q_target, axis=1) * done_multiplier
        
        # Train
        feed = {
            self.state: states,
            self.obs: obs,
            self.field_importance: field_selections,
            self.selected_field_oh: field_selections,
            self.mutation_actions_oh: np.zeros((len(batch), self.n_actions)),  # Placeholder
            self.field_td_target: targets
        }
        
        if summarize:
            summary, _ = sess.run([self.field_summary_op, self.field_train_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            sess.run(self.field_train_op, feed_dict=feed)
        
        sess.run(self.list_update_target_ops)
    
    def train_mutation_policy(self, sess, batch, step_train, summarize=False, writer=None):
        """Train the low-level mutation action policy.
        
        Args:
            sess: TensorFlow session
            batch: Training batch
            step_train: Training step number
            summarize: Whether to create summaries
            writer: TensorFlow summary writer
        """
        # Process batch for mutation training
        obs, actions, rewards, obs_next, contexts, field_selections, done = self.process_mutation_batch(batch)
        
        # Get target Q-values
        feed = {
            self.obs: obs_next,
            self.mutation_context: contexts,
            self.selected_field_oh: field_selections
        }
        mutation_q_target = sess.run(self.mutation_q_values_target, feed_dict=feed)
        
        # Calculate TD targets
        done_multiplier = -(done - 1)
        targets = rewards + self.gamma * np.max(mutation_q_target, axis=1) * done_multiplier
        
        # Train
        feed = {
            self.obs: obs,
            self.mutation_context: contexts,
            self.selected_field_oh: field_selections,
            self.mutation_actions_oh: actions,
            self.mutation_td_target: targets
        }
        
        if summarize:
            summary, _ = sess.run([self.mutation_summary_op, self.mutation_train_op], feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            sess.run(self.mutation_train_op, feed_dict=feed)
    
    def train_field_decoder(self, sess, dataset, step_train, summarize=False, writer=None):
        """Train the field importance decoder.
        
        Args:
            sess: TensorFlow session
            dataset: Dataset of field trajectories and importance scores
            step_train: Training step number
            summarize: Whether to create summaries
            writer: TensorFlow summary writer
            
        Returns:
            Expected probability for curriculum learning
        """
        trajectories, mamba_features, field_importance = self.process_decoder_dataset(dataset)
        
        feed = {
            self.field_trajectory: trajectories,
            self.mamba_features: mamba_features,
            self.field_importance_target: field_importance
        }
        
        if summarize:
            summary, _, decoder_probs = sess.run(
                [self.decoder_summary_op, self.decoder_train_op, self.decoder_probs], 
                feed_dict=feed)
            writer.add_summary(summary, step_train)
        else:
            _, decoder_probs = sess.run([self.decoder_train_op, self.decoder_probs], feed_dict=feed)
        
        # Calculate expected probability for curriculum learning
        prob = np.sum(np.multiply(decoder_probs, field_importance), axis=1)
        expected_prob = np.mean(prob)
        
        return expected_prob
    
    def compute_field_importance_reward(self, sess, field_trajectories, mamba_features, 
                                      field_selections):
        """Compute intrinsic reward for field importance learning.
        
        Args:
            sess: TensorFlow session
            field_trajectories: Trajectories of field mutations
            mamba_features: Mamba feature representations
            field_selections: Selected fields (one-hot)
            
        Returns:
            Intrinsic reward values
        """
        feed = {
            self.field_trajectory: field_trajectories,
            self.mamba_features: mamba_features
        }
        decoder_probs = sess.run(self.decoder_probs, feed_dict=feed)
        
        # Calculate P(field|trajectory) as intrinsic reward
        prob = np.sum(np.multiply(decoder_probs, field_selections), axis=1)
        return prob
    
    def process_field_batch(self, batch):
        """Process batch for field selection training."""
        # Extract and format data for field selection training
        states = np.stack(batch[:, 0])
        obs = np.stack(batch[:, 1])
        field_selections = np.stack(batch[:, 2])  # One-hot encoded
        rewards = np.stack(batch[:, 3])
        states_next = np.stack(batch[:, 4])
        obs_next = np.stack(batch[:, 5])
        done = np.stack(batch[:, 6])
        
        return states, obs, field_selections, rewards, states_next, obs_next, done
    
    def process_mutation_batch(self, batch):
        """Process batch for mutation action training."""
        obs = np.stack(batch[:, 0])
        actions = np.stack(batch[:, 1])  # One-hot encoded mutation actions
        rewards = np.stack(batch[:, 2])
        obs_next = np.stack(batch[:, 3])
        contexts = np.stack(batch[:, 4])  # Mutation contexts
        field_selections = np.stack(batch[:, 5])  # One-hot field selections
        done = np.stack(batch[:, 6])
        
        return obs, actions, rewards, obs_next, contexts, field_selections, done
    
    def process_decoder_dataset(self, dataset):
        """Process dataset for decoder training."""
        trajectories = np.stack(dataset[:, 0])
        mamba_features = np.stack(dataset[:, 1])
        field_importance = np.stack(dataset[:, 2])
        
        # Downsample trajectories
        trajectories = trajectories[:, ::self.traj_skip, :]
        
        return trajectories, mamba_features, field_importance
    
    def get_field_importance_update(self, field_idx, reward):
        """Calculate field importance score update.
        
        Args:
            field_idx: Index of mutated field
            reward: Reward received
            
        Returns:
            Updated importance score for the field
        """
        # Simple exponential moving average update
        current_importance = 1.0 / self.n_fields  # Default uniform importance
        normalized_reward = max(0, min(1, reward / 100.0))
        
        updated_importance = (
            (1 - self.field_importance_learning_rate) * current_importance +
            self.field_importance_learning_rate * normalized_reward
        )
        
        return updated_importance
    
    def create_mutation_context(self, field_idx, packet_info, mutation_history):
        """Create context vector for mutation action selection.
        
        Args:
            field_idx: Selected field index
            packet_info: Information about current packet
            mutation_history: History of mutations on this field
            
        Returns:
            Context vector for mutation decision
        """
        context = np.zeros(self.mutation_context_dim)
        
        # Field characteristics
        context[0] = field_idx / self.n_fields  # Normalized field index
        context[1] = packet_info.get('field_length', 0) / 1500.0  # Normalized length
        context[2] = packet_info.get('field_entropy', 0) / 8.0  # Normalized entropy
        
        # Mutation history features
        context[3] = len(mutation_history) / 100.0  # Normalized mutation count
        context[4] = mutation_history.get('success_rate', 0)  # Historical success rate
        context[5] = mutation_history.get('avg_reward', 0) / 10.0  # Normalized avg reward
        
        # Fill remaining context with packet-level features
        context[6:] = np.random.random(self.mutation_context_dim - 6)  # Placeholder for additional features
        
        return context