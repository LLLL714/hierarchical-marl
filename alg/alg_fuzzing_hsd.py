"""Fuzzing adaptation of Hierarchical Skill Discovery (HSD) algorithm.

This module adapts the original HSD algorithm for protocol fuzzing:
- High-level policy: Protocol field selection with 90/10 strategy
- Low-level policy: Mutation action selection with multi-criteria reasoning
- Integrated Mamba feature processing
- Three-tier reward system
"""

import tensorflow as tf
import numpy as np
import sys
import networks_fuzzing


class FuzzingHSDAlg(object):
    """Fuzzing adaptation of Hierarchical Skill Discovery algorithm."""

    def __init__(self, config_alg, config_fuzzing, n_fields, l_state, l_obs, l_mutation_actions, nn):
        """Initialize fuzzing HSD algorithm.
        
        Args:
            config_alg: Dictionary of general RL params
            config_fuzzing: Dictionary of fuzzing-specific params
            n_fields: Number of protocol fields (replaces n_agents)
            l_state: State dimension including Mamba features
            l_obs: Observation dimension
            l_mutation_actions: Number of mutation actions (6)
            nn: Dictionary with neural net sizes
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_mutation_actions = l_mutation_actions
        self.n_fields = n_fields
        self.nn = nn
        
        # General RL parameters
        self.tau = config_alg['tau']
        self.lr_Q = config_alg['lr_Q']
        self.lr_actor = config_alg['lr_actor']
        self.lr_decoder = config_alg['lr_decoder']
        self.gamma = config_alg['gamma']
        
        # Fuzzing-specific parameters
        self.mutation_actions = config_fuzzing.get('mutation_actions', 
                                                  ['insert', 'delete', 'flip', 'replace', 'shuffle', 'copy'])
        self.field_selection_steps = config_fuzzing.get('field_selection_steps', 10)
        self.exploit_ratio = config_fuzzing.get('exploit_ratio', 0.9)
        self.mamba_feature_dim = config_fuzzing.get('mamba_feature_dim', 256)
        self.use_mamba_features = config_fuzzing.get('use_mamba_features', True)
        
        # Trajectory parameters for decoder
        self.traj_length = self.field_selection_steps
        self.traj_skip = config_fuzzing.get('traj_skip', 2)
        self.traj_length_downsampled = int(np.ceil(self.traj_length / self.traj_skip))
        self.use_state_difference = config_fuzzing.get('use_state_difference', True)
        if self.use_state_difference:
            self.traj_length_downsampled -= 1
            
        # Observation truncation for decoder
        self.obs_truncate_length = config_fuzzing.get('obs_truncate_length', None)
        assert (self.obs_truncate_length is None) or (self.obs_truncate_length <= self.l_obs)
        
        # Low-level algorithm type
        self.low_level_alg = config_fuzzing.get('low_level_alg', 'iql')
        assert self.low_level_alg in ['reinforce', 'iac', 'iql']
        if self.low_level_alg == 'iac':
            self.lr_V = config_alg['lr_V']
            
        # Initialize computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops, self.list_update_target_ops_low = self.get_assign_target_ops()
        self.create_train_op_high()
        self.create_train_op_low()
        self.create_train_op_decoder()
        
        # TF summaries
        self.create_summary()
        
    def create_networks(self):
        """Create neural networks for fuzzing HSD."""
        
        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')
        self.field_selection = tf.placeholder(tf.float32, [None, self.n_fields], 'field_selection')
        
        # Mamba features placeholder
        if self.use_mamba_features:
            self.mamba_features = tf.placeholder(tf.float32, [None, self.mamba_feature_dim], 'mamba_features')
        else:
            self.mamba_features = tf.zeros([tf.shape(self.obs)[0], self.mamba_feature_dim])
            
        # Field importance tracking
        self.field_importance = tf.placeholder(tf.float32, [None, self.n_fields], 'field_importance')
        
        # Decoder for field selection discovery
        if self.obs_truncate_length:
            self.traj = tf.placeholder(dtype=tf.float32, 
                                     shape=[None, self.traj_length_downsampled, self.obs_truncate_length])
        else:
            self.traj = tf.placeholder(dtype=tf.float32, 
                                     shape=[None, self.traj_length_downsampled, self.l_obs])
        
        with tf.variable_scope("FuzzingDecoder"):
            self.decoder_out, self.decoder_probs = networks_fuzzing.fuzzing_decoder(
                self.traj, self.traj_length_downsampled, self.nn['n_h_decoder'], self.n_fields)
        
        # Feature processor for Mamba integration
        with tf.variable_scope("FeatureProcessor"):
            self.feature_processor = networks_fuzzing.FeatureProcessor(
                mamba_dim=self.mamba_feature_dim, output_dim=128)
            
            # Extract protocol features from observations
            protocol_features = self.obs[:, :64]  # First 64 dims as protocol features
            self.processed_features = self.feature_processor.build_network(
                self.mamba_features, protocol_features, is_training=True)
        
        # High-level policy: Field selection network
        with tf.variable_scope("FieldSelector_main"):
            self.field_selector = networks_fuzzing.FieldSelectorNetwork(
                obs_dim=128, n_fields=self.n_fields, 
                n_h1=self.nn['n_h1'], n_h2=self.nn['n_h2'])
            self.field_logits, self.field_probs = self.field_selector.build_network(
                self.processed_features, self.field_importance, is_training=True)
        
        with tf.variable_scope("FieldSelector_target"):
            self.field_selector_target = networks_fuzzing.FieldSelectorNetwork(
                obs_dim=128, n_fields=self.n_fields,
                n_h1=self.nn['n_h1'], n_h2=self.nn['n_h2'])
            self.field_logits_target, self.field_probs_target = self.field_selector_target.build_network(
                self.processed_features, self.field_importance, is_training=False)
        
        # High-level Q-values using QMIX approach
        with tf.variable_scope("FieldQ_main"):
            self.field_qs = networks_fuzzing.fuzzing_qmix_single(
                self.processed_features, self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        
        with tf.variable_scope("FieldQ_target"):
            self.field_qs_target = networks_fuzzing.fuzzing_qmix_single(
                self.processed_features, self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        
        self.argmax_field_Q = tf.argmax(self.field_qs, axis=1)
        self.argmax_field_Q_target = tf.argmax(self.field_qs_target, axis=1)
        
        # Field selection Q-value extraction
        self.field_actions_1hot = tf.placeholder(tf.float32, [None, self.n_fields], 'field_actions_1hot')
        self.field_q_selected = tf.reduce_sum(tf.multiply(self.field_qs, self.field_actions_1hot), axis=1)
        self.mixer_q_input = tf.reshape(self.field_q_selected, [-1, 1])  # Single field selection
        
        self.field_q_target_selected = tf.reduce_sum(tf.multiply(self.field_qs_target, self.field_actions_1hot), axis=1)
        self.mixer_target_q_input = tf.reshape(self.field_q_target_selected, [-1, 1])
        
        # Mixing network for field coordination (simplified for single field)
        with tf.variable_scope("FieldMixer_main"):
            self.field_mixer = networks_fuzzing.fuzzing_qmix_mixer(
                self.mixer_q_input, self.state, self.l_state, 1, self.nn['n_h_mixer'])
        
        with tf.variable_scope("FieldMixer_target"):
            self.field_mixer_target = networks_fuzzing.fuzzing_qmix_mixer(
                self.mixer_target_q_input, self.state, self.l_state, 1, self.nn['n_h_mixer'])
        
        # Low-level policy: Mutation action selection
        self.field_features = tf.placeholder(tf.float32, [None, 16], 'field_features')  # Field-specific features
        self.mutation_history = tf.placeholder(tf.float32, [None, self.l_mutation_actions], 'mutation_history')
        
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
            with tf.variable_scope("MutationPolicy_main"):
                self.mutation_network = networks_fuzzing.MutationActionNetwork(
                    obs_dim=128, field_features_dim=16, n_mutations=self.l_mutation_actions,
                    n_h1=self.nn['n_h1_low'], n_h2=self.nn['n_h2_low'])
                self.mutation_logits, mutation_probs_raw = self.mutation_network.build_network(
                    self.processed_features, self.field_features, self.mutation_history, is_training=True)
            
            # Apply epsilon-greedy exploration
            self.mutation_probs = (1 - self.epsilon) * mutation_probs_raw + \
                                self.epsilon / float(self.l_mutation_actions)
            self.mutation_action_samples = tf.multinomial(tf.log(self.mutation_probs), 1)
        
        if self.low_level_alg == 'iac':
            with tf.variable_scope("MutationV_main"):
                # Value function for mutation actions
                combined_input = tf.concat([self.processed_features, self.field_features], axis=1)
                self.mutation_V = tf.layers.dense(combined_input, 1, activation=None, name='mutation_v')
            
            with tf.variable_scope("MutationV_target"):
                combined_input_target = tf.concat([self.processed_features, self.field_features], axis=1)
                self.mutation_V_target = tf.layers.dense(combined_input_target, 1, activation=None, name='mutation_v')
        
        # Low-level Q-functions for IQL
        if self.low_level_alg == 'iql':
            with tf.variable_scope("MutationQ_main"):
                self.mutation_Q = networks_fuzzing.fuzzing_mutation_q(
                    self.processed_features, self.field_features,
                    self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_mutation_actions)
            
            with tf.variable_scope("MutationQ_target"):
                self.mutation_Q_target = networks_fuzzing.fuzzing_mutation_q(
                    self.processed_features, self.field_features,
                    self.nn['n_h1_low'], self.nn['n_h2_low'], self.l_mutation_actions)
            
            self.argmax_mutation_Q = tf.argmax(self.mutation_Q, axis=1)
            self.mutation_actions_1hot = tf.placeholder(tf.float32, [None, self.l_mutation_actions], 'mutation_actions_1hot')
    
    def get_assign_target_ops(self):
        """Get operations for updating target networks."""
        list_initial_ops = []
        list_update_ops = []
        list_update_ops_low = []
        
        # Field selector target update
        field_main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_main')
        field_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_target')
        
        for main_var, target_var in zip(field_main_vars, field_target_vars):
            list_initial_ops.append(target_var.assign(main_var))
            list_update_ops.append(target_var.assign(self.tau * main_var + (1 - self.tau) * target_var))
        
        # Field Q target update
        fieldq_main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldQ_main')
        fieldq_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldQ_target')
        
        for main_var, target_var in zip(fieldq_main_vars, fieldq_target_vars):
            list_initial_ops.append(target_var.assign(main_var))
            list_update_ops.append(target_var.assign(self.tau * main_var + (1 - self.tau) * target_var))
        
        # Field mixer target update
        mixer_main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldMixer_main')
        mixer_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldMixer_target')
        
        for main_var, target_var in zip(mixer_main_vars, mixer_target_vars):
            list_initial_ops.append(target_var.assign(main_var))
            list_update_ops.append(target_var.assign(self.tau * main_var + (1 - self.tau) * target_var))
        
        # Low-level target updates
        if self.low_level_alg == 'iac':
            v_main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationV_main')
            v_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationV_target')
            
            for main_var, target_var in zip(v_main_vars, v_target_vars):
                list_initial_ops.append(target_var.assign(main_var))
                list_update_ops_low.append(target_var.assign(self.tau * main_var + (1 - self.tau) * target_var))
        
        elif self.low_level_alg == 'iql':
            mutq_main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationQ_main')
            mutq_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationQ_target')
            
            for main_var, target_var in zip(mutq_main_vars, mutq_target_vars):
                list_initial_ops.append(target_var.assign(main_var))
                list_update_ops_low.append(target_var.assign(self.tau * main_var + (1 - self.tau) * target_var))
        
        return list_initial_ops, list_update_ops, list_update_ops_low
    
    def select_field(self, obs, mamba_features, field_importance, epsilon, sess):
        """Select protocol field using 90/10 strategy.
        
        Args:
            obs: Observations
            mamba_features: Mamba extracted features
            field_importance: Historical field importance scores
            epsilon: Exploration parameter
            sess: TensorFlow session
            
        Returns:
            Selected field index
        """
        feed_dict = {
            self.obs: obs,
            self.mamba_features: mamba_features,
            self.field_importance: field_importance
        }
        
        field_probs = sess.run(self.field_probs, feed_dict=feed_dict)
        
        # Sample from probability distribution
        field_idx = np.random.choice(self.n_fields, p=field_probs[0])
        
        return field_idx
    
    def select_mutation_action(self, obs, mamba_features, field_features, mutation_history, epsilon, sess):
        """Select mutation action for given field.
        
        Args:
            obs: Observations
            mamba_features: Mamba features
            field_features: Field-specific features
            mutation_history: Historical mutation performance
            epsilon: Exploration parameter
            sess: TensorFlow session
            
        Returns:
            Selected mutation action index
        """
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            feed_dict = {
                self.obs: obs,
                self.mamba_features: mamba_features,
                self.field_features: field_features,
                self.mutation_history: mutation_history,
                self.epsilon: epsilon
            }
            action = sess.run(self.mutation_action_samples, feed_dict=feed_dict)
            return action.flatten()[0]
        
        elif self.low_level_alg == 'iql':
            feed_dict = {
                self.obs: obs,
                self.mamba_features: mamba_features,
                self.field_features: field_features,
                self.mutation_history: mutation_history
            }
            q_values = sess.run(self.mutation_Q, feed_dict=feed_dict)
            
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.l_mutation_actions)
            else:
                return np.argmax(q_values[0])
    
    def create_train_op_high(self):
        """Create training operations for high-level policy (field selection)."""
        self.td_target_high = tf.placeholder(tf.float32, [None], 'td_target_high')
        self.loss_Q_high = tf.reduce_mean(tf.square(self.td_target_high - tf.squeeze(self.field_mixer)))
        self.Q_opt_high = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_op_high = self.Q_opt_high.minimize(self.loss_Q_high)
    
    def create_train_op_low(self):
        """Create training operations for low-level policy (mutation actions)."""
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            self.mutation_actions_taken = tf.placeholder(tf.float32, [None, self.l_mutation_actions], 'mutation_actions_taken')
            log_probs = tf.log(tf.reduce_sum(tf.multiply(self.mutation_probs, self.mutation_actions_taken), axis=1) + 1e-15)
            
            if self.low_level_alg == 'reinforce':
                log_probs_reshaped = tf.reshape(log_probs, [-1, self.traj_length])
                self.traj_reward = tf.placeholder(tf.float32, [None], 'traj_reward')
                self.mutation_policy_loss = -tf.reduce_mean(tf.reduce_sum(log_probs_reshaped, axis=1) * self.traj_reward)
            
            elif self.low_level_alg == 'iac':
                # Value function training
                self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
                self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.mutation_V)))
                self.V_opt = tf.train.AdamOptimizer(self.lr_V)
                self.V_op = self.V_opt.minimize(self.loss_V)
                
                # Policy training
                self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
                self.V_td_error = self.V_td_target - self.V_evaluated
                self.mutation_policy_loss = -tf.reduce_mean(tf.multiply(log_probs, self.V_td_error))
            
            self.mutation_policy_opt = tf.train.AdamOptimizer(self.lr_actor)
            self.mutation_policy_op = self.mutation_policy_opt.minimize(self.mutation_policy_loss)
        
        elif self.low_level_alg == 'iql':
            self.td_target_mutation = tf.placeholder(tf.float32, [None], 'td_target_mutation')
            self.td_error_mutation = self.td_target_mutation - tf.reduce_sum(
                tf.multiply(self.mutation_Q, self.mutation_actions_1hot), axis=1)
            self.loss_mutation_IQL = tf.reduce_mean(tf.square(self.td_error_mutation))
            self.mutation_IQL_opt = tf.train.AdamOptimizer(self.lr_Q)
            self.mutation_IQL_op = self.mutation_IQL_opt.minimize(self.loss_mutation_IQL)
    
    def create_train_op_decoder(self):
        """Create training operations for field selection decoder."""
        self.onehot_field_selection = tf.placeholder(tf.float32, [None, self.n_fields], 'onehot_field_selection')
        self.decoder_loss = tf.losses.softmax_cross_entropy(self.onehot_field_selection, self.decoder_out)
        self.decoder_opt = tf.train.AdamOptimizer(self.lr_decoder)
        self.decoder_op = self.decoder_opt.minimize(self.decoder_loss)
    
    def create_summary(self):
        """Create TensorFlow summaries for monitoring."""
        # High-level summaries
        summaries_Q_high = [tf.summary.scalar('loss_Q_high', self.loss_Q_high)]
        self.summary_op_Q_high = tf.summary.merge(summaries_Q_high)
        
        # Low-level summaries
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            summaries_policy = [tf.summary.scalar('mutation_policy_loss', self.mutation_policy_loss)]
            self.summary_op_policy = tf.summary.merge(summaries_policy)
        
        if self.low_level_alg == 'iac':
            summaries_V = [tf.summary.scalar('V_loss', self.loss_V)]
            self.summary_op_V = tf.summary.merge(summaries_V)
        
        if self.low_level_alg == 'iql':
            summaries_mutation = [tf.summary.scalar('loss_mutation_IQL', self.loss_mutation_IQL)]
            self.summary_op_mutation = tf.summary.merge(summaries_mutation)
        
        # Decoder summaries
        summaries_decoder = [tf.summary.scalar('decoder_loss', self.decoder_loss)]
        self.summary_op_decoder = tf.summary.merge(summaries_decoder)
    
    def train_field_selection(self, sess, batch, step_train, summarize=False, writer=None):
        """Train high-level field selection policy."""
        # Process batch for field selection training
        n_steps, state, obs, mamba_features, field_actions, reward, state_next, obs_next, mamba_features_next, done = self.process_batch_high(batch)
        
        # Get target field selections
        feed_dict = {
            self.obs: obs_next,
            self.mamba_features: mamba_features_next,
            self.field_importance: np.ones((len(obs_next), self.n_fields)) / self.n_fields  # Uniform for now
        }
        target_field_qs = sess.run(self.field_qs_target, feed_dict=feed_dict)
        argmax_fields = np.argmax(target_field_qs, axis=1)
        
        # Convert to one-hot
        field_actions_target_1hot = np.zeros([len(obs_next), self.n_fields])
        field_actions_target_1hot[np.arange(len(obs_next)), argmax_fields] = 1
        
        # Get target Q-values
        feed_dict = {
            self.state: state_next,
            self.field_actions_1hot: field_actions_target_1hot,
            self.obs: obs_next,
            self.mamba_features: mamba_features_next
        }
        Q_tot_target = sess.run(self.field_mixer_target, feed_dict=feed_dict)
        
        done_multiplier = -(done - 1)
        target = reward + self.gamma * np.squeeze(Q_tot_target) * done_multiplier
        
        # Train
        feed_dict = {
            self.state: state,
            self.td_target_high: target,
            self.obs: obs,
            self.mamba_features: mamba_features,
            self.field_actions_1hot: field_actions
        }
        
        if summarize:
            summary, _ = sess.run([self.summary_op_Q_high, self.Q_op_high], feed_dict=feed_dict)
            writer.add_summary(summary, step_train)
        else:
            _ = sess.run(self.Q_op_high, feed_dict=feed_dict)
        
        sess.run(self.list_update_target_ops)
    
    def train_mutation_actions(self, sess, batch, step_train, summarize=False, writer=None):
        """Train low-level mutation action policy."""
        if self.low_level_alg == 'iql':
            # Process batch for mutation training
            n_steps, obs, mamba_features, field_features, mutation_actions, rewards, obs_next, mamba_features_next, field_features_next, mutation_history, done = self.process_batch_low(batch)
            
            # Get target values
            feed_dict = {
                self.obs: obs_next,
                self.mamba_features: mamba_features_next,
                self.field_features: field_features_next,
                self.mutation_history: mutation_history
            }
            Q_target = sess.run(self.mutation_Q_target, feed_dict=feed_dict)
            done_multiplier = -(done - 1)
            target = rewards + self.gamma * np.max(Q_target, axis=1) * done_multiplier
            
            feed_dict = {
                self.obs: obs,
                self.mamba_features: mamba_features,
                self.field_features: field_features,
                self.mutation_actions_1hot: mutation_actions,
                self.mutation_history: mutation_history,
                self.td_target_mutation: target
            }
            
            if summarize:
                summary, _ = sess.run([self.summary_op_mutation, self.mutation_IQL_op], feed_dict=feed_dict)
                writer.add_summary(summary, step_train)
            else:
                _ = sess.run(self.mutation_IQL_op, feed_dict=feed_dict)
            
            sess.run(self.list_update_target_ops_low)
    
    def train_decoder(self, sess, dataset, step_train, summarize=False, writer=None):
        """Train field selection decoder."""
        dataset = np.array(dataset)
        obs = np.stack(dataset[:, 0])
        field_selections = np.stack(dataset[:, 1])
        
        # Downsample observations
        obs_downsampled = obs[:, ::self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :, :self.obs_truncate_length]
        if self.use_state_difference:
            obs_downsampled = obs_downsampled[:, 1:, :] - obs_downsampled[:, :-1, :]
        
        feed_dict = {
            self.onehot_field_selection: field_selections,
            self.traj: obs_downsampled
        }
        
        if summarize:
            summary, _, decoder_probs = sess.run([self.summary_op_decoder, self.decoder_op, self.decoder_probs], feed_dict=feed_dict)
            writer.add_summary(summary, step_train)
        else:
            _, decoder_probs = sess.run([self.decoder_op, self.decoder_probs], feed_dict=feed_dict)
        
        # Calculate expected probability
        prob = np.sum(np.multiply(decoder_probs, field_selections), axis=1)
        expected_prob = np.mean(prob)
        
        return expected_prob
    
    def process_batch_high(self, batch):
        """Process batch for high-level training."""
        # Extract components from batch
        state = np.stack(batch[:, 0])
        obs = np.stack(batch[:, 1])
        mamba_features = np.stack(batch[:, 2])
        field_actions = np.stack(batch[:, 3])
        reward = np.stack(batch[:, 4])
        state_next = np.stack(batch[:, 5])
        obs_next = np.stack(batch[:, 6])
        mamba_features_next = np.stack(batch[:, 7])
        done = np.stack(batch[:, 8])
        
        n_steps = len(state)
        return n_steps, state, obs, mamba_features, field_actions, reward, state_next, obs_next, mamba_features_next, done
    
    def process_batch_low(self, batch):
        """Process batch for low-level training."""
        # Extract components from batch
        obs = np.stack(batch[:, 0])
        mamba_features = np.stack(batch[:, 1])
        field_features = np.stack(batch[:, 2])
        mutation_actions = np.stack(batch[:, 3])
        rewards = np.stack(batch[:, 4])
        obs_next = np.stack(batch[:, 5])
        mamba_features_next = np.stack(batch[:, 6])
        field_features_next = np.stack(batch[:, 7])
        mutation_history = np.stack(batch[:, 8])
        done = np.stack(batch[:, 9])
        
        n_steps = len(obs)
        return n_steps, obs, mamba_features, field_features, mutation_actions, rewards, obs_next, mamba_features_next, field_features_next, mutation_history, done
    
    def compute_intrinsic_reward(self, sess, field_traj_obs, field_selection):
        """Compute intrinsic reward based on field selection consistency."""
        # Downsample trajectory
        obs_downsampled = field_traj_obs[::self.traj_skip, :]
        if self.obs_truncate_length:
            obs_downsampled = obs_downsampled[:, :self.obs_truncate_length]
        
        if self.use_state_difference:
            obs_downsampled = obs_downsampled[1:, :] - obs_downsampled[:-1, :]
        
        # Reshape for batch processing
        obs_batch = np.expand_dims(obs_downsampled, axis=0)
        
        decoder_probs = sess.run(self.decoder_probs, feed_dict={self.traj: obs_batch})
        prob = np.sum(np.multiply(decoder_probs[0], field_selection))
        
        return prob