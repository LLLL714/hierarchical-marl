"""Implementation of hierarchical fuzzing reinforcement learning based on HSD.

This adapts the HSD framework for protocol fuzzing by:
- High-level policy: Selects protocol fields to mutate (90% reward-based, 10% random)
- Low-level policy: Selects mutation operations for chosen fields
- Decoder: Learns to predict field selection from mutation history for intrinsic rewards
"""

import tensorflow as tf
import numpy as np
import sys
import logging

# Import fuzzing-specific networks
from . import networks_fuzzing


class AlgFuzzingHSD(object):
    """Fuzzing-specific adaptation of HSD algorithm."""

    def __init__(self, config_alg, config_h, config_fuzzing, n_fields, l_state, l_obs, l_mutation_actions, nn):
        """Initialize fuzzing HSD algorithm.
        
        Args:
            config_alg: Dictionary of general RL params
            config_h: Dictionary of HSD params (adapted for fuzzing)
            config_fuzzing: Dictionary of fuzzing-specific params
            n_fields: Number of protocol fields (replaces n_agents)
            l_state: State dimension
            l_obs: Observation dimension per field
            l_mutation_actions: Number of mutation action types
            nn: Neural network configuration dictionary
        """
        self.l_state = l_state
        self.l_obs = l_obs
        self.l_mutation_actions = l_mutation_actions
        self.n_fields = n_fields
        self.nn = nn

        # RL parameters
        self.tau = config_alg['tau']
        self.lr_Q = config_alg['lr_Q']
        self.lr_actor = config_alg['lr_actor']
        self.lr_decoder = config_alg['lr_decoder']
        self.gamma = config_alg['gamma']

        # Fuzzing-specific parameters
        self.field_selection_ratio = config_fuzzing.get('field_selection_reward_ratio', 0.9)
        self.mutation_history_length = config_fuzzing.get('mutation_history_length', 10)
        self.max_episode_steps = config_fuzzing.get('max_episode_steps', 1000)

        # HSD parameters adapted for fuzzing
        self.mutation_steps_per_assign = config_h['steps_per_assign']  # Steps per field selection
        self.traj_skip = config_h['traj_skip']
        self.traj_length_downsampled = int(np.ceil(self.mutation_steps_per_assign / self.traj_skip))
        self.use_state_difference = config_h['use_state_difference']
        if self.use_state_difference:
            self.traj_length_downsampled -= 1

        self.obs_truncate_length = config_h['obs_truncate_length']
        assert (self.obs_truncate_length is None) or (self.obs_truncate_length <= self.l_obs)

        # Low-level algorithm for mutation action selection
        self.low_level_alg = config_h['low_level_alg']
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

        # Logging
        self.logger = logging.getLogger(__name__)

    def create_networks(self):
        """Create neural networks for fuzzing HSD."""

        # Placeholders
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'obs')
        self.field_selection = tf.placeholder(tf.float32, [None, self.n_fields], 'field_selection')

        # Decoder p(field_selection|mutation_history)
        # Predicts which fields were selected based on mutation trajectory
        if self.obs_truncate_length:
            self.mutation_traj = tf.placeholder(dtype=tf.float32, 
                                              shape=[None, self.traj_length_downsampled, self.obs_truncate_length])
        else:
            self.mutation_traj = tf.placeholder(dtype=tf.float32, 
                                              shape=[None, self.traj_length_downsampled, self.l_obs])

        with tf.variable_scope("FieldSelectionDecoder"):
            self.decoder_out, self.decoder_probs = networks_fuzzing.field_selection_decoder(
                self.mutation_traj, self.traj_length_downsampled, self.nn['n_h_decoder'], self.n_fields)

        # Low-level policy for mutation action selection
        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
            with tf.variable_scope("MutationPolicy_main"):
                probs = networks_fuzzing.mutation_actor(
                    self.obs, self.field_selection, self.nn['n_h1_low'], 
                    self.nn['n_h2_low'], self.l_mutation_actions)
            self.mutation_probs = (1 - self.epsilon) * probs + self.epsilon / float(self.l_mutation_actions)
            self.mutation_action_samples = tf.multinomial(tf.log(self.mutation_probs), 1)

        if self.low_level_alg == 'iac':
            with tf.variable_scope("MutationV_main"):
                self.V_mutation = networks_fuzzing.mutation_critic(
                    self.obs, self.field_selection, self.nn['n_h1_low'], self.nn['n_h2_low'])
            with tf.variable_scope("MutationV_target"):
                self.V_mutation_target = networks_fuzzing.mutation_critic(
                    self.obs, self.field_selection, self.nn['n_h1_low'], self.nn['n_h2_low'])

        # Low-level Q-functions for mutation actions
        if self.low_level_alg == 'iql':
            with tf.variable_scope("QMutation_main"):
                self.Q_mutation = networks_fuzzing.mutation_Q_network(
                    self.obs, self.field_selection, self.nn['n_h1_low'], 
                    self.nn['n_h2_low'], self.l_mutation_actions)
            with tf.variable_scope("QMutation_target"):
                self.Q_mutation_target = networks_fuzzing.mutation_Q_network(
                    self.obs, self.field_selection, self.nn['n_h1_low'], 
                    self.nn['n_h2_low'], self.l_mutation_actions)
            self.argmax_Q_mutation = tf.argmax(self.Q_mutation, axis=1)
            self.mutation_actions_1hot = tf.placeholder(tf.float32, [None, self.l_mutation_actions], 'mutation_actions_1hot')

        # High-level field selection network (QMIX-style)
        # Individual field selection networks
        with tf.variable_scope("FieldSelector_main"):
            self.field_qs = networks_fuzzing.field_selector_network(
                self.obs, self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        with tf.variable_scope("FieldSelector_target"):
            self.field_qs_target = networks_fuzzing.field_selector_network(
                self.obs, self.nn['n_h1'], self.nn['n_h2'], self.n_fields)

        self.argmax_field_Q = tf.argmax(self.field_qs, axis=1)
        self.argmax_field_Q_target = tf.argmax(self.field_qs_target, axis=1)

        # To extract Q-value from field_qs and field_qs_target
        self.field_actions_1hot = tf.placeholder(tf.float32, [None, self.n_fields], 'field_actions_1hot')
        self.field_q_selected = tf.reduce_sum(tf.multiply(self.field_qs, self.field_actions_1hot), axis=1)
        self.field_mixer_q_input = tf.reshape(self.field_q_selected, [-1, self.n_fields])

        self.field_q_target_selected = tf.reduce_sum(tf.multiply(self.field_qs_target, self.field_actions_1hot), axis=1)
        self.field_mixer_target_q_input = tf.reshape(self.field_q_target_selected, [-1, self.n_fields])

        # Mixing network for field selection
        with tf.variable_scope("FieldMixer_main"):
            self.field_mixer = networks_fuzzing.field_selection_mixer(
                self.field_mixer_q_input, self.state, self.l_state, self.n_fields, self.nn['n_h_mixer'])
        with tf.variable_scope("FieldMixer_target"):
            self.field_mixer_target = networks_fuzzing.field_selection_mixer(
                self.field_mixer_target_q_input, self.state, self.l_state, self.n_fields, self.nn['n_h_mixer'])

    def get_assign_target_ops(self):
        """Get target network assignment operations."""
        list_initial_ops = []
        list_update_ops = []
        list_update_ops_low = []

        # Field selector networks
        list_FieldSelector_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_main')
        map_name_FieldSelector_main = {v.name.split('main')[1]: v for v in list_FieldSelector_main}
        list_FieldSelector_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldSelector_target')
        map_name_FieldSelector_target = {v.name.split('target')[1]: v for v in list_FieldSelector_target}

        if len(list_FieldSelector_main) != len(list_FieldSelector_target):
            raise ValueError("Field selector main and target network lengths do not match")

        for name, var in map_name_FieldSelector_main.items():
            list_initial_ops.append(map_name_FieldSelector_target[name].assign(var))

        for name, var in map_name_FieldSelector_main.items():
            list_update_ops.append(map_name_FieldSelector_target[name].assign(
                self.tau * var + (1 - self.tau) * map_name_FieldSelector_target[name]))

        # Field mixer networks
        list_FieldMixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldMixer_main')
        map_name_FieldMixer_main = {v.name.split('main')[1]: v for v in list_FieldMixer_main}
        list_FieldMixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FieldMixer_target')
        map_name_FieldMixer_target = {v.name.split('target')[1]: v for v in list_FieldMixer_target}

        if len(list_FieldMixer_main) != len(list_FieldMixer_target):
            raise ValueError("Field mixer main and target network lengths do not match")

        for name, var in map_name_FieldMixer_main.items():
            list_initial_ops.append(map_name_FieldMixer_target[name].assign(var))

        for name, var in map_name_FieldMixer_main.items():
            list_update_ops.append(map_name_FieldMixer_target[name].assign(
                self.tau * var + (1 - self.tau) * map_name_FieldMixer_target[name]))

        # Low-level mutation networks
        if self.low_level_alg == 'iac':
            list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationV_main')
            map_name_V_main = {v.name.split('main')[1]: v for v in list_V_main}
            list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'MutationV_target')
            map_name_V_target = {v.name.split('target')[1]: v for v in list_V_target}
            if len(list_V_main) != len(list_V_target):
                raise ValueError("Mutation V main and target network lengths do not match")
            for name, var in map_name_V_main.items():
                list_initial_ops.append(map_name_V_target[name].assign(var))
            for name, var in map_name_V_main.items():
                list_update_ops_low.append(map_name_V_target[name].assign(
                    self.tau * var + (1 - self.tau) * map_name_V_target[name]))

        elif self.low_level_alg == 'iql':
            list_QMutation_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'QMutation_main')
            map_name_QMutation_main = {v.name.split('main')[1]: v for v in list_QMutation_main}
            list_QMutation_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'QMutation_target')
            map_name_QMutation_target = {v.name.split('target')[1]: v for v in list_QMutation_target}
            if len(list_QMutation_main) != len(list_QMutation_target):
                raise ValueError("Mutation Q main and target network lengths do not match")
            for name, var in map_name_QMutation_main.items():
                list_initial_ops.append(map_name_QMutation_target[name].assign(var))
            for name, var in map_name_QMutation_main.items():
                list_update_ops_low.append(map_name_QMutation_target[name].assign(
                    self.tau * var + (1 - self.tau) * map_name_QMutation_target[name]))

        return list_initial_ops, list_update_ops, list_update_ops_low

    def run_mutation_actor(self, list_obs, field_selections, epsilon, sess):
        """Get mutation actions for all fields as a batch.

        Args:
            list_obs: List of observation vectors, one per field
            field_selections: np.array where each row is a 1-hot vector for field selection
            epsilon: Exploration parameter
            sess: TF session

        Returns: np.array of mutation action integers
        """
        obs = np.array(list_obs)

        if self.low_level_alg == 'reinforce' or self.low_level_alg == 'iac':
            feed = {self.obs: obs, self.field_selection: field_selections, self.epsilon: epsilon}
            actions = sess.run(self.mutation_action_samples, feed_dict=feed)
        elif self.low_level_alg == 'iql':
            feed = {self.obs: obs, self.field_selection: field_selections}
            actions_argmax = sess.run(self.argmax_Q_mutation, feed_dict=feed)
            actions = np.zeros(self.n_fields, dtype=int)
            for idx in range(self.n_fields):
                if np.random.rand() < epsilon:
                    actions[idx] = np.random.randint(0, self.l_mutation_actions)
                else:
                    actions[idx] = actions_argmax[idx]

        return actions.flatten()

    def assign_fields(self, list_obs, epsilon, sess, field_importance_scores=None):
        """Get field selection using 90% reward-based, 10% random strategy.
        
        Args:
            list_obs: List of observation vectors, one per field
            epsilon: Exploration parameter
            sess: TF session
            field_importance_scores: Optional importance scores for reward-based selection

        Returns: np.array of field indices to mutate
        """
        obs = np.array(list_obs)
        
        # Get Q-values for all fields
        feed = {self.obs: obs}
        Q_values = sess.run(self.field_qs, feed_dict=feed)
        
        field_indices = []
        
        for field_idx in range(self.n_fields):
            if np.random.rand() < self.field_selection_ratio:
                # 90% reward-based selection
                if field_importance_scores is not None and np.sum(field_importance_scores) > 0:
                    # Use importance scores if available
                    probabilities = field_importance_scores / np.sum(field_importance_scores)
                    selected_field = np.random.choice(self.n_fields, p=probabilities)
                else:
                    # Use Q-values for selection
                    if np.random.rand() < epsilon:
                        selected_field = np.random.randint(0, self.n_fields)
                    else:
                        selected_field = np.argmax(Q_values[field_idx])
            else:
                # 10% random exploration
                selected_field = np.random.randint(0, self.n_fields)
                
            field_indices.append(selected_field)
            
        return np.array(field_indices)

    def compute_intrinsic_reward(self, sess, mutation_trajectories, field_selections):
        """Compute intrinsic reward based on field selection predictability.
        
        Args:
            sess: TF session
            mutation_trajectories: List of mutation observation trajectories
            field_selections: Field selection vectors
            
        Returns: np.array of intrinsic rewards for each field
        """
        # Prepare trajectory data
        trajs = []
        for traj in mutation_trajectories:
            if len(traj) >= self.mutation_steps_per_assign:
                # Downsample trajectory
                downsampled = traj[-self.mutation_steps_per_assign::self.traj_skip]
                if self.use_state_difference and len(downsampled) > 1:
                    # Use state differences
                    downsampled = [downsampled[i+1] - downsampled[i] for i in range(len(downsampled)-1)]
                
                # Truncate observations if needed
                if self.obs_truncate_length:
                    downsampled = [obs[:self.obs_truncate_length] for obs in downsampled]
                    
                # Pad to required length
                while len(downsampled) < self.traj_length_downsampled:
                    downsampled.append(np.zeros_like(downsampled[0] if downsampled else np.zeros(self.l_obs)))
                    
                trajs.append(downsampled[:self.traj_length_downsampled])
            else:
                # Trajectory too short, pad with zeros
                if self.obs_truncate_length:
                    obs_dim = self.obs_truncate_length
                else:
                    obs_dim = self.l_obs
                trajs.append([np.zeros(obs_dim) for _ in range(self.traj_length_downsampled)])

        trajs = np.array(trajs)
        
        # Get decoder predictions
        feed = {self.mutation_traj: trajs}
        log_probs = sess.run(self.decoder_out, feed_dict=feed)
        
        # Calculate intrinsic rewards
        intrinsic_rewards = []
        for i, field_selection in enumerate(field_selections):
            # Convert field selection to index
            field_idx = np.argmax(field_selection)
            
            # Intrinsic reward is negative log probability (encourages unpredictability)
            if i < len(log_probs):
                reward = -log_probs[i, field_idx]
            else:
                reward = 0.0
                
            intrinsic_rewards.append(reward)
            
        return np.array(intrinsic_rewards)

    def create_train_op_high(self):
        """Create training operation for high-level field selection policy."""
        # Placeholders for training
        self.reward_high = tf.placeholder(tf.float32, [None], 'reward_high')
        self.done_high = tf.placeholder(tf.float32, [None], 'done_high')
        self.state_next = tf.placeholder(tf.float32, [None, self.l_state], 'state_next')
        self.obs_next = tf.placeholder(tf.float32, [None, self.l_obs], 'obs_next')

        # Target Q calculation
        with tf.variable_scope("FieldSelector_target", reuse=True):
            field_qs_next_target = networks_fuzzing.field_selector_network(
                self.obs_next, self.nn['n_h1'], self.nn['n_h2'], self.n_fields)
        
        field_q_next_max = tf.reduce_max(field_qs_next_target, axis=1)
        field_q_next_mixer_input = tf.reshape(field_q_next_max, [-1, self.n_fields])
        
        with tf.variable_scope("FieldMixer_target", reuse=True):
            field_mixer_next = networks_fuzzing.field_selection_mixer(
                field_q_next_mixer_input, self.state_next, self.l_state, self.n_fields, self.nn['n_h_mixer'])

        target_q = self.reward_high + self.gamma * field_mixer_next.flatten() * (1.0 - self.done_high)
        target_q = tf.stop_gradient(target_q)

        # Loss calculation
        td_error = target_q - self.field_mixer.flatten()
        self.loss_high = tf.reduce_mean(tf.square(td_error))

        # Optimizer
        self.train_op_high = tf.train.AdamOptimizer(self.lr_Q).minimize(self.loss_high)

    def create_train_op_low(self):
        """Create training operation for low-level mutation policy."""
        if self.low_level_alg == 'iql':
            # Placeholders
            self.reward_low = tf.placeholder(tf.float32, [None], 'reward_low')
            self.done_low = tf.placeholder(tf.float32, [None], 'done_low')
            self.obs_next_low = tf.placeholder(tf.float32, [None, self.l_obs], 'obs_next_low')
            self.field_selection_next = tf.placeholder(tf.float32, [None, self.n_fields], 'field_selection_next')

            # Target Q calculation
            with tf.variable_scope("QMutation_target", reuse=True):
                Q_mutation_next_target = networks_fuzzing.mutation_Q_network(
                    self.obs_next_low, self.field_selection_next, self.nn['n_h1_low'], 
                    self.nn['n_h2_low'], self.l_mutation_actions)

            Q_next_max = tf.reduce_max(Q_mutation_next_target, axis=1)
            target_q_low = self.reward_low + self.gamma * Q_next_max * (1.0 - self.done_low)
            target_q_low = tf.stop_gradient(target_q_low)

            # Current Q values
            Q_selected = tf.reduce_sum(tf.multiply(self.Q_mutation, self.mutation_actions_1hot), axis=1)

            # Loss
            td_error_low = target_q_low - Q_selected
            self.loss_low = tf.reduce_mean(tf.square(td_error_low))

            # Optimizer
            self.train_op_low = tf.train.AdamOptimizer(self.lr_actor).minimize(self.loss_low)

        # Add other low-level algorithms (reinforce, iac) as needed

    def create_train_op_decoder(self):
        """Create training operation for field selection decoder."""
        # Target field selections (ground truth)
        self.target_field_selections = tf.placeholder(tf.float32, [None, self.n_fields], 'target_field_selections')
        
        # Cross-entropy loss
        self.loss_decoder = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target_field_selections, 
                logits=self.decoder_out))

        # Optimizer
        self.train_op_decoder = tf.train.AdamOptimizer(self.lr_decoder).minimize(self.loss_decoder)

    def create_summary(self):
        """Create TensorFlow summaries for monitoring."""
        tf.summary.scalar('loss_high', self.loss_high)
        if hasattr(self, 'loss_low'):
            tf.summary.scalar('loss_low', self.loss_low)
        tf.summary.scalar('loss_decoder', self.loss_decoder)
        self.summary_op = tf.summary.merge_all()

    def train_policy_high(self, sess, batch, step, summarize=False, writer=None):
        """Train high-level field selection policy."""
        # Extract batch data (adapt to fuzzing batch format)
        states, obs_batch, field_actions, rewards, states_next, obs_next, dones = batch
        
        # Convert field actions to one-hot
        field_actions_1hot = np.zeros((len(field_actions), self.n_fields))
        for i, action in enumerate(field_actions):
            if isinstance(action, (list, np.ndarray)):
                field_actions_1hot[i] = action
            else:
                field_actions_1hot[i, action] = 1.0

        # Training feed
        feed = {
            self.state: states,
            self.obs: obs_batch,
            self.field_actions_1hot: field_actions_1hot,
            self.reward_high: rewards,
            self.state_next: states_next,
            self.obs_next: obs_next,
            self.done_high: dones
        }

        if summarize and writer:
            _, loss, summary = sess.run([self.train_op_high, self.loss_high, self.summary_op], feed_dict=feed)
            writer.add_summary(summary, step)
        else:
            _, loss = sess.run([self.train_op_high, self.loss_high], feed_dict=feed)

        return loss

    def train_policy_low(self, sess, batch, step, summarize=False, writer=None):
        """Train low-level mutation policy."""
        if self.low_level_alg == 'iql':
            # Extract batch data
            obs_batch, mutation_actions, rewards, obs_next, field_selections, dones = batch
            
            # Convert actions to one-hot
            actions_1hot = np.zeros((len(mutation_actions), self.l_mutation_actions))
            for i, action in enumerate(mutation_actions):
                actions_1hot[i, action] = 1.0

            # Training feed
            feed = {
                self.obs: obs_batch,
                self.field_selection: field_selections,
                self.mutation_actions_1hot: actions_1hot,
                self.reward_low: rewards,
                self.obs_next_low: obs_next,
                self.field_selection_next: field_selections,  # Assume same for next
                self.done_low: dones
            }

            _, loss = sess.run([self.train_op_low, self.loss_low], feed_dict=feed)
            return loss

    def train_decoder(self, sess, dataset, step, summarize=False, writer=None):
        """Train field selection decoder."""
        # Extract trajectories and field selections from dataset
        trajs = []
        field_selections = []
        
        for sample in dataset:
            traj, field_selection = sample
            trajs.append(traj)
            field_selections.append(field_selection)
            
        trajs = np.array(trajs)
        field_selections = np.array(field_selections)
        
        # Training feed
        feed = {
            self.mutation_traj: trajs,
            self.target_field_selections: field_selections
        }
        
        _, loss, probs = sess.run([self.train_op_decoder, self.loss_decoder, self.decoder_probs], feed_dict=feed)
        
        # Calculate expected probability for curriculum
        expected_prob = np.mean(np.max(probs, axis=1))
        
        return expected_prob