import gym
import tensorflow as tf
import numpy as np
import os
import sys
from collections import namedtuple
import random
import itertools
import platformer_environment
import pygame


##### Double DQN class used to predict and train the policy network #####
class DQN():
    """
    Deep Q Network which estimates the Q-value.

    Used for both the principle and target network.
    """

    def __init__(self, scope="_", summaries_path=None):
        self.scope = scope

        self.summary_writer = None # Write Tensorboard summaries to disc

        # Allows creation of multiple variables w/o references
        with tf.variable_scope(scope):
            # Create CNN
            self._create_model()
            if summaries_path:
                summary_path = os.path.join(summaries_path, "summaries_{}".format(scope))
                if not os.path.exists(summary_path):
                    os.makedirs(summary_path)
                self.summary_writer = tf.summary.FileWriter(summary_path)


    ##### Define personal functions used for building the CNN #####
    def init_weights(self, shape):
        """
        Initialize the weights

        Args:
            shape: List describing shape of weight tensor
                   [patch h, patch w, num. input channels/input features, num. features]

        Returns:
            A variable of given shape intiialized with xavier initialization
        """
        initializer = tf.contrib.layers.xavier_initializer()

        return tf.Variable(initializer(shape))

    def init_bias(self, shape):
        """
        Initialize the bias

        Args:
            shape: List representing size of current layer

        Returns:
            A variable of same length and constants of 0.01
        """
        init_bias_vals = tf.constant(0.01, shape=shape)

        return tf.Variable(init_bias_vals)

    def conv2d(self, x, W, strides=[1,1,1,1], padding='SAME'):
        """
        Create convolution for convolutional layer

        Args:
            x: Input data of shape [batch, H, W, Channels]
            W: The kernel of shape [filter H, filter W, num. input channels/input features, num. output features]
            strides: List of num. steps taken in each direction
            padding: whether to use padding or not
        Returns:
            A convolution on x using the kernel W
        """

        return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    def batch_norm(self, x, phase):
        """
        Apply batch normalization

        Args:
            x: Input data
            phase: Placeholder boolean --> True when training

        Returns:
            The batch norm layer applied to its input
        """
        return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)

    def max_pool(self, x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
        """
        Apply max pooling

        Args:
            x: Input data of shape [batch, h, w, channels]
            ksize: List with shape of the pool window
            strides: List as steps taken in each direction
            padding: Whether to use padding or not

        Returns:
            The input with max pooling applied
        """
        # x: input data --> [batch, h, w, channels]
        # ksize: shape of pool window
        # strides: steps taken in each direction

        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def convolutional_layer(self, input_x, shape):
        """
        Creates a convolutional layer

        Args:
            shape: list of shape [patch h, patch w, num. input channels/input features, num. output features]

        """
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])

        return tf.nn.relu(self.conv2d(input_x, W) + b)

    def normal_full_layer(self, input_layer, size):
        """
        Create a dense layer

        size --> [num. neurons]
        """
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])

        return tf.matmul(input_layer, W) + b


    def _create_model(self):
        """
        Create the CNN model:

        input x -> convolutional layer + batch norm + max pooling ->
        convolutional layer + batch norm + max pooling -> dense layer -> output layer
        """

        # PLACEHOLDERS
        # Input = 4 frames : 120x152x4
        self.X = tf.placeholder(tf.uint8, shape=[None, 120, 152, 4], name='X')
        # y from target network
        self.y_opt = tf.placeholder(tf.float32, shape=[None], name='y')
        # Phase placeholder for batch_norm. True only when training
        self.phase = tf.placeholder(tf.bool)
        # Action selected. This is just the action, use associated q(s, a) to calculate loss
        self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')

        # LAYERS
        X = tf.to_float(self.X) / 255.0

        convo_1 = self.convolutional_layer(X, shape=[3,3,4,32])
        batch_norm_1 = self.batch_norm(convo_1, phase=self.phase)
        max_pool_1 = self.max_pool(batch_norm_1)

        convo_2 = self.convolutional_layer(max_pool_1, shape=[3,3,32,64])
        batch_norm_2 = self.batch_norm (convo_2, phase=self.phase)
        max_pool_2 = self.max_pool(batch_norm_2)

        convo_3 = self.convolutional_layer(max_pool_2, shape=[3,3,64,128])
        batch_norm_3 = self.batch_norm (convo_3, phase=self.phase)
        max_pool_3 = self.max_pool(batch_norm_3)

        # h = 120 / 2*2*2 and w = 152 / 2*2*2
        flat = tf.reshape(max_pool_3, [-1, 15 * 19 * 128])
        full_layer_1 = tf.nn.relu(self.normal_full_layer(flat, size=1024))

        # REQUIRE THE CLASS ITSELF TO SEND AN ACTION LIST LATER
        self.y_pred = self.normal_full_layer(full_layer_1, size=len(action_list))

        # Chooses index in y_pred corresponding to action taken
        self.chosen_state = tf.gather_nd(self.y_pred, tf.stack([tf.reshape(tf.range(tf.shape(self.y_pred)[0]), (-1,)), self.actions], axis=-1))

        # LOSS FUNCTION
        # self.y_opt = q*(s, a), sent calculated
        self.loss = tf.reduce_mean(tf.square(self.y_opt - self.chosen_state))

        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss)
        ])

    def predict(self, sess, s):
        """
        Uses the given session and state to predict q(s, a)

        Args:
            sess: A tensorflow session
            s: State of shape [batch size, num. frames, height, width]

        Returns:
            Tensor with the shape [batch size, num. actions] containing estimated q-values
        """

        return sess.run(self.y_pred, feed_dict={self.X: s, self.phase: False})

    # Add phase = True or false to predict and update

    def update(self, sess, s, a, y):
        """
        Trains the policy network using y from the following optimal q(s', a')
        on q(s, a)

        Args:
            sess: A tensorflow Session
            s: State of shape [batch size, num. frames, height, width]
            a: Action taken of size [batch size]
            y: List of optimal q(s', a'), of shape [batch size]

        returns:
            The loss of the batch between y and q(s, a)
        """

        summaries, global_step1, _, loss = sess.run(
            [self.summaries, tf.train.get_or_create_global_step(), self.train_op, self.loss],
            feed_dict={self.X: s, self.y_opt: y, self.actions: a, self.phase: True}
        )

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step1)

        return loss

class StateProcessor():
    """
    Processes a raw cartpole image, resizes it, and converts it to grayscale.
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[800, 1200, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            # crop height 300 to 300 + 400 pixels and width 0 to 0 + 1200 pixels
            self.output = tf.image.crop_to_bounding_box(self.output, 0, 0, 800, 1020)
            self.output = tf.image.resize_images(self.output, [120, 152], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [800, 1200, 3] cartpole RGB State

        Returns:
            A processed [120, 152] state representing grayscale values.
        """

        return sess.run(self.output, feed_dict={self.input_state: state})

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another

    Args:
        sess: Tensorflow session instance
        estimator1: Estimator to copy the paramaters from
        estimator2: Estimator to copy the parameters to
    """

    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


##### INITIALIZE VARIABLES #####
tf.reset_default_graph()

# Store display in dummy for google cloud
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize pygame and create the environment
pygame.init()
env = platformer_environment.driver()

# Possible actions to take
action_dict = {0:'right', 1:'left', 2:'jump', 3:'stop'}
action_list = np.arange(4)

# Directory to save checkpoints / graphs
experiment_dir = os.path.abspath("./experiments/platform_game")

# Global step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
policy_network = DQN(scope='q', summaries_path=experiment_dir)
target_estimator = DQN(scope='target_q')

# Class which will preprocess our state image (convert to greyscale, crop and reduce dimensions)
state_processor = StateProcessor()

# Rest of the variables
batch_size = 32
num_episodes = 10000
update_target_estimator_every = 7500
discount_factor = 0.99
replay_memory_size = 100000
replay_memory_init_size = 10000
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_decay = 50000
reset_episode_every = 1000

##### Main Deep Q Learning Implementation #####
# Run session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Create replay memory to store experience replays
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    # Keep track of some statistics
    #stats = plotting.EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # Access a previous model or create a dir to save them
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    # Create a saver
    saver = tf.train.Saver()

    # Load checkpoint if one exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver = tf.train.import_meta_graph('{}.meta'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Get time step to apply accurate decay on continuing models
    total_t = sess.run(tf.train.get_or_create_global_step()) ### I HAVE TO INCLUDE GLOBAL STEP IN THE MINIMIZER LOSS FUNCTION
    print("Starting at step: {}".format(total_t))

    # Decrease epsilon over num. steps (prob. that we choose a random action)
    epsilons = np.linspace(epsilon_max, epsilon_min, epsilon_decay)

    # Policy
    policy = make_epsilon_greedy_policy(
        policy_network,
        len(action_list)
    )

    ### Filling the replay memory ###
    print("Populating replay memory...")
    state = env.set_rand_level() # Reset the state
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        if i % 1000 == 0:
            print("On step {}".format(i))
        action_probs = policy(sess, state, epsilons[total_t])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = env.initiate_action(action_dict[action_list[action]])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done or (i % reset_episode_every == reset_episode_every - 1):
            state = env.set_rand_level()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state


    ### Double DQN Algorithm ###
    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(sess, checkpoint_path, global_step=global_step)

        # Reset the environment
        state = env.set_rand_level()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            policy_network.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, policy_network, target_estimator)
                print("\nCopied model parameters to target network")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.initiate_action(action_dict[action_list[action]])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = policy_network.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]


            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = policy_network.update(sess, states_batch, action_batch, targets_batch)

            if done or t == reset_episode_every:
                break


            state = next_state
            total_t += 1

        print("Total reward this episode: {}".format(episode_rewards[i_episode]))

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        policy_network.summary_writer.add_summary(episode_summary, total_t)
        policy_network.summary_writer.flush()

        #yield total_t, plotting.EpisodeStats(
        #    episode_lengths=stats.episode_lengths[:i_episode+1],
        #    episode_rewards=stats.episode_rewards[:i_episode+1])

    # env.monitor.close() -- deprecated
