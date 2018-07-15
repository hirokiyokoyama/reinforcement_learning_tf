import tensorflow as tf

class ExperienceHistory:
    def __init__(self, state_shape,
                 history_size = 100000,
                 variable_collections=['history']):
        variables = []
        self._state_shape = state_shape
        self._history_size = history_size
        self._state_history = tf.Variable(tf.zeros([history_size]+list(state_shape), dtype=tf.float32),
                                          trainable = False, collections=variable_collections)
        variables.append(self._state_history)
        self._action_history = tf.Variable(-tf.ones([history_size], dtype=tf.int32),
                                           trainable = False, collections=variable_collections)
        variables.append(self._action_history)
        self._reward_history = tf.Variable(tf.zeros([history_size], dtype=tf.float32),
                                           trainable = False, collections=variable_collections)
        variables.append(self._reward_history)
        
        self._history_mask = tf.greater_equal(self._action_history, 0)
        self._history_pointer = tf.Variable(tf.constant(0, dtype=tf.int32),
                                            trainable = False,  collections=variable_collections)
        variables.append(self._history_pointer)
        
        self.saver = tf.train.Saver(var_list=variables)
        self.initializer = tf.group(map(lambda v: v.initializer, variables))
        self.history_count = tf.reduce_sum(tf.cast(self._history_mask, tf.int32))

    def append(self, action, reward, next_state):
        hist_ops = []
        hist_ops.append(self._action_history[self._history_pointer].assign(action))
        hist_ops.append(self._reward_history[self._history_pointer].assign(reward))
        hist_ops.append(self._state_history[self._history_pointer].assign(next_state))
        with tf.control_dependencies(hist_ops):
            return self._history_pointer.assign((self._history_pointer + 1) % self._history_size)

    def append_init(self, state):
        return self.append(-1, 0., state)

    def sample(self, size):
        inds = tf.random_shuffle(tf.where(self._history_mask)[:,0])[:size]
        states = tf.gather(self._state_history, (inds-1) % self._history_size)
        actions = tf.gather(self._action_history, inds)
        rewards = tf.gather(self._reward_history, inds)
        next_states = tf.gather(self._state_history, inds)
        return states, actions, rewards, next_states

class GymExecutor:
    def __init__(self, env, action_fn, history,
                 summary_writer = None,
                 variable_collections = ['history'],
                 image_size = (84,84)):
        self._history = history
        self._env = env
        self._new_episode = True
        self._next_action = None
        image_size = list(image_size)
        
        self._state_ph = tf.placeholder(tf.float32, shape=[None,None,3])
        state = tf.image.resize_images(self._state_ph, image_size)
        self._action_ph = tf.placeholder(tf.int32, shape=[])
        self._reward_ph = tf.placeholder(tf.float32, shape=[])
        probs, self._action = action_fn(state)
        self._hist_op = self._history.append(self._action_ph, self._reward_ph, state)

        if summary_writer is None:
            summary_writer = tf.summary.FileWriter('/tmp/gym')
        self._summary_writer = summary_writer

        variables = []
        self.total_step = tf.Variable(0, trainable=False, collections=variable_collections)
        variables.append(self.total_step)
        self.episode_step = tf.Variable(0, trainable=False, collections=variable_collections)
        variables.append(self.episode_step)
        self.episode_count = tf.Variable(0, trainable=False, collections=variable_collections)
        variables.append(self.episode_count)
        self.episode_reward = tf.Variable(0., trainable=False, collections=variable_collections)
        variables.append(self.episode_reward)
        self._init_episode_op = tf.group([self.episode_reward.assign(0.),
                                          self.episode_step.assign(0),
                                          self.episode_count.assign_add(1)])
        self._step_op = tf.group([self.total_step.assign_add(1),
                                  self.episode_step.assign_add(1),
                                  self.episode_reward.assign_add(self._reward_ph)])
        self._step_summary = tf.summary.merge([tf.summary.scalar('max_prob', tf.reduce_max(probs)),
                                               tf.summary.scalar('min_prob', tf.reduce_min(probs))])
        self._episode_summary = tf.summary.merge([tf.summary.scalar('episode_reward', self.episode_reward),
                                                  tf.summary.scalar('episode_steps', self.episode_step)])

        self._init_op = tf.group([self._history.initializer]+map(lambda v: v.initializer, variables))

    def initialize(self, sess):
        sess.run(self._init_op)
        self._new_episode = True
        
    def step(self, sess):
        fetch_list = [self._hist_op, self._action, self._step_summary, self.total_step]
        if self._new_episode:
            action = -1
            reward = 0.
            state = self._env.reset()
            done = False
            fetch_list.append(self._init_episode_op)
        else:
            action = self._next_action
            state, reward, done, _ = self._env.step(action)
            fetch_list.append(self._step_op)
        state = state/255.
        reward = reward/100.
        feed_dict = {self._state_ph: state,
                     self._action_ph: action,
                     self._reward_ph: reward}
        _, self._next_action, step_summary, total_step, _ = sess.run(fetch_list, feed_dict)
        self._summary_writer.add_summary(step_summary, total_step)
        if done:
            episode_summary, episode_count = sess.run([self._episode_summary, self.episode_count])
            self._summary_writer.add_summary(episode_summary, episode_count)
        self._new_episode = done
        return state, action, reward, done
