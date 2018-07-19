import tensorflow as tf
import numpy as np

def stratified_sample(probs, n):
    N = tf.shape(probs)[0:1]
    c = tf.cumsum(probs)
    c = c/c[-1]
    borders = tf.linspace(0.,1.,n+1)
    right = borders[1:]

    c = tf.expand_dims(c, 0)
    right = tf.expand_dims(right, 1)
    greater_mask = tf.cast(tf.greater(c, right), tf.int32)
    _cum_num = tf.reduce_sum(greater_mask, 1)
    cum_num = tf.concat([N,_cum_num[:-1]],0)
    num = cum_num - _cum_num
    unif = tf.contrib.distributions.Uniform(low=0., high=tf.cast(num, tf.float32))
    local_inds = tf.cast(unif.sample(), tf.int32)
    begin = N - cum_num
    return local_inds + begin

class PrioritizedHistory:
    def __init__(self, name_to_shape_dtype,
                 capacity = 100000,
                 variable_collections=['history']):
        variables = []
        self._capacity = capacity

        self._histories = {}
        for name, (shape, dtype) in name_to_shape_dtype.iteritems():
            self._histories[name] = tf.Variable(tf.zeros([capacity]+list(shape), dtype=dtype),
                                                trainable = False, collections=variable_collections)
            variables.append(self._histories[name])
        
        self._weights = tf.Variable(tf.zeros([capacity], dtype=tf.float32),
                                    trainable = False, collections=variable_collections)
        variables.append(self._weights)
        
        self._size = tf.Variable(tf.constant(0, dtype=tf.int32),
                                          trainable = False,  collections=variable_collections)
        variables.append(self._size)
        
        self.saver = tf.train.Saver(var_list=variables)
        self.initializer = tf.group(map(lambda v: v.initializer, variables))

    def append(self, name_to_value, weight):
        weight = tf.convert_to_tensor(weight)
        name_to_value = {name: tf.convert_to_tensor(value) for name, value in name_to_value.iteritems()}
        inds = tf.where(tf.less(self._weights, weight))
        accepted = tf.greater(tf.shape(inds)[0], 0)
        def insert():
            ind = inds[0,0]
            ops = [self._weights[(ind+1):].assign(self._weights[ind:-1])]
            for name, value in name_to_value.iteritems():
                ops.append(self._histories[name][(ind+1):].assign(self._histories[name][ind:-1]))
            with tf.control_dependencies(ops):
                ops = [self._weights[ind].assign(weight)]
                for name, value in name_to_value.iteritems():
                    ops.append(self._histories[name][ind].assign(value))
                ops.append(self._size.assign(tf.reduce_min([self._size+1, self._capacity])))
                with tf.control_dependencies(ops):
                    return tf.cast(ind, tf.int32)
        return tf.cond(accepted, insert, lambda: -1)

    def update_weight(self, ind, weight):
        ind = tf.convert_to_tensor(ind)
        old_weight = self._weights[ind]
        weight = tf.convert_to_tensor(weight)
        def first_less():
            inds = tf.where(tf.less(self._weights, weight))
            return tf.cond(tf.greater(tf.shape(inds)[0], 0),
                           lambda: tf.cast(inds[0,0], tf.int32),
                           lambda: tf.constant(self._capacity-1, dtype=tf.int32))
        def last_greater():
            inds = tf.where(tf.greater(self._weights, weight))
            return tf.cond(tf.greater(tf.shape(inds)[0], 0),
                           lambda: tf.cast(inds[-1,0], tf.int32),
                           lambda: tf.constant(self._capacity-1, dtype=tf.int32))
        new_ind = tf.cond(tf.greater(weight, old_weight), first_less, last_greater)
        def up():
            ops = [self._weights[ind:new_ind].assign(self._weights[(ind+1):(new_ind+1)])]
            for hist in self._histories.itervalues():
                ops.append(hist[ind:new_ind].assign(hist[(ind+1):(new_ind+1)]))
            return tf.group(ops)
        def down():
            ops = [self._weights[(new_ind+1):(ind+1)].assign(self._weights[new_ind:ind])]
            for hist in self._histories.itervalues():
                ops.append(hist[(new_ind+1):(ind+1)].assign(hist[new_ind:ind]))
            return tf.group(ops)
        values = {name: hist[ind] for name, hist in self._histories.iteritems()}
        with tf.control_dependencies(values.values()):
            shift = tf.cond(tf.greater(new_ind, ind), up, down)
        with tf.control_dependencies([shift]):
            ops = [self._weights[new_ind].assign(weight)]
            for name, value in values.iteritems():
                ops.append(self._histories[name][new_ind].assign(value))
            with tf.control_dependencies(ops):
                return tf.identity(new_ind)
    
    def sample(self, size):
        inds = stratified_sample(self._weights[:self._size], size)
        return {name: tf.gather(hist, inds) for name, hist in self._histories.iteritems()}

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
        self._action_history = tf.Variable(tf.ones([history_size], dtype=tf.int32)*(-1000),
                                           trainable = False, collections=variable_collections)
        variables.append(self._action_history)
        self._reward_history = tf.Variable(tf.zeros([history_size], dtype=tf.float32),
                                           trainable = False, collections=variable_collections)
        variables.append(self._reward_history)
        
        self._history_mask = tf.greater(self._action_history, -1)
        self._history_pointer = tf.Variable(tf.constant(0, dtype=tf.int32),
                                            trainable = False,  collections=variable_collections)
        variables.append(self._history_pointer)
        
        self.saver = tf.train.Saver(var_list=variables)
        self.initializer = tf.group(map(lambda v: v.initializer, variables))
        self.history_count = tf.reduce_sum(tf.cast(self._history_mask, tf.int32))

    def append(self, state, action, reward):
        hist_ops = []
        hist_ops.append(self._state_history[self._history_pointer].assign(state))
        hist_ops.append(self._action_history[self._history_pointer].assign(action))
        hist_ops.append(self._reward_history[self._history_pointer].assign(reward))
        with tf.control_dependencies(hist_ops):
            return self._history_pointer.assign((self._history_pointer + 1) % self._history_size)

    def sample(self, size):
        inds = tf.random_shuffle(tf.where(self._history_mask)[:,0])[:size]
        states = tf.gather(self._state_history, inds)
        actions = tf.gather(self._action_history, inds)
        rewards = tf.gather(self._reward_history, inds)
        next_states = tf.gather(self._state_history, (inds+1) % self._history_size)
        terminal = tf.equal(tf.gather(self._action_history, (inds+1) % self._history_size), -1)
        return states, actions, rewards, next_states, terminal

class GymExecutor:
    def __init__(self, env, action_fn,
                 history = None,
                 summary_writer = None,
                 variable_collections = ['history'],
                 frame_skip = 4,
                 preprocess_observation_fn = lambda x: x,
                 preprocess_reward_fn = lambda x: x):
        self._history = history
        self._env = env
        self._state_val = None
        self._frame_skip = frame_skip
        
        self._state_ph = tf.placeholder(tf.float32)
        state = preprocess_observation_fn(self._state_ph)
        self._action_ph = tf.placeholder(tf.int32, shape=[])
        self._reward_ph = tf.placeholder(tf.float32, shape=[])
        reward = preprocess_reward_fn(self._reward_ph)
        probs, self._action = action_fn(state)
        if self._history is not None:
            self._hist_op = self._history.append(state, self._action_ph, reward)
        else:
            self._hist_op = tf.no_op()

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
                                  self.episode_reward.assign_add(reward)])
        self._step_summary = tf.summary.merge([tf.summary.scalar('max_prob', tf.reduce_max(probs)),
                                               tf.summary.scalar('min_prob', tf.reduce_min(probs))])
        self._episode_summary = tf.summary.merge([tf.summary.scalar('episode_reward', self.episode_reward),
                                                  tf.summary.scalar('episode_steps', self.episode_step)])

        ops = map(lambda v: v.initializer, variables)
        if self._history is not None:
            ops += [self._history.initializer]
        self._init_op = tf.group(ops)

    def initialize(self, sess):
        sess.run(self._init_op)
        self._state_val = None
        
    def step(self, sess):
        if self._state_val is None:
            self._state_val = np.repeat(np.expand_dims(self._env.reset(), 0), self._frame_skip, 0)
            init_or_step = self._init_episode_op
        else:
            init_or_step = self._step_op
        action = sess.run(self._action, {self._state_ph: self._state_val})
        next_states = [None]*self._frame_skip
        rewards = [0.]*self._frame_skip
        done = False
        for i in range(self._frame_skip):
            if not done:
                next_states[i], rewards[i], done, _ = self._env.step(action)
            else:
                next_states[i] = next_states[i-1]
        next_state = np.stack(next_states)
        reward = np.mean(rewards)
        fetch_list = [self._hist_op, self._step_summary, self.total_step, init_or_step]
        feed_dict = {self._state_ph: self._state_val,
                     self._action_ph: action,
                     self._reward_ph: reward}
        _, step_summary, total_step, _ = sess.run(fetch_list, feed_dict)
        self._state_val = next_state
        self._summary_writer.add_summary(step_summary, total_step)
        if done:
            fetch_list = [self._hist_op, self._episode_summary, self.episode_count]
            feed_dict = {self._state_ph: self._state_val,
                         self._action_ph: -1,
                         self._reward_ph: 0.}
            _, epi_summary, epi_cnt = sess.run(fetch_list, feed_dict)
            self._summary_writer.add_summary(epi_summary, epi_cnt)
            self._state_val = None
        return next_state, action, reward, done
