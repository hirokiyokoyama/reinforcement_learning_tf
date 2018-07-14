import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, q_fn, state_shape,
                 summary_writer = None,
                 history_size=5000,
                 batch_size=32,
                 learning_rate=0.0001,
                 gamma=0.995):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        if summary_writer is None:
            summary_writer = tf.summary.FileWriter('/tmp/DQN')
        self.summary_writer = summary_writer

        self.global_step = tf.Variable(0, trainable=False)
        self.episode_step = tf.Variable(0, trainable=False)
        self.trained_episodes = tf.Variable(0, trainable=False)
        self.episode_reward = tf.Variable(0., trainable=False)
        self.init_episode_op = tf.group([self.episode_reward.assign(0.),
                                         self.episode_step.assign(0),
                                         self.trained_episodes.assign_add(1)])
        self.episode_summary = tf.summary.merge([tf.summary.scalar('episode_reward', self.episode_reward),
                                                 tf.summary.scalar('episode_steps', self.episode_step)])
        
        self.state_history = tf.Variable(tf.zeros([history_size]+list(state_shape), dtype=tf.float32),
                                         trainable = False, collections=['history'])
        self.action_history = tf.Variable(-tf.ones([history_size], dtype=tf.int32),
                                          trainable = False, collections=['history'])
        self.reward_history = tf.Variable(tf.zeros([history_size], dtype=tf.float32),
                                          trainable = False, collections=['history'])
        self.history_mask = tf.greater_equal(self.action_history, 0)
        self.history_count = tf.reduce_sum(tf.cast(self.history_mask, tf.int32))
        self.history_pointer = tf.Variable(tf.constant(0, dtype=tf.int32),
                                           trainable = False,  collections=['history'])
        self.history_saver = tf.train.Saver(var_list=tf.get_collection('history'))
        self.history_initializer = tf.group(map(lambda v: v.initializer, tf.get_collection('history')))

        history_inds = tf.random_shuffle(tf.where(self.history_mask)[:,0])[:batch_size]
        sampled_states = tf.gather(self.state_history, (history_inds-1) % history_size)
        sampled_actions = tf.gather(self.action_history, history_inds)
        sampled_rewards = tf.gather(self.reward_history, history_inds)
        sampled_next_states = tf.gather(self.state_history, history_inds)
        with tf.variable_scope('q', reuse=False):
            target_q = q_fn(sampled_next_states, is_training=False)
        target_q = sampled_rewards + self.gamma * tf.reduce_max(target_q, 1)
        target_q = tf.stop_gradient(target_q)
        with tf.variable_scope('q', reuse=True):
            _q = q_fn(sampled_states, is_training=True)
        __q = tf.gather_nd(_q, tf.stack([tf.range(batch_size), sampled_actions], 1))
        self.loss = tf.reduce_mean(tf.square(target_q - __q))
        self.train_summary = tf.summary.merge([tf.summary.histogram('Q', _q),
                                               tf.summary.scalar('loss', self.loss)])
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(self.loss, global_step=self.global_step)

        self.state = tf.placeholder(tf.float32, shape=state_shape)
        self.action = tf.placeholder(tf.int32, shape=[])
        self.reward = tf.placeholder(tf.float32, shape=[]) 
        self.episode_op = tf.group([self.episode_step.assign_add(1),
                                    self.episode_reward.assign_add(self.reward)])
       
        hist_ops = []
        hist_ops.append(self.state_history[self.history_pointer].assign(self.state))
        hist_ops.append(self.action_history[self.history_pointer].assign(self.action))
        hist_ops.append(self.reward_history[self.history_pointer].assign(self.reward))
        with tf.control_dependencies(hist_ops):
            self.hist_op = self.history_pointer.assign((self.history_pointer + 1) % history_size)

        with tf.variable_scope('q', reuse=True):
            q = q_fn(tf.expand_dims(self.state, 0), is_training=False)
        probs = tf.nn.softmax(q)[0]
        self.step_summary = tf.summary.merge([tf.summary.scalar('max_prob', tf.reduce_max(probs)),
                                              tf.summary.scalar('min_prob', tf.reduce_min(probs))])
        self.next_action = tf.distributions.Categorical(probs=probs).sample()

    def init_episode(self, sess, state):
        return self.step(sess, state, -1, 0.)
    
    def step(self, sess, state, action=None, reward=None):
        if action is not None:
            if action == -1:
                s, t = sess.run([self.episode_summary, self.trained_episodes])
                self.summary_writer.add_summary(s, t)
                sess.run(self.init_episode_op)
            a, _, _, s = sess.run([self.next_action, self.hist_op, self.episode_op, self.step_summary],
                                  {self.state: state, self.action: action, self.reward: reward})
            self.summary_writer.add_summary(s)
            return a
        return sess.run(self.next_action, {self.state: state})

    def update(self, sess):
        count = sess.run(self.history_count)
        if count >= self.batch_size:
            loss,_,s,t = sess.run([self.loss, self.train_op, self.train_summary, self.global_step])
            self.summary_writer.add_summary(s, t)
            return loss
        else:
            return None

if __name__=='__main__':
    import gym
    import os
    from nets import atari_cnn

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    LOG_DIR = os.path.join(DATA_DIR, 'log')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        import cv2
        def show(x):
            cv2.imshow('State', x[:,:,::-1])
            cv2.waitKey(1)
    else:
        show = lambda x: None

    env = gym.make('SpaceInvaders-v0')
    def q_fn(x, is_training=True):
        return atari_cnn(x, num_classes=env.action_space.n, is_training=is_training)
    dqn = DQN(q_fn, env.observation_space.shape,
              summary_writer=tf.summary.FileWriter(LOG_DIR))

    sess = tf.Session()
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(dqn.history_initializer)

    count = 0
    while True:
        state = env.reset()/255.
        show(state)
        done = False
        action = dqn.init_episode(sess, state)
        while not done:
            state, reward, done, meta = env.step(action)
            state = state/255.
            reward = reward/100.
            action = dqn.step(sess, state, action, reward)
            if count % 8 == 0:
                loss = dqn.update(sess)
            show(state)
            count += 1
        saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=dqn.global_step)
