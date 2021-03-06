import tensorflow as tf
import numpy as np

class QLearning:
    def __init__(self, q_fn, prob_fn, gamma=0.95):
        self.gamma = gamma
        self._q_fn = q_fn
        self._prob_fn = prob_fn

    def train(self, states, actions, rewards, next_states, terminal):
        batch_size = tf.shape(actions)[0]
        with tf.variable_scope('target_q'):
            target_q = self._q_fn(next_states, is_training=True)
        _target_q = tf.reduce_max(target_q, 1)
        _target_q = tf.where(terminal, tf.zeros_like(_target_q), _target_q)
        _target_q = (1-self.gamma) * rewards + self.gamma * _target_q
        _target_q = tf.stop_gradient(_target_q)
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            q = self._q_fn(states, is_training=True)
        _q = tf.gather_nd(q, tf.stack([tf.range(batch_size), actions], 1))
        loss = tf.losses.huber_loss(_target_q, _q)

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q/')
        target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q/')
        q_dict = {v.op.name:v for v in q_vars}
        ops = [v.assign(q_dict[v.op.name.replace('target_q/', 'q/')]) for v in target_q_vars]
        copy_op = tf.group(ops)
        
        return {'q': q,                # [N,num_actions]
                'target_q': target_q,  # [N]
                'loss': loss,          # [N]
                'copy_op': copy_op,
                'q_variables': q_vars,
                'target_q_variables': target_q_vars}

    def action(self, states, is_training=False):
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            q = self._q_fn(states, is_training=is_training)
        probs = self._prob_fn(q)
        return {'q': q,
                'action_probabilities': probs,
                'actions': tf.distributions.Categorical(probs=probs).sample()}

def boltzmann(q, temperature=1.):
    return tf.nn.softmax(q/temperature)

def epsilon_greedy(q, eps=0.1):
    shape = tf.shape(q)
    n = shape[0]
    m = shape[1]
    inds = tf.argmax(q, 1, output_type=tf.int32)
    subs = tf.stack([tf.range(n), inds], 1)
    return tf.scatter_nd(subs, tf.ones([n])*(1.-eps), shape) + eps/tf.cast(m, tf.float32)

if __name__=='__main__':
    import gym
    import os
    import sys
    from nets import cnn
    from replay import ExperienceHistory, GymExecutor

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    LOG_DIR = os.path.join(DATA_DIR, 'log')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    global_step = tf.Variable(0, trainable=False)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        TRAIN = False
        print 'Test mode.'
    else:
        TRAIN = True
        print 'Training mode.'
    ATARI = True
    ENV_NAME = 'SpaceInvaders-v0' #'Breakout-v0', 'CartPole-v1'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TRAIN_INTERVAL = 8
    COPY_INTERVAL = 2500
    SAVE_INTERVAL = 1000
    IMAGE_SIZE = [84,84]
    FRAME_SKIP = 4
    GAMMA = 0.95
    if TRAIN:
        PROB_FN = lambda q: epsilon_greedy(q, tf.train.exponential_decay(0.9, global_step, 1000000, 0.01)+0.1)
    else:
        PROB_FN = PROB_FN = lambda q: epsilon_greedy(q, 0.)
    HISTORY_SIZE = 20000
    MIN_HISTORY_SIZE = 10000
    BATCH_NORM_DECAY = 0.999

    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        def show():
            env.render()
    else:
        show = lambda: None

    env = gym.make(ENV_NAME)

    if ATARI:
        def q_fn(x, is_training=True):
            return cnn(x, num_classes=env.action_space.n,
                       is_training = is_training,
                       decay = BATCH_NORM_DECAY)
    else:
        def q_fn(x, is_training=True):
            dim = np.prod(env.observation_space.shape)
            w = tf.Variable(tf.truncated_normal([FRAME_SKIP*dim, env.action_space.n]))
            b = tf.Variable(tf.zeros([1, env.action_space.n]))
            return tf.matmul(x, w) + b
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    dqn = QLearning(q_fn, PROB_FN, gamma=GAMMA)

    history = None
    target_q_variables = []
    if TRAIN:
        if ATARI:
            history = ExperienceHistory(IMAGE_SIZE+[FRAME_SKIP],
                                        history_size=HISTORY_SIZE)
        else:
            dim = np.prod(env.observation_space.shape)
            history = ExperienceHistory([FRAME_SKIP*dim],
                                        history_size=HISTORY_SIZE)
            
        out = dqn.train(*history.sample(BATCH_SIZE))
        loss = tf.reduce_mean(out['loss'])
        q = out['q']
        target_q = out['target_q']
        copy_op = out['copy_op']
        target_q_variables = out['target_q_variables']
        # this must be before calling action_fn: to avoid updating moving averages of executor
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)

        train_summary = tf.summary.merge([tf.summary.histogram('Q', q),
                                          tf.summary.histogram('target_Q', target_q),
                                          tf.summary.scalar('loss', loss)])
        
    def action_fn(state):
        out = dqn.action(tf.expand_dims(state,0), is_training=True)
        return out['action_probabilities'][0], out['actions'][0]
    if ATARI:
        def preprocess_obs(x):
            x = tf.cast(x, tf.float32)/255.
            x.set_shape((FRAME_SKIP,)+env.observation_space.shape)
            x = tf.image.resize_images(x, IMAGE_SIZE)
            x = tf.reduce_mean(x, 3)
            return tf.transpose(x, [1,2,0])
        preprocess_reward = lambda x: x/100.
    else:
        preprocess_obs = lambda x: tf.reshape(x, [-1])
        preprocess_reward = lambda x: x
    executor = GymExecutor(env, action_fn, history,
                           summary_writer=summary_writer,
                           frame_skip = FRAME_SKIP,
                           preprocess_observation_fn=preprocess_obs,
                           preprocess_reward_fn=preprocess_reward)

    sess = tf.Session()
    vars_to_save = list(set(tf.global_variables())-set(target_q_variables))
    saver = tf.train.Saver(vars_to_save)
    latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR, latest_filename=ENV_NAME)
    if latest_ckpt is not None:
        print 'Restored from %s.' % latest_ckpt
        saver.restore(sess, latest_ckpt)
    else:
        print 'Starting with a new model.'
        sess.run(tf.global_variables_initializer())
    if TRAIN:
        sess.run(copy_op)
    executor.initialize(sess)

    count = 0
    while True:
        state, action, reward, done = executor.step(sess)
        show()
        if done:
            print 'Episode done.'

        if TRAIN and count % TRAIN_INTERVAL == 0:
            if sess.run(history.history_count) >= MIN_HISTORY_SIZE:
                _, loss_val, summary, step = sess.run([train_op, loss, train_summary, global_step])
                summary_writer.add_summary(summary, step)
                print 'loss = ', loss_val
                if step % COPY_INTERVAL == 0:
                    sess.run(copy_op)
                if step % SAVE_INTERVAL == 0:
                    saver.save(sess, os.path.join(MODEL_DIR, ENV_NAME),
                               global_step=global_step,
                               latest_filename=ENV_NAME)
        count += 1
