import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, q_fn,
                 gamma=0.95,
                 temperature=1.):
        self.gamma = gamma
        self.temperature = temperature

    def train(self, states, actions, rewards, next_states):
        batch_size = tf.shape(actions)[0]
        with tf.variable_scope('target_q'):
            target_q = q_fn(next_states, is_training=True)
        _target_q = tf.reduce_max(target_q, 1)
        _target_q = tf.where(tf.less(actions, 0), tf.zeros_like(_target_q), _target_q)
        _target_q = (1-self.gamma) * rewards + self.gamma * _target_q
        _target_q = tf.stop_gradient(_target_q)
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            q = q_fn(states, is_training=True)
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

    def action(self, state):
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            q = q_fn(tf.expand_dims(state, 0), is_training=False)
        probs = tf.nn.softmax(q/self.temperature)[0]
        return {'q': q,
                'action_probabilities': probs,
                'action': tf.distributions.Categorical(probs=probs).sample()}

if __name__=='__main__':
    import gym
    import os
    from nets import cnn
    from replay import ExperienceHistory, GymExecutor

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    LOG_DIR = os.path.join(DATA_DIR, 'log')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    BATCH_SIZE = 32
    LEARNING_RATE = 0.00003
    TRAIN_INTERVAL = 8
    COPY_INTERVAL = 40000
    IMAGE_SIZE = [84,84]
    GAMMA = 0.95
    TEMPERATURE = 1.
    HISTORY_SIZE = 20000
    BATCH_NORM_DECAY = 0.999

    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        import cv2
        def show(x):
            cv2.imshow('State', x[:,:,::-1])
            cv2.waitKey(1)
    else:
        show = lambda x: None

    env = gym.make('SpaceInvaders-v0')
    def q_fn(x, is_training=True):
        return cnn(x, num_classes=env.action_space.n,
                   is_training = is_training,
                   decay = BATCH_NORM_DECAY)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    dqn = DQN(q_fn, gamma=GAMMA, temperature=TEMPERATURE)
    def action_fn(state):
        out = dqn.action(state)
        return out['action_probabilities'], out['action']
    history = ExperienceHistory(IMAGE_SIZE+[3], history_size=HISTORY_SIZE)
    executor = GymExecutor(env, action_fn, history,
                           summary_writer=summary_writer,
                           image_size=IMAGE_SIZE)

    global_step = tf.Variable(0, trainable=False)
    out = dqn.train(*history.sample(BATCH_SIZE))
    loss = tf.reduce_mean(out['loss'])
    q = out['q']
    target_q = out['target_q']
    copy_op = out['copy_op']
    vars_to_save = list(set(tf.global_variables())-set(out['target_q_variables']))
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=global_step)

    train_summary = tf.summary.merge([tf.summary.histogram('Q', q),
                                      tf.summary.histogram('target_Q', target_q),
                                      tf.summary.scalar('loss', loss)])

    sess = tf.Session()
    saver = tf.train.Saver(vars_to_save)
    latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())
    executor.initialize(sess)

    count = 0
    while True:
        state, action, reward, done = executor.step(sess)
        show(state)
        if done:
            print 'Episode done.'
        if count % COPY_INTERVAL == 0:
            sess.run(copy_op)
        if count % TRAIN_INTERVAL == 0:
            if sess.run(history.history_count) >= 10000:
                _, loss_val, summary, step = sess.run([train_op, loss, train_summary, global_step])
                summary_writer.add_summary(summary, step)
                print 'loss = ', loss_val
            
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=global_step)
        count += 1
