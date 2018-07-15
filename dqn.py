import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, q_fn,
                 gamma=0.995,
                 temperature=1.):
        self.gamma = gamma
        self.temperature = temperature

    def train(self, states, actions, rewards, next_states):
        batch_size = tf.shape(actions)[0]
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            target_q = q_fn(next_states, is_training=False)
        target_q = (1-self.gamma) * rewards + self.gamma * tf.reduce_max(target_q, 1)
        target_q = tf.stop_gradient(target_q)
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            sampled_q = q_fn(states, is_training=True)
        _q = tf.gather_nd(sampled_q, tf.stack([tf.range(batch_size), actions], 1))
        loss = tf.square(target_q - _q)
        return {'q': sampled_q,        # [N,num_actions]
                'target_q': target_q,  # [N]
                'loss': loss}          # [N]

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
    LEARNING_RATE = 0.00001
    TRAIN_INTERVAL = 8
    IMAGE_SIZE = [84,84]
    HISTORY_SIZE = 100000

    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        import cv2
        def show(x):
            cv2.imshow('State', x[:,:,::-1])
            cv2.waitKey(1)
    else:
        show = lambda x: None

    env = gym.make('SpaceInvaders-v0')
    def q_fn(x, is_training=True):
        return cnn(x, num_classes=env.action_space.n, is_training=is_training)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    dqn = DQN(q_fn)
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
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=global_step)

    train_summary = tf.summary.merge([tf.summary.histogram('Q', q),
                                      tf.summary.scalar('loss', loss)])

    sess = tf.Session()
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())
    executor.initialize(sess)

    count = 0
    while True:
        count += 1
        state, action, reward, done = executor.step(sess)
        show(state)
        if done:
            print 'Episode done.'
        if count % TRAIN_INTERVAL == 0:
            if sess.run(history.history_count) >= 10000:
                _, loss_val, summary, step = sess.run([train_op, loss, train_summary, global_step])
                summary_writer.add_summary(summary, step)
                print 'loss = ', loss_val
            
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=global_step)
