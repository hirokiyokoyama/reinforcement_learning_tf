import tensorflow as tf
slim = tf.contrib.slim

# input shape: [None,210,160,3]
def atari_cnn(x, num_classes=6, is_training=True):
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
        # [224,160]
        net = tf.pad(x, tf.constant([[0,0], [7,7], [0,0], [0,0]]), 'CONSTANT')
        net = slim.conv2d(x, 64, [7,7])
        net = slim.conv2d(x, 32, [1,1])
        net = slim.conv2d(x, 64, [3,3])

        # [112,80]
        net = slim.conv2d(net, 128, [3,3], stride=2)
        net = slim.conv2d(net, 64, [1,1])
        net = slim.conv2d(net, 128, [3,3])
        
        # [56,40]
        net = slim.conv2d(net, 256, [3,3], stride=2)
        net = slim.conv2d(net, 128, [1,1])
        net = slim.conv2d(net, 256, [3,3])
        
        # [28,20]
        net = slim.conv2d(net, 512, [3,3], stride=2)
        net = slim.conv2d(net, 256, [1,1])
        net = slim.conv2d(net, 512, [3,3])
        
        # [14,10]
        net = slim.conv2d(net, 1024, [3,3], stride=2)
        net = slim.conv2d(net, 512, [1,1])
        net = slim.conv2d(net, 1024, [3,3])

        # [7,5]
        net = slim.conv2d(net, 2048, [3,3], stride=2)
        net = slim.conv2d(net, 1024, [1,1])
        # [4,3]
        net = slim.conv2d(net, 2048, [4,3], padding='VALID')
        net = slim.conv2d(net, 1024, [1,1])
        # [1,1]
        net = slim.conv2d(net, num_classes, [4,3], padding='VALID')
    net = tf.reshape(net, [-1,num_classes])
    return net
