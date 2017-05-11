import tensorflow as tf

from utils import conv2d, max_pool, deconv2d, iou_loss


class fcn(object):

    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 512, 512, 1], name = 'input_data')
        self.label = tf.placeholder(tf.float32, [None, 512, 512], name='y')
        self.learning_rate = 0.0005
        self.set_up()

    def set_up(self):

        with tf.variable_scope('conv1'):
            network = conv2d(self.input, [7, 7], 32, scope='conv1_1')
            network = conv2d(network, [3, 3], 32, scope='conv1_2')
            network = max_pool(network, 'pool1')    # downsample

        with tf.variable_scope('conv1'):
            network = conv2d(network, [3, 3], 64, scope='conv1_1')
            network = conv2d(network, [3, 3], 64, scope='conv1_2')
            network = max_pool(network, 'pool2')    # downsample

        with tf.variable_scope('conv1'):
            network = conv2d(network, [3, 3], 128, scope='conv1_1')
            network = conv2d(network, [3, 3], 128, scope='conv1_2')

        with tf.variable_scope('deconv1'):
            network = deconv2d(network, [3, 3], 64, scope='deconv1_1')  # upsample
            network = deconv2d(network, [3, 3], 64, stride=1, scope='deconv1_1')

        with tf.variable_scope('deconv2'):
            network = deconv2d(network, [3, 3], 32, scope='deconv1_1')  # upsample
            network = deconv2d(network, [3, 3], 32, stride=1, scope='deconv1_1')

        with tf.variable_scope('out_class'):
            logits = conv2d(network, [3, 3], 2, bn=False, relu=False, scope='logits')

        self.pred_prob = tf.nn.softmax(logits, name='predictions')[:,:,:,1]
        self.pred = tf.argmax(logits, 3)
        self.loss = iou_loss(self.pred_prob, self.label)
        self.train_score = iou_loss(tf.cast(self.pred, tf.float32), self.label)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-4).minimize(self.loss)




if __name__ == '__main__':
    Fcn = fcn()
    vars = tf.global_variables()
    for i in vars:
        print i.name, '      ', i.get_shape()



