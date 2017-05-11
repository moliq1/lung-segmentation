import tensorflow as tf


def weight_variable(shape, std=0.01):
    # print 'create weights , shape:', shape
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    # print 'cread biases, shape:', shape
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

def batch_norm(inputs, epsilon = 1e-3):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='bn_scale')
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='bn_beta')
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)


def conv2d(inputs, kernel_size, b_shape, std=0.01, bn=True, relu=True, scope=None):
    # with tf.variable_scope(scope, 'Conv'):
    num_in = int(inputs.get_shape()[-1])
    weights_shape = [kernel_size[0], kernel_size[1], num_in, b_shape]
    # print weights_shape
    w = weight_variable(weights_shape, std)
    b = bias_variable([b_shape])
    conv = tf.nn.bias_add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1],
                                       padding='SAME'), b)
    if bn:
        conv = batch_norm(conv)
    if relu:
        conv = tf.nn.relu(conv)
    return conv


def iou_loss(y_pred, y, smooth=1.0):

    intersection = tf.reduce_sum(y_pred * y)
    loss = (2. * intersection + smooth) / (tf.reduce_sum(y) + tf.reduce_sum(y_pred) + smooth)
    return -loss


def deconv2d(inputs, kernel_size, out_channel, stride=2 , bn = True, scope=None):

    num_in = int(inputs.get_shape()[-1])
    in_shape = tf.shape(inputs)
    output_shape = tf.pack([in_shape[0], in_shape[1]*stride, in_shape[2]*stride, out_channel])
    w = weight_variable([kernel_size[0], kernel_size[1], out_channel, num_in])
    b = bias_variable([out_channel])
    conv = tf.nn.bias_add(tf.nn.conv2d_transpose(inputs, w, output_shape, strides=[1, stride, stride, 1],
                                                 padding='SAME'), b)
    if bn:
        conv = batch_norm(conv)

    return tf.nn.relu(conv)



def max_pool(inputs,  scope=None, kernel_size=[2,2], stride=2):
    with tf.variable_scope(scope, 'MaxPool'):
        return tf.nn.max_pool(inputs, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(inputs, kernel_size=[2,2], stride=1, scope=None):
    with tf.variable_scope(scope, 'AvgPool'):
        return tf.nn.avg_pool(inputs, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1], padding='SAME')

def fc(inputs, num_units_out, scope=None, bn = False):
    with tf.variable_scope(scope, 'FC'):
        num_units_in = inputs.get_shape()[-1]
        w = weight_variable([int(num_units_in), num_units_out], std=0.01)
        b = bias_variable([num_units_out])
        wx_plus_b =  tf.nn.xw_plus_b(inputs, w, b)
        if bn:
            wx_plus_b = batch_norm(wx_plus_b)
        return tf.nn.relu(wx_plus_b)

def dropout(inputs, keep_prob=0.5, scope=None):
    if keep_prob < 1:
        with tf.name_scope(scope, 'Dropout'):
            return tf.nn.dropout(inputs, keep_prob)
    else:
        return inputs

def flatten(inputs, scope=None):
    if len(inputs.get_shape()) < 2:
        raise ValueError('Inputs must be have a least 2 dimensions')
    dims = inputs.get_shape()[1:]
    k = dims.num_elements()
    with tf.name_scope(scope, 'Flatten'):
        return tf.reshape(inputs, [-1, k])


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon = 1e-3):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)