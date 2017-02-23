import math

import numpy as np
import tensorflow as tf
import random

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def global_step_variable():
    return tf.Variable(0, name='global_step', trainable=False)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        #  tf.summary.histogram('histogram', var)


def inputs(n_input, n_classes):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_input], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

        # Add these to be retreived when importing meta data
        # tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, x)
        # tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, y_)
    return x, y_


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def image_summary(tensor, side):
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(tensor, [-1, side, side, 1])
        tf.summary.image('input', image_shaped_input, 10)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, conv2d=False, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            if conv2d:
                conv_size = 5
                weights = tf.Variable(tf.random_normal([conv_size, conv_size, input_dim,
                                                        output_dim]))
            else:
                weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            if conv2d:
                strides = 1
                preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, strides,
                                                                           strides, 1],
                                           padding='SAME')
            else:
                preactivate = tf.matmul(input_tensor, weights)
            preactivate += biases
            #  tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        #  tf.summary.histogram('activations', activations)
        return activations


def keep_probability():
    with tf.name_scope('input'):
        keep_prob = tf.placeholder(tf.float32, name="keep_probability")
    # tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, keep_prob)
    return keep_prob


def dropout(tensor, keep_prob):
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        # dropped = tf.nn.dropout(hidden2, keep_prob)
        # dropped = tf.nn.dropout(hidden3, keep_prob)
        # dropped = tf.nn.dropout(fc1, keep_prob)
    return tf.nn.dropout(tensor, keep_prob)


def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        # diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


def rmse_loss(logits, labels):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, logits))))


def train(cross_entropy, learning_rate, global_step):
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            cross_entropy, global_step=global_step)
    return train_step

def tf_learning_rate(starter_learning_rate, global_step):
    decay_step = 100
    decay_rate = 0.90
    staircase=True
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
            global_step, decay_step, decay_rate, staircase=staircase)
    tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate


def measure_accuracy(logits, labels):
    # Accuracy is placed under tf.GraphKey.SUMMARIES due to summary.scalar(), not accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return correct_prediction, accuracy

def mlp(n_input, n_classes):
    hidden1 = 3000
    hidden2 = 3000
    keep_prob = keep_probability()
    x, y_ = inputs(n_input, n_classes)

    layer1 = nn_layer(x, n_input, hidden1, 'layer1')
    layer1_drop = dropout(layer1, keep_prob)
    layer2 = nn_layer(layer1_drop, hidden1, hidden2, 'layer2')
    layer2_drop = dropout(layer2, keep_prob)
    y = nn_layer(layer2_drop, hidden2, n_classes, 'output', act=tf.identity)

    return x, y_, keep_prob, y, None, None

def mlp2(n_input, n_classes):
    weight_dir = 'weight/'
    weight = weight_dir + 'cnn_02-21-2017_21:24:28-10'
    meta = weight + ".meta"
    saver = tf.train.import_meta_graph(meta)
    #  config = tf.ConfigProto(
            #  device_count = {'GPU': 0}
            #  )
    sess = tf.Session()
    saver.restore(sess, weight)

    tensor = sess.graph.get_tensor_by_name("output/activation:0")

    hidden1 = 5000
    hidden2 = 5000
    keep_prob = keep_probability()
    x, y_ = inputs(n_input, n_classes)

    layer1 = nn_layer(x, n_input, hidden1, 'layer1')
    layer1_drop = dropout(layer1, keep_prob)
    layer2 = nn_layer(layer1_drop, hidden1, hidden2, 'layer2')
    layer2_drop = dropout(layer2, keep_prob)
    y = nn_layer(layer2_drop, hidden2, n_classes, 'output', act=tf.identity)
    y = tf.concat([y, tensor], 1)

    return x, y_, keep_prob, y, None

def cnn(n_input, n_classes, rotate=True):
    side = int(math.sqrt(n_input))

    def input_transform(x):
        if type(x) is tf.Tensor:
            transform = tf.reshape(x, shape=[-1, side, side, 1])
            if rotate:
                with tf.variable_scope("image_transform"):
                    transform = tf.contrib.image.rotate(transform, random.uniform(0,6.28))
            return transform
        else:
            return np.reshape(x, [-1, side, side, 1])

    keep_prob = keep_probability()
    x, y_ = inputs(n_input, n_classes)

    x = input_transform(x)

    conv1 = nn_layer(x, 1, 32, 'layer1', conv2d=True)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = nn_layer(conv1, 32, 64, 'layer2', conv2d=True)
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, 25 * 25 * 64])
    fc1 = nn_layer(fc1, 25 * 25 * 64, 1024, 'fc_layer1')

    #  Do not apply softmax activation yet, see loss()
    dropped = dropout(fc1, keep_prob)
    y = nn_layer(dropped, 1024, n_classes, 'output', act=tf.identity)

    return x, y_, keep_prob, y, input_transform, None


def cnn256(n_input, n_classes, rotate=False):
    side = int(math.sqrt(n_input))

    def input_transform(x):
        if type(x) is tf.Tensor:
            transform = tf.reshape(x, shape=[-1, side, side, 1])
            if rotate:
                with tf.variable_scope("image_transform"):
                    transform = tf.contrib.image.rotate(transform, random.uniform(0,6.28))
            return transform
        else:
            return np.reshape(x, [-1, side, side, 1])

    keep_prob = keep_probability()
    x, y_ = inputs(n_input, n_classes)

    x = input_transform(x)

    filter1 = 32
    filter2 = 64
    red = 25
    out = 1024

    conv1 = nn_layer(x, 1, filter1, 'layer1', conv2d=True)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = nn_layer(conv1, filter1, filter2, 'layer2', conv2d=True)
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, red * red * filter2])
    fc1 = nn_layer(fc1, red * red * filter2, out, 'fc_layer1')

    #  Do not apply softmax activation yet, see loss()
    dropped = dropout(fc1, keep_prob)
    y = nn_layer(dropped, out, n_classes, 'output', act=tf.identity)

    return x, y_, keep_prob, y, input_transform, None

def cnn2(n_input, n_classes, rotate=False):
    additional_input = 192
    side = int(math.sqrt(n_input))

    def input_transform(x):
        if type(x) is tf.Tensor:
            transform = tf.reshape(x, shape=[-1, side, side, 1])
            if rotate:
                with tf.variable_scope("image_transform"):
                    transform = tf.contrib.image.rotate(transform, random.uniform(0,6.28))
                #  transform = tf.contrib.image.rotate(transform, random.uniform(0,6.28))
            return transform
        else:
            return np.reshape(x, [-1, side, side, 1])

    keep_prob = keep_probability()
    x, y_ = inputs(n_input, n_classes)

    x = input_transform(x)

    conv1 = nn_layer(x, 1, 32, 'layer1', conv2d=True)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = nn_layer(conv1, 32, 64, 'layer2', conv2d=True)
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, 25 * 25 * 64])
    fc1 = nn_layer(fc1, 25 * 25 * 64, 1024, 'fc_layer1')
    #  dropped = dropout(fc1, keep_prob)

    #  il1 = nn_layer(dropped, 1024, n_classes, 'integration_layer1')
    il1 = nn_layer(fc1, 1024, n_classes, 'integration_layer1')
    il1 = tf.nn.l2_normalize(il1,1)
    x2 = tf.placeholder(tf.float32, [None, additional_input], name='x-input2')
    il1 = tf.concat([il1, x2], 1)
    #  il1 = tf.concat(1, [dropped, x2])
    #  il1 = tf.concat([dropped, x2], 1) # tf1.0

    #  il2 = nn_layer(il1, 1024 + additional_input, 5000, 'integration_layer2')
    il2 = nn_layer(il1, n_classes + additional_input, 3000, 'integration_layer2')
    dropped2 = dropout(il2, keep_prob)

    il3 = nn_layer(dropped2, 3000, 3000, 'integration_layer3')
    dropped3 = dropout(il3, keep_prob)

    y = nn_layer(dropped3, 3000, n_classes, 'output', act=tf.identity)

    return x, y_, keep_prob, y, input_transform, x2
