# coding: utf-8
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from time import gmtime, strftime
from IPython import embed
from scipy import misc



class ImageRecognition:
    def __init__(self):
        # self.data is in order of [1D image, one-hot labels, identifier]
        self.current_step = 0
        pass

    def process_images(self, single_side, num_images, image_dir, save=False):
        # input parameters
        n_input = single_side * single_side
        self.n_input = n_input + 192

        train_im = np.zeros(shape=(num_images, n_input))  # initiate array
        for i in range(1, num_images):
            im = misc.imread(image_dir + str(i) + ".jpg")  # transform to matrix
            train_im[i - 1] = im.ravel()  # Shift index due to numpy's zero-indexing
        print train_im.shape

        train_csv = pd.read_csv('train.csv')
        print train_csv.shape  # training has the "species" column

        train_idx = train_csv['id'] - 1  # subtract 1 to take care of zero-indexing
        train = train_im[train_idx]  # extract training pictures from images
        print train.shape

        labels = pd.get_dummies(train_csv['species'])  # one-hot encoding
        labels = labels.as_matrix()  # convert dataframe to matrix
        print labels.shape

        additional = train_csv.drop(['id', 'species'], axis=1).as_matrix() * 255

        train_and_label = np.hstack((train, additional, labels))  # combine matrix column-wise

        self.data = np.column_stack((train_and_label, train_csv['id']))

        if (save):
            with open('data_additional_' + str(single_side) + '.pkl', 'wb') as output:
                pickle.dump(self.data, output, -1)
                pickle.dump(self.n_input, output, -1)

    def load_processed_data(self, data_dir):
        with open(data_dir, 'rb') as input:
            self.data = pickle.load(input)
            self.n_input = pickle.load(input)

    def cv(self, train_num):
        self.train_num = train_num
        temp = np.copy(self.data)
        np.random.shuffle(temp)

        self.identifier_train = temp[:, -1][:train_num]
        self.identifier_test = temp[:, -1][train_num:]

        temp = temp[:, :-1]

        self.train = temp[:, :self.n_input][:train_num]
        self.labels = temp[:, self.n_input:][:train_num]
        self.test_image = temp[:, :self.n_input][train_num:]
        self.test_labels = temp[:, self.n_input:][train_num:]

        print "Train: {}".format(self.train.shape)
        print "Labels: {}".format(self.labels.shape)
        print "Test image: {}".format(self.test_image.shape)
        print "Test labels: {}".format(self.test_labels.shape)

    def mp(self, **kwargs):
        # Parameters
        learning_rate = kwargs['learning_rate']
        training_epochs = kwargs['training_epochs']
        batch_size = kwargs['batch_size']
        display_step = kwargs['display_step']

        # Network Parameters
        # n_input: from class
        n_hidden_1 = kwargs['n_hidden_1']
        n_hidden_2 = kwargs['n_hidden_2']
        n_classes = kwargs['n_classes']

        # tf Graph input
        x = tf.placeholder("float", [None, self.n_input])
        y = tf.placeholder("float", [None, n_classes])

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
                tf.summary.histogram('histogram', var)

        # Create model
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        pred = multilayer_perceptron(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()  # used older version of global_variables_initializer

        pred_labels = tf.argmax(pred, 1)

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(self.train_num / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    begin = total_batch * batch_size
                    end = begin + batch_size
                    batch_x = self.train[begin:end]
                    batch_y = self.labels[begin:end]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    #  print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
                    print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c)
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print "Train Accuracy:", accuracy.eval({x: self.train, y: self.labels})
                    print "Test Accuracy:", accuracy.eval({x: self.test_image, y: self.test_labels})
                    print
            print "Optimization Finished!"

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print "Train Accuracy:", accuracy.eval({x: self.train, y: self.labels})
            print "Test Accuracy:", accuracy.eval({x: self.test_image, y: self.test_labels})
            predict_labels = pred_labels.eval(feed_dict={x: self.test_image})
            prediction = pred.eval(feed_dict={x: self.test_image})
            train_labels = pred_labels.eval(feed_dict={x: self.train})
            train = pred.eval(feed_dict={x: self.train})
            #  print prediction2.eval(feed_dict={x: self.test_image})[0]

        print predict_labels
        print prediction

    def next_batch(self, batch_size):
        # Keep looping
        modulus = self.train.shape[0] / batch_size
        begin = (self.current_step % modulus) * batch_size
        end = begin + batch_size
        self.current_step += 1
        print "Epoch {}. train[{},{}]".format(self.current_step / modulus, begin, end)
        return self.train[begin:end], self.labels[begin:end]


    def training(self, **kwargs):
        # Import data
        # mnist = input_data.read_data_sets(FLAGS.data_dir,
        #                                   one_hot=True,
        #                                   fake_data=FLAGS.fake_data)

        # Parameters
        learning_rate = kwargs['learning_rate']
        training_epochs = kwargs['training_epochs']
        batch_size = kwargs['batch_size']
        display_step = kwargs['display_step']

        # Network Parameters
        # n_input: from class
        n_hidden_1 = kwargs['n_hidden_1']
        n_hidden_2 = kwargs['n_hidden_2']
        n_classes = kwargs['n_classes']

        sess = tf.InteractiveSession()
        # Create a multilayer model.

        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, self.n_input], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

        side = int(math.sqrt(self.n_input))

        # with tf.name_scope('input_reshape'):
        #     image_shaped_input = tf.reshape(x, [-1, side, side, 1])
        #     tf.summary.image('input', image_shaped_input, 10)

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

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
                tf.summary.histogram('histogram', var)

        def maxpool2d(x, k=2):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        def plotNNFilter(units):
            filters = units.shape[3]
            plt.figure(1, figsize=(20,20))
            n_columns = 6
            n_rows = math.ceil(filters / n_columns) + 1
            for i in range(filters):
                plt.subplot(n_rows, n_columns, i+1)
                plt.title('Filter ' + str(i))
                plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

        def getActivations(layer,stimuli):
            units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
            plotNNFilter(units)

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
                    tf.summary.histogram('pre_activations', preactivate)
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

        # hidden1 = nn_layer(x, self.n_input, n_hidden_1, 'layer1')
        # hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
        # hidden3 = nn_layer(hidden2, n_hidden_2, n_hidden_2, 'layer3')

        x = tf.reshape(x, shape=[-1, side, side, 1])

        conv1 = nn_layer(x, 1, 32, 'layer1', conv2d=True)
        conv1 = maxpool2d(conv1, k=2)

        conv2 = nn_layer(conv1, 32, 64, 'layer2', conv2d=True)
        conv2 = maxpool2d(conv2, k=2)

        fc1 = tf.reshape(conv2, [-1, 25*25*64])
        fc1 = nn_layer(fc1, 25*25*64, 1024, 'fc_layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            # dropped = tf.nn.dropout(hidden2, keep_prob)
            # dropped = tf.nn.dropout(hidden3, keep_prob)
            dropped = tf.nn.dropout(fc1, keep_prob)

        # Do not apply softmax activation yet, see below.
        y = nn_layer(dropped, 1024, n_classes, 'output', act=tf.identity)

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
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(
                cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        curr_time = str(strftime("%m-%d-%Y_%H:%M:%S", gmtime()))
        train_writer = tf.summary.FileWriter('output/train_' + curr_time, sess.graph)
        test_writer = tf.summary.FileWriter('output/test_' + curr_time)
        tf.global_variables_initializer().run()

        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        def feed_dict(train):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            # if train or FLAGS.fake_data:
            if train:
                xs, ys = self.next_batch(100)
                xs = np.reshape(xs, [-1, 100, 100, 1])
                k = 0.9
            else:
                # xs, ys = mnist.test.images, mnist.test.labels
                xs, ys = self.test_image, self.test_labels
                xs = np.reshape(xs, [-1, 100, 100, 1])
                k = 1.0
            return {x: xs, y_: ys, keep_prob: k}

        for i in range(training_epochs):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
                print('Test accuracy at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    ir = ImageRecognition()
    # ir.process_images(100, 1584, 'images/processed_100/', True)
    # ir.process_images(100, 1584, 'images/processed_100/', True)
    # ir.process_images(350, 1584, 'images/processed_350/', True)
    # ir.load_processed_data('data_100.pkl')
    ir.load_processed_data('data_100.pkl')
    # ir.load_processed_data('data_additional_100.pkl')
    ir.cv(800)
    ir.training(
        learning_rate=0.001,
        training_epochs=101,
        batch_size=100,
        display_step=1,
        n_hidden_1=10000,
        n_hidden_2=10000,
        n_classes=99
    )
