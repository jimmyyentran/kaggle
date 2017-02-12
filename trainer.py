import math
import numpy as np
from time import strftime, gmtime
import tensorflow as tf

import model_helper


class Trainer():
    def __init__(self, **kwargs):
        # Train Parameters
        self.learning_rate = kwargs['learning_rate']
        self.training_epochs = kwargs['training_epochs']
        self.batch_size = kwargs['batch_size']
        self.display_step = kwargs['display_step']

        # Network Parameters
        self.n_input = kwargs['n_input']
        self.n_hidden_1 = kwargs['n_hidden_1']
        self.n_hidden_2 = kwargs['n_hidden_2']
        self.n_classes = kwargs['n_classes']
        self.keep_prob = kwargs['keep_prob']

        # Data
        self.train = kwargs['train']
        self.train_labels = kwargs['train_labels']
        self.test = kwargs['test']
        self.test_labels = kwargs['test_labels']

        self.current_step = 0
        self.train_len = self.train.shape[0]
        self.modulus = 0
        self.modulus_set = False

    def next_batch(self, batch_size):
        if not self.modulus_set:
            self.modulus = int(math.ceil(self.train.shape[0] / float(batch_size)))
            self.modulus_set = True
        begin = (self.current_step % self.modulus) * batch_size
        end = begin + batch_size
        if end >= self.train_len:
            end = self.train_len - 1
        print "Epoch {}. train[{},{}]".format(self.current_step / self.modulus, begin, end)
        self.current_step += 1
        return self.train[begin:end], self.train_labels[begin:end]

    def cnn(self, n_input, n_classes):
        side = int(math.sqrt(self.n_input))

        def input_transform(x):
            if type(x) is tf.Tensor:
                return tf.reshape(x, shape=[-1, side, side, 1])
            else:
                return np.reshape(x, [-1, side, side, 1])

        keep_prob = model_helper.keep_probability()
        x, y_ = model_helper.inputs(n_input, n_classes)

        x = input_transform(x)

        conv1 = model_helper.nn_layer(x, 1, 32, 'layer1', conv2d=True)
        conv1 = model_helper.maxpool2d(conv1, k=2)

        conv2 = model_helper.nn_layer(conv1, 32, 64, 'layer2', conv2d=True)
        conv2 = model_helper.maxpool2d(conv2, k=2)

        fc1 = tf.reshape(conv2, [-1, 25 * 25 * 64])
        fc1 = model_helper.nn_layer(fc1, 25 * 25 * 64, 1024, 'fc_layer1')

        # Do not apply softmax activation yet, see loss()
        dropped = model_helper.dropout(fc1, keep_prob)
        y = model_helper.nn_layer(dropped, 1024, self.n_classes, 'output', act=tf.identity)

        return x, y_, keep_prob, y, input_transform

    def generate_model(self, model):
        if model.lower() == 'cnn':
            model_function = self.cnn

        sess = tf.InteractiveSession()

        # Model tensors
        x, y_, keep_prob, y, input_transform = model_function(self.n_input, self.n_classes)

        cross_entropy = model_helper.loss(y, y_)

        train_step = model_helper.train(cross_entropy, self.learning_rate)

        correct_prediction, accuracy = model_helper.measure_accuracy(y, y_)

        # Summary handler
        merged = tf.summary.merge_all()
        curr_time = str(strftime("%m-%d-%Y_%H:%M:%S", gmtime()))
        train_writer = tf.summary.FileWriter('output/train_' + curr_time, sess.graph)
        test_writer = tf.summary.FileWriter('output/test_' + curr_time)
        tf.global_variables_initializer().run()

        # Saver
        saver = tf.train.Saver()

        # Train the model, and also write summaries.
        # Every ith step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries
        def feed_dict(train):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            # if train or FLAGS.fake_data:
            if train:
                xs, ys = self.next_batch(self.batch_size)
                k = self.keep_prob
            else:
                # xs, ys = mnist.test.images, mnist.test.labels
                xs, ys = self.test, self.test_labels
                k = 1.0

            if model.lower() == 'cnn':
                # side = 100
                # xs = np.reshape(xs, [-1, side, side, 1])
                xs = input_transform(xs)

            return {x: xs, y_: ys, keep_prob: k}

        for i in range(self.training_epochs):
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
                    saver.save(sess, 'output/model_' + curr_time, global_step=i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()

        # reset for next batch
        tf.reset_default_graph()
