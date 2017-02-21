import math
import numpy as np
import tensorflow as tf
import random

import model


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
        self.integrated = kwargs['integrated']
        self.rotate = kwargs['rotate']

        self.name = kwargs['name']

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

    def generate_model(self, mdl):
        if mdl.lower() == 'cnn':
            model_function = model.cnn
        elif mdl.lower() == 'mlp':
            model_function = model.mlp

        sess = tf.InteractiveSession()

        # Model tensors
        #  x, y_, keep_prob, y, input_transform = model_function(self.n_input, self.n_classes)
        x, y_, keep_prob, y, input_transform, _ = model_function(self.n_input, self.n_classes)

        cross_entropy = model.loss(y, y_)

        #  learning_rate = model.tf_learning_rate(0.001)
        #  train_step = model.train(cross_entropy, learning_rate)
        #  train_step = model.train(cross_entropy, self.learning_rate)

        global_step = model.global_step_variable()
        learning_rate = model.tf_learning_rate(self.learning_rate, global_step)
        train_step = model.train(cross_entropy, learning_rate, global_step)

        correct_prediction, accuracy = model.measure_accuracy(y, y_)

        # Summary handler
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('output/train_' + self.name, sess.graph)
        test_writer = tf.summary.FileWriter('output/test_' + self.name)

        # Saver
        saver = tf.train.Saver()

        # initialize the variables
        tf.global_variables_initializer().run()

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

            if mdl.lower() == 'cnn':
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
                    if i % 1000 == 999:
                        saver.save(sess, 'model/model_' + self.name, global_step=i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()

        # reset for next batch
        tf.reset_default_graph()

        # close and restore memory
        sess.close()

    def generate_model2(self, mdl):
        if mdl.lower() == 'cnn':
            model_function = model.cnn
        elif mdl.lower() == 'mlp':
            model_function = model.mlp
        elif mdl.lower() == 'cnn_mlp':
            model_function = model.cnn2

        sess = tf.InteractiveSession()

        # Model tensors
        x, y_, keep_prob, y, input_transform, x2 = model_function(10000,
                self.n_classes, rotate=self.rotate)

        cross_entropy = model.loss(y, y_)

        global_step = model.global_step_variable()
        learning_rate = model.tf_learning_rate(self.learning_rate, global_step)
        train_step = model.train(cross_entropy, learning_rate, global_step)
        #  train_step = model.train(cross_entropy, self.learning_rate)

        correct_prediction, accuracy = model.measure_accuracy(y, y_)

        # Summary handler
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('output/train_' + self.name, sess.graph)
        test_writer = tf.summary.FileWriter('output/test_' + self.name)

        # Saver
        saver = tf.train.Saver()

        # initialize the variables
        tf.global_variables_initializer().run()

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

            xs2 = xs[:, 10000:]
            xs = xs[:, :10000]

            if mdl.lower() == 'cnn' or mdl.lower() == 'cnn_mlp':
                xs = input_transform(xs)

            return {x: xs, y_: ys, keep_prob: k, x2:xs2}

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
                    if i % 1000 == 999:
                        saver.save(sess, 'model/model_' + self.name, global_step=i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()

        # reset for next batch
        tf.reset_default_graph()

        # close and restore memory
        sess.close()
