import model
import tensorflow as tf

class Evaluator():
    def __init__(self, **kwargs):
        self.n_input = kwargs['n_input']
        self.n_classes = kwargs['n_classes']
        self.test = kwargs['test']
        self.test_labels = kwargs['test_labels']

    def evaluate_model(self, mdl):
        if mdl.lower() == 'cnn':
            model_function = model.cnn

        sess = tf.InteractiveSession()

        # Model tensors
        x, y_, keep_prob, y, input_transform = model_function(self.n_input, self.n_classes)

        cross_entropy = model.loss(y, y_)

        train_step = model.train(cross_entropy, self.learning_rate)

        correct_prediction, accuracy = model.measure_accuracy(y, y_)

        # # Summary handler
        # merged = tf.summary.merge_all()
        # curr_time = str(strftime("%m-%d-%Y_%H:%M:%S", gmtime()))
        # train_writer = tf.summary.FileWriter('output/train_' + curr_time, sess.graph)
        # test_writer = tf.summary.FileWriter('output/test_' + curr_time)

        # initialize the variables
        # tf.global_variables_initializer().run()

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

        # for i in range(self.training_epochs):
        #     if i % 10 == 0:  # Record summaries and test-set accuracy
        #         summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        #         test_writer.add_summary(summary, i)
        #         print('Test accuracy at step %s: %s' % (i, acc))
        #     else:  # Record train set summaries, and train
        #         if i % 100 == 99:  # Record execution stats
        #             run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #             run_metadata = tf.RunMetadata()
        #             summary, _ = sess.run([merged, train_step],
        #                                   feed_dict=feed_dict(True),
        #                                   options=run_options,
        #                                   run_metadata=run_metadata)
        #             train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        #             train_writer.add_summary(summary, i)
        #             saver.save(sess, 'output/model_' + curr_time, global_step=i)
        #             print('Adding run metadata for', i)
        #         else:  # Record a summary
        #             summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        #             train_writer.add_summary(summary, i)
        # train_writer.close()
        # test_writer.close()

        # reset for next batch
        tf.reset_default_graph()

        # close and restore memory
        sess.close()
