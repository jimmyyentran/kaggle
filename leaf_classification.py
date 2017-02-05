# coding: utf-8
from scipy import misc
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import pickle
from IPython import embed


class ImageRecognition:
    def __init__(self):
        #self.data is in order of [1D image, one-hot labels, identifier]
        pass

    def process_images(self, single_side, num_images, image_dir, save=False):
        #input parameters
        n_input = single_side * single_side
        self.n_input = n_input

        train_im = np.zeros(shape=(num_images, n_input)) # initiate array
        for i in range(1,num_images):
            im = misc.imread(image_dir + str(i) + ".jpg") # transform to matrix
            train_im[i-1] = im.ravel() # Shift index due to numpy's zero-indexing
        print train_im.shape

        train_csv = pd.read_csv('train.csv')
        print train_csv.shape # training has the "species" column

        train_idx = train_csv['id'] - 1 # subtract 1 to take care of zero-indexing
        train = train_im[train_idx] # extract training pictures from images
        print train.shape

        labels = pd.get_dummies(train_csv['species']) # one-hot encoding
        labels = labels.as_matrix() # convert dataframe to matrix
        print labels.shape

        train_and_label = np.hstack((train, labels)) # combine matrix column-wise

        self.data = np.column_stack((train_and_label, train_csv['id']))

        if(save):
            with open('data_'+str(single_side)+'.pkl', 'wb') as output:
                pickle.dump(self.data, output, -1)
                pickle.dump(n_input, output, -1)

    def load_processed_data(self, data_dir):
        with open(data_dir, 'rb') as input:
            self.data = pickle.load(input)
            self.n_input = pickle.load(input)

    def cv(self, train_num):
        self.train_num = train_num
        temp = np.copy(self.data)
        np.random.shuffle(temp)

        self.identifier_train = temp[:,-1][:train_num]
        self.identifier_test = temp[:,-1][train_num:]

        temp = temp[:,:-1]

        self.train = temp[:,:self.n_input][:train_num]
        self.labels = temp[:,self.n_input:][:train_num]
        self.test_image = temp[:,:self.n_input][train_num:]
        self.test_labels = temp[:,self.n_input:][train_num:]

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


        # batch size of 10 results in nan costs

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
        init = tf.initialize_all_variables() # used older version of global_variables_initializer

        pred_labels=tf.argmax(pred,1)

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(self.train_num/batch_size)
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
                    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)
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

        embed()

if __name__ == "__main__":
    ir = ImageRecognition()
    #  ir.process_images(100, 1584, 'images/processed_100/', True)
    ir.load_processed_data('data_100.pkl')
    ir.cv(790)
    ir.mp(
            learning_rate = 0.0008,
            training_epochs = 10,
            batch_size = 100,
            display_step = 1,
            num_examples = 790,
            n_hidden_1 = 3000,
            n_hidden_2 = 3000,
            n_classes = 99
    )
