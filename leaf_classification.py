
# coding: utf-8

# In[1]:

from scipy import misc
import numpy as np
import pandas as pd


# In[2]:

#input parameters
single_side = 350
n_input = single_side * single_side
num_images = 1584

image_dir = 'images/processed_2/'

train_im = np.zeros(shape=(num_images, n_input)) # initiate array
for i in range(1,num_images):
    im = misc.imread(image_dir + str(i) + ".jpg") # transform to matrix
    train_im[i-1] = im.ravel() # Shift index due to numpy's zero-indexing
print train_im.shape


# Image vector contains 1584 images each with 122500 columns containing the greyscale value

# In[3]:

train_csv = pd.read_csv('train.csv')
print train_csv.shape # training has the "species" column


# In[4]:

train_idx = train_csv['id'] - 1 # subtract 1 to take care of zero-indexing
train = train_im[train_idx] # extract training pictures from images
print train.shape


# In[5]:

labels = pd.get_dummies(train_csv['species']) # one-hot encoding
labels = labels.as_matrix() # convert dataframe to matrix
print labels.shape


# In[6]:

train_num = 790

temp = np.hstack((train, labels)) # combine matrix column-wise
np.random.shuffle(temp)
train = temp[:,:n_input][:train_num]
labels = temp[:,n_input:][:train_num]
test_image = temp[:,:n_input][train_num:]
test_labels = temp[:,n_input:][train_num:]

print train.shape
print labels.shape
print test_image.shape
print test_labels.shape


# In[24]:

np.save('data_grayscale.npy', temp)


# In[7]:

import tensorflow as tf


# In[19]:

# Parameters
learning_rate = 0.0008
training_epochs = 10
batch_size = 100
display_step = 1
num_examples = 700

# Network Parameters
n_hidden_1 = 1000 # 1st layer number of features
n_hidden_2 = 1000 # 2nd layer number of features
# n_input - from above
n_classes = 99 # one-hot shows 99 species

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# batch size of 10 results in nan costs

# In[20]:

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


# In[21]:

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
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


# In[22]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            begin = total_batch * batch_size
            end = begin + batch_size
            batch_x = train[begin:end]
            batch_y = labels[begin:end]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost)
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             print "Train Accuracy:", accuracy.eval({x: train, y: labels})
#             print "Test Accuracy:", accuracy.eval({x: test_image, y: test_labels})
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Train Accuracy:", accuracy.eval({x: train, y: labels})
    print "Test Accuracy:", accuracy.eval({x: test_image, y: test_labels})
    


# In[ ]:

train[980:1000]


# In[ ]:



