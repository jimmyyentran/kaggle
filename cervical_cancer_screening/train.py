import tensorflow as tf
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
import sys

sys.path.append('/home/jt2/Workspace/models/slim')
from nets import inception

from tf_utils import *

class Train:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def init_train(self):
        with tf.Graph().as_default():
            dataset = self.data_loader.get_split('train', FLAGS.data_dir)
            data_tensors = load_batch(dataset, 
                    height=FLAGS.image_size,
                    width=FLAGS.image_size)
            images = data_tensors['images']
            labels = data_tensors['labels']

            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

            labels = tf.subtract(labels, 1)

            # Specify the loss function:
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            one_hot_labels = tf.cast(one_hot_labels, tf.int64)
            slim.losses.softmax_cross_entropy(logits, one_hot_labels)
            total_loss = slim.losses.get_total_loss()

            # Create some summaries to visualize the training process:
            tf.summary.scalar('losses/Total Loss', total_loss)

            # Specify the optimizer and create the train op:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            # Run the training:
            final_loss = slim.learning.train(
                    train_op,
                    logdir=FLAGS.log_dir,
                    init_fn=get_init_fn(),
                    number_of_steps=FLAGS.number_of_steps)


        print('Finished training. Last batch loss %f' % final_loss)
