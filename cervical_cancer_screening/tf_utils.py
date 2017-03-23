import os
import tensorflow as tf
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
import sys
sys.path.append('/home/jt2/Workspace/models/slim')
from preprocessing import inception_preprocessing

def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False, shuffle=True):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32,
            common_queue_min=8, shuffle=shuffle)
    image_raw, label, label_text, filename = data_provider.get(['image', 'label', 'label_text', 'filename'])


    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels, label_texts, filenames = tf.train.batch(
            [image, image_raw, label, label_text, filename],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)

    # Pack to dir
    packed = dict(images=images, images_raw=images_raw, labels=labels,
            label_texts=label_texts, filenames=filenames)

    return packed

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)

    for i in variables_to_restore:
        print i.name

    return slim.assign_from_checkpoint_fn(
            os.path.join(FLAGS.checkpoints_dir, 'inception_v1.ckpt'),
            variables_to_restore)

