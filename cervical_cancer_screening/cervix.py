import os
import tensorflow as tf

import sys
sys.path.append('/home/jt2/Workspace/models/slim/datasets')
import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {'train': 1480, 'test': 10000}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'Image of cerix',
    'label': 'Types of cervix',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.
  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/height': tf.FixedLenFeature((), tf.int64),
      'image/width': tf.FixedLenFeature((), tf.int64),
      'image/colorspace': tf.FixedLenFeature((), tf.string),
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
          default_value=-1),
      'image/class/text': tf.FixedLenFeature([], tf.string, default_value='')
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'label_text': slim.tfexample_decoder.Tensor('image/class/text')
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)
