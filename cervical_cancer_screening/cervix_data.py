import sys
sys.path.append('/home/jt2/Workspace/models/inception/inception/')
from dataset import Dataset

class CervixData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset):
    super(CervixData, self).__init__('Cervix', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 3

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1480
    if self.subset == 'validation':
      return 0

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any ImageNet %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the ImageNet data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take several hours depending on the speed of your '
          'computer and network connection\n')
    print('Please see README.md for instructions on how to build '
          'the ImageNet dataset using download_and_preprocess_imagenet.\n')
    print('Note that the raw data size is 300 GB and the processed data size '
          'is 150 GB. Please ensure you have at least 500GB disk space.')
