import tensorflow as tf
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
import sys
import os

sys.path.append('/home/jt2/Workspace/models/slim')
from datasets import dataset_utils


def download_model(url, checkpoints_dir):
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

def get_variables_stats(variables, variable_values):
    for i in xrange(len(variables)):
        arr = np.array(all_var[i])
    print "%d %-20.20s mean: %10.10s var: %s" % (i, variables[i].name,
            np.mean(arr), np.var(arr))

def set_flag(key, value):
    try:
        if type(value) is str:
            tf.app.flags.DEFINE_string(key, value, '')
        elif type(value) is int:
            tf.app.flags.DEFINE_integer(key, value, '')
        elif type(value) is float:
            tf.app.flags.DEFINE_float(key, value, '')
        else:
            print "set_flag ERROR"
    except Exception:
        FLAGS.__setattr__(key, value)
