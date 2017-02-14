import pickle

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from IPython import embed
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
import seaborn as sns

from leaf_classification import ImageRecognition


def load_session(mdl_loc):
    meta = mdl_loc + ".meta"
    saver = tf.train.import_meta_graph(meta)

    sess = tf.Session()
    saver.restore(sess, mdl_loc)
    return sess


def get_tensor(session, tensor_name):
    tensor = session.graph.get_tensor_by_name(tensor_name)
    return tensor


def predict(sess, tensor, x_input):
    output = sess.run(tensor,
                      feed_dict={'input/keep_probability:0': 1.0,
                                 'input_1/x-input:0': x_input})
    return output


def save_prediction(array, dir, name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(dir + name, 'wb') as output:
        pickle.dump(array, output, -1)


def load_predictions(pred_loc):
    with open(pred_loc, 'rb') as input:
        data = pickle.load(input)
    return data


def compare_with_true(prediction, labels):
    best_guess_index = np.argmax(prediction, 1)
    actual_index = np.argmax(labels, 1)
    correct_predictions = np.equal(best_guess_index, actual_index)
    correct_percentage = np.mean(correct_predictions)
    return best_guess_index, actual_index, correct_predictions, correct_percentage


def correct_count(correct_predictions, actual_labels, one_hot_names):
    correct_indices = actual_labels[correct_predictions]
    incorrect_indices = actual_labels[-correct_predictions]

    num_occurrences_correct = np.bincount(correct_indices)
    num_occurrences_incorrect = np.bincount(incorrect_indices)

    counts = pd.DataFrame(index=one_hot_names, data={'correct': num_occurrences_correct,
                                                     'incorrect': num_occurrences_incorrect,
                                                     'total':num_occurrences_correct +
                                                             num_occurrences_incorrect})
    counts['percent'] = counts['correct'] / counts['total']

    return counts

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Accent):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    sns.set(context="paper", rc={"font.size":5})
    plt.figure(figsize=(24,10))
    ax = sns.heatmap(cm, linewidths=0.1)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.yticks(tick_marks, classes, rotation=0)
    plt.xticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     if cm[i,j] != 0:
    #         plt.text(j, i, cm[i, j],
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def best_worse(df):
    col_name = 'correct'
    col_total_name = 'total'

    df[col_name]
    embed()


def load_session_and_save_prediction(**kwargs):
    test = kwargs['test']
    model_location = kwargs['model_location']
    tensor_name = kwargs['tensor_name']
    save_dir = kwargs['save_dir']
    save_name = kwargs['save_name']

    sess = load_session(model_location)
    y_ = get_tensor(sess, tensor_name)
    prediction = predict(sess, y_, test)
    save_prediction(prediction, save_dir, save_name)


def evaluate(**kwargs):
    test_labels = kwargs['test_labels']
    identifier = kwargs['id']
    one_hot_names = kwargs['one_hot_names']
    save_dir = kwargs['save_dir']
    save_name = kwargs['save_name']

    pred = load_predictions(save_dir + save_name)
    best_guess_index, actual_index, correct_predictions, correct_percentage = compare_with_true(
        pred, test_labels)
    print correct_percentage

    # count = correct_count(correct_predictions, actual_index, one_hot_names)
    # cm = confusion_matrix(actual_index, best_guess_index)
    # plot_confusion_matrix(cm, one_hot_names)

    embed()



if __name__ == "__main__":
    ir = ImageRecognition()
    ir.load_processed_data('data_no_id_100.pkl')
    test = ir.data[:, :-99]
    label = ir.data[:, -99:]
    metadata = dict(
        test=test,
        test_labels=label,
        id=ir.identifier,
        one_hot_names=ir.one_hot_names,
        model_location="output/model_02-13-2017_05:24:36-99",
        tensor_name="output/activation:0",
        save_dir="predictions/",
        save_name="model_02-13-2017_05:24:36-99.pkl")

    # load_session_and_save_prediction(**metadata)
    evaluate(**metadata)
