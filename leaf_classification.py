# coding: utf-8
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
from IPython import embed
from scipy import misc
from sklearn.model_selection import KFold
from time import strftime, gmtime
from trainer import Trainer

class ImageRecognition:
    def __init__(self):
        # self.data is in order of [1D image, one-hot labels, identifier]
        self.current_step = 0
        self.train_index = []
        self.test_index = []
        self.names = []
        pass

    def process_csv(self, save=False):
        train_csv = pd.read_csv('train.csv')

        labels = pd.get_dummies(train_csv['species'])  # one-hot encoding
        self.one_hot_names = labels.columns
        labels = labels.as_matrix()

        train_data = train_csv.drop(['id', 'species'], axis=1).as_matrix()
        self.n_input = train_data.shape[1]

        #  with open('predictions/model_02-20-2017_02:10:32-999.pkl', 'rb') as cnn:
            #  data_cnn = pickle.load(cnn)

        #  embed()
        self.data = np.column_stack((train_data, labels))
        self.identifier = train_csv['id']

        if (save):
            with open('data_csv_only.pkl', 'wb') as output:
                pickle.dump(self.data, output, -1)
                pickle.dump(self.identifier, output, -1)
                pickle.dump(self.one_hot_names, output, -1)
                pickle.dump(self.n_input, output, -1)


    def process_images(self, single_side, num_images, image_dir, save=False):
        # input parameters
        self.n_input = single_side * single_side

        train_im = np.zeros(shape=(num_images, self.n_input))  # initiate array
        for i in range(1, num_images):
            im = misc.imread(image_dir + str(i) + ".jpg")  # transform to matrix
            train_im[i - 1] = im.ravel()  # Shift index due to numpy's zero-indexing
        print "All 1D images: ", train_im.shape

        train_csv = pd.read_csv('train.csv')
        print "Train csv", train_csv.shape  # training has the "species" column

        train_idx = train_csv['id'] - 1  # subtract 1 to take care of zero-indexing
        train = train_im[train_idx]  # extract training pictures from images
        print "Train ID images:", train.shape

        labels = pd.get_dummies(train_csv['species'])  # one-hot encoding
        self.one_hot_names = labels.columns
        labels = labels.as_matrix()  # convert dataframe to matrix
        print "Labels one-hot", labels.shape

        # additional = train_csv.drop(['id', 'species'], axis=1).as_matrix() * 255

        # train_and_label = np.hstack((train, additional, labels))  # combine matrix column-wise

        # self.data = np.column_stack((train_and_label, train_csv['id']))
        self.data = np.column_stack((train, labels))
        self.identifier = train_csv['id']

        if (save):
            # with open('data_additional_' + str(single_side) + '.pkl', 'wb') as output:
            with open('data_no_id_' + str(single_side) + '.pkl', 'wb') as output:
                pickle.dump(self.data, output, -1)
                pickle.dump(self.identifier, output, -1)
                pickle.dump(self.one_hot_names, output, -1)
                pickle.dump(self.n_input, output, -1)

    def load_processed_data(self, data_dir):
        with open(data_dir, 'rb') as input:
            self.data = pickle.load(input)
            self.identifier = pickle.load(input)
            self.one_hot_names = pickle.load(input)
            self.n_input = pickle.load(input)

    def cv(self, num_splits):
        # K-fold with shuffle
        kf = KFold(n_splits=num_splits, shuffle=True)

        # for train, test in kf.split(temp):
        for train, test in kf.split(self.data):
            self.train_index.append(train)
            self.test_index.append(test)

    def train(self, params):
        if len(self.train_index) > params['train_times']:
            min = params['train_times']
        else:
            min = len(self.train_index)

        for i in range(min):
            cv_train_data = self.data[self.train_index[i]]
            cv_test_data = self.data[self.test_index[i]]

            train = cv_train_data[:, :self.n_input]
            train_labels = cv_train_data[:, self.n_input:]
            test = cv_test_data[:, :self.n_input]
            test_labels = cv_test_data[:, self.n_input:]

            print "Set %s)" % (i + 1) + \
                  " Train: {}".format(train.shape) + \
                  " Train Labels: {}".format(train_labels.shape) + \
                  " Test: {}".format(test.shape) + \
                  " Test labels: {}".format(test_labels.shape)

            model = params['model']

            curr_time = str(strftime("%m-%d-%Y_%H:%M:%S", gmtime()))
            name = model+'_'+curr_time

            self.names.append(name)
            params.update(dict(train=train,
                               train_labels=train_labels,
                               test=test,
                               test_labels=test_labels,
                               n_input=self.n_input,
                               name=name))

            #  Trainer(**params).generate_model('cnn')
            Trainer(**params).generate_model(model)

    def save_run_metadata(self):
        with open('metadata_' + self.names[0] + '.pkl', 'wb') as output:
            pickle.dump(self.names, output, -1)
            pickle.dump(self.train_index, output, -1)
            pickle.dump(self.test_index, output, -1)


if __name__ == "__main__":
    ir = ImageRecognition()
    ir.process_csv(save=True)
    # ir.process_images(100, 1584, 'images/processed_100/', True)
    # ir.process_images(350, 1584, 'images/processed_350/', True)
    # ir.load_processed_data('data_100.pkl')
    # ir.load_processed_data('data_additional_100.pkl')
    #  ir.load_processed_data('data_no_id_100.pkl')
    #  ir.load_processed_data('data_csv_only.pkl')
    #  ir.cv(10)
    #  params = dict(learning_rate=0.001,
                #  training_epochs=1000,
                #  batch_size=100,
                #  display_step=1,
                #  n_hidden_1=10000,
                #  n_hidden_2=10000,
                #  n_classes=99,
                #  keep_prob=0.9,
                #  train_times=1,
                #  model="mlp")
    #  ir.train(params)
    #  ir.save_run_metadata()
