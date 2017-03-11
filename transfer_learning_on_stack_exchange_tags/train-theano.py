#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from IPython import embed
import re

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '20000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '2000'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
_EVALUATE_ = os.environ.get('EVALUATE', False)
_CSV_FILE = os.environ.get('CSV_FILE', 'preprocess/biology_light_punctuation_with_stop.csv')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
	print "Epoch: ", epoch
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
	    #print "Evaluating"
            #loss = model.calculate_loss(X_train, y_train)
            #losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            #print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            ## Adjust the learning rate if loss increases
            #if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            #    learning_rate = learning_rate * 0.5  
            #    print "Setting learning rate to %f" % learning_rate
            #sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
	    if i % 1000 == 0:
		print "Training ex", i
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def load_data_set(csv_file, vocabulary_size):
	# Read the data and append SENTENCE_START and SENTENCE_END tokens
	print "Reading CSV file..."
	#  with open('data/reddit-comments-2015-08.csv', 'rb') as f:
	with open(csv_file, 'rb') as f:
	    reader = csv.reader(f, skipinitialspace=True)
	    reader.next()
	    # Split full comments into sentences
	    #  sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
	    sentences = itertools.chain(*[nltk.sent_tokenize(x[2].decode('utf-8').lower()) for x in reader])
	    # Append SENTENCE_START and SENTENCE_END
	    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
	print "Parsed %d sentences." % (len(sentences))
	    
	# Tokenize the sentences into words
	tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
	
	pickle_dir = 'token'
	#with open(picke_dir + csv_filehhhhhhhhhhhhhhh

	# Count the word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print "Found %d unique words tokens." % len(word_freq.items())

	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size-1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

	print "Using vocabulary size %d." % vocabulary_size
	print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

	# Replace all words not in our vocabulary with the unknown token
	for i, sent in enumerate(tokenized_sentences):
	    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
	return tokenized_sentences, index_to_word, word_to_index

if not _EVALUATE_:
    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)

    if _MODEL_FILE != None:
	m = re.findall('\d+',model_location)
	vocabulary_size = m[1]
	model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
        load_model_parameters_theano(_MODEL_FILE, model)

    tokenized_sentences, index_to_word, word_to_index = load_data_set(csv_file, vocabulary_size)

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    
    print "Begin training"
    train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
else:
    print "Start IPython for Evaluation"

    def generate_sentence(model):
        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
    	unknown_count = 0
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            next_word_probs = model.forward_propagation(new_sentence)
            sampled_word = word_to_index[unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
		sys.out.write('.')
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
	    #sys.stdout.write(str(index_to_word[sampled_word]) + ": "+ str(next_word_probs[-1][sampled_word]) + " ")
            new_sentence.append(sampled_word)
    	#sys.stdout.write("uk= " + str(unknown_count) + " ")
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
	#sys.stdout.write('|')
        return sentence_str

    def gen(num_sentences, senten_min_length):
        for i in range(num_sentences):
            sent = []
            # We want long sentences, not sentences with one or two words
            while len(sent) < senten_min_length:
                sent = generate_sentence(model)
	    print ""
            print " ".join(sent)

    def import_model(model_location, csv_file=_CSV_FILE):
	m = re.findall('\d+',model_location)
	vocab_size = int(m[1])
	hidden_d = int(m[0])
	print "Vocab size= %d, hidden dimensions= %d" % (vocab_size, hidden_d)
	model2 = RNNTheano(vocab_size, hidden_dim=hidden_d)
	load_model_parameters_theano(model_location, model2)
	_, idx_to_word, word_to_idx = load_data_set(csv_file, vocab_size)
	return model2, idx_to_word, word_to_idx
	#model = model2
	#word_to_index = word_to_idx
	#idx_to_word = index_to_word

    if _MODEL_FILE != None:
	model, index_to_word, word_to_index = import_model(_MODEL_FILE)

    embed()
