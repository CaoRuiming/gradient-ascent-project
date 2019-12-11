import tensorflow as tf
import numpy as np
import string
from functools import reduce

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 30

def pad_corpus(sents):
    """
    DO NOT CHANGE:

    arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param french: list of French sentences
    :param english: list of English sentences
    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
    """
    padded_sentences = []
    for line in sents:
        padded = line[:WINDOW_SIZE - 1]
        padded += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded)-1)
        padded_sentences.append(padded)

    return padded_sentences

def get_data(train_file, validation_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # TODO: load and concatenate training data from training file.
    labeldict = {"true": 5, "mostly-true": 4, "half-true": 3, "barely-true": 2, "false": 1, "pants-fire": 0}
    # labeldict = {"true": 1, "mostly-true": 1, "half-true": 1, "barely-true": 1, "false": 0, "pants-fire": 0}
    vocab = []
    train = []
    train_labels = []
    with open(train_file, 'r') as f:
        for line in f:
            l = line.split('\t')
            text = l[2].replace('-', ' ')
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            train.append(text.split())
            vocab += text.split()
            train_labels.append(labeldict[l[1]])
        f.close()

    val = []
    val_labels = []
    # TODO: load and concatenate validation data from validation file.
    with open(validation_file, 'r') as f:
        for line in f:
            l = line.split('\t')
            text = l[2].replace('-', ' ')
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            val.append(text.split())
            vocab += text.split()
            val_labels.append(labeldict[l[1]])
        f.close()

    test = []
    test_labels = []
    # TODO: load and concatenate testing data from testing file.
    with open(test_file, 'r') as f:
        for line in f:
            l = line.split('\t')
            text = l[2].replace('-', ' ')
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            test.append(text.split())
            vocab += text.split()
            test_labels.append(labeldict[l[1]])
        f.close()

    vocab += ["*PAD*", "*STOP*"]
    vocab = set(vocab)
    dictionary = {w: i for i, w in enumerate(list(vocab))}
    train = pad_corpus(train)
    val = pad_corpus(val)
    test = pad_corpus(test)

    train_i = []
    val_i = []
    test_i = []

    # print(dictionary['*PAD*'])
    for statement in train:
        n_state = []
        for word in statement:
            n_state.append(dictionary[word])
        train_i.append(n_state)
    for statement in val:
        n_state = []
        for word in statement:
            n_state.append(dictionary[word])
        val_i.append(n_state)
    for statement in test:
        n_state = []
        for word in statement:
            n_state.append(dictionary[word])
        test_i.append(n_state)

    return (train_i, train_labels, val_i, val_labels, test_i, test_labels, dictionary, labeldict)
