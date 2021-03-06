import numpy as np
from model import LstmModel, LstmDropoutModel, HybridModel
from preprocess import get_data, PAD_TOKEN, STOP_TOKEN
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from random import shuffle, randint, sample
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='GAP')
'../data/train.tsv', '../data/valid.tsv', '../data/test.tsv'
parser.add_argument('--test-sentence', type=str, default=None,
                    help='Single sentence for model to evaluate truthiness')
parser.add_argument('--train-file', type=str, default='../data/train.tsv',
                    help='TSV file containing train data')
parser.add_argument('--val-file', type=str, default='../data/valid.tsv',
                    help='TSV file containing validation data')
parser.add_argument('--test-file', type=str, default='../data/test.tsv',
                    help='TSV file containing test data')
args = parser.parse_args()

def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: Average loss
    """
    batches = int(len(train_input) / model.batch_size)
    b_size = model.batch_size * model.window_size
    total_loss = 0.0

    progress = tqdm(range(batches))
    progress.set_description('Training ' + str(type(model).__name__))
    for batch in progress:
        with tf.GradientTape() as tape:
            inputs = train_input[batch*model.batch_size:(batch+1)*model.batch_size]
            prbs = model.call(tf.convert_to_tensor(inputs), initial_state=None)
            labels = train_labels[batch*model.batch_size:(batch+1)*model.batch_size]

            loss = model.loss(prbs, labels)
            total_loss += loss
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss / batches

def test(model, test_input, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: accuracy of model predictions
    """
    batches = int(len(test_input) / model.batch_size)
    total_correct = 0.0
    total = 0.0
    total_square_error = 0.0

    for batch in range(batches):
        inputs = test_input[batch*model.batch_size:(batch+1)*model.batch_size]
        prbs = model.call(tf.convert_to_tensor(inputs), initial_state=None)
        guesses = tf.math.argmax(prbs, axis=1).numpy()

        labels = test_labels[batch*model.batch_size:(batch+1)*model.batch_size]


        for i in range(model.batch_size):
            total += 1
            total_square_error += (guesses[i] - labels[i]) ** 2
            if guesses[i]==labels[i]:
                total_correct += 1

    return total_correct / total, total_square_error / total


def tune_hyperparameters(vocab_size, num_labels, train_input, train_labels, validation_input, validation_labels):
    """
    Uses the training and validation data to

    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :returns: None
    """
    embedding_size_opts = [20, 40, 80, 160, 240]
    lr_opts = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]
    rnn_size_opts = [64, 128, 256, 512]

    best = None
    highest_acc = 0

    for eopt in embedding_size_opts:
        for lopt in lr_opts:
            for ropt in rnn_size_opts:
                print("Training with {}, {}, {}".format(eopt, lopt, ropt))
                # CHANGE TO BE THE MODEL THAT YOU WANT
                # Model also needs to be altered to take in values for hyperparams
                model = HybridModel(vocab_size, num_labels, eopt, lopt, ropt)
                for i in range(10):
                    train(model, train_input, train_labels)
                accuracy = test(model, validation_input, validation_labels)
                print("ACCURACY: {}".format(accuracy))
                if accuracy > highest_acc:
                    best = (eopt, lopt, ropt)
                    highest_acc = accuracy


    print('Best parameters: {}'.format(best))
    print("Best score: {}".format(highest_acc))

def main():
    train_input, train_labels, val_input, val_labels, test_input, test_labels, dictionary, labels, encoded_test_sentence = get_data(args.train_file, args.val_file, args.test_file, args.test_sentence)
    # tune_hyperparameters(len(dictionary), len(labels), train_input, train_labels, test_input, test_labels)
    # return

    if args.test_sentence == None:
        lmodel = LstmModel(len(dictionary), len(set(labels.values())))
        train(lmodel, train_input, train_labels)

        dmodel = LstmDropoutModel(len(dictionary), len(set(labels.values())))
        train(dmodel, train_input, train_labels)

    hmodel = HybridModel(len(dictionary), len(set(labels.values())))
    train(hmodel, train_input, train_labels)

    if args.test_sentence == None:
        lval, _ = test(lmodel, val_input, val_labels)
        print("LSTM VALIDATION ACCURACY: {}".format(lval))
        laccuracy, lmse = test(lmodel, test_input, test_labels)
        print("LSTM TEST ACCURACY: {}".format(laccuracy))
        print("LSTM TEST MSE: {}".format(lmse))
        print('----------')

        dval, _ = test(dmodel, val_input, val_labels)
        print("LSTMDROPOUT VALIDATION ACCURACY: {}".format(dval))
        daccuracy, dmse = test(dmodel, test_input, test_labels)
        print("LSTMDROPOUT TEST ACCURACY: {}".format(daccuracy))
        print("LSTMDROPOUT TEST MSE: {}".format(dmse))
        print('----------')

        hval, _ = test(hmodel, val_input, val_labels)
        print("HYBRID VALIDATION ACCURACY: {}".format(hval))
        haccuracy, hmse = test(hmodel, test_input, test_labels)
        print("HYBRID TEST ACCURACY: {}".format(haccuracy))
        print("HYBRID TEST MSE: {}".format(hmse))

        print('----------BENCHMARKS----------')
        rand_val_predictions = [randint(0, len(set(labels.values()))-1) for _ in range(len(val_labels))]
        rand_test_predictions = [randint(0, len(set(labels.values()))-1) for _ in range(len(test_labels))]
        print("RANDOM VALIDATION ACCURACY: {}".format(len([x for x,y in zip(val_labels,rand_val_predictions) if x == y])/len(val_labels)))
        print("RANDOM TEST ACCURACY: {}".format(len([x for x,y in zip(test_labels,rand_test_predictions) if x == y])/len(test_labels)))
        print("RANDOM TEST MSE: {}".format(np.sum(np.square(np.subtract(rand_test_predictions, test_labels)))/len(test_labels)))
        print('----------')

        val_label_counts = []
        test_label_counts = []
        for label in sorted(list(set(labels.values()))):
            val_label_counts.append(len([x for x in val_labels if x == label]))
            test_label_counts.append(len([x for x in test_labels if x == label]))
        val_maj_label = np.argmax(val_label_counts)
        test_maj_label = np.argmax(test_label_counts)
        maj_val_predictions = [val_maj_label] * len(test_labels)
        maj_test_predictions = [test_maj_label] * len(test_labels)
        print("MAJORITY VALIDATION ACCURACY: {}".format(len([x for x,y in zip(val_labels,maj_val_predictions) if x == y])/len(val_labels)))
        print("MAJORITY TEST ACCURACY: {}".format(len([x for x,y in zip(test_labels,maj_test_predictions) if x == y])/len(test_labels)))
        print("MAJORITY TEST MSE: {}".format(np.sum(np.square(np.subtract(maj_test_predictions, test_labels)))/len(test_labels)))

    print('----------HYBRID MODEL RESULTS----------')
    reverse_dictionary = {v:k for k,v in dictionary.items()}
    reverse_labels = {v:k for k,v in labels.items()}
    if encoded_test_sentence != None:
        sentence = [reverse_dictionary[word_id] for word_id in encoded_test_sentence]
        sentence = [word for word in sentence if word != STOP_TOKEN]
        sentence = [word for word in sentence if word != PAD_TOKEN]
        prbs = hmodel.call(tf.convert_to_tensor([encoded_test_sentence]), initial_state=None)
        guess = tf.math.argmax(prbs, axis=1).numpy()[0]
        print('Sentence:', ' '.join(sentence))
        print('Predicted Label:', reverse_labels[guess])
    else:
        examples = list(sample(range(len(test_input)), k=10))
        for index in examples:
            sentence = [reverse_dictionary[word_id] for word_id in test_input[index]]
            sentence = [word for word in sentence if word != STOP_TOKEN]
            sentence = [word for word in sentence if word != PAD_TOKEN]
            prbs = hmodel.call(tf.convert_to_tensor(test_input), initial_state=None)
            guesses = tf.math.argmax(prbs, axis=1).numpy()
            print('Sentence:', ' '.join(sentence))
            print('True Label:', reverse_labels[test_labels[index]])
            print('Predicted Label:', reverse_labels[guesses[index]])
            print('----------')

if __name__ == '__main__':
    main()
