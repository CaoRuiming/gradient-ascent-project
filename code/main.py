import numpy as np
from model import LstmModel, LstmDropoutModel, HybridModel
from preprocess import get_data
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

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

    for batch in range(batches):
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

    for batch in range(batches):
        inputs = test_input[batch*model.batch_size:(batch+1)*model.batch_size]
        prbs = model.call(tf.convert_to_tensor(inputs), initial_state=None)
        guesses = tf.math.argmax(prbs, axis=1)

        labels = test_labels[batch*model.batch_size:(batch+1)*model.batch_size]

        for i in range(model.batch_size):
            total += 1
            if guesses[i]==labels[i]:
                total_correct += 1

    return total_correct / total


def tune_hyperparameters(vocab_size, num_labels, train_input, train_labels, validation_input, validation_labels):
    """
    Uses the training and validation data to

    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :returns: None
    """
    """
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0.05, patience = 5)

    callback = [es]

    # Tune the embedding size, learning rate, and RNN size
    fit_params = {
        'callbacks': callback,
        'epochs': 20,
        'batch_size': 64,
        'validation_data': (validation_input, validation_labels),
        'verbose': 0
    }

    embedding_size_opts = [20, 40, 80, 160, 240]
    lr_opts = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]
    rnn_size_opts = [128, 256, 512]

    params_options = {
        'embedding_size': embedding_size_opts,
        'learning_rate': lr_opts,
        'rnn_size': rnn_size_opts
    }

    rs = RandomizedSearchCV(
        model,
        param_distributions = params_options,
        fit_params = fit_params,
        scoring = 'accuracy',
        cv = 3,
        verbose = 1
    )

    print(train_labels)
    print(len(train_labels))
    rs.fit(train_input, train_labels)
    """

    embedding_size_opts = [20, 40, 80, 160, 240]
    lr_opts = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]
    rnn_size_opts = [128, 256, 512]

    best = None
    highest_acc = 0

    for eopt in embedding_size_opts:
        for lopt in lr_opts:
            for ropt in rnn_size_opts:
                print("Training {}, {}, {}".format(eopt, lopt, ropt))
                model = LstmModel(vocab_size, num_labels, eopt, lopt, ropt)
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
    # Currently loads up the validation file, but can change it to load the test file
    # I tried both and they had the same issue
    train_input, train_labels, test_input, test_labels, dictionary, labels = get_data('../data/train.tsv', '../data/valid.tsv')
    #print("THIS MODEL IS CURRENTLY IN HYPERPARAMETER TUNING MODE")
    #tune_hyperparameters(len(dictionary), len(labels), train_input, train_labels, test_input, test_labels)
    model = LstmModel(len(dictionary), len(labels), 40, 0.001, 256)
    for i in range(20):
        loss = train(model, train_input, train_labels)
        print("AVG LOSS: {}".format(loss))
        accuracy = test(model, test_input, test_labels)
        print("ACCURACY: {}".format(accuracy))

if __name__ == '__main__':
    main()
