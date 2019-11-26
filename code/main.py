import numpy as np
from model import LstmModel, LstmDropoutModel, HybridModel
from preprocess import get_data
from sklearn.model_selection import RandomizedSearchCV

def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    windows = int(len(train_labels) / model.window_size)
    batches = int(windows / model.batch_size)
    b_size = model.batch_size * model.window_size

    for batch in range(batches):
        with tf.GradientTape() as tape:
            inputs = train_input[batch*b_size:(batch+1)*b_size]
            inputs = tf.reshape(inputs, [model.batch_size, model.window_size])
            prbs = model.call(inputs, initial_state=None, training=True)

            labels = train_labels[batch*b_size:(batch+1)*b_size]
            labels = tf.reshape(labels, [model.batch_size, model.window_size])

            loss = model.loss(prbs, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_input, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: accuracy of model predictions
    """
    windows = int(len(test_labels) / model.window_size)
    batches = int(windows / model.batch_size)
    b_size = model.batch_size * model.window_size
    total_correct = 0.0

    for batch in range(batches):
        inputs = test_input[batch*b_size:(batch+1)*b_size]
        inputs = tf.reshape(inputs, [model.batch_size, model.window_size])
        prbs = model.call(inputs, initial_state=None, training=True)

        labels = test_labels[batch*b_size:(batch+1)*b_size]
        labels = tf.reshape(labels, [model.batch_size, model.window_size])

        # sum the ones that are incorrect and return this

def tune_hyperparameters(vocab_size, train_input, train_labels, validation_input, validation_labels):
    """
    Uses the training and validation data to

    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :returns: None
    """
    # Tune the embedding size, learning rate, and RNN size
    model = LstmModel(vocab_size)
    fit_params = {
        'epochs': 15,
        'batch_size': 64,
        'validation_data': (validation_input, validation_labels)
    }

    embedding_size_opts = [20, 40, 80, 160, 240]
    lr_opts = [0.01, 0.001, 0.005, 0.0001, 0.0005]
    rnn_size_opts = [64, 128, 256, 512]

    params_options = {
        'embedding_size': embedding_size_opts,
        'learning_rate': lr_opts,
        'rnn_size': rnn_size_opts
    }

    rs = RandomizedSearchCV(
        model,
        param_distributions = params_options,
        fit_params = fit_params
    )

    rs.fit(train_input, train_labels)

def main():
    print('hello, world!')
    train_input, train_labels, test_input, test_labels = # get data somehow
    model = #pick a model
    train(model, train_input, train_labels)
    test(model, test_input, test_labels)

if __name__ == '__main__':
    main()
