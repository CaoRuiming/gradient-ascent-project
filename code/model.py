import tensorflow as tf
import numpy as np

class LstmModel(tf.keras.Model):
    def __init__(self, vocab_size, num_labels):

        super(LstmModel, self).__init__()

        self.learning_rate = 0.001
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.window_size = 30
        self.embedding_size = 160
        self.batch_size = 64
        self.rnn_size = 128

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_size
        )
        self.lstm = tf.keras.layers.LSTM(
            self.rnn_size,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(
            self.num_labels,
            activation='softmax'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, initial_state=None):
        """
        Performs a forward pass on given `inputs` and `initial_state`.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :param training: boolean representing if call is during training
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
        """
        x = self.embedding(inputs)
        x, _, _ = self.lstm(x, initial_state=initial_state)
        x = tf.reshape(x, [x.shape[0], self.window_size*self.rnn_size])
        x = self.dense(x)
        return x

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, num_labels) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probs
            )
        )

class LstmDropoutModel(tf.keras.Model):
    def __init__(self, vocab_size, num_labels):

        super(LstmDropoutModel, self).__init__()

        self.learning_rate = 0.001
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.window_size = 30
        self.embedding_size = 160
        self.batch_size = 64
        self.rnn_size = 256

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_size
        )
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.lstm = tf.keras.layers.LSTM(
            self.rnn_size,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(
            self.num_labels,
            activation='softmax'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, initial_state=None, training=False):
        """
        Performs a forward pass on given `inputs` and `initial_state`.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :param training: boolean representing if call is during training
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
        """
        x = self.embedding(inputs)
        x = self.dropout(x, training)
        x, _, _ = self.lstm(x, initial_state=initial_state)
        x = tf.reshape(x, [x.shape[0], self.window_size*self.rnn_size])
        x = self.dense(x)
        return x

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probs
            )
        )

class HybridModel(tf.keras.Model):
    def __init__(self, vocab_size, num_labels):

        super(HybridModel, self).__init__()

        self.learning_rate = 0.001
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.window_size = 30
        self.embedding_size = 240
        self.batch_size = 64
        self.rnn_size = 256

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_size
        )
        self.conv = tf.keras.layers.Conv1D(
            self.embedding_size, # dimension of output
            5, # size of kernel
            2,
            padding='SAME'
        )
        self.max_pool = tf.keras.layers.MaxPool1D(
            pool_size=2,
            padding='VALID'
        )
        self.lstm = tf.keras.layers.LSTM(
            self.rnn_size,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(
            self.num_labels,
            activation='softmax'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs, initial_state=None, training=False):
        """
        Performs a forward pass on given `inputs` and `initial_state`.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :param training: boolean representing if call is during training
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
        """
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.max_pool(x)
        x, _, _ = self.lstm(x, initial_state=initial_state)
        x = tf.reshape(x, [x.shape[0], x.shape[1]*x.shape[2]])
        x = self.dense(x)
        return x

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probs
            )
        )
