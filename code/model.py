import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.learning_rate = 0.01
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 40
        self.batch_size = 64
        self.rnn_size = 256

        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.Embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_size
        )
        self.LSTM = tf.keras.layers.LSTM(
            self.rnn_size,
            return_sequences=True,
            return_state=True
        )
        self.Dense = tf.keras.layers.Dense(
            self.vocab_size,
            activation='softmax'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs, initial_state=None):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
       
        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        x = self.Embedding(inputs)
        x, _, _ = self.LSTM(x, initial_state=initial_state)
        x = self.Dense(x)

        return x

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        # tf.keras.losses.sparse_categorical_crossentropy()
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels,
                probs
            )
        )
