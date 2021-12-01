from __future__ import absolute_import
import csv
from matplotlib import pyplot as plt
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_nn_ops import MaxPool
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Dropout
from tensorflow.math import exp, sqrt, square
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 1
        self.num_classes = 101 # or 75
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.num_epochs = 10
        self.hidden_dim = 32

        #optimizer
        self.optimization = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Can change no. of filers and blocks

        self.architecture = [
                Conv2D(32,5,1,padding="same",
                   activation="relu", name="block1_conv1"),
                # Conv2D(32,5,1,padding="same",
                #    activation="relu", name="block1_conv2"),
                MaxPool2D(2, name="block1_pool"),
                # Conv2D(64,5,1,padding="same",
                #    activation="relu", name="block2_conv1"),
                # Conv2D(64,5,1,padding="same",
                #    activation="relu", name="block2_conv2"),
                # MaxPool2D(2, name="block2_pool"),
                Flatten(),
                Dense(self.hidden_dim, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax") # can change to relu
       ]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: 
        :return: logits 
        """
        l = inputs
        for layer in self.architecture:
            l = layer(l)
            
        return l

    def loss(self, logits, labels): 
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        # print(logits.shape)
        # print(labels.shape)
        # print(logits)
        # print(labels)

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))
        #return tf.keras.losses.sparse_categorical_crossentropy(labels,predictions,from_logits=False)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels 
        
        :param logits: 
        :param labels: 

        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # print(logits)
        # print(labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    # def accuracy(self,logits, labels):
    #     """
    #     Calculates the model's accuracy by comparing the number 
    #     of correct predictions with the correct answers.
    #     :param probabilities: result of running model.call() on test inputs
    #     :param labels: test set labels
    #     :return: Float (0,1) that contains batch accuracy
    #     """
    #     # TODO: calculate the batch accuracy
    #     result = np.sum(np.argmax(logits, axis=1) == labels)
    #     return result/labels.shape[0]
