from __future__ import absolute_import
import csv
from matplotlib import pyplot as plt
from numpy.core.arrayprint import format_float_positional
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_nn_ops import MaxPool
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Dropout, Activation, BatchNormalization
# from tf.keras.layers.BatchNormalization import BatchNormalization
from tensorflow.math import exp, sqrt, square
import numpy as np
import random
import math
# import tensorflow.keras as keras
# # from keras.datasets import mnist
# from keras.layers import Dense, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Activation
# from keras.layers import Dropout

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.num_classes = 101
        self.batch_size = 250
        self.num_epochs = 10
        self.hidden_dim = 500

        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
    
        #optimizer
        self.optimization = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Can change no. of filers and blocks

        self.architecture = [
                Conv2D(32,5,1,padding="same",
                   activation="relu", name="block1_conv1"),
                Conv2D(32,5,1,padding="same",
                    activation="relu", name="block1_conv2"),
                MaxPool2D(2, name="block1_pool"),
                Conv2D(128,5,1,padding="same",
                   activation="relu", name="block2_conv1"),
                Conv2D(128,5,1,padding="same",
                   activation="relu", name="block2_conv2"),
                MaxPool2D(2, name="block2_pool"),
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

        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels 
        
        :param logits: 
        :param labels: 

        
        :return: the accuracy of the model as a Tensor
        """

        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
