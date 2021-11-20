import csv
from __future__ import absolute_import
from matplotlib import pyplot as plt
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_nn_ops import MaxPool
import os
import tensorflow as tf
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

        self.batch_size = 100
        self.num_classes = 10
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.num_epochs = 10

        #optimizer
        self.optimization = tf.keras.optimizers.Adam(learning_rate=0.001)


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: 
        :return: logits 
        """

        pass

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        pass

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels 
        
        :param logits: 
        :param labels: 

        
        :return: the accuracy of the model as a Tensor
        """
        pass




def main():
    tsv_file = open('data/groundtruth_train.tsv')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    print(load_categories())
    # for row in read_tsv:
    #     print(row[0])

if __name__ == '__main__':
    main()
