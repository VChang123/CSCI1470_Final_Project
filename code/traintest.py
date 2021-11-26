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
from preprocess import load_data
from model import Model

def segmentation(image):
    pass

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    #creates a and index range
    num_examples = np.arange(train_inputs.shape[0])
    #shuffles the inputs
    random_index = tf.random.shuffle(num_examples)
    #gets the random inputs
    random_inputs = tf.gather(train_inputs, random_index)
    #gets the random labels
    random_labels = tf.gather(train_labels, random_index)
  
    #loop through data in batches
    for i in range(0, random_inputs.shape[0], model.batch_size):
        if(i + model.batch_size > random_inputs.shape[0]):
            break
        
        #maybe change the shape of the inputs depending on what they are
        batch_input = random_inputs[i : i + model.batch_size, :, :, :]
        batch_label = random_labels[i : i + model.batch_size]

        #calculated logit and loss
        with tf.GradientTape() as tape:
            logits = model.call(batch_input, is_testing = False)
            loss = model.loss(logits, batch_label)
            model.loss_list.append(loss)

        #gets the gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        #applies the optimizer
        model.optimization.apply_gradients(zip(gradients, model.trainable_variables))



def test_characters(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    accuracy = 0
    num_batches = 0

    #loop through the data in batches
    for i in range(0, test_inputs.shape[0], model.batch_size):
        if(i + model.batch_size > test_inputs.shape[0]):
            break
        #get the batched inputs and labels
        #change the shape of the inputs depending on the size of the inputs
        batch_input = test_inputs[i : i + model.batch_size, :, :,:]
 
        batch_label = test_labels[i : i + model.batch_size]

        #get logits and calculated accuracy
        logits = model.call(batch_input, is_testing = True)
        accuracy += model.accuracy(logits, batch_label)
        num_batches+=1

    return accuracy/num_batches


def test_expressions(model, test_inputs, test_labels):
    accuracy = 0
    num_batches = 0

    #segement all the images

    #loop through the data in batches
    for i in range(0, test_inputs.shape[0], model.batch_size):
        if(i + model.batch_size > test_inputs.shape[0]):
            break
        #get the batched inputs and labels
        batch_input = test_inputs[i : i + model.batch_size, :, :, :]
 
        batch_label = test_labels[i : i + model.batch_size]

        #get logits and calculated accuracy
        logits = model.call(batch_input, is_testing = True)
        #create a fucntion that measures accuracy for expressions
        accuracy += model.accuracy(logits, batch_label)
        num_batches+=1

    return accuracy/num_batches

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 


    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"



    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    pass



if __name__ == '__main__':
    main()