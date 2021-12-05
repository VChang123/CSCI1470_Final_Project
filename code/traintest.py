from __future__ import absolute_import
from matplotlib import pyplot as plt
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_math_ops import exp
from tensorflow.python.ops.gen_nn_ops import MaxPool
import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_data
from model import Model

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
    # #creates a and index range
    # num_examples = np.arange(train_inputs.shape[0])
    # #shuffles the inputs
    # random_index = tf.random.shuffle(num_examples)
    # #gets the random inputs
    # random_inputs = tf.gather(train_inputs, random_index)
    # #gets the random labels
    # random_labels = tf.gather(train_labels, random_index)
  
    #loop through data in batches
    accuracy = 0
    j = 0
    for i in range(0, train_inputs.shape[0], model.batch_size):
        if(i + model.batch_size > train_inputs.shape[0]):
            break
        #maybe change the shape of the inputs depending on what they are
        batch_input = train_inputs[i : i + model.batch_size]
        batch_label = train_labels[i : i + model.batch_size]
        # print(batch_label)
        # print(batch_input)

        #calculated logit and loss
        with tf.GradientTape() as tape:
            logits = model.call(batch_input)
            loss = model.loss(logits, batch_label)
            
            accuracy += model.accuracy(logits, batch_label)
            model.loss_list.append(loss)

        #gets the gradient
        gradients = tape.gradient(loss, model.trainable_variables)
        #applies the optimizer
        model.optimization.apply_gradients(zip(gradients, model.trainable_variables))
        j+=1

    return model.loss_list, accuracy/j



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
    for i in range(0, len(test_inputs), model.batch_size):
        if(i + model.batch_size > len(test_inputs)):
            break
        #get the batched inputs and labels
        #change the shape of the inputs depending on the size of the inputs
        batch_input = test_inputs[i : i + model.batch_size]
 
        batch_label = test_labels[i : i + model.batch_size]

        #get logits and calculated accuracy
        logits = model.call(batch_input)
        accuracy += model.accuracy(logits, batch_label)
        num_batches+=1

    return accuracy/num_batches


def test_expressions(model, test_inputs, test_labels):
    """
    FIX THIS FUNCTION LATER
    """
    accuracy = 0

    #segement all the images

    #loop through the data in batches

    for i in range(len(test_inputs)):
        #get the batched inputs and labels

        expression_input = test_inputs[i]
        expression_input = np.array(expression_input)
        expression_input = np.reshape(expression_input, (-1,32,32,1))
        expression_label = test_labels[i]

        #get logits and calculated accuracy
        logits = model.call(expression_input)
        #create a fucntion that measures accuracy for expressions
        accuracy += model.accuracy(logits, expression_label)

    return accuracy/len(test_inputs)

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
    #get and load data
    train_inputs, train_labels, test_inputs, test_labels, test_char_inputs, test_char_labels = get_data()

    

    # print(train_inputs.shape)
    # print(train_labels.shape)


    print("Preprocessing Completed!")

    model = Model()

    #train model
    for i in range(5):
        loss_list, accuracy = train(model, train_inputs, train_labels)
        print("Epoch",i , " ", accuracy)
        print("Loss:", tf.reduce_mean(model.loss_list))
    visualize_loss(loss_list)

    print("Accuracy for Training", accuracy)
    
    # test model on characters

    acc_1 = test_characters(model, test_char_inputs, test_char_labels)
     
    print("Accuracy for testing characters: ", acc_1)

    acc = test_expressions(model, test_inputs, test_labels)

    print("Accuracy for testing expression", acc)
    # test model from expression
    pass



if __name__ == '__main__':
    main()