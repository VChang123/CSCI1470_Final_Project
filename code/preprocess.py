import os
import re
import random
import argparse
import numpy as np
from PIL import Image
import csv
import imageio
from extract import Extractor
from one_hot import encode

# def load_categories():
    
#     with open('data/categories.txt', 'r') as desc:
        
#         lines = desc.readline()
#         categories = lines.split(" ")
        
#         return categories

# def load_data(resize):
#     # Find all the Image Files
#     image_files = open("./data/groundtruth_train.tsv")
#     read_tsv = csv.reader(image_files, delimiter="\t")

#     symbols = load_categories()
#     # print(symbols)

#     images = []

#     a = 0
    
#     for row in read_tsv:
#         #image path
#         path = row[0]
#         path = path + ".png"
#         image_path = os.path.join("./data/train/", path)

#         # Open the Image
#         img = Image.open(image_path)
#         # print(img)
        
#         # Rescale the Image to the Appropriate Size
#         img = img.resize(resize)

#         # Convert from ints to floats
#         img = np.array(img)

#         images.append(img)

#         # label = row[1]

#         # label = label.split()

#         # final_label = []

#         # for i in label:
#         #     if i in symbols:
#         #         final_label.append(i)
#         #     else:
#         #         j = re.sub("\_*", " " , i)
#         #         j = re.sub("\{*", " ", j)
#         #         j = re.sub("\}*", " ", j)
#         #         j = re.sub("\{*", " ", j)
                    
#         # print(final_label)
#         # if a == 1:
#         #     return

#         # a+=1




#         # # arr1 = label.split()
#         # # labels = []
#         # # for i in arr1:
#         # #     if i in symbols:
#         # # return



#     images = np.asarray(images)

#     # Convert from ints to floats
#     images = images.astype('float32')

#     # Scale from [0,255] to [-1,1]
#     images = images/255

#     return images

def get_data():
    extractor = Extractor(32, "2012")
    train_data, test_data, test_data_char = extractor.pixels()

    train_inputs = []
    train_labels = []

    test_inputs = []
    test_labels = []

    test_inputs_char = []
    test_labels_char = []


    for i in train_data:
        train_inputs.append(i['features'])
        train_labels.append(i['label'])

    train_inputs = np.array(train_inputs)
    train_inputs = np.reshape(train_inputs, (-1,32,32,1))

    train_labels = [encode(train_label, extractor.classes) for train_label in train_labels]
    train_labels = np.asarray(train_labels)

    for i in test_data:
        test_input = []
        test_label = []

        for j in i:
            test_input.append(j['features'])
            test_label.append(j['label'])
        
        test_label = np.array(test_label)
        test_label = [encode(test_i, extractor.classes) for test_i in test_label]

        test_inputs.append(test_input)
        test_labels.append(test_label)

    for i in test_data_char:
        test_inputs_char.append(i['features'])
        test_labels_char.append(i['label'])

    test_inputs_char = np.array(test_inputs_char)
    test_inputs_char = np.reshape(test_inputs_char, (-1,32,32,1))

    test_labels_char = [encode(test_label_char, extractor.classes) for test_label_char in test_labels_char]
    test_labels_char = np.array(test_labels_char)

    return train_inputs, train_labels, test_inputs, test_labels, test_inputs_char, test_labels_char
    
if __name__ == '__main__':
    train_inputs, train_labels, test_inputs, test_labels, test_inputs_char, test_labels_char = get_data()
  
