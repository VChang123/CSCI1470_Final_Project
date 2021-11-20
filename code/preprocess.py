import os
import re
import random
import argparse
import numpy as np
from PIL import Image
import csv
import imageio

def load_data(resize):
    # Find all the Image Files
    image_files = open("./data/groundtruth_train.tsv")
    read_tsv = csv.reader(image_files, delimiter="\t")

    images = []
    
    for row in read_tsv:
        #image path
        path = row[0]
        path = path + ".png"
        image_path = os.path.join("./data/train/", path)

        # Open the Image
        img = Image.open(image_path)
        # print(img)
        
        # Rescale the Image to the Appropriate Size
        img = img.resize(resize)

        # Convert from ints to floats
        img = np.array(img)

        images.append(img)
        break

    images = np.asarray(images)

    # Convert from ints to floats
    images = images.astype('float32')

    # Scale from [0,255] to [-1,1]
    images = images/255

    return images

    
if __name__ == '__main__':
   output = load_data(resize=(32,32))
   for i in output:
       for j in i:
           print(j)
   
  
