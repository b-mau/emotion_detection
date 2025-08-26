from PIL import Image
import numpy as np

import glob

image_list = []  # Stores 1d array of images as list
labels = []  # Stores label of image as a number from 0 to 6


for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/angry*.jpg'):
    im = Image.open(filename)  # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(0)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/disgust*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(1)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/fear*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(2)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/happy*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(3)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/neutral*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(4)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/sad*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(5)  # Append number to labels list

for filename in glob.glob('C:/Users/bertr/archive (2)/images/train/surprise*.jpg'):
    im = Image.open(filename) # Opens image
    image_list.append(im.flatten())  # Turn image into 1d array
    labels.append(6)  # Append number to labels list

image_list = np.asarray(image_list)
labels = np.asarray(labels)
