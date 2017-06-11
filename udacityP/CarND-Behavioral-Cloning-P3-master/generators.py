import os
import csv
from random import shuffle
samples = []
with open('E:/train/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import helper

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                name = 'E:/train/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                #print(batch_sample[3])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #new_image, new_angle = helper.generate_new_image(center_image, center_angle)
                #images.append(new_image)
                #angles.append(new_angle)
            # trim image to only see section with road
            print(len(images))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

ch, row, col = 3, 160, 320  # Trimmed image format