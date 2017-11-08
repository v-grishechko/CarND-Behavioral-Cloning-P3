import csv
import cv2
import numpy as np

samples = []

with open('data_set/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'data_set/IMG/' + batch_sample[0].split('/')[-1]                
                left_name = 'data_set/IMG/' + batch_sample[1].split('/')[-1]
                right_name = 'data_set/IMG/' + batch_sample[2].split('/')[-1]
                #Images
                center_image = process_image(cv2.imread(center_name))
                left_image = process_image(cv2.imread(left_name))
                right_image = process_image(cv2.imread(right_name))
                #Angles
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
               
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle) 

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def process_image(image, top = 70, bottom = 25):
    return image

# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

ch, row, col = 3, 160, 320  # Trimmed image format
epoch = 5

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D

#Model
print(keras.__version__)
model = Sequential()
model.load_weights("transfer.h5")
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (1, 1))))
model.add(Conv2D(24, 5, 5,  subsample=(2,2), activation='relu'))
print(model.layers[-1].output_shape)
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
print(model.layers[-1].output_shape)
model.add(Conv2D(48, 3, 3, activation='relu'))
print(model.layers[-1].output_shape)
model.add(Conv2D(64, 3, 3, activation='relu'))
print(model.layers[-1].output_shape)
model.add(Flatten())

print(model.layers[-1].output_shape)
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),\
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
