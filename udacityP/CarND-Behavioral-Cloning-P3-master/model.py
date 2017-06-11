import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

import helper
from generators import *

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as tf
#tf.python.control_flow_ops = tf

number_of_epochs = 8
number_of_samples_per_epoch = 2000
number_of_validation_samples = 640
learning_rate = 1e-4
activation_relu = 'relu'

# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

#model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3), output_shape=(64, 64, 3)))

model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=( 160, 320,ch),
        ))
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(16, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(28, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(40, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

# create two generators for training and validation
#train_gen = helper.generate_next_batch()
#validation_gen = helper.generate_next_batch()

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

'''
history=model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)
'''
# finally save our model and weights
model.save('model5.h5')
