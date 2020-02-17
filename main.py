from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, ActivityRegularization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Adadelta
from models import *
from functions import *
# Data preprocessing and splitting

num_classes = 10
x_train = pd.read_pickle('./data/train_processed1.pkl')
# x_train = np.load('./datasets/train_processed1.npy')
# x_test = pd.read_pickle('./datasets/test_images.pkl')
Y_train = pd.read_csv('./data/train_labels.csv')
x_test = np.load('./data/test_processed1.npy')

# Preprocessing

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
x_train = np.asarray(x_train)
# x_train = preprocess(x_train)


# x_test = pd.read_pickle('./datasets/test_images.pkl')
# x_test = preprocess(x_test)

# chage csv format to numpy array for training labels

y_train = []
for i in range(x_train.shape[0]):
    y_train.append(Y_train.iloc[i]['Category'])

y_train = np.asarray(y_train)
y_train = y_train.reshape(np.shape(y_train)[0], 1)
y_train = keras.utils.to_categorical(y_train, num_classes)

# Reshaping the array to 4-dims so that it can work with the Keras API

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
X_Test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Splitting training data into Train-Dev

X_Training, X_Validation, Y_Training, Y_Validation = train_test_split(x_train, y_train, test_size=0.0625)

print('X_Training shape:', X_Training.shape)
print('X_Validation shape:', X_Validation.shape)
print('Y_Training shape:', Y_Training.shape)
print('Y_Validation shape:', Y_Validation.shape)
print('X_Test shape:', X_Test.shape)
print('Number of images in training set', X_Training.shape[0])
print('Number of images in validation set', X_Validation.shape[0])
print('Number of images in test set', X_Test.shape[0])

# Original mnist data
from keras.datasets import mnist

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
img_rows_mnist, img_cols_mnist = x_train_mnist.shape[1], x_train_mnist.shape[1]

x_train_mnist = x_train_mnist.astype('float32')
x_test_mnist = x_test_mnist.astype('float32')
x_train_mnist /= 255
x_test_mnist /= 255

y_train_mnist = keras.utils.to_categorical(y_train_mnist, num_classes)
y_test_mnist = keras.utils.to_categorical(y_train_mnist, num_classes)

train = np.zeros((x_train_mnist.shape[0], img_rows, img_cols))
test = np.zeros((x_test_mnist.shape[0], img_rows, img_cols))

for i in range(x_train_mnist.shape[0]):
    train[i, :, :] = np.pad(x_train_mnist[i, :, :], pad_width=18, mode='constant', constant_values=0)

for i in range(x_test_mnist.shape[0]):
    test[i, :, :] = np.pad(x_test_mnist[i, :, :], pad_width=18, mode='constant', constant_values=0)

x_train_mnist = train.reshape(x_train_mnist.shape[0], img_rows, img_cols, 1)
x_test_mnist = test.reshape(x_test_mnist.shape[0], img_rows, img_cols, 1)

print('x_train_mnist shape:', x_train_mnist.shape)
print(x_train_mnist.shape[0], 'mnist train samples')
print(x_test_mnist.shape[0], 'mnist test samples')

### Data Augmentation
augment_size = 30000
image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False)
# fit data for zca whitening
image_generator.fit(X_Training, augment=True)
# get transformed images
randidx = np.random.randint(X_Training.shape[0], size=augment_size)
x_augmented = X_Training[randidx].copy()
y_augmented = Y_Training[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                            batch_size=augment_size, shuffle=False).next()[0]
# append augmented data to trainset
X_Tr = np.concatenate((X_Training, x_augmented))
Y_Tr = np.concatenate((Y_Training, y_augmented))
print(X_Tr.shape[0])

######################################################################################

# Training a model from models.py

model = model3()
model.summary()
epochs = 20
batch_size = 128

model.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=X_Training, y=Y_Training, epochs = epochs, batch_size= batch_size)

# Evaluate the model on validation set
Y_Prediction = model.predict_classes(x= X_Validation, batch_size= batch_size)
Y_Prediction = keras.utils.to_categorical(Y_Prediction, num_classes)
print('')
print('Validation Accuracy for CNN is: ')
accuracy_score(Y_Validation, Y_Prediction)

#################################################################################
# in case we want to have the predictions on test set in a .csv file
# Test set prediction and creating csv for Kaggle submission

Y_Test = model.predict_classes(X_Test, batch_size= batch_size)
csvWriter(Y_Test,9)