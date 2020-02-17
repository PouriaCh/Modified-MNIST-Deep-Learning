# We have all the models in this file
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, ActivityRegularization
import tensorflow as tf
# Importing the required Keras modules containing model and layers
def model1():
    # Creating a Sequential Model and adding the layers
    model = Sequential()

    model.add(Conv2D(128, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.10))

    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(4, 4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation=tf.nn.softmax))  # Since we have ten classes

    return model


# Importing the required Keras modules containing model and layers
def model2():
    # Creating a Sequential Model and adding the layers
    model = Sequential()

    model.add(Conv2D(128, kernel_size=(7, 7)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation=tf.nn.softmax))  # Since we have ten classes

    return model


# Importing the required Keras modules containing model and layers
def model3():
    # Creating a Sequential Model and adding the layers
    model = Sequential()

    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=tf.nn.softmax))  # Since we have ten classes

    return model