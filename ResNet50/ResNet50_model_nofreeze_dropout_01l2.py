#!/usr/bin/env python
# coding: utf-8

import os
import sys
import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageio import imread
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#set up parameters
target_side_len = 224
batch_size = 64
epochs = 100
optimizer_choice = Adam
learning_rate = 0.0001
drop_rate = 0.5
l2_rate = 0.01

# define f1 score
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1(y_true, y_pred): 
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2*((precision_value * recall_value)/(precision_value + recall_value + K.epsilon()))


train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda x: preprocess_input(x, mode='tf'))
    
    
test_datagen = ImageDataGenerator(preprocessing_function=lambda x: preprocess_input(x, mode='tf'))
    
train_generator = train_datagen.flow_from_directory(
    './dataset/train',
    target_size=(target_side_len, target_side_len),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='binary')

dev_generator = test_datagen.flow_from_directory(
    './dataset/dev',
    target_size=(target_side_len, target_side_len),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='binary')



initial_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

# Create the model
model = Sequential()
model.add(initial_model)
model.add(MaxPooling2D(7))
model.add(Flatten())
model.add(Dropout(rate = drop_rate))
model.add(Dense(64,kernel_regularizer=l2(l2_rate),
                activity_regularizer=l1(l2_rate)))
model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation('sigmoid'))

my_optimizer = optimizer_choice(lr=learning_rate)
model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy', f1, precision, recall])

csv_logger = CSVLogger('ResNet50_results_nofreeze_5drop_regularizer.csv', append=False, separator=';')

filepath="current_model_weights/Resnet50_new_weights_5drop_regu-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples//train_generator.batch_size,
        epochs=epochs,
        callbacks=[csv_logger,checkpoint],
        validation_data=dev_generator,
        validation_steps=dev_generator.samples//dev_generator.batch_size, 
        verbose=1)

model.save('ResNet_new_5dropout_regularizer_weight.h5')

