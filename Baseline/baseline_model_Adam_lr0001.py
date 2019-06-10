import os
import sys
import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageio import imread
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

target_side_len = 150
batch_size = 16
epochs = 50

def train(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
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
    
    csv_logger = CSVLogger('result_adam_default.csv', append=True, separator=';')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=epochs,
        callbacks=[csv_logger],
        validation_data=dev_generator,
        validation_steps=800)
    
def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    # model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))

    model.add(Activation('sigmoid'))
    
    return model

model = build_model()
# my_optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.1)
my_optimizer = Adam(lr=0.0005,  beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False )
model.compile(optimizer= my_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
train(model)
