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
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

# define f1 score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

target_side_len = 224
batch_size = 16
epochs = 100

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

initial_model = VGG16(include_top=False, weights='imagenet', pooling='max')
input = Input(shape=(224, 224, 3), name='image_input')
x = Flatten()(initial_model(input))
x = Dense(200, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
model = Model(inputs=input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])

csv_logger = CSVLogger('VGG16_results.csv', append=True, separator=';')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=epochs,
        callbacks=[csv_logger],
        validation_data=dev_generator,
        validation_steps=800)
