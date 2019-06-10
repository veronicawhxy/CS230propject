import os
import sys
import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageio import imread
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#set up parameters
target_side_len = 224
batch_size = 16
epochs = 100
optimizer_choice = Adam
learning_rate = 0.0001

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

initial_model = ResNet50(include_top=False, weights='imagenet',input_tensor=None, input_shape=(224,224, 3))

output = initial_model.output 
output = Flatten(name='flatten')(output)
output = Dense(256, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(1, activation='sigmoid')(output)

for layer in initial_model.layers[:25]:
    layer.trainable = False
    
model = Model(inputs=initial_model.input, outputs=output)
my_optimizer = optimizer_choice(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy', f1])

csv_logger = CSVLogger('ResNet50_results.csv', append=True, separator=';')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=epochs,
        callbacks=[csv_logger],
        validation_data=dev_generator,
        validation_steps=800)
