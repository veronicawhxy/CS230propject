#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#set up parameters
target_side_len = 224
batch_size = 16
epochs = 50
optimizer_choice = Adam
learning_rate = 0.0001


# In[ ]:


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


# In[ ]:


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


# In[ ]:


def f1(y_true, y_pred): 
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2*((precision_value * recall_value)/(precision_value + recall_value + K.epsilon()))


# In[ ]:


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


# In[ ]:


initial_model = ResNet50(include_top=False, weights='imagenet',input_tensor=None, input_shape=(224,224, 3))


# In[ ]:


for layer in initial_model.layers[:170]:
    layer.trainable = False
for layer in initial_model.layers[170:]:
    layer.trainable = True


# In[ ]:


#output = initial_model.output 
#output = Conv
#output = Flatten(name='flatten')(output)
#output = Dense(256, activation='relu')(output)
#output = BatchNormalization()(output)
# #output = Dropout(0.5)(output)
#output = Dense(1, activation='sigmoid')(output)


# In[ ]:


#for i, layer in enumerate(initial_model.layers):
#    print(i, layer.name)


# In[ ]:


model = Sequential()
model.add(initial_model)
#model.summary()


# In[ ]:


model.add(Conv2D(4096, (3, 3), padding='same', input_shape=initial_model.output.shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
#model.summary()


# In[ ]:


model.add(Conv2D(4096, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.summary()


# In[ ]:


model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
#model.summary()


# In[ ]:


model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))


# In[ ]:


#model.summary()


# In[ ]:


model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


#model.summary()


# In[ ]:


my_optimizer = optimizer_choice(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy', f1, precision, recall])


# In[ ]:


# csv_logger = CSVLogger('ResNet50_results.csv', append=True, separator=';')
csv_logger = CSVLogger('ResNet50_results_withprere_dropout0.2_freeze170_topWithBaseline.csv', append=True, separator=';')


# In[ ]:


model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples//train_generator.batch_size,
        epochs=epochs,
        callbacks=[csv_logger],
        validation_data=dev_generator,
        validation_steps=dev_generator.samples//dev_generator.batch_size, 
        verbose=1)


# In[ ]:




