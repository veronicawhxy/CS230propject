#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[3]:


#set up parameters
target_side_len = 224
batch_size = 64
epochs = 100
optimizer_choice = Adam
learning_rate = 0.00005
# drop_rate = 0.5


# In[4]:


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


# In[5]:


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


# In[6]:


def f1(y_true, y_pred): 
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2*((precision_value * recall_value)/(precision_value + recall_value + K.epsilon()))


# In[7]:


train_datagen = ImageDataGenerator(
#         rescale=1./255,
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


# In[8]:


# a = train_generator.next()


# In[9]:


# initial_model = VGG16(include_top=False, weights='imagenet', pooling='max', input_shape = (224,224,3))
# initial_model = VGG16(include_top=False, pooling='max', input_shape = (224,224,3))
initial_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
# for layer in initial_model.layers[:-4]:
#     layer.trainable = False
# for layer in initial_model.layers:
#     print(layer, layer.trainable)


# In[10]:


# Create the model
model = Sequential()
model.add(initial_model)


# In[11]:


# Add new layers
# model.add(Flatten())
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.2))


# In[12]:


model.add(MaxPooling2D(7))
model.add(Flatten())


# In[13]:

# model.add(Dropout(rate = drop_rate))
model.add(Dense(64))
model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.2))


# In[14]:


model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[15]:


# my_optimizer = optimizer_choice(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
my_optimizer = optimizer_choice(lr=learning_rate)
model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['accuracy', f1, precision, recall])


# In[16]:


csv_logger = CSVLogger('ResNet50_results_nofreeze_nodrop_00005lr.csv', append=False, separator=';')

filepath="current_model_weights/Resnet50_weights_nodrop_00005lr-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# In[ ]:


model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples//train_generator.batch_size,
        epochs=epochs,
        callbacks=[csv_logger,checkpoint],
        validation_data=dev_generator,
        validation_steps=dev_generator.samples//dev_generator.batch_size, 
        verbose=1)

model.save('ResNet_new_nodropout_00005lr_weight.h5')

# In[ ]:


#model.summary()

