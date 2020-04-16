
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
import os
import random

# Reading file 'labels.csv.
# Please read file labels to get full understanding of what is being loaded.
df = pd.read_csv('labels.csv')

# Test of the loaded DataFrame.
df.head()

# Storing the length of df in variable n and then printing it for visual purposes.
n = len(df)
print('The length of of DataFrame/n is: ', n, '\n')

# Creating a 'set' of the types of weapons we are training the model for and storing it in variable 'gun'
gun = set(df['gun'])
print('Printing the set of "gun": ', gun, '\n')

# Storing the length of the of the set into n_class.
n_class = len(gun)
print('The length of gun variable now stored in n_class', n_class, '\n')

# Giving a value to each class in a dictionary format.
class_to_num = dict(zip(gun, range(n_class)))
print('Printing class_to_num: ', class_to_num, '\n')
num_to_class = dict(zip(range(n_class), gun))
print('Printing num_to_class: ', num_to_class, '\n')

# Shaping data in to a 329 by 299 by 299 by 3 layer and the data type is an unsigned integer.
# This is where the layering effect is being utilized.
width = 299
X = np.zeros((n, width, width, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# Printing for example.
print('Printing X: ', X)

# creating all true where both are true.
try:
    for i in tqdm(range(n)):
        X[i] = cv2.resize(cv2.imread('Train/%s.png' % df['id'][i]), (width, width))
        y[i][class_to_num[df['gun'][i]]] = 1
        print('printing X[i]: ', X[i])
        print('printing y[i]: ', y[i])
except Exception as e:
    print(e)



from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

# 'get_features' is a method that receives models from inceptionv3 and xception, and X layer
# preprocessing the images assuring each image has some form of likeness so they may be compared.
def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=64, verbose=1)
    return features

# Calling 'get_features' and sending InceptionV3 Model with our 'X' Layer.
# InceptionV3 is a Deep Convolutional Architecture Developed by Google for
# image recognition.
inception_features = get_features(InceptionV3, X)


xception_features = get_features(Xception, X)
features = np.concatenate([inception_features, xception_features], axis=-1)

inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(features, y, batch_size=128, epochs=100, validation_split=0.1)

model.save("my_model4.h5")

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
print('h.history keys: ', h.history.keys())
plt.subplot(1, 2, 2)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()
