import numpy as np
import cv2
import json
import pandas as pd
import time

import os
import matplotlib.pyplot as plt

import PIL.Image as Image

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
BASE_DIR = "./cassava-leaf-disease-classification/"
#%%
new_model = tf.keras.models.load_model(('my_keras_model[3].h5'),custom_objects={'KerasLayer':hub.KerasLayer})
#new_model = tf.keras.models.load_model('my_keras_model[3].h5')
#%%
filename = 'finalized_model.sav'
pickle.dump(new_model, open(filename, 'wb'))
#%%
preds = []
ss = pd.read_csv(BASE_DIR + 'sample_submission.csv')
#%%
for image in ss.image_id:
    img = keras.preprocessing.image.load_img(BASE_DIR + 'test_images/' + image)    
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, [192, 192])
    img = np.expand_dims(img, 0)
    prediction = new_model.predict(img)
    preds.append(np.argmax(prediction))
#%%
my_submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})
my_submission.to_csv('my_submission.csv', index=False) 