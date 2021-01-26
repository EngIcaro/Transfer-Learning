#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:21:48 2021

@author: icaro
"""
#%%
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
# ESTE TRECHO DE CÓDIGO UTILIZA O CLASSIFICADOR JÁ TREINADO APENAS PARA FAZER CLASSIFICAÇÃO
#%% 
# criando uma constante com o shape da imagem 
IMAGE_SIZE = (192,192)
# Pegando o classificador já treinado
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4")
])
#%% lendo uma imagem como exemplo
img = cv2.imread(BASE_DIR+'train_images/336550.jpg')
img2 = cv2.resize(img, IMAGE_SIZE)/255.0
cv2.imshow('ok', img2)
#%% printando como deve ser a entrada da rede, pois a rede é treinada atraves de batch
print(img2[np.newaxis, ...].shape)
#%%
result = classifier.predict(img2[np.newaxis, ...])
#%%
predict_label_index = np.argmax(result)
print(predict_label_index)


#%%
# ESTE TRECHO DE CÓDIGO VAI UTILIZAR A REDE JÁ TREINADA E ACRESCENTAR MAIS CAMADAS
#%% Vai ler o JSON e criar um dicionário que vai relacionar os labels com o nome de cada folha
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
print(json.dumps(map_classes, indent=4))
#%%
# Lendo o csv de treinamento e criando uma nova coluna que relaciona a label com o nome da folha
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train
df_train.label= df_train.label.astype('str')
#%%
# Definindo o batch_size, target size(tamanho da largura e altura), quantidae de passos por época no treino e validação 
batch_size = 500
tgt_size = 192
STEPS_PER_EPOCH = len(df_train)*0.8 / batch_size
val_steps = len(df_train)*0.2/ batch_size
#%%
# A função DataGenerator gera lotes de dados de imagem de tensor com aumento de dados em tempo real
# A função flow_from_dataframe pega o dataframe criado, o diretório das imagens e gera os batches necessários
train_generator = ImageDataGenerator(validation_split = 0.2,
                                     rescale = 1.0/255,
                                     zoom_range = 0.2,
                                     horizontal_flip = True,
                                     vertical_flip = True,
                                     fill_mode = 'nearest',
                                     shear_range = 0.2,
                                     height_shift_range = 0.2,
                                     width_shift_range = 0.2) \
    .flow_from_dataframe(df_train,
                         directory = os.path.join(BASE_DIR, "train_images"),
                         subset = "training",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (tgt_size, tgt_size),
                         batch_size = batch_size,
                         class_mode = "sparse")

validation_generator = ImageDataGenerator(validation_split = 0.2,
                                     rescale = 1.0/255) \
    .flow_from_dataframe(df_train,
                         directory = os.path.join(BASE_DIR, "train_images"),
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (tgt_size, tgt_size),
                         batch_size = batch_size,
                         class_mode = "sparse")
#%%
# Pegando os vetores das features da rede treianda e colocando as camadas congeladas
pre_model_without_top_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/4",
               trainable=False, input_shape=(192,192,3), arguments=dict(batch_norm_momentum=0.997))

#%%
# Setando o número de folhas e adiconando a nova camada de cassificação
num_of_leaf = 5
model = tf.keras.Sequential([
    pre_model_without_top_layer,
    layers.Dense(num_of_leaf,   activation='softmax')
    ])

model.summary()

#%%
# Compilando o modelo com a função loss, otimizador adam e a métrica é a acurácia
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
#%%
# Salvando apenas os melhores pesos no [3].h5 
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model[3].h5",
save_best_only=True)
# A função EarlyStopping vai interromper o treinamento quando não tiver uma melhora no progresso em uma quantidad de época(patience)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=4,
restore_best_weights=True)

history = model.fit_generator(generator=train_generator, epochs=50,
                              validation_data=validation_generator,
                              validation_steps=val_steps,
                              callbacks = [checkpoint_cb, early_stopping_cb],
                              steps_per_epoch=STEPS_PER_EPOCH)
#model.save('my_model[3].h5')
#%%
#history_dict = history.history
# Save it under the form of a json file
history_dict = history.history
# Save it under the form of a json file
pd.DataFrame.from_dict(history_dict).to_csv('history.csv',index=False)
#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()