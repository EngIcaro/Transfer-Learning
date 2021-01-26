#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import json
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import time
#%% Criando uma constante com o diretório da base de dados
BASE_DIR = "./cassava-leaf-disease-classification/"
#%%
# Diretório raiz que será usado para os logs do tensorrboard 
# E uma função que vai gerar uma pata de sub-diretório baseado na data e tempo atual
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'

#%% Vai ler o JSON e criar um dicionário que vai relacionar os labels com o nome de cada folha
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
print(json.dumps(map_classes, indent=4))
#%%
# Printando o número de observações na base de treinamento
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")
#%%
# Lendo o csv de treinamento e criando uma nova coluna que relaciona a label com o nome da folha
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train
df_train.label= df_train.label.astype('str')
#%%
# Definindo o batch_size, target size(tamanho da largura e altura), quantidae de passos por época no treino e validação 
batch_size = 2500
tgt_size = 100
STEPS_PER_EPOCH = len(df_train)*0.8 / batch_size
val_steps = len(df_train)*0.2/ batch_size
#%%
train_generator = ImageDataGenerator(validation_split = 0.2,
                                     preprocessing_function = None,
                                     zoom_range = 0.2,
                                     cval = 0.2,
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
                         batch_size = 500,
                         class_mode = "sparse")

validation_generator = ImageDataGenerator(validation_split = 0.2) \
    .flow_from_dataframe(df_train,
                         directory = os.path.join(BASE_DIR, "train_images"),
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (tgt_size, tgt_size),
                         batch_size = 500,
                         class_mode = "sparse")
# %%
num_classes = 5
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(tgt_size, tgt_size, 3)),
  layers.Flatten(input_shape = [tgt_size,tgt_size]),
  layers.Dense(150, activation='relu'),
  layers.Dense(50, activation='relu'),
  layers.Dense(5,   activation='softmax')
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Flatten(),
# layers.Dense(128, activation='relu'),
#  layers.Dense(num_classes)
])
# %%
model.summary()
# %%
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])
#%%
#history = model.fit(train_generator, epochs=50,
#                    validation_data=validation_generator,
#                    verbose = 2,
#                    steps_per_epoch=STEPS_PER_EPOCH)
#%%
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model[2].h5",
save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
restore_best_weights=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit_generator(generator=train_generator, epochs=40,
                              validation_data=validation_generator,
                              validation_steps=val_steps,
                              callbacks = [checkpoint_cb, early_stopping_cb, tensorboard_cb],
                              steps_per_epoch=STEPS_PER_EPOCH)   
#model.save("my_keras_model.h5")
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
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%