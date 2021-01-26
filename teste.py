#https://www.kaggle.com/ihelon/cassava-leaf-disease-exploratory-data-analysis
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import json
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models, layers
import pandas as pd
import seaborn as sn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_efficientnets import EfficientNetB0
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#%%
BASE_DIR = "./cassava-leaf-disease-classification/"
#%%
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
print(json.dumps(map_classes, indent=4))
#%%
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")
#%%
# Let's take a look at the dimensions of the first 300 images
# As you can see below, all images are the same size (600, 800, 3)
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)
#%%
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

df_train["class_name"] = df_train["label"].map(map_classes)

df_train
#%%
plt.figure(figsize=(8, 4))
sn.countplot(y="class_name", data=df_train);
#%%
def visualize_batch(image_ids, labels):
    plt.figure(figsize=(16, 12))
    
    for ind, (image_id, label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Class: {label}", fontsize=12)
        plt.axis("off")
    
    plt.show()
tmp_df = df_train.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_name"].values

visualize_batch(image_ids, labels)
#%%
batch_size = 32
STEPS_PER_EPOCH = len(df_train)*0.8 / batch_size
validation_steps = len(df_train)*0.2/ batch_size
epochs = 20
target_size = 224
#%%
df_train.label= df_train.label.astype('str')
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
                         target_size = (target_size, target_size),
                         batch_size = BASE_DIR,
                         class_mode = "sparse")

validation_generator = ImageDataGenerator(validation_split = 0.2) \
    .flow_from_dataframe(df_train,
                         directory = os.path.join(BASE_DIR, "train_images"),
                         subset = "validation",
                         x_col = "image_id",
                         y_col = "label",
                         target_size = (target_size, target_size),
                         batch_size = batch_size,
                         class_mode = "sparse")
#%%
def create_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    #model = models.Sequential()

   # model.add(EfficientNetB0(include_top = False, weights = 'imagenet',
    #                     input_shape = (target_size, target_size, 3)))

    #model.add(layers.GlobalAveragePooling2D())
    #model.add(layers.Dense(5, activation = "softmax"))

    #model.compile(optimizer = Adam(lr = 0.001),
    #          loss = "sparse_categorical_crossentropy",
    #          metrics = ["acc"])
    return model
#%%
model = create_model()
model.summary()
#%%
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
history = model.fit_generator(generator=train_generator, epochs=20,
                              validation_data=validation_generator,
                          steps_per_epoch=STEPS_PER_EPOCH)   
#%%
model_save = ModelCheckpoint('./best_baseline_model.h5', 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, 
                              patience = 2, min_delta = 0.001, 
                              mode = 'min', verbose = 1)


history = model.fit(
    train_generator,
    steps_per_epoch = str(steps_per_epoch),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = str(validation_steps),
    callbacks = [model_save, early_stop, reduce_lr]
)
# %%
tf.__version__
keras.__version__
# %%
