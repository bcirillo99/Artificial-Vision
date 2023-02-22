import csv 
import os
import cv2
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import losses
from utils import *
from AdaBoost import *


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


# File path

csv_file_train = "dataframe_train_rus.csv"
csv_file_val = "dataframe_val.csv"
train_data_dir = "/mnt/sdc1/2022_va_gr04/training_caip_contest/"

# Hyper parameters
split_ratio=0.2
batch_size = 64
width, height = 224, 224
patience = 20
max_lr = 0.0005
base_lr = max_lr/5
max_m = 0.98
base_m = 0.85
cyclical_momentum = True
cycles = 2.35

# Data augumentation
shear_range=0.5   
zoom_range=0.2  
brightness_range=(0.6,1.2)
horizontal_flip=True
vertical_flip=True

#training
n_epoch = 70
class_mode = "raw"
############ Data Augumentation ############

### DATA GENERATOR
# get data from the folder, perform the preprocessing and the data augmentation,
# and arranges them in batches

# this is the preprocessing configuration we will use for training
train_datagen = ImageDataGenerator() 

                 

df_val = pd.read_csv(csv_file_val)
df_val["label"] = df_val["label"].astype("str")
d = dict([(key, 0) for key in range(0,81)])
for elem in df_val["label"].tolist():
    d[int(elem)-1]+=1
print("Number of element per classes in the Validation set: ")
print(d)

#Sort classes and set them to 2-digit labels
labels = np.array(df_val["label"].tolist())
labels_int = [int(c) for c in labels]
labels_2_digits = ["%02d" % c for c in labels_int]
classes_int = np.unique(labels_int)
classes_sorted = np.sort(classes_int)
classes = ["%02d" % c for c in classes_sorted]

df_val["label"] = labels_2_digits
print('Validation set generator:')
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df_val,
    directory=train_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
    classes=classes,
    validate_filenames=False,
    shuffle=False) 

df_train = pd.read_csv(csv_file_train)
labels = np.array(df_train["label"].tolist())
labels_int = [int(c) for c in labels]
labels_2_digits = ["%02d" % c for c in labels_int]
df_train["label"] = labels_2_digits

label = tf.expand_dims(np.array(df_train["label"].astype("float32").tolist(),dtype="float32"),-1)
adaboost = AdaBoost(train_data_dir, df_train, classes, label, epochs=n_epoch, patience=20,reduce_factor=0.1, M=3)

adaboost.fit(val_generator,M=3,verbose=True)

val_generator.reset()
aar_m,mmae_m,sigma_m,mae_m,mae_j_list = adaboost.evaluate(val_generator, df_val, verbose=True)
print('Val aar:', aar_m)
print('Val mmae:', mmae_m)
print('Val sigma:', sigma_m)
print('Val mae:', mae_m)
print('Val maej_list:', mae_j_list)

print("Done.")