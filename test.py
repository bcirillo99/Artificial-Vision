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
import argparse

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


############ Data Augumentation ############

### DATA GENERATOR
# get data from the folder, perform the preprocessing and the data augmentation,
# and arranges them in batches

train_datagen = ImageDataGenerator() 

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='GTA_CAIP_Contest_Code/foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='GTA_CAIP_Contest_Code/foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()

csv_file_test = args.data                  
test_data_dir = args.images
csv_results =  args.results

batch_size = 64
width, height = 224, 224
n_epoch = 70




df_test = pd.read_csv(csv_file_test)
df_test.columns = ["filename","label"]
df_test["label"] = df_test["label"].astype("str")
d = dict([(key, 0) for key in range(0,81)])
for elem in df_test["label"].tolist():
    d[int(elem)-1]+=1
print("Number of element per classes in the Validation set: ")
print(d)

#Sort classes and set them to 2-digit labels
labels = np.array(df_test["label"].tolist())
labels_int = [int(c) for c in labels]
labels_2_digits = ["%02d" % c for c in labels_int]
classes_int = np.unique(labels_int)
classes_sorted = np.sort(classes_int)
classes = ["%02d" % c for c in classes_sorted]

df_test["label"] = labels_2_digits
print('Validation set generator:')
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=test_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
    classes=classes,
    validate_filenames=False,
    shuffle=False) 


label = tf.expand_dims(np.array(df_test["label"].astype("float32").tolist(),dtype="float32"),-1)
adaboost = AdaBoost(test_data_dir, df_test, classes, label, epochs=n_epoch, patience=20,reduce_factor=0.1, M=3)
adaboost.load("RUSboost/model","RUSboost/alphas_adaboost.csv")

val_generator.reset()
pred = adaboost.predict(val_generator, verbose=True)

df_results = pd.DataFrame()
df_results["filename"]= df_test["filename"]
df_results["label"]= pred.numpy().astype(int)
df_results.to_csv(csv_results,header=False,index=False)
