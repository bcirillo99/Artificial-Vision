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
from clr import CyclicLR
from utils import *

pid = os.getpid()
pgid = os.getpgid(pid)
print("My pid is: ", pid)
line = "kill -9 "+str(pid)
with open("kill.bash","w") as bash:
    bash.write(line)
# Use this command before the training
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


workstation="Diem"

# File path
if workstation == "Kaggle":
  csv_file = "/kaggle/input/gta2022-train-csv/training_caip_contest.csv"
  dataset_folder = "/kaggle/input/gta2022-train/training_caip_contest/"
  train_data_dir = '/kaggle/working/dataset/training/'
  val_data_dir = '/kaggle/working/dataset/validation/'
elif workstation == "Diem":
  csv_file_train = "dataframe_train.csv"
  csv_file_val = "dataframe_val.csv"
  train_data_dir = "/mnt/sdc1/2022_va_gr04/training_caip_contest/"
else:
  csv_file = "training_caip_contest.csv"
  dataset_folder = "content/dataset/training_caip_contest/"
  train_data_dir = 'dataset/training/'
  val_data_dir = 'dataset/validation/'

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
n_epoch = 50
model_path="model_resnet_regr_reduce_lr"
log_path="logs_resnet_regr_reduce_lr"

regression = True
if regression:
    class_mode = "raw"
else:
    class_mode = "categorical"
layer_inf = "conv1_conv"
layer_sup = "conv2_block3_add"
############ Data Augumentation ############

### DATA GENERATOR
# get data from the folder, perform the preprocessing and the data augmentation,
# and arranges them in batches

# this is the preprocessing configuration we will use for training
train_datagen = ImageDataGenerator() 

# this is the preprocessing configuration we will use for validation:
# rescaling only
#val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split = split_ratio)


############# Loss function ############

"""
Used for testing
y_pred=[9,4,12,14,16,20,22,22,28,28,31,37,39]
y_true=[10,2,13,14,15,19,21,22,29,30,33,39]
y_pred=np.array(y_pred)-1
y_true=np.array(y_true)-1
#to_categorical considers the first class as the age 0

x=to_categorical(y_pred)
y_pred= np.array(x[:-1])
print(y_pred)

y_true=np.array(to_categorical(y_true))
"""

############# Model ########
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
# Add the preprocessing/augmentation layers.
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.RandomFlip(mode='horizontal_and_vertical')(x)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.2)(x)


base_model = tf.keras.applications.ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    input_tensor=(x),
    include_top=False)
X = base_model.output

X = GlobalAveragePooling2D()(X)

X = Dense(512)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

if regression:
    X = Dense(81, activation='relu')(X)
    output = Dense(1, activation='linear', name='out')(X)
    model = Model(inputs, output)
else:
    output = Dense(81, activation='softmax', name='out')(X)
    model = Model(inputs, output)

if layer_inf is not None and layer_sup is not None:
    layer_inf,layer_sup = getLayerIndexByName(model, layer_inf, layer_sup)
    print("Layer non trainable:")
    for layer in model.layers[layer_inf:layer_sup+1]:
        print(layer.name)
        layer.trainable = False
model.summary()

############ Train ###########

df_train = pd.read_csv(csv_file_train)
df_train["label"] = df_train["label"].astype("float32")
d = dict([(key, 0) for key in range(1,82)])
for elem in df_train["label"].tolist():
    d[elem]+=1
print("Number of element per classes in the Training set: ")
len(df_train)
print(d)
print('Training set generator:')
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=train_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_mode,
    validate_filenames=False)              

df_val = pd.read_csv(csv_file_val)
df_val["label"] = df_val["label"].astype("float32")
d = dict([(key, 0) for key in range(1,82)])
for elem in df_val["label"].tolist():
    d[elem]+=1
print("Number of element per classes in the Validation set: ")
len(df_val)
print(d)
print('Validation set generator:')
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df_val,
    directory=train_data_dir,
    x_col='filename',
    y_col='label',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode=class_mode,
    validate_filenames=False) 



training_steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps_per_epoch = val_generator.n//val_generator.batch_size
print("Total number of batches per epoch = training:", training_steps_per_epoch, "and validation:", validation_steps_per_epoch)
print("training set samples: ", train_generator.n)


print("Regression")
model.compile(
    loss=aar_regr,      
    optimizer=SGD(0.0001),
    metrics=[aar_metric_regr,mae_regr,sigma_regr,accuracy_regr])


#callbacks

tensorboard = TensorBoard(log_dir=log_path)

modelcheckpoint = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, 
                        mode='auto', restore_best_weights=True)

iterations = round(train_generator.n/batch_size*n_epoch)
iterations = list(range(0,iterations+1))
step_size = len(iterations)/(cycles)
clr =  CyclicLR(base_lr=base_lr,
                max_lr=max_lr,
                step_size=step_size,
                max_m=max_m,
                base_m=base_m,
                cyclical_momentum=cyclical_momentum)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, 
                              verbose=1, mode='auto')



history = model.fit(train_generator, steps_per_epoch=training_steps_per_epoch,
                            validation_data=val_generator, validation_steps=validation_steps_per_epoch,
                            epochs=n_epoch,
                            verbose=1,
                            callbacks=[reduce_lr, early_stop, tensorboard, modelcheckpoint]
                            )
#class_weight=class_weights

score = model.evaluate(val_generator, verbose=True)
print('Val loss:', score[0])
print('Val aar:', score[1])
print('Val mae:', score[2])
print('Val sigma:', score[3])

print("Done.")
"""
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor

def build_model(neurons_per_layer=[64, 64], activation='relu', optimizer='adam'):
  # compute the input shape from the train_dataset outer variable
  input_shape=[len(train_dataset.keys())]

  # create an empty sequential model
  model = keras.Sequential()
  # add the first Dense layer with the input_shape information
  model.add(Dense(neurons_per_layer[0],
                  activation=activation,
                  input_shape=input_shape))
  # add the intermediate layers
  for neurons in neurons_per_layer[1:]:
    model.add(Dense(neurons,
                    activation=activation))
  # add the last output layer
  model.add(Dense(1)) #implicitamente stiamo dicendo che la funzione di attivazione è f(x)=x

  # compile the model
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

  model = build_model()
def loss(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)


    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = 0.5*mae + 0.5*sigma
    #tf.print(mae,mmae,sigma,aar)
    return aar
#clf = KerasClassifier(twoLayerFeedForward, epochs=100, batch_size=20, verbose=0)
clf = KerasRegressor(build_model, epochs=1000, batch_size=20, verbose=1)
boosted_ann = AdaBoostRegressor(base_estimator= clf,n_estimators=3)
boosted_ann.fit(normed_train_data, train_labels)"""