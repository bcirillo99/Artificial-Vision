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
model_path="new_model_resnet_class_w"
log_path="new_logs_resnet_class_w"

regression = False
class_mode = "raw"
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




df_train = pd.read_csv(csv_file_train)
print(len(df_train))
df_train["label"] = df_train["label"].astype("str")

d = dict([(key, 0) for key in range(0,81)])
for elem in df_train["label"].tolist():
    d[int(elem)-1]+=1
print("Number of element per classes in the Training set: ")
print(d)

#Sort classes and set them to 2-digit labels
labels = np.array(df_train["label"].tolist())
labels_int = [int(c) for c in labels]
labels_2_digits = ["%02d" % c for c in labels_int]
classes_int = np.unique(labels_int)
classes_sorted = np.sort(classes_int)
classes = ["%02d" % c for c in classes_sorted]

#Set class_weight to macro-classes (0-9, 10-19,..., 70+)
eight_classes = np.arange(8).tolist()
print(eight_classes)
eight_labels = np.floor(np.divide(labels_int,10)).astype(int)
eight_labels = [7 if  i == 8 else i for i in eight_labels]
print(np.unique(eight_labels))

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=eight_classes, y=eight_labels)
class_weights = dict(enumerate(class_weights))
class_weights[8] = class_weights[7]
print(class_weights)
for keys in class_weights.keys():
  df_train.loc[np.floor(np.divide(df_train.label.astype(int),10)).astype(int) == keys, "w_col"] = class_weights[keys] 

df_train["label"] = labels_2_digits
print(df_train)
print('Training set generator:')
print(len(df_train))
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=train_data_dir,
    x_col='filename',
    y_col='label',
    weight_col='w_col',
    target_size=(height, width),
    batch_size=batch_size,
    classes = classes,
    class_mode="categorical",
    validate_filenames=False,
    shuffle=False)                    

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
    validate_filenames=False) 


"""base_model = load_model('model', compile=False)
selected_layer1 = getLayerIndexByName(base_model, layer_sup)"""

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

output = Dense(81, activation='softmax', name='out')(X)
model = Model(base_model.input, output)

if layer_inf is not None and layer_sup is not None:
    layer_inf,layer_sup = getLayerIndexByNames(model, layer_inf, layer_sup)
    print("Layer non trainable:")
    for layer in model.layers[layer_inf:layer_sup+1]:
        print(layer.name)
        layer.trainable = False
model.summary()


############ Train ###########
training_steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps_per_epoch = val_generator.n//val_generator.batch_size
print("Total number of batches per epoch = training:", training_steps_per_epoch, "and validation:", validation_steps_per_epoch)
print("training set samples: ", train_generator.n)


print("Clasification")
model.compile(
    loss=aar_class,      
    optimizer=SGD(0.005),
    metrics=[aar_metric_class,mae_class,sigma_class])


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


#print(train_generator.next())
"""history = model.fit(train_generator, steps_per_epoch=training_steps_per_epoch,
                            validation_data=val_generator, validation_steps=validation_steps_per_epoch,
                            epochs=n_epoch,
                            verbose=1,
                            callbacks=[reduce_lr, early_stop, tensorboard, modelcheckpoint]
                            )"""


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