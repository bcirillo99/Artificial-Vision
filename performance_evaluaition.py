import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
from utils import *

def plot_confusion_matrix(y, pred, title):
  # MLP confusion matrix
  conf_matrix = confusion_matrix(y, pred,normalize='true')
  # Visualization commands
  print("Confusion Matrix")
  print(conf_matrix[:6][:6])

  fig, ax = plt.subplots(figsize=(30, 30))
  plt.matshow(conf_matrix)
  plt.title(title)
  plt.colorbar()
  plt.ylabel('True Label')
  plt.xlabel('Predicated Label')
  plt.savefig('confusion_matrix.jpg')



csv_file_train = "dataframe_train.csv"
csv_file_val = "dataframe_val.csv"
train_data_dir = "/mnt/sdc1/2022_va_gr04/training_caip_contest/"

batch_size = 64
width, height = 224, 224

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


#Load the model
model = load_model("model_resnet_class_w", compile=False)

model.compile(
    loss=aar_class,      
    optimizer=SGD(0.005),
    metrics=[aar_metric_class,mae_class,sigma_class])

positions = np.arange(1,82, dtype=np.float32)
y_val=np.array(df_val["label"].astype("float32").tolist())

pred=model.predict(val_generator,verbose=True)
prod = pred*positions
pred = tf.reduce_sum(prod,axis=1,keepdims=True)
y_val_pred = tf.round(pred)

classes = [5., 15., 25., 35., 45., 55., 65., 75]

plot_confusion_matrix(y_val, np.squeeze(y_val_pred), 'confusion matrix')