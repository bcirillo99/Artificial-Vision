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


####### LOSS functions ########

# Regression
def aar_regr(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    y_true = y_true/100
    range_list = [(0.01,0.10),(0.11,0.20),(0.21,0.30),(0.31,0.40),(0.41,0.50),(0.51,0.60),(0.61,0.70),(0.71,1)]
    

    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)


    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = 0.5*mmae + 0.5*sigma
    #tf.print(mae,mmae,sigma,aar)
    return aar

# Classification
def aar_class(y_true,y_pred):
    #tf.print(y_true,y_pred)
    #tf.print(y_true, summarize=-1)
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    positions = tf.range(1,82, dtype=tf.float32)

    prod_true = y_true*positions
    prod_pred = y_pred*positions

    y_true = tf.reduce_sum(prod_true,axis=1,keepdims=True)
    y_pred = tf.reduce_sum(prod_pred,axis=1,keepdims=True)

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)

    maej_list = tf.convert_to_tensor(maej_list)
    nonzero_indices = tf.squeeze(tf.where(maej_list))
    maej_list = tf.gather(maej_list, nonzero_indices)
    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = mmae + sigma
    #tf.print(mae,mmae,sigma,aar)
    return aar
    
######### METRIC functions #########

# Regression
def accuracy_regr(y_true,y_pred):
    y_pred = y_pred*100
    y_pred = tf.round(y_pred)
    cond = tf.equal(y_true,y_pred)
    res = tf.math.count_nonzero(cond)/K.int_shape(y_true)[1]
    return res

def mae_regr(y_true,y_pred):
    y_pred = y_pred*100
    return tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)

def sigma_regr(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    y_pred = y_pred*100
    y_pred = tf.round(y_pred)
    #tf.print(y_true, summarize=-1)
    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)

    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))

    return sigma

def aar_metric_regr(y_true,y_pred):
  mae_func = tf.keras.losses.MeanAbsoluteError()
  range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
  y_pred = y_pred*100
  y_pred = tf.round(y_pred)
  #tf.print(y_true, summarize=-1)
  #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

  maej_list = []
  for (inf, sup) in range_list:
      sup = tf.constant(sup,dtype='float32')
      inf = tf.constant(inf,dtype='float32')
      cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
      cond = tf.squeeze(cond)
      #tf.print(cond, summarize=-1)
      indices_j = tf.where(cond)
      y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
      y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
      mae_j = mae_func(y_true_j,y_pred_j)
      maej_list.append(mae_j)


  mmae= tf.reduce_mean(maej_list)
  mae = mae_func(y_true,y_pred)
  #tf.print(maej_list)
  sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
  aar = tf.maximum(tf.constant([0.]),5-mmae)+tf.maximum(tf.constant([0.]),5-sigma)
  #tf.print(mae,mmae,sigma,aar)
  return aar

def mmae_regr(y_true,y_pred):
  mae_func = tf.keras.losses.MeanAbsoluteError()
  range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
  y_pred = y_pred*100
  y_pred = tf.round(y_pred)
  #tf.print(y_true, summarize=-1)
  #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

  maej_list = []
  for (inf, sup) in range_list:
      sup = tf.constant(sup,dtype='float32')
      inf = tf.constant(inf,dtype='float32')
      cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
      cond = tf.squeeze(cond)
      #tf.print(cond, summarize=-1)
      indices_j = tf.where(cond)
      y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
      y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
      mae_j = mae_func(y_true_j,y_pred_j)
      maej_list.append(mae_j)


  mmae= tf.reduce_mean(maej_list)
  return mmae

# Classification
def mae_class(y_true,y_pred):
    positions = tf.range(1,82, dtype=tf.float32)
    positions = tf.range(1,82, dtype=tf.float32)

    prod_true = y_true*positions
    prod_pred = y_pred*positions

    y_true = tf.reduce_sum(prod_true,axis=1,keepdims=True)
    y_pred = tf.reduce_sum(prod_pred,axis=1,keepdims=True)

    y_pred = tf.round(y_pred)
    return tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)

def sigma_class(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    positions = tf.range(1,82, dtype=tf.float32)

    prod_true = y_true*positions
    prod_pred = y_pred*positions

    y_true = tf.reduce_sum(prod_true,axis=1,keepdims=True)
    y_pred = tf.reduce_sum(prod_pred,axis=1,keepdims=True)

    y_pred = tf.round(y_pred)
    #tf.print(y_true, summarize=-1)
    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)

    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))

    return sigma

def aar_metric_class(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    positions = tf.range(1,82, dtype=tf.float32)

    prod_true = y_true*positions
    prod_pred = y_pred*positions

    y_true = tf.reduce_sum(prod_true,axis=1,keepdims=True)
    y_pred = tf.reduce_sum(prod_pred,axis=1,keepdims=True)
    y_pred = tf.round(y_pred)
    #tf.print(y_true, summarize=-1)
    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)

    
    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = tf.maximum(tf.constant([0.]),5-mmae)+tf.maximum(tf.constant([0.]),5-sigma)
    #tf.print(mae,mmae,sigma,aar)
    return aar

def mmae_class(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    positions = tf.range(1,82, dtype=tf.float32)

    prod_true = y_true*positions
    prod_pred = y_pred*positions

    y_true = tf.reduce_sum(prod_true,axis=1,keepdims=True)
    y_pred = tf.reduce_sum(prod_pred,axis=1,keepdims=True)
    y_pred = tf.round(y_pred)
    #tf.print(y_true, summarize=-1)
    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)


    mmae= tf.reduce_mean(maej_list)
    return mmae

######### BOOST METRICS ############

def aar(y_true,y_pred):
    mae_func = tf.keras.losses.MeanAbsoluteError()
    range_list = [(1,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),(71,100)]
    #tf.print(y_true, summarize=-1)
    #range_list = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,100)]

    maej_list = []
    for (inf, sup) in range_list:
        sup = tf.constant(sup,dtype='float32')
        inf = tf.constant(inf,dtype='float32')
        cond = tf.logical_and(tf.greater_equal(y_true,inf),tf.less_equal(y_true,sup))
        cond = tf.squeeze(cond)
        #tf.print(cond, summarize=-1)
        indices_j = tf.where(cond)
        y_true_j = tf.squeeze(tf.gather(y_true,indices_j),[-1])
        y_pred_j = tf.squeeze(tf.gather(y_pred,indices_j),[-1])
        mae_j = mae_func(y_true_j,y_pred_j)
        maej_list.append(mae_j)


    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = tf.maximum(tf.constant([0.]),5-mmae)+tf.maximum(tf.constant([0.]),5-sigma)
    return aar[0],mmae,sigma,mae


###### OTHER #######
def getLayerIndexByNames(model, layername_inf, layername_sup):
        for idx, layer in enumerate(model.layers):
            if layer.name == layername_inf:
                print(layer.name)
                idx_inf = idx
            if layer.name == layername_sup:
                print(layer.name)
                idx_sup = idx
        return idx_inf,idx_sup

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx

def error_per_age(model,generator,df,train_data_dir):
  d = dict([(key, 0) for key in range(1,82)])
  label=np.expand_dims(np.array(df["label"].tolist()),-1)
  val_generator = generator.flow_from_dataframe(
        dataframe=df,
        directory=train_data_dir,
        x_col='filename',
        y_col='label',
        target_size=(height, width),
        batch_size=5,
        class_mode='raw',
        shuffle=False,
        validate_filenames=False)
  
  pred = model.predict(val_generator,verbose=1)
  tot_error = np.sum(np.squeeze(np.not_equal(pred,label)))
  for key in d.keys():
    q = "label == "+str(key)
    df_i=df.query(q)
    if len(df_i)>0:
      val_generator = generator.flow_from_dataframe(
        dataframe=df_i,
        directory=train_data_dir,
        x_col='filename',
        y_col='label',
        target_size=(height, width),
        batch_size=5,
        class_mode='raw',
        shuffle=False,
        validate_filenames=False)
      pred_i = model.predict(val_generator,verbose=1)
      pred_i = np.round(pred_i*100)
      error=np.sum(np.squeeze(np.not_equal(pred_i,np.expand_dims(np.ones(len(df_i))*key,-1))))
      perc_i = float(error)/float(tot_error)
      d[key] = {"n_error":error,"perc":float(error)/float(len(df_i))}
  print(d)
  d["tot"]=tot_error
  print(tot_error)
  return d
