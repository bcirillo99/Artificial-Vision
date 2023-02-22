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

####### LOSS functions ########


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
    aar = 0.5*mmae + 0.5*sigma

    #tf.print(mae,mmae,sigma,aar)
    return aar


    
######### METRIC functions #########

# Classification
def mae_class(y_true,y_pred):
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
    maej_list = tf.convert_to_tensor(maej_list)
    nonzero_indices = tf.squeeze(tf.where(maej_list))
    maej_list = tf.gather(maej_list, nonzero_indices)
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

    maej_list = tf.convert_to_tensor(maej_list)
    nonzero_indices = tf.squeeze(tf.where(maej_list))
    maej_list = tf.gather(maej_list, nonzero_indices)
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

    maej_list = tf.convert_to_tensor(maej_list)
    nonzero_indices = tf.squeeze(tf.where(maej_list))
    maej_list = tf.gather(maej_list, nonzero_indices)
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

    maej_list = tf.convert_to_tensor(maej_list)
    nonzero_indices = tf.squeeze(tf.where(maej_list))
    maej_list = tf.gather(maej_list, nonzero_indices)
    mmae= tf.reduce_mean(maej_list)
    mae = mae_func(y_true,y_pred)
    #tf.print(maej_list)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(maej_list,mae))))
    aar = tf.maximum(tf.constant([0.]),5-mmae)+tf.maximum(tf.constant([0.]),5-sigma)
    return aar[0],mmae,sigma,mae, maej_list


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

