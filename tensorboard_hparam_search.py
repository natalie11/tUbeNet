# -*- coding: utf-8 -*-
"""
Hyperparamter search

Created on Thu Aug 26 14:55:31 2021

@author: Natal
"""

#Import libraries
import os
import pickle
from functools import partial
import numpy as np
import random

import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator#, ImageDisplayCallback, MetricDisplayCallback
#from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

from sklearn.metrics import precision_recall_curve, f1_score, make_scorer
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras.backend as K

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 100			         	# number of1 epoch for training CNN
steps_per_epoch = 100		        # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
batch_size = 2		 	       	    # batch size for training CNN
class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
n_classes=2
dataset_weighting = (1)

# Training and prediction options
binary_output = True	           	# save as binary (True) or softmax (False)
save_model = False		        	# save model structure and weights? Yes=True, No=False


""" Paths and filenames """
# Training data
data_path = 'F:/Paired datasets/train/headers'
val_path = 'F:/Paired datasets/test/headers'

#----------------------------------------------------------------------------------------------------------------------------------------------
""" Create Data Directory and Generators for train/test"""
# Load data headers into a list
header_filenames=os.listdir(data_path)
headers = []
for file in header_filenames: #Iterate through header files
    file=os.path.join(data_path,file)
    with open(file, "rb") as f:
        data_header = pickle.load(f) # Unpickle DataHeader object
    headers.append(data_header) # Add to list of headers

# Create empty data directory    
data_dir = DataDir([], image_dims=[], 
                   image_filenames=[], 
                   label_filenames=[], 
                   data_type=[])

# Fill directory from headers
for header in headers:
    data_dir.list_IDs.append(header.ID)
    data_dir.image_dims.append(header.image_dims)
    data_dir.image_filenames.append(header.image_filename+'.npy')
    if header.label_filename is not None:
        data_dir.label_filenames.append(header.label_filename+'.npy')
    else: data_dir.label_filenames.append(header.label_filename)
    data_dir.data_type.append('float32')

params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': dataset_weighting,
	       'shuffle': False}

data_generator=DataGenerator(data_dir, **params)

# Import vaildation data header
header_filenames=os.listdir(val_path)
headers = []
for file in header_filenames: #Iterate through header files
    file=os.path.join(val_path,file)
    with open(file, "rb") as f:
        data_header = pickle.load(f) # Unpickle DataHeader object
    headers.append(data_header) # Add to list of headers
    
# Create empty data directory    
val_dir = DataDir([], image_dims=[], 
                   image_filenames=[], 
                   label_filenames=[], 
                   data_type=[])

# Fill directory from headers
for header in headers:
    val_dir.list_IDs.append(header.ID)
    val_dir.image_dims.append(header.image_dims)
    val_dir.image_filenames.append(header.image_filename+'.npy')
    val_dir.label_filenames.append(header.label_filename+'.npy')
    val_dir.data_type.append('float32')

vparams = {'batch_size': batch_size,
  'volume_dims': volume_dims, 
  'n_classes': n_classes,
  'dataset_weighting': [1],
	       'shuffle': False}

val_generator=DataGenerator(val_dir, **vparams)

""" Build Model """
# create partial for  to pass to complier
#custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)
#custom_loss.__name__ = "custom_loss" #partial doesn't copy name or module attribute from function
#custom_loss.__module__ = tube.weighted_crossentropy.__module__

def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

# Define hparam space
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.001, 0.0005, 0.0001]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.3]))
HP_LOSS = hp.HParam('loss', hp.Discrete(['DiceBCELoss', 'catagorical_crossentropy']))
HP_ALPHA = hp.HParam('alpha', hp.Discrete([0.1, 0.2, 0.3]))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','rmsprop']))

# Set metrics to log
METRIC_ACCURACY = 'accuracy'



with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_LR, HP_DROPOUT, HP_LOSS, HP_ALPHA],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


        
def train_test_model(hparams, data_generator, val_generator):
  model = tube.tUbeNet(n_classes=2, input_height=64, input_width=64, input_depth=64,
            learning_rate=hparams[HP_LR], loss=hparams[HP_LOSS], metrics=['accuracy', tube.recall, tube.precision, tube.dice], 
            dropout=hparams[HP_DROPOUT], alpha=hparams[HP_ALPHA])
  
  model.fit(data_generator, epochs=1)
  _, accuracy = model.evaluate(val_generator)
  return accuracy

def run(run_dir, hparams, data_generator, val_generator):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams, data_generator, val_generator)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

""" Conduct search """        
session_num = 0

for session_num in range(6):
    
    lr = random.choice(HP_LR.domain.values)
    dropout = random.choice(HP_DROPOUT.domain.values)
    loss = random.choice(HP_LOSS.domain.values)
    alpha = random.choice(HP_ALPHA.domain.values)
    
    hparams = {
        HP_LR: lr,
        HP_DROPOUT: dropout,
        HP_LOSS: loss,
        HP_ALPHA: alpha
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams, data_generator, val_generator)
    session_num += 1