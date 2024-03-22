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
#import datetime
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator#, ImageDisplayCallback, MetricDisplayCallback
import keras_tuner
from keras import metrics
#----------------------------------------------------------------------------------------------------------------------------------------------
class Dice(metrics.Metric):

  def __init__(self, name='dice', **kwargs):
    super().__init__(name=name, **kwargs)
    self.recall = self.add_weight(name='recall', initializer='zeros')
    self.precision = self.add_weight(name='precision', initializer='zeros')
    self.score = self.add_weight(name='score', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = ops.cast(y_true, "bool")
    y_pred = ops.cast(y_pred, "bool")
    
    recall_values=recall_logical(y_true, y_pred)
    precision_values=precision_logical(y_true, y_pred)
    dice_values = np.divide(np.multiply(2,np.multiply(precision_values,recall_values)),np.add(precision_values,recall_values))

    # TP = ops.sum(ops.logical_and(ops.equal(y_true,1),ops.equal(y_pred,1)))
    # FP = ops.sum(ops.logical_and(ops.equal(y_true,0),ops.equal(y_pred,1)))
    # FN = ops.sum(ops.logical_and(ops.equal(y_true,1),ops.equal(y_pred,0)))
    # recall_values = ops.divide(TP,ops.add(TP,FN))
    # precision_values = ops.divide(TP,ops.add(TP,FP))
    # dice_values = ops.divide(ops.multiply(2,ops.multiply(precision_values,recall_values)),ops.add(precision_values,recall_values))
    
    recall_values = ops.cast(recall_values, self.dtype)
    precision_values = ops.cast(precision_values, self.dtype)
    dice_values = ops.cast(dice_values, self.dtype)

    self.recall.assign_add(ops.sum(recall_values))
    self.precision.assign_add(ops.sum(precision_values))
    self.score.assign_add(ops.sum(dice_values))

  def result(self):
    return self.score

  def reset_states(self):
    self.true_positives.assign(0)
    self.recall.assign(0)
    self.precision.assign(0)
    self.score.assign(0)

def create_model(lr=0.01, alpha=0.2, dropout = 0.25, loss=None):
    tubenet = tUbeNet(n_classes=2, input_dims=(64,64,64), dropout=dropout, alpha=alpha)
    model = tubenet.create(learning_rate=lr, loss=loss, metrics=['accuracy', tube.recall, tube.precision, Dice()])
    return model

def build_with_tuner(hp):
    dropout = hp.Choice("dropout", [0.0,0.1,0.2,0.4])
    alpha = hp.Choice("alpha", [0.0,0.1,0.2,0.4])
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    loss = hp.Choice("loss",['DICE BCE', 'weighted categorical crossentropy'])
    # call existing model-building code with the hyperparameter values.
    create_model(lr=lr, alpha=alpha, dropout=dropout, loss=loss)
    return model

if __name__ == '__main__':
#----------------------------------------------------------------------------------------------------------------------------------------------
    """Set hard-coded parameters and file paths:"""
    
    # Paramters
    volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
    n_epochs = 10			         	# number of1 epoch for training CNN
    steps_per_epoch = 30		        # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    batch_size = 6		 	       	    # batch size for training CNN
    class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
    n_classes=2
    dataset_weighting = (4,5,1,1)
    
    # Training and prediction options
    binary_output = True	           	# save as binary (True) or softmax (False)
    save_model = False		        	# save model structure and weights? Yes=True, No=False
    
    
    """ Paths and filenames """
    # Training data
    data_path = 'F:/Paired datasets/train/headers'
    
    # Validation data
    val_path = 'F:/Paired datasets/test/headers' # Set to None is not using validation data
    
    # Model
    #model_path = 'F:/Paired datasets/models/sws'
    #updated_model_filename = None # model will be saved under this name
    
    # Image output
    #output_filename = 'F:/Paired datasets/pred_sws'
    
    #----------------------------------------------------------------------------------------------------------------------------------------------
    """ Create Data Directory"""
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
                       data_type=[], exclude_region=[])
    
    # Fill directory from headers
    for header in headers:
        data_dir.list_IDs.append(header.ID)
        data_dir.image_dims.append(header.image_dims)
        data_dir.image_filenames.append(header.image_filename+'.npy')
        if header.label_filename is not None:
            data_dir.label_filenames.append(header.label_filename+'.npy')
        else: data_dir.label_filenames.append(header.label_filename)
        data_dir.data_type.append('float32')
        data_dir.exclude_region.append((None,None,None))
    
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
                       data_type=[], exclude_region=[])
    
    # Fill directory from headers
    for header in headers:
        val_dir.list_IDs.append(header.ID)
        val_dir.image_dims.append(header.image_dims)
        val_dir.image_filenames.append(header.image_filename+'.npy')
        val_dir.label_filenames.append(header.label_filename+'.npy')
        val_dir.data_type.append('float32')
        val_dir.exclude_region.append((None,None,None))
    
    vparams = {'batch_size': batch_size,
      'volume_dims': volume_dims, 
      'n_classes': n_classes,
      'dataset_weighting': None,
    	       'shuffle': False}
    
    val_generator=DataGenerator(val_dir, **vparams)

    
    """ Conduct search """        
    tuner = keras_tuner.RandomSearch(
            hypermodel=build_with_tuner,
            objective="dice",
            max_trials=10,
            executions_per_trial=2,
            overwrite=True,
            directory='C:/Users/Natalie/Documents/GitHub/tUbeNet/logs/keras_tuner',
            project_name="tubenet_hparams",
            )
    
    tuner.search_space_summary()
    
    tuner.search(data_generator, steps_per_epoch=30, epochs=10, validation_data=val_generator, validation_steps=30)
    
    
    """ Results """
    tuner.results_summary()
    
    best_model = tuner.get_best_models(num_models=1)
    best_model.summary
    

