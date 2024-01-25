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
#from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV #RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, make_scorer
#----------------------------------------------------------------------------------------------------------------------------------------------
def create_model(lr=0.01, alpha=0.2, dropout = 0.25, loss=None):
    model = tube.tUbeNet(n_classes=2, input_height=64, input_width=64, input_depth=64,
            learning_rate=lr, loss=loss, metrics=['accuracy', tube.recall, tube.precision, tube.dice], 
            dropout=dropout, alpha=alpha)
    return model

if __name__ == '__main__':
#----------------------------------------------------------------------------------------------------------------------------------------------
    """Set hard-coded parameters and file paths:"""
    
    # Paramters
    volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
    n_epochs = 100			         	# number of1 epoch for training CNN
    steps_per_epoch = 100		        # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    batch_size = 2		 	       	    # batch size for training CNN
    class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
    n_classes=2
    dataset_weighting = (30,60,10)
    
    # Training and prediction options
    binary_output = True	           	# save as binary (True) or softmax (False)
    save_model = False		        	# save model structure and weights? Yes=True, No=False
    
    
    """ Paths and filenames """
    # Training data
    data_path = 'F:/Paired datasets/train/headers'
    
    # Validation data
    #val_path = None # Set to None is not using validation data
    
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
    
    
    """ Create Data Generator """
    params = {'batch_size': batch_size,
              'volume_dims': volume_dims, 
              'n_classes': n_classes,
              'dataset_weighting': dataset_weighting,
    	       'shuffle': False}
    
    data_generator=DataGenerator(data_dir, **params)
    
    
    """ Build Model """
    # create partial for  to pass to complier
    custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)
    custom_loss.__name__ = "custom_loss" #partial doesn't copy name or module attribute from function
    custom_loss.__module__ = tube.weighted_crossentropy.__module__            
    
    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=2)
    
    """ Conduct search """        
    # Define K-fold cross validation
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Define search space
    space = dict()
    space['lr'] = [0.0001, 0.001, 0.01]
    space['alpha'] = [0.1, 0.2, 0.4]
    space['dropout']=[0.1,0.2,0.4]
    space['loss']=[custom_loss, 'catagorical_crossentropy']
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Nadam']
    #param_grid = dict(optimizer=optimizer)
    
    f1=make_scorer(f1_score)
    search = RandomizedSearchCV(estimator=model, n_iter=5, param_distributions=space, n_jobs=-2, cv=3, scoring=f1, verbose=10)
    result = search.fit(data_generator)
    
    
    """ Results """
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
