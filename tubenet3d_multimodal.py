# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import pickle
from functools import partial
import numpy as np
import datetime
from model import tUbeNet
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator, ImageDisplayCallback, MetricDisplayCallback, FilterDisplayCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
volume_dims = (64,64,64)    	 	# Size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 50			         	# Number of epoch for training CNN
steps_per_epoch = 10		        # Number of steps (batches of samples) to yield from generator before declaring one epoch finished
batch_size = 6		 	       	    # Batch size for training CNN
n_classes=2                         # Number of classes
dataset_weighting = [1]             # Relative weighting when pulling training data from multiple datasets
loss = "DICE BCE"	        	    # "DICE BCE", "focal" or "weighted categorical crossentropy"
lr0 = 1e-4                          # Initial learning rate
class_weights = (1, 7)	        	# if using weighted loss: relative weighting of background to blood vessel classes
augment = True       	        	# Augment training data, True/False
attention = False

# Training and prediction options
use_saved_model = True	        	# use previously saved model structure and weights? Yes=True, No=False
fine_tune = True                    # prepare model for fine tuning by replacing classifier and freezing shallow layers? Yes=True, No=False
binary_output = False	           	# save as binary (True) or softmax (False)
save_model = True		        	# save model structure and weights? Yes=True, No=False
prediction_only = False             # if True -> training is skipped

""" Paths and filenames """
# Training data
data_path = '[path to preprocessed data headers folder]'

# Validation data
val_path = '[path to preprocessed validation data headers folder (optional)]'  # Set to None is not using validation data

# Model
model_path = '[path to model folder]'
model_filename =  '[model filename]'  # filepath for model weights is using an exisiting model, else set to None
updated_model_filename = '[updated model filename]' # trained model will be saved under this name

# Image output
output_path = '[path to predictions folder]' # predicted segmentations will be saved here

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
    data_dir.exclude_region.append((None,None,None)) #region to be left out of training for use as validation data (under development)
    
""" Manually set excluded region (to reserve for validation)"""
# This section is under development
# data_dir.exclude_region[0]=(None, None, (300,500)) # This exludes the bottom 200 'rows' of pixels from the training set


""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': dataset_weighting,
          'augment':augment,
	       'shuffle': False}

data_generator=DataGenerator(data_dir, **params)


""" Load or Build Model """
# callbacks              
#time_callback = tube.TimeHistory()		      
#stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)
tubenet = tUbeNet(n_classes=n_classes, input_dims=volume_dims, attention=attention)

if use_saved_model:
    # Load exisiting model with or without fine tuning adjustment (fine tuning -> classifier replaced and first 10 layers frozen)
       model = tubenet.load_weights(filename=os.path.join(model_path,model_filename), loss=loss, class_weights=class_weights, learning_rate=lr0, 
                                 metrics=['accuracy', tube.recall, tube.precision, tube.dice],
                                 freeze_layers=0, fine_tune=fine_tune)

else:
    model = tubenet.create(learning_rate=lr0, loss=loss, class_weights=class_weights, 
                           metrics=['accuracy', tube.recall, tube.precision, tube.dice])

""" Train and save model """
if not prediction_only and header.label_filename is not None:
    #Log files
    date = datetime.datetime.now()
    filepath = os.path.join(model_path,"{}_model_checkpoint.h5".format(date.strftime("%d%m%y")))
    log_dir = os.path.join(model_path,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    #Callbacks
    schedule = partial(tube.piecewise_schedule, lr0=lr0, decay=0.9)
    checkpoint = ModelCheckpoint(filepath, monitor='dice', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=False)
    imageCallback = ImageDisplayCallback(data_generator,log_dir=os.path.join(log_dir,'images')) 
    filterCallback = FilterDisplayCallback(log_dir=os.path.join(log_dir,'filters')) #experimental
    metricCallback = MetricDisplayCallback(log_dir=log_dir)
        
	# Create directory of validation data
    if val_path is not None:
        # Import data header
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
        

        """ Manually set excluded region (to reserve for validation)"""
        # This section is under development
        # val_dir.exclude_region[0]=(None, None, (0,300)) # This exludes the top 300'rows' of pixels that were used for training

        vparams = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': None,
          'augment': False,
	       'shuffle': False}
        
        val_generator=DataGenerator(val_dir, **vparams)
        
        # TRAIN with validation
        history=model.fit_generator(generator=data_generator, validation_data=val_generator, validation_steps=4, epochs=n_epochs, steps_per_epoch=steps_per_epoch, 
                                    callbacks=[LearningRateScheduler(schedule), checkpoint, tbCallback, imageCallback, filterCallback, metricCallback])

    else:
        # TRAIN without validation
        history=model.fit_generator(generator=data_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch, 
                                    callbacks=[LearningRateScheduler(schedule), checkpoint, tbCallback, imageCallback, filterCallback, metricCallback])
   
    # SAVE MODEL
    if save_model:
        #model.save(os.path.join(model_path,updated_model_filename))
        model.save_weights(os.path.join(model_path,updated_model_filename), save_format='h5')

    """ Plot ROC """
    # Create directory of validation data
    if val_path is not None:
        # Import data header
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
                                
        """ Manually set excluded region (to reserve for validation)"""
        # This section is under development
        #val_dir.exclude_region[0]=(None, None, (0,300)) # This exludes the top 300'rows' of pixels that were used for training
        
        optimised_thresholds=tube.roc_analysis(model=model, data_dir=val_dir, volume_dims=volume_dims, batch_size=batch_size, overlap=None, 
                                               classes=(0,1), save_prediction=True, prediction_filename=output_path, binary_output=binary_output)

else:
    """Predict segmentation only - non training"""
    tube.predict_segmentation(model=model, data_dir=data_dir,
                        volume_dims=volume_dims, batch_size=batch_size, overlap=24, classes=list(range(n_classes)), 
                        binary_output=binary_output, save_output= True, path=output_path)
