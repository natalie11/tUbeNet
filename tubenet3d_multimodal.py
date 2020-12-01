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
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator, MetricDisplayCallback #, ImageDisplayCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 2		         	# number of1 epoch for training CNN
steps_per_epoch = 50		        # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
batch_size = 2		 	       	    # batch size for training CNN
class_weights = (1,1,1,1) 	        	# relative weighting of background to blood vessel classes
n_classes=4
dataset_weighting = (1,1,1,1)

# Training and prediction options
use_saved_model = False	        	# use previously saved model structure and weights? Yes=True, No=False
fine_tuning = False                 # prepare model for fine tuning by replacing classifier and freezing shallow layers? Yes=True, No=False
binary_output = False	           	# save as binary (True) or softmax (False)
save_model = True		        	# save model structure and weights? Yes=True, No=False
prediction_only = False             # if True -> training is skipped

""" Paths and filenames """
# Training data
data_path = "F:\COVID-CNN\paired_datasets\headers"

# Validation data
val_path = "F:\COVID-CNN//validation_dataset\headers" # Set to None is not using validation data

# Model
model_path = 'F:\COVID-CNN\paired_datasets'
model_filename = None # If not using an exisiting model, else set to None
updated_model_filename = 'COVID_test1_5epochs' # model will be saved under this name

# Image output
output_filename = 'F:\COVID-CNN\predictions'

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
    data_dir.label_filenames.append(header.label_filename+'.npy')
    data_dir.data_type.append('float32')


""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': dataset_weighting,
	       'shuffle': False}

data_generator=DataGenerator(data_dir, **params)


""" Load or Build Model """
# create partial for  to pass to complier
custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)
custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
custom_loss.__module__ = tube.weighted_crossentropy.__module__

# callbacks              
#time_callback = tube.TimeHistory()		      
#stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)


if use_saved_model:
    # Load exisiting model with or without fine tuning adjustment (fine tuning -> classifier replaced and first 10 layers frozen)
    model_gpu, model = tube.load_saved_model(model_path=model_path, filename=model_filename,
                         learning_rate=1e-5, n_gpus=2, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision],
                         freeze_layers=10, fine_tuning=fine_tuning, n_classes=n_classes)
else:
    model_gpu, model = tube.tUbeNet(n_classes=n_classes, input_height=volume_dims[1], input_width=volume_dims[2], input_depth=volume_dims[0], 
                                    n_gpus=2, learning_rate=1e-5, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision, tube.dice])

""" Train and save model """
if not prediction_only:
    #Log files
    date = datetime.datetime.now()
    filepath = os.path.join(model_path,"{}_model_checkpoint".format(date.strftime("%d%m%y")))
    log_dir = (os.path.join(model_path,'logs'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    #Callbacks
    schedule = partial(tube.piecewise_schedule, lr0=1e-5, decay=0.9)
    filepath = os.path.join(model_path,"multimodal_checkpoint")
    checkpoint = ModelCheckpoint(filepath, monitor='dice', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    #imageCallback = ImageDisplayCallback(data_generator,log_dir=log_dir)
    metricCallback = MetricDisplayCallback(log_dir=log_dir)
    
    #Fit
    history=model_gpu.fit_generator(generator=data_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch, 
                                    callbacks=[LearningRateScheduler(schedule), checkpoint, tbCallback, metricCallback])
    
    # SAVE MODEL
    if save_model:
    	tube.save_model(model, model_path, updated_model_filename)

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
                           data_type=[])
        
        # Fill directory from headers
        for header in headers:
            val_dir.list_IDs.append(header.ID)
            val_dir.image_dims.append(header.image_dims)
            val_dir.image_filenames.append(header.image_filename+'.npy')
            val_dir.label_filenames.append(header.label_filename+'.npy')
            val_dir.data_type.append('float32')
        
        tube.predict_segmentation(model_gpu=model_gpu, data_dir=val_dir,
                        volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1,2,3), 
                        binary_output=True, save_output= True, prediction_filename = 'prediction', path=output_filename)

else:
    """Predict segmentation only - non training"""
    tube.predict_segmentation(model_gpu=model_gpu, data_dir=data_dir,
                        volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
                        binary_output=True, save_output= True, prediction_filename = 'prediction', path=output_filename)
