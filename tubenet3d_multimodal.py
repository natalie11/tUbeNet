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
from tUbeNet_classes import DataDir, DataGenerator, ImageDisplayCallback, MetricDisplayCallback, SaveModelCallback
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 500			         	# number of1 epoch for training CNN
steps_per_epoch = 100		        # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
batch_size = 20		 	       	    # batch size for training CNN
midline = False                      # include midline?
if not midline:
    class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
    n_classes = 2                   # =2 for binary mask, =3 for mask and midline
else:    
    class_weights = (1,7,150) 	        	# relative weighting of background to blood vessel classes
    n_classes = 3                   # =2 for binary mask, =3 for mask and midline
dataset_weighting = (100./3.,100./3.,100./3.,100./3.)

# Training and prediction options
use_saved_model = True	        	# use previously saved model structure and weights? Yes=True, No=False
resume = True                       # resume training with saved model
fine_tuning = False                 # prepare model for fine tuning by replacing classifier and freezing shallow layers? Yes=True, No=False
binary_output = False	           	# save as binary (True) or softmax (False)
save_model = True		        	# save model structure and weights? Yes=True, No=False
prediction_only = False             # if True -> training is skipped

""" Paths and filenames """
# Training data
#data_path = "F:\\Paired datasets\\train\\headers"
data_path = "/mnt/data2/natalie_tubenet_data2/train/headers"

# Validation data
#val_path = "F:\\Paired datasets\\test\\headers" # Set to None is not using validation data
data_path = "/mnt/data2/natalie_tubenet_data2/test/headers"

# Model
model_path = '/mnt/data2/natalie_tubenet_data2/model4/'
model_filename = 'multimodal_checkpoint.data-00001-of-00002' #None # If not using an exisiting model, else set to None
updated_model_filename = 'multimodal_cropped_100epochs_1000steps_Oct5' # model will be saved under this name

# Image output
output_filename = '/mnt/data2/natalie_tubenet_data2/prediction/'

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
                   midline_filenames=[],
                   downsample_filenames=[],
                   downsample_factor=[],
                   data_type=[])

# Fill directory from headers
for header in headers:
    data_dir.list_IDs.append(header.ID)
    data_dir.image_dims.append(header.image_dims)
    data_dir.image_filenames.append(header.image_filename+'.npy')
    data_dir.label_filenames.append(header.label_filename+'.npy')
    if midline:
        data_dir.midline_filenames.append(header.midline_filename+'.npy')
    else:
        data_dir.midline_filenames.append(None) 
    data_dir.downsample_filenames.append(header.downsample_filename+'.npy')
    data_dir.downsample_factor.append(header.downsample_factor)
    data_dir.data_type.append('float32')


""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': dataset_weighting,
	       'shuffle': False}

data_generator = DataGenerator(data_dir, **params)

""" Load or Build Model """
# create partial for  to pass to complier
custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)
custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
custom_loss.__module__ = tube.weighted_crossentropy.__module__

# callbacks              
#time_callback = tube.TimeHistory()		      
#stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)
lr_init = 1e-4
lr_decay = 0.99

#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
if True:
    metrics = ['accuracy', tube.recall, tube.precision, tube.dice]
    model_gpu, model = tube.tUbeNet(n_classes=n_classes, input_height=volume_dims[1], input_width=volume_dims[2], input_depth=volume_dims[0], 
                                    n_gpus=2, learning_rate=lr_init, loss=custom_loss, metrics=metrics)

""" Train and save model """
if not prediction_only:
    #Log files
    date = datetime.datetime.now()
    filepath = os.path.join(model_path,"{}_model_checkpoint".format(date.strftime("%d%m%y")))
    log_dir = (os.path.join(model_path,'logs'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    elif not resume: # Delete old logs if not resuming training
        import shutil
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        
    #Callbacks
    schedule = partial(tube.piecewise_schedule, lr0=lr_init, decay=lr_decay)
    filepath = os.path.join(model_path,"multimodal_checkpoint")
    checkpoint = ModelCheckpoint(filepath, monitor='dice', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    imageCallback = ImageDisplayCallback(data_generator,log_dir=log_dir)
    metricCallback = MetricDisplayCallback(log_dir=log_dir)
    
    savemodelCallback = SaveModelCallback(data_generator,path=model_path)
    if use_saved_model:
        wfile,initial_epoch = savemodelCallback.get_most_recent_saved_model()
        model_gpu.load_weights(wfile)
        if not resume:
            initial_epoch = 0
    else:
        initial_epoch = 0
    
    #Fit
    history = model_gpu.fit(data_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch,initial_epoch=initial_epoch, 
                                    callbacks=[LearningRateScheduler(schedule), checkpoint, tbCallback, imageCallback, metricCallback, savemodelCallback])
    
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
                           midline_filenames=[],
                           downsample_filenames=[],
                           downsample_factor=[],
                           data_type=[])
        
        # Fill directory from headers
        for header in headers:
            val_dir.list_IDs.append(header.ID)
            val_dir.image_dims.append(header.image_dims)
            val_dir.image_filenames.append(header.image_filename+'.npy')
            val_dir.label_filenames.append(header.label_filename+'.npy')
            if midline:
                val_dir.midline_filenames.append(header.midline_filename+'.npy')
            else:
                val_dir.midline_filenames.append(None)
            val_dir.data_type.append('float32')
            val_dir.downsample_factor.append(header.downsample_factor)
        
        optimised_thresholds=tube.roc_analysis(model=model_gpu, data_dir=val_dir, volume_dims=volume_dims, batch_size=batch_size, overlap=None, classes=(0,1), save_prediction=True, prediction_filename=output_filename)

else:
    """Predict segmentation only - non training"""
    tube.predict_segmentation(model_gpu=model_gpu, data_dir=data_dir,
                        volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
                        binary_output=True, save_output= True, prediction_filename = 'prediction', path=output_filename)
                        
# Model 3: Liver (fixed labels) and LS tumour only, lr=1e-4, decay=0.95 ('/mnt/data2/natalie_tubenet_data2/model3/'). Max dice ~ 0.936
# Model 4: Liver (fixed labels) and LS tumour only, lr=1e-4, decay=0.99 ('/mnt/data2/natalie_tubenet_data2/model4/')
