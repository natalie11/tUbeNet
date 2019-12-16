# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
from functools import partial
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras.backend as K

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 100			         	      # number of epoch for training CNN
steps_per_epoch = 1000		         	 	      # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
batch_size = 2		 	       	   # batch size for training CNN
use_saved_model = False	        	# use saved model structure and weights? Yes=True, No=False
save_model = True		        	   # save model structure and weights? Yes=True, No=False
fine_tuning = False               # prepare model for fine tuning by replacing classifier and freezing shallow layers
class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
binary_output = False	           	# save as binary (True) or softmax (False)
n_classes=2

""" Paths and filenames """
# Training data
path = "F:\\Paired datasets\\train"
image_filenames = os.listdir(os.path.join(path,"data"))
label_filenames = os.listdir(os.path.join(path,"labels"))

# Validation data
val_path = "F:\\Paired datasets\\test"
X_test_filenames = os.listdir(os.path.join(val_path,"data"))
y_test_filenames = os.listdir(os.path.join(val_path,"labels"))

# Model
model_path = 'F:\\Paired datasets'
model_filename = ''
updated_model_filename = 'multimodal_cropped_100epochs_1000steps'
output_filename = 'output'

#----------------------------------------------------------------------------------------------------------------------------------------------

# create partial for  to pass to complier
custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)

# callbacks              
#time_callback = tube.TimeHistory()		      
#stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)

""" Create data Directory """
for i in range(len(image_filenames)):
    image_filenames[i]=os.path.join(path,'data\\'+image_filenames[i])
    label_filenames[i]=os.path.join(path,'labels\\'+label_filenames[i])

dt=np.dtype([('list_ID', object), ('image_filename', object), ('label_filename', object), 
             ('image_dims', int,(3,)), ('data_type', object)])
data = np.array([('CT', image_filenames[0], label_filenames[0], (489,667,544), 'float32'),
              ('HREM', image_filenames[1], label_filenames[1], (312,3071,2223), 'float32'),
              ('RSOM', image_filenames[2], label_filenames[2], (376,221,191), 'float32')], dtype=dt)

data_dir = DataDir(data['list_ID'], image_dims=data['image_dims'], image_filenames=data['image_filename'], 
                   label_filenames=data['label_filename'], data_type=data['data_type'])

""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
	       'shuffle': False}

training_generator=DataGenerator(data_dir, **params)
#
""" Load or Build Model """
if use_saved_model:
    model_gpu, model = tube.load_saved_model(model_path=model_path, filename=model_filename,
                         learning_rate=1e-5, n_gpus=2, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision],
                         freeze_layers=10, fine_tuning=fine_tuning, n_classes=n_classes)
else:
    model_gpu, model = tube.tUbeNet(n_classes=n_classes, input_height=volume_dims[1], input_width=volume_dims[2], input_depth=volume_dims[0], 
                                    n_gpus=2, learning_rate=1e-5, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision])

""" Train and save model """
#TRAIN
schedule = partial(tube.piecewise_schedule, lr0=1e-5, decay=0.9)
filepath = os.path.join(model_path,"multimodal_checkpoint")
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
history=model_gpu.fit_generator(generator=training_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch, 
                                callbacks=[LearningRateScheduler(schedule), checkpoint])

# SAVE MODEL
if save_model:
	tube.save_model(model, model_path, updated_model_filename)

""" Plot ROC """
# Create directory of validation data
for i in range(len(X_test_filenames)):
    X_test_filenames[i]=os.path.join(val_path,'data\\'+X_test_filenames[i])
    y_test_filenames[i]=os.path.join(val_path,'labels\\'+y_test_filenames[i])
    
val_data = np.array([('CT', X_test_filenames[0], y_test_filenames[0], (163,667,544), 'float32'),
              ('HREM', X_test_filenames[1], y_test_filenames[1], (104,3071,2223), 'float32'),
              ('RSOM', X_test_filenames[2], y_test_filenames[2], (125,221,191), 'float32')], dtype=dt)

val_data_dir = DataDir(val_data['list_ID'], image_dims=val_data['image_dims'], image_filenames=val_data['image_filename'], 
                   label_filenames=val_data['label_filename'], data_type=val_data['data_type'])


optimised_thresholds=tube.roc_analysis(model=model_gpu, data_dir=val_data_dir, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), save_prediction=True, prediction_filename='F:\Paired datasets\prediction')

#""" Predict Segmentation """
#whole_img_pad = tube.data_preprocessing(image_filename=whole_img_filename, downsample_factor=downsample_factor, pad_array=pad_array)
#tube.predict_segmentation(model_gpu=model_gpu, image_stack=whole_img_pad,
#                         volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
#                         binary_output=True, save_output= True, prediction_filename = 'prediction', path=path)