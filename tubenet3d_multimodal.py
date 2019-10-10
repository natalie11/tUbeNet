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

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 4           	# factor by which images are downsampled in x and y dimensions 
pad_array = 1024	           	   # size images are padded up to, to achieve n^2 x n^2 structure 
volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 3			         	      # number of epoch for training CNN
batch_size = 2		 	       	   # batch size for training CNN
n_rep = 500		         	 	      # number of training cycle repetitions
use_saved_model = False	        	# use saved model structure and weights? Yes=True, No=False
save_model = False		        	   # save model structure and weights? Yes=True, No=False
fine_tuning = False               # prepare model for fine tuning by replacing classifier and freezing shallow layers
class_weights = (1,5) 	        	# relative weighting of background to blood vessel classes
binary_output = True 	           	# save as binary (True) or softmax (False)
n_classes=2

# Paths and filenames
path = "F:\\Paired datasets"
image_filenames = os.listdir(os.path.join(path,"data"))
label_filenames = os.listdir(os.path.join(path,"labels"))
whole_img_filename = os.path.join(path,"GFP_2044_2459.tif")
model_path = "G:\\Vessel Segmentation\\saved_weights\\3D"
model_filename = 'HREM_Adam_50000x2epochs_excl0.1percentVessels_4xdownsampled_1-5weighting_FineTunedCT_3114cycles'
updated_model_filename = 'updated'
output_filename = 'output'


#----------------------------------------------------------------------------------------------------------------------------------------------

# create partial for  to pass to complier
custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)

#callbacks                
time_callback = tube.TimeHistory()		      
stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)

""" Create data Directory """
dt=np.dtype([('list_ID', object), ('modality', object), ('image_filename', object), ('label_filename', object), 
             ('image_dims', int,(3,)), ('data_type', object)])
data = np.array([('ID_1', 'CT', image_filenames[0], label_filenames[0], (682,1024,1024), 'float32'),
              ('ID_2', 'HREM', image_filenames[1], label_filenames[1], (416,4096,4096), 'float32'),
              ('ID_3', 'Retinal Angio', image_filenames[2], label_filenames[2], (64,512,512), 'float32'),
              ('ID_4', 'RSOM', image_filenames[3], label_filenames[3], (501,256,256), 'float32')], dtype=dt)

data_dir = DataDir(data['list_ID'], image_dims=data['image_dims'], image_filenames=data['image_filename'], 
                   label_filenames=data['label_filename'], data_type=data['data_type'])

""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
	       'shuffle': False}

training_generator=DataGenerator(data_dir, **params)


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
model_gpu.fit_generator(generator=training_generator)

# SAVE MODEL
if save_model:
	tube.save_model(model, model_path, updated_model_filename)

""" Predict Segmentation """
whole_img_pad = tube.data_preprocessing(image_filename=whole_img_filename, downsample_factor=downsample_factor, pad_array=pad_array)
tube.predict_segmentation(model_gpu=model_gpu, image_stack=whole_img_pad,
                         volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
                         binary_output=True, save_output= True, prediction_filename = 'prediction', path=path)