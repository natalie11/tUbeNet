# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
from sklearn.model_selection import train_test_split
from functools import partial
import tUbeNet_functions as tube

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 4           	# factor by which images are downsampled in x and y dimensions 
pad_array = 1024	           	   # size images are padded up to, to achieve n^2 x n^2 structure 
volume_dims = (64,128,128)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 3			         	      # number of epoch for training CNN
batch_size = 2		 	       	   # batch size for training CNN
n_rep = 500		         	 	   # number of training cycle repetitions
use_saved_model = True	        	# use saved model structure and weights? Yes=True, No=False
save_model = False		        	   # save model structure and weights? Yes=True, No=False
fine_tuning = False               # prepare model for fine tuning by replacing classifier and freezing shallow layers
class_weights = (1,5) 	        	# relative weighting of background to blood vessel classes
binary_output = True 	           	# save as binary (True) or softmax (False)

# Paths and filenames
path = "F:\Paired datasets\HREM"
img_filename = os.path.join(path,"GFP_2044_2459.tif")
seg_filename = os.path.join(path,"Monica_seg_binary_2044_2459.tif")
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


""" Load and Preprocess data """
img_pad, seg_pad, classes = tube.data_preprocessing(image_filename=img_filename, label_filename=seg_filename, 
                                                    downsample_factor=downsample_factor, pad_array=pad_array)
test_size = int(round(img_pad.shape[0]*0.25))
img_pad, img_test, seg_pad, seg_test = train_test_split(img_pad, seg_pad, test_size=test_size, random_state = 42)

""" Load or Build Model """
if use_saved_model:
    model_gpu, model = tube.load_saved_model(model_path=model_path, filename=model_filename,
                         learning_rate=1e-5, n_gpus=2, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision],
                         freeze_layers=10, fine_tuning=fine_tuning, n_classes=len(classes))
else:
    model_gpu, model = tube.tUbeNet(n_classes=len(classes), input_height=volume_dims[1], input_width=volume_dims[2], input_depth=volume_dims[0], 
                                    n_gpus=2, learning_rate=1e-5, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision])

""" Train and save model """
#TRAIN
training_cycle_list, accuracy_list, precision_list, recall_list = tube.train_model(model=model, model_gpu=model_gpu, 
                                                                                   image_stack=img_pad, labels=seg_pad, 
                                                                                   image_test=img_test, labels_test=seg_test, 
                                                                                   volume_dims=volume_dims, batch_size=batch_size, n_rep=n_rep, n_epochs=n_epochs,
                                                                                   path=model_path, model_filename=updated_model_filename, output_filename=output_filename)

# SAVE MODEL
if save_model:
	tube.save_model(model, model_path, updated_model_filename)

""" Predict Segmentation """
whole_img_pad = tube.data_preprocessing(image_filename=whole_img_filename, downsample_factor=downsample_factor, pad_array=pad_array)
tube.predict_segmentation(model_gpu=model_gpu, image_stack=whole_img_pad,
                         volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
                         binary_output=True, save_output= True, prediction_filename = 'prediction', path=path)