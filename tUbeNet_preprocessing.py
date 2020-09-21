# -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load data from images and convert into numpy arrays

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataHeader
import argparse

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

parser = argparse.ArgumentParser(description="tUbNet training and valadation data processing script")

parser.add_argument("--downsample_factor", help="factor by which images are downsampled in x and y dimensions (default %(default)s)",
                    type=int, default=1)
parser.add_argument("--val_fraction", help="fraction of data to use for validation",
                    type=float)
parser.add_argument("-id","--identifier", help="include a string that can be used to identify this dataset (eg. modality)",
                    type=str, required=True)

parser.add_argument("--pad_array", help="set size to pad images to in x-y dimension (must be larger than image. If padding, no_crop flag is assumed)",
                    type=int)
parser.add_argument("--no_crop", help="include this flag if you do not want the images to be cropped",
                    action="store_true")

parser.add_argument("--image_filename", help="path and filename for image to be pre-processed",
                    type=str, required=True)
parser.add_argument("--label_filename", help="path and filename for labels to be pre-processed",
                    type=str, required=True)
parser.add_argument("--output_dir", help="path to folder in which output images will be saved",
                    type=str)

args=parser.parse_args()

# Paramters
downsample_factor = args.downsample_factor    
val_fraction = args.val_fraction                  
pad_array = args.pad_array 
if pad_array is not None: 
    no_crop = True # do not crop is padding
else:
    no_crop = args.no_crop         	      
modality = args.identifier


""" Paths and filenames """
image_filename = args.image_filename
label_filename = args.label_filename

# Validation data
output_dir = args.output_dir
output_name = os.path.join(output_dir, modality)

""" Parameters - when not calling from command line """
## Paramters
#downsample_factor = 1               	# factor by which images are downsampled in x and y dimensions 
#val_fraction = 0.25                   # fraction of data to use for validation  
#no_crop = False
#pad_array = None	           	       # size images are padded up to, to achieve n^2 x n^2 structure 
#modality = 'HREM'
#
#
#""" Paths and filenames """
#image_filename = ""
#label_filename = ""
#
## Validation data
#output_dir = "F:\\Paired datasets"
#output_name = os.path.join(output_dir, modality)
#----------------------------------------------------------------------------------------------------------------------------------------------

# Use preprocessing function to open images and labels, downsample and pad as neccessary, scale between 0 and 1, and find number of classes
data, labels, classes = tube.data_preprocessing(image_filename=image_filename, label_filename=label_filename,
                                           downsample_factor=downsample_factor, pad_array=pad_array, no_crop=no_crop)

# Split into test and train
if val_fraction is not None:
    n_training_imgs = int(data.shape[0]-np.floor(data.shape[0]*val_fraction))

    train_data = data[0:n_training_imgs,...]
    train_labels = labels[0:n_training_imgs,...]

    test_data = data[n_training_imgs:,...]
    test_labels = labels[n_training_imgs:,...]

    # Save as data as numpy arrays, and header as pickel dump
    output_file_train_data = str(output_name+"_train_data")
    output_file_train_labels = str(output_name+"_train_labels")
    output_file_test_data = str(output_name+"_test_data")
    output_file_test_labels = str(output_name+"_test_labels")   
    
    np.save(output_file_train_data,train_data)
    np.save(output_file_train_labels,train_labels)

    header_train = DataHeader(modality=modality, image_dims=train_data.shape, image_filename=output_file_train_data, 
                   label_filename=output_file_train_labels)
    header_train.save(str(output_name+'_train_header'))
    
    np.save(output_file_test_data,test_data)
    np.save(output_file_test_labels,test_labels)
    
    header_test = DataHeader(modality=modality, image_dims=test_data.shape, image_filename=output_file_test_data, 
                   label_filename=output_file_test_labels)
    header_test.save(str(output_name+'_test_header'))
    print("Processed data and header files saved to "+str(output_dir))
else:
    output_file_data = str(output_name+"_data")
    output_file_labels = str(output_name+"_labels")
    np.save(output_file_data, data)
    np.save(output_file_labels, labels)

    header = DataHeader(modality=modality, image_dims=data.shape, image_filename=output_file_data, 
                   label_filename=output_file_labels)
    header.save(str(output_name+'_header'))
    print("Processed data and header files saved to "+str(output_dir))