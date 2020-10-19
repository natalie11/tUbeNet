# -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data(and labels) and convert into numpy arrays


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 1               # factor by which images are downsampled in x and y dimensions 
pad_array = None	           	    # size images are padded up to, set to None if not padding 

# Note: these optios only work when labels are defined
val_fraction = 0.25                 # fraction of data to use for validation, set 1 0 if not creating a validation set
crop = True                         # crop images if there are large sections of background containing no vessels

""" Paths and filenames """
# Data directory
path = "F:\\Paired datasets"
image_filename = "image_data\\filtered_dataset_crop.tif"
label_filename = "image_labels\\filtered_dataset_Labels.tif"  # Set to None if not using labels

# Output directory
output_path = "F:\\Paired datasets"
output_name = "RSOM"


#----------------------------------------------------------------------------------------------------------------------------------------------
# Define paths using os.path.join
image_filename = os.path.join(path, image_filename)
if label_filename is not None:
    label_filename = os.path.join(path, label_filename)
if val_fraction > 0:
    train_folder = os.path.join(output_path,"train")
    test_folder = os.path.join(output_path,"test")

if label_filename is not None:
    data, labels, classes = tube.data_preprocessing(image_filename=image_filename, label_filename=label_filename,
                                           downsample_factor=downsample_factor, pad_array=pad_array)
else:
    data = tube.data_preprocessing(image_filename=image_filename, downsample_factor=downsample_factor, pad_array=pad_array)
    
# Set data type
data = data.astype('float32')
if label_filename is not None:
    labels = labels.astype('int8')

# Crop
if crop:
    iz, ix, iy = np.where(labels[...]!=0) # find instances of non-zero values in X_test along axis 1
    labels = labels[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1] # use this to index y_test and y_pred
    data = data[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]
    print(data.shape)


# Split into test and train
if val_fraction > 0:
    n_training_imgs = int(data.shape[0]-np.floor(data.shape[0]*val_fraction))
    
    train_data = data[0:n_training_imgs,...]
    train_labels = labels[0:n_training_imgs,...]
    
    test_data = data[n_training_imgs:,...]
    test_labels = labels[n_training_imgs:,...]
    
    # Save as numpy arrays
    np.save(os.path.join(train_folder,str(output_name)+"train"),train_data)
    np.save(os.path.join(train_folder,str(output_name)+"train_labels"),train_labels)
    
    np.save(os.path.join(test_folder,str(output_name)+"test"),test_data)
    np.save(os.path.join(test_folder,str(output_name)+"test_labels"),test_labels)
else:
    np.save(os.path.join(output_path, output_name), data)
    if label_filename is not None: np.save(os.path.join(output_path, str(output_name)+"_labels"), labels)