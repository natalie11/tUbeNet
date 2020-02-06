# -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load data from images and convert into numpy arrays

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 1               	# factor by which images are downsampled in x and y dimensions 
pad_array = 256	           	       # size images are padded up to, to achieve n^2 x n^2 structure 
val_fraction = 0.00                   # fraction of data to use for validation  

""" Paths and filenames """
path = "F:\\Paired datasets"
image_filename = os.path.join(path,"image_data\\GFP_2044_2459.tif")
label_filename = os.path.join(path,"image_labels\\Monica_seg_binary_2044_2459.tif")

# Validation data
output_path = "F:\\Paired datasets"
output_name = "Retinal_angio"
train_folder = os.path.join(output_path,"train")
test_folder = os.path.join(output_path,"test")

#----------------------------------------------------------------------------------------------------------------------------------------------


data, labels, classes = tube.data_preprocessing(image_filename=image_filename, label_filename=label_filename,
                                           downsample_factor=downsample_factor, pad_array=None)

# Set data type
data = data.astype('float32')
labels = labels.astype('int8')

iz, ix, iy = np.where(labels[...]!=0) # find instances of non-zero values in X_test along axis 1
labels = labels[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1] # use this to index y_test and y_pred
data = data[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]
print(data.shape)


# Split into test and train
n_training_imgs = int(data.shape[0]-np.floor(data.shape[0]*val_fraction))

train_data = data[0:n_training_imgs,...]
train_labels = labels[0:n_training_imgs,...]

test_data = data[n_training_imgs:,...]
test_labels = labels[n_training_imgs:,...]

# Save as numpy arrays
np.save(os.path.join(train_folder,str(output_name)+"train_crop"),train_data)
np.save(os.path.join(train_folder,str(output_name)+"train_labels_crop"),train_labels)

np.save(os.path.join(test_folder,str(output_name)+"test_crop"),test_data)
np.save(os.path.join(test_folder,str(output_name)+"test_labels_crop"),test_labels)