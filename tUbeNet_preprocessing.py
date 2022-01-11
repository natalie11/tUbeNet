# -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data(and labels) and convert into numpy arrays


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataHeader

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 1               # factor by which images are downsampled in x and y dimensions 
pad_array = None	           	    # size images are padded up to, set to None if not padding 

# Note: these optios only work when labels are defined
val_fraction = 0                 # fraction of data to use for validation, set 1 0 if not creating a validation set
crop = False                         # crop images if there are large sections of background containing no vessels

""" Paths and filenames """
# Data directory
path = 'C:/Users/Natal/Documents/CABI/Vessel data/SWS_to_try/corrosioncast'
image_filename = "20200512134413_127481_crop500.tiff"
label_filename = None

# Output directory
output_path = "C:/Users/Natal/Documents/CABI/Vessel data/SWS_to_try/preprocessed"
output_name = "corosioncast_20200512134413_127481_crop500"


#----------------------------------------------------------------------------------------------------------------------------------------------
# Define path for data & labels if necessary
image_filename = os.path.join(path, image_filename)
if label_filename is not None:
    label_filename = os.path.join(path, label_filename)


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
    # Create folders
    train_folder = os.path.join(output_path,"train")
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    test_folder = os.path.join(output_path,"test")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    # Save train data
    train_name=os.path.join(train_folder,str(output_name)+"_train")
    header_folder=os.path.join(train_folder, "headers")
    if not os.path.exists(header_folder):
        os.makedirs(header_folder)
    header_name=os.path.join(header_folder,str(output_name)+"_train_header")
    
    np.save(train_name,train_data)
    np.save(str(train_name)+"_labels",train_labels)
    header = DataHeader(ID=output_name, image_dims=train_labels.shape, image_filename=train_name, 
                        label_filename=str(train_name)+"_labels")
    header.save(header_name)
    print("Processed training data and header files saved to "+str(train_folder))
    
    # Save test data
    test_name=os.path.join(test_folder,str(output_name)+"_test")
    header_folder=os.path.join(test_folder, "headers")
    if not os.path.exists(header_folder):
        os.makedirs(header_folder)
    header_name=os.path.join(header_folder,str(output_name)+"_test_header")
    
    np.save(test_name,test_data)
    np.save(str(test_name)+"_labels",test_labels)
    header = DataHeader(ID=output_name, image_dims=test_labels.shape, image_filename=test_name, 
                        label_filename=str(test_name)+"_labels")
    header.save(header_name)
    print("Processed test data and header files saved to "+str(test_folder))
    
else:
    header_folder=os.path.join(output_path, "headers")
    if not os.path.exists(header_folder):
        os.makedirs(header_folder)
    header_name=os.path.join(header_folder,str(output_name)+"_header")
    
    # Save data as numpy array
    np.save(os.path.join(output_path, output_name), data)
    
    if label_filename is not None: 
        # Save labels as numpy array
        np.save(os.path.join(output_path, str(output_name)+"_labels"), labels)
        header = DataHeader(ID=output_name, image_dims=labels.shape, image_filename=os.path.join(output_path, output_name),
                            label_filename=os.path.join(output_path, str(output_name)+"_labels"))
        header.save(header_name)
    else:
        # Save header with label_filename=None
        header = DataHeader(ID=output_name, image_dims=data.shape, image_filename=os.path.join(output_path, output_name),
                            label_filename=None)
        header.save(str(header_name)+'_header')
    
    print("Processed data and header files saved to "+str(output_path))
        