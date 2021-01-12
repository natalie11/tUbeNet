# -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data(and labels) and convert into numpy arrays


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
join = os.path.join
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataHeader
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
data_downsample_factor = 1               # factor by which images are downsampled in x and y dimensions 
pad_array = None	           	    # size images are padded up to, set to None if not padding 
crop_array = None #np.asarray([[0,1060],[0,1260],[0,1260]])

# Note: these optios only work when labels are defined
val_fraction = 0.25                 # fraction of data to use for validation, set 1 0 if not creating a validation set
crop = True                         # crop images if there are large sections of background containing no vessels

""" Paths and filenames """
# Data directory
sim = False
if not sim:
    path = "/home/simon/Dropbox/natalie_vessel_library"
    if False:
        image_filename = '/mnt/data2/tubenet_data/train/real/X/LS174T_yanan.nii' # join("image_data","LS174T_yanan.tif")
        label_filename = '/mnt/data2/tubenet_data/train/real/Y/LS174T_yanan_labels.nii' #join("image_labels","LS174T_yanan_labels.tif")  # Set to None if not using labels
        #midline_filename = "/mnt/data2/tubenet_data/midline/LS174T_yanan_midline.nii"
        midline_filename = '/mnt/data/data/tubenet_data/train/real/midline/LS174T_yanan_midline.nii'
        output_name = "ls174t_tumour"
    else:
        image_filename = join("image_data","liver_hrem.tif")
        #label_filename = join("image_labels","liver_hrem_labels.tif")  # Set to None if not using labels
        label_filename = join("image_labels","liver_hrem_labels_fixed.tif")  # Set to None if not using labels
        midline_filename = "/mnt/data2/tubenet_data/midline/liver_hrem_midline.nii"    
        output_name = "liver_hrem"
else: # simulation
    if False:
        path = "/home/simon/Dropbox/corrosion_cast_129/129_I_4x"
        image_filename = join(path,'mask.nii')
        label_filename = image_filename
    else:
        path = "/mnt/data2/normal_vessel_test2"
        stub = '468b963a-28e8-11eb-9fc6-d1d0fb1a5aa1'
        image_filename = join(path,'volume','{}.nii'.format(stub))
        label_filename = image_filename
        midline_filename = join(path,'volume_midline','{}_midline.nii'.format(stub))
        output_name = "sim_1"

# Output directory
output_path = "/mnt/data2/natalie_tubenet_data2"

downsample_factor = 10 # Factor by which labels are resampled as a map for bathc creation

#----------------------------------------------------------------------------------------------------------------------------------------------
# Define path for data & labels if necessary

# Create folders and filepaths
if not os.path.exists(output_path):
    os.makedirs(output_path)
train_folder = os.path.join(output_path,"train")
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
test_folder = os.path.join(output_path,"test")
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
train_filename = os.path.join(train_folder,str(output_name)+"_train")
test_filename = os.path.join(test_folder,str(output_name)+"_test")

train_header_folder = os.path.join(train_folder, "headers")
if not os.path.exists(train_header_folder):
    os.makedirs(train_header_folder)
train_header_filename = os.path.join(train_header_folder,str(output_name)+"_train_header")

test_header_folder = os.path.join(test_folder, "headers")
if not os.path.exists(test_header_folder):
    os.makedirs(test_header_folder)
test_header_filename = os.path.join(test_header_folder,str(output_name)+"_test_header")

train_label_filename = str(train_filename)+"_labels"
test_label_filename = str(test_filename)+"_labels"

train_midline_filename = str(train_filename)+"_midline"
test_midline_filename = str(test_filename)+"_midline"

train_downsample_filename = str(train_filename)+"_downsampled"
test_downsample_filename = str(test_filename)+"_downsampled"


# LABELS AND HEADERS ------

label_filename = os.path.join(path, label_filename)
labels, classes = tube.label_preprocessing(label_filename=label_filename, downsample_factor=data_downsample_factor, pad_array=pad_array, crop_array=crop_array, binary=True)
labels[labels>0.] = 1
labels = labels.astype('int8')

print('Labels dims: {}'.format(labels.shape))

# Crop
if crop:
    iz, ix, iy = np.where(labels[...]!=0) # find instances of non-zero values in X_test along axis 1
    labels = labels[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1] # use this to index y_test and y_pred

# Split into test and train
if val_fraction > 0:
    n_training_imgs = int(labels.shape[0]-np.floor(labels.shape[0]*val_fraction))
    
print('Labels croped dims: {}. #Train:{}, #Test:{}'.format(labels.shape,n_training_imgs,labels.shape[0]-n_training_imgs))

train_labels = labels[0:n_training_imgs,...]
test_labels = labels[n_training_imgs:,...]

# Create downsampled training labels
train_downsample_factor_arr = np.asarray([downsample_factor,downsample_factor,downsample_factor],dtype='int')
while True:
    train_downsample_dims = np.ceil(np.asarray(train_labels.shape)/train_downsample_factor_arr).astype('int')
    if np.any(train_downsample_dims<64):
        train_downsample_factor_arr[train_downsample_dims<64] = np.ceil(train_downsample_factor_arr[train_downsample_dims<64] / 2).astype('int')
    else:
        break
train_downsample_labels = np.zeros(train_downsample_dims,dtype='int16')
for i in range(0,train_labels.shape[0],train_downsample_factor_arr[0]):
    for j in range(0,train_labels.shape[1],train_downsample_factor_arr[1]):
        for k in range(0,train_labels.shape[2],train_downsample_factor_arr[2]):
            ii,jj,kk = np.round(np.array([i,j,k])/train_downsample_factor_arr).astype('int')
            train_downsample_labels[ii,jj,kk] = np.sum(train_labels[i:i+train_downsample_factor_arr[0],j:j+train_downsample_factor_arr[1],k:k+train_downsample_factor_arr[2]])
            
# Create downsampled testing labels
test_downsample_factor_arr = np.asarray([downsample_factor,downsample_factor,downsample_factor],dtype='int')
while True:
    test_downsample_dims = np.ceil(np.asarray(test_labels.shape)/test_downsample_factor_arr).astype('int')
    if np.any(test_downsample_dims<64):
        test_downsample_factor_arr[test_downsample_dims<64] = np.ceil(test_downsample_factor_arr[test_downsample_dims<64] / 2).astype('int')
    else:
        break
test_downsample_labels = np.zeros(test_downsample_dims,dtype='int16')
for i in range(0,test_labels.shape[0],test_downsample_factor_arr[0]):
    for j in range(0,test_labels.shape[1],test_downsample_factor_arr[1]):
        for k in range(0,test_labels.shape[2],test_downsample_factor_arr[2]):
            ii,jj,kk = np.round(np.array([i,j,k])/test_downsample_factor_arr).astype('int')
            test_downsample_labels[ii,jj,kk] = np.sum(test_labels[i:i+test_downsample_factor_arr[0],j:j+test_downsample_factor_arr[1],k:k+test_downsample_factor_arr[2]])

print('Downsample dimensions: train:{}, test:{}'.format(train_downsample_labels.shape,test_downsample_labels.shape))

# Save training labels and header
train_labels = train_labels.astype('int8')
np.save(train_label_filename,train_labels)
np.save(train_downsample_filename+'.npy',train_downsample_labels)
header = DataHeader(ID=output_name, image_dims=train_labels.shape, image_filename=train_filename, 
                    downsample_filename=train_downsample_filename, label_filename=train_label_filename,
                    downsample_factor=train_downsample_factor_arr,
                    midline_filename=train_midline_filename)
header.save(train_header_filename)
print("Processed training data and header files saved to "+str(train_folder))

# Save testing labels and header
np.save(test_label_filename,test_labels)
np.save(test_downsample_filename,test_downsample_labels)
header = DataHeader(ID=output_name, image_dims=test_labels.shape, image_filename=test_filename, 
                    downsample_filename=test_downsample_filename, label_filename=test_label_filename,
                    downsample_factor=test_downsample_factor_arr,
                    midline_filename=test_midline_filename)
header.save(test_header_filename)
print("Processed test data and header files saved to "+str(test_folder))

# IMAGE DATA --------------

image_filename = os.path.join(path, image_filename)
data = tube.data_preprocessing(image_filename=image_filename, downsample_factor=data_downsample_factor, pad_array=pad_array, crop_array=crop_array)

# Set data type
if sim:
    data *= 127
data = data.astype('float32')
if crop:
    data = data[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]
    
train_data = data[0:n_training_imgs,...]
test_data = data[n_training_imgs:,...]

print('Data dims: {}'.format(data.shape))

np.save(train_filename,train_data)
np.save(test_filename,test_data)


# MIDLINE ----------------        

midline_filename = os.path.join(path, midline_filename)
midline,_ = tube.label_preprocessing(label_filename=midline_filename, downsample_factor=data_downsample_factor, pad_array=pad_array, crop_array=crop_array, binary=True)
midline = midline.astype('int8')

# Crop
if crop:
    midline = midline[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]

train_midline = midline[0:n_training_imgs,...]
test_midline = midline[n_training_imgs:,...]

print('Midline dims: {}'.format(midline.shape))

np.save(train_midline_filename,train_midline)
np.save(test_midline_filename,test_midline)
       
