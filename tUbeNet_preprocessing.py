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
crop_array = np.asarray([[0,1060],[0,1260],[0,1260]])

# Note: these optios only work when labels are defined
val_fraction = 0.25                 # fraction of data to use for validation, set 1 0 if not creating a validation set
crop = True                         # crop images if there are large sections of background containing no vessels

""" Paths and filenames """
# Data directory
sim = True
if not sim:
    path = "/home/simon/Dropbox/natalie_vessel_library"
    image_filename = join("image_data","LS174T_yanan.tif")
    label_filename = join("image_labels","LS174T_yanan_labels.tif")  # Set to None if not using labels
else: # simulation
    #path = "/home/simon/Desktop/Share/normal_vessel_test/volume"
    #image_filename = join(path,'764a00ec-1a96-11eb-b037-81b9c433b1bc.nii')
    path = "/home/simon/Dropbox/corrosion_cast_129/129_I_4x"
    image_filename = join(path,'mask.nii')
    label_filename = image_filename

# Output directory
output_path = "/mnt/data2/natalie_tubenet_data"
output_name = "sim_covid_lung_cast_1"
downsample_factor = 10 # Factor by which labels are resampled as a map for bathc creation

#----------------------------------------------------------------------------------------------------------------------------------------------
# Define path for data & labels if necessary


for dtype in ['label','data']:

    if dtype=='data':
        image_filename = os.path.join(path, image_filename)
        data = tube.data_preprocessing(image_filename=image_filename, downsample_factor=data_downsample_factor, pad_array=pad_array, crop_array=crop_array)

        # Set data type
        if sim:
            data *= 127
        data = data.astype('float16')
        if crop:
            data = data[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]
            
        train_data = data[0:n_training_imgs,...]
        test_data = data[n_training_imgs:,...]
        
    elif dtype=='label' and         if label_filename is not None::
        label_filename = os.path.join(path, label_filename)
        labels, classes = tube.label_preprocessing(label_filename=label_filename, downsample_factor=data_downsample_factor, pad_array=pad_array, crop_array=crop_array)
        labels[labels>0.] = 1
        labels = labels.astype('int8')

        # Crop
        if crop:
            iz, ix, iy = np.where(labels[...]!=0) # find instances of non-zero values in X_test along axis 1
            labels = labels[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1] # use this to index y_test and y_pred

        # Split into test and train
        if val_fraction > 0:
            n_training_imgs = int(labels.shape[0]-np.floor(labels.shape[0]*val_fraction))
        
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
        label_filename = str(train_name)+"_labels"
        np.save(label_filename,train_labels)
        downsample_filename = str(train_name)+"_downsampled"
        np.save(downsample_filename+'.npy',train_downsample_labels)
        header = DataHeader(ID=output_name, image_dims=train_labels.shape, image_filename=train_name, 
                            downsample_filename=downsample_filename, label_filename=label_filename,
                            downsample_factor=train_downsample_factor_arr)
        header.save(header_name)
        print("Processed training data and header files saved to "+str(train_folder))
        
        # Save test data
        test_name=os.path.join(test_folder,str(output_name)+"_test")
        header_folder=os.path.join(test_folder, "headers")
        if not os.path.exists(header_folder):
            os.makedirs(header_folder)
        header_name=os.path.join(header_folder,str(output_name)+"_test_header")
        
        np.save(test_name,test_data)
        label_filename = str(test_name)+"_labels"
        np.save(label_filename,test_labels)
        downsample_filename = str(test_name)+"_downsampled"
        np.save(downsample_filename,test_downsample_labels)
        header = DataHeader(ID=output_name, image_dims=test_labels.shape, image_filename=test_name, 
                            downsample_filename=downsample_filename, label_filename=label_filename,
                            downsample_factor=test_downsample_factor_arr)
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
        
