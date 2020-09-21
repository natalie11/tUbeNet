# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:41:55 2018

@author: Natalie

Summary: Simple 11 layer autoencoding CNN with RELU activation, softmax output, catagorical crossentropy loss.
	    Data pre-processing is set up to take .tif images (liver vascularture HREM data, eosin contrast), 
	    which are scaled between 0 and 1 and split into test and train sets, before being subsampled into smaller 128x128 arrays.
	    HREM images are downsampled by a factor of four before being segmented
"""

import os

# import basic librarys
import numpy as np

# import preprocessing tools
from sklearn.model_selection import train_test_split

# import reuired objects and fuctions from keras
# Model object and load_model function
from keras.models import Model, model_from_json #load_model
# core layers
from keras.layers import Input #Dense, Dropout, Activation, Flatten
# CNN layers
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU, Dropout
# utilities
from keras.utils import multi_gpu_model #np_utils
# opimiser
from keras.optimizers import Adam
# checkpoint
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
# import time for recording time for each epoch
import time
# import tensor flow
import tensorflow as tf
# set backend as tensor flow
from keras import backend as K
K.set_image_dim_ordering('tf')

# import Image for loading data
from PIL import Image
from skimage import io

from functools import partial
import json

# HARD-CODED PARAMETERS
downsample_factor = 4 	# factor by which images are downsampled in x and y dimensions 
pad_array = 1024		# size images are padded up to, to achieve n^2 x n^2 structure 
subsample_size = 128		# size of image sub-arrays to be passed to CNN (n^2 x n^2) 
epoch_number = 500		# number of epoch for training CNN
batch_size = 32			# batch size for training CNN
test_size = 0.25		# proportion of data to be used for testing
use_saved_model = True 	# use saved model structure and weights? Yes=True, No=False
save_model = True 		# save model structure and weights? Yes=True, No=False
class_weights = (1,4) 	# relative weighting of background to blood vessel classes

# PATHS AND FILENAMES
path = "G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2"
img_filename = os.path.join(path,"GFP_2044_2459.tif")
seg_filename = os.path.join(path,"Monica_seg_binary_2044_2459.tif")
model_path = "D:\\HREM predicated segmentations\\Histories and models"
model_filename = 'PreTrained345epochsCT_Trained312img_500epochs'
updated_model_filename = 'PreTrained345epochsCT_Trained312img_1000epochs'
history_filename = 'PreTrained345epochsCT_Trained312img_1000epochs_history'

# FUNCTIONS
# define function to save images
def save_image(array, filename):
	image = Image.fromarray(array)
	image.save(filename)

# define function to subsample array into overlapping 128x128 squares
# n is the factor by which each dimension is to be divided
def subsample(array, subsample_size):	
	dims = array.shape # get dimensions of image array, dim[0]=no of images, dim[1]=image height
	n = dims[1]/subsample_size # find number of subarrays (n) that will fit arcoss image (assumes square image)
	sub_array = np.zeros([dims[0], int(n*n), subsample_size, subsample_size, dims[3]], dtype='float32') #where n*n determines position of subarray in the image
	for i in range(int(n)):
		for j in range(int(n)):
			sub_array[:,int(n*i+j),:,:,:] = array[:, i*subsample_size:(i+1)*subsample_size, j*subsample_size:(j+1)*subsample_size,:]
	sub_array = np.reshape(sub_array,(int(dims[0]*n*n), subsample_size, subsample_size, dims[3]))
	return sub_array

# reconstruct image from subarrays
def reconstruct(array, img_height, img_width, subsample_size):
	dims = array.shape
	n = img_height/subsample_size # find number of subarrays that fit across the original image
	nobjects = np.array(dims[0]/(n*n)) # calculate number of images before subsampling took place
	recon_array = np.zeros([int(nobjects), img_height, img_width], dtype='float32')
	for img in range(int(nobjects)): #reconstruct one image at a time
		for i in range(int(n)):
			for j in range(int(n)):	
				recon_array[img, i*subsample_size:(i+1)*subsample_size, j*subsample_size:(j+1)*subsample_size] = array[int((n*n*img)+(n*i+j)), :, :]
	return recon_array

# class for recording time
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# from https://github.com/keras-team/keras/issues/1625
class TimedStopping(Callback):
#    Stop training when enough time has passed.
#    Arguments
#        seconds: maximum time before stopping.
#        verbose: verbosity mode.
    
    def __init__(self, seconds=None, verbose=0):
        super(Callback, self).__init__()

        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)

time_callback = TimeHistory()		      
stop_time_callback = TimedStopping(seconds=86400, verbose=1) 
#stop_loss_callback = EarlyStopping(monitor='val_loss', min_delta=0.000005, patience=50, mode='min')

# LOAD DATA
print('Loading images from '+str(img_filename))
img=io.imread(img_filename)
seg=io.imread(seg_filename)
    
# PREPROCESSING
print('DATA PREPROCESSING')
print('Downsampling by a factor of '+str(downsample_factor))
from skimage.measure import block_reduce
img=block_reduce(img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)
#seg = seg[:,:,:,0] # get rid of extra dimensions
seg=block_reduce(seg, block_size=(1, downsample_factor, downsample_factor), func=np.max) #max instead of mean to maintain binary image

# Normalise 
print('Rescaling data between 0 and 1')
img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1
seg = (seg-np.amin(seg))/(np.amax(seg)-np.amin(seg))


# Seems that the CNN needs 2^n data dimensions (i.e. 64, 128, 256, etc.)
# Set the images to 1024x1024 (2^10) arrays
print('Padding arrays')
xpad=(pad_array-img.shape[1])//2
ypad=(pad_array-img.shape[2])//2

img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img

seg_pad = np.zeros([seg.shape[0],pad_array,pad_array], dtype='float32')
seg_pad[0:seg.shape[0],xpad:seg.shape[1]+xpad,ypad:seg.shape[2]+ypad] = seg

print('Shape of padded image array:')
print(img_pad.shape) 

# split into test and training sets
print('Splitting into train and test sets')
img_train, img_test, seg_train, seg_test = train_test_split(img_pad, seg_pad, test_size=test_size, random_state=42)
print('Size of train set = '+str(img_train.shape[0]))
print('Size of test set = '+str(img_test.shape[0]))

# Find the number of unique classes in segmented training set
classes = np.unique(seg_train)
n_classes = len(classes)
print('Number of classes = '+str(n_classes))
# Find image height and Width
img_height, img_width = img_train.shape[1:3] 

# Reshape data, specifying depth as 1 (allows matrix to be passed as input to conv2d)
img_train = img_train.reshape(img_train.shape[0], img_height, img_width, 1)
img_test = img_test.reshape(img_test.shape[0], img_height, img_width, 1)

# Convert to one-hot encoding
print('One Hot Encoding segmentation classes')
seg_train_ohe = np.zeros([seg_train.shape[0],img_height, img_width, n_classes],dtype='int')
for i,cl in enumerate(classes):
    index = np.where(seg_train==cl)
    tmp = np.zeros([seg_train.shape[0], img_height, img_width],dtype='int')
    tmp[index] = 1
    seg_train_ohe[:,:,:,i] = tmp

seg_test_ohe = np.zeros([seg_test.shape[0], img_height, img_width, n_classes],dtype='int')
for i,cl in enumerate(classes):
    inds = np.where(seg_test==cl)
    tmp = np.zeros([seg_test.shape[0], img_height, img_width],dtype='int')
    tmp[inds] = 1
    seg_test_ohe[:,:,:,i] = tmp
    
# break up image into subsections to avoid OOM error
print('Resampling data into '+str(subsample_size)+'x'+str(subsample_size)+' subarrays')
sub_img_train = subsample(img_train, subsample_size)
sub_img_test = subsample(img_test, subsample_size)
sub_seg_train_ohe = subsample(seg_train_ohe, subsample_size)
sub_seg_test_ohe = subsample(seg_test_ohe, subsample_size)

del img, seg, img_pad, seg_pad	#remove un-used variables

# BUILDING CNN
print('CNN')

##Weighted catagorical cross entropy
def weighted_crossentropy_Nat(y_true, y_pred, weights):

	weight_mask = y_true[...,0] * weights[0] + y_true[...,1] * weights[1]

	return K.categorical_crossentropy(y_true, y_pred,) * weight_mask

# create partial to pass to complier
custom_loss=partial(weighted_crossentropy_Nat, weights=class_weights)
# define model
def create_model(nClasses , input_height, input_width):

    """
    Adapted from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """

    inputs = Input((input_height, input_width, 1))
    conv1 = Conv2D(32, (3, 3), activation= 'linear', padding='same')(inputs)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation= 'linear', padding='same')(activ1)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(activ1)
    drop1 = Dropout(0.25)(pool1)   
      
    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(drop1)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(activ2)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(activ2)
    drop2 = Dropout(0.25)(pool2)
     

    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(drop2)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(activ3)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(activ3)
    drop3 = Dropout(0.5)(pool3)
    

    conv4 = Conv2D(256, (3, 3), activation='linear', padding='same')(drop3)
    activ4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='linear', padding='same')(activ4)
    activ4 = LeakyReLU(alpha=0.2)(conv4)			
    pool4 = MaxPooling2D(pool_size=(2, 2))(activ4)
    drop4 = Dropout(0.5)(pool4)
    
  
    conv5 = Conv2D(512, (3, 3), activation='linear', padding='same')(drop4)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='linear', padding='same')(activ5)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(activ5)    
    drop5 = Dropout(0.5)(pool5)
    
      
    conv6 = Conv2D(1024, (3, 3), activation='linear', padding='same')(drop5)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='linear', padding='same')(activ6)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    

    up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)    
    conv7 = Conv2D(512, (3, 3), activation='linear', padding='same')(up7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(512, (3, 3), activation='linear', padding='same')(activ7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)   

    up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='linear', padding='same')(up8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Conv2D(256, (3, 3), activation='linear', padding='same')(activ8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    

    up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='linear', padding='same')(up9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)
    conv9 = Conv2D(128, (3, 3), activation='linear', padding='same')(activ9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)

    up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='linear', padding='same')(up10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='linear', padding='same')(activ10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)

    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='linear', padding='same')(up11)
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    conv11 = Conv2D(32, (3, 3), activation='linear', padding='same')(activ11)    
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    
    conv12 = Conv2D(n_classes, (1, 1), activation='softmax')(conv11)

    # create model on CPU
    with tf.device("/cpu:0"):	
	    model = Model(inputs=[inputs], outputs=[conv12])
    # tell model to run on 2 gpus
    model_gpu = multi_gpu_model(model, gpus=2) 
    model_gpu.compile(optimizer=Adam(lr=1e-5), loss=custom_loss, metrics=['accuracy'])
    return model_gpu, model

# Load saved model or create new model
mfile = os.path.join(model_path,model_filename+'.h5') # file containing weights
jsonfile = os.path.join(model_path,model_filename+'.json') # file containing model template in json format
if use_saved_model:
	print('Loading model')
	# open json
	json_file = open(jsonfile, 'r')
	# load model from json
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights(mfile)
	# Replace classifier (if fine-tuning)
#	if fine_tuning:
#		new_classifier = Conv3D(n_classes, (1, 1), activation='softmax')(conv11)
#		model = Model(inputs=[inputs], outputs=[new_classifier])
#	model_gpu = multi_gpu_model(model, gpus=2) 
	model_gpu.compile(optimizer=Adam(lr=1e-5), loss=custom_loss, metrics=['accuracy'])
else:   
	print('Building model')
	model_gpu, model = create_model(n_classes, input_height = subsample_size,input_width = subsample_size)
print('Template model structure')
model.summary()
print('Multi-GPU model structure')
model_gpu.summary()

## TRAINING CNN

# Train model on training images and masks, run through all subarrays of all images
print('Fiting model to training set')
print('Number of epochs = '+str(epoch_number))
history = model_gpu.fit(sub_img_train, sub_seg_train_ohe, batch_size=batch_size, epochs=epoch_number, callbacks=[time_callback, stop_time_callback], verbose=1)

# SAVE MODEL


if save_model:
	print('Saving model')
	# serialize model to JSON
	model_json = model.to_json()
	updated_mfile = os.path.join(model_path,updated_model_filename+'.h5') # file containing weights
	updated_jsonfile = os.path.join(model_path,updated_model_filename+'.json') # file containing model template in json format
	with open(updated_jsonfile, "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights(updated_mfile)
	# save history
	history_path = os.path.join(model_path, history_filename+'.json')
	time_path = os.path.join(model_path, history_filename+'_time.json')
	with open(history_path, 'w') as f:
		json.dump(history.history,f)
	with open(time_path, 'w') as f:
		json.dump(time_callback.times,f)
	     
# TESTING CNN

# Evaluate model on test set
print('Evaluating model on test set')
score = model_gpu.evaluate(sub_img_test[:,:,:,:], sub_seg_test_ohe[:,:,:,:], verbose=1)
print(score)

# Get predicted segmentations (encoded)
print('Using model to predict segmentation for test set')
seg_pred_ohe = model_gpu.predict(sub_img_test[:,:,:,:],verbose=1)

# Convert one-hot encoding back to a label image
print('Converting OHE array into labelled image')																																																																																																																																					
class_pred = np.argmax(seg_pred_ohe,axis=3) # find most probable class for each pixel
seg_pred = np.zeros([seg_pred_ohe.shape[0],seg_pred_ohe.shape[1],seg_pred_ohe.shape[2]],dtype=seg_test.dtype)
for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
    seg_pred[class_pred==i] = cls
    
# Recontstruct image from subarrays
recon_seg_pred = reconstruct(seg_pred, pad_array, pad_array, subsample_size)

# Save segmented image
print('Saving segmented images')
for i in range(seg_test.shape[0]):
	array = recon_seg_pred[i, :, :]
	filename = os.path.join(path,str(i+1)+"_pred_"+str(epoch_number)+"epochs.tif")
	save_image(array, filename)
print('Saving true segmented images')
for i in range(seg_test.shape[0]):
	array = seg_test[i, :, :]
	filename = os.path.join(path,str(i+1)+"_test_"+str(epoch_number)+"epochs.tif")
	save_image(array, filename)
	
	
## SEGMENT UNLABELLED DATA

del img_test, img_train, seg_test, seg_train, seg_pred, recon_seg_pred, class_pred

# Load unlabelled data
print('LOADING UNLABELLED DATA')
path = "G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2"
img_filename = os.path.join(path,"GFP.tif")

print('Loading images from '+str(img_filename))
img=io.imread(img_filename)

print('Downsampling by a factor of '+str(downsample_factor))
from skimage.measure import block_reduce
img=block_reduce(img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)

# Normalise 
print('Rescaling data between 0 and 1')
img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1

# Set the images to 1024x1024 (2^10) arrays
print('Padding arrays')
xpad=(pad_array-img.shape[1])//2
ypad=(pad_array-img.shape[2])//2

img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img

del img

img_pad = img_pad.reshape(img_pad.shape[0], pad_array, pad_array, 1)

print('Resampling data into '+str(subsample_size)+'x'+str(subsample_size)+' subarrays')
img_pad = subsample(img_pad, subsample_size)

print('Using model to predict segmentation for unlabelled data')
seg_pred_ohe = model_gpu.predict(img_pad[:,:,:,:],verbose=1)

print('Converting OHE array into labelled image')																																																																																																																																					
class_pred = np.argmax(seg_pred_ohe,axis=3) # find most probable class for each pixel
seg_pred = np.zeros([seg_pred_ohe.shape[0],seg_pred_ohe.shape[1],seg_pred_ohe.shape[2]],dtype=img_pad.dtype)
for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
    seg_pred[class_pred==i] = cls
    
del class_pred
    
# Recontstruct image from subarrays
recon_seg_pred = reconstruct(seg_pred, pad_array, pad_array, subsample_size)

del seg_pred

# Save segmented image
print('Saving segmented images')
for i in range(recon_seg_pred.shape[0]):
	array = recon_seg_pred[i, :, :]
	filename = os.path.join(path,str(i+1)+"_wholestack_pred.tif")
	save_image(array, filename)
