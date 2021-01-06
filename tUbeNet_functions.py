# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import random
import math

# import required objects and fuctions from keras
from tensorflow.keras.models import Model, model_from_json #load_model
# CNN layers
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, LeakyReLU, Dropout#, AveragePooling3D, Reshape, Flatten, Dense, Lambda
# utilities
from tensorflow.keras.utils import multi_gpu_model, to_categorical #np_utils
# opimiser
from tensorflow.keras.optimizers import Adam
# checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, LearningRateScheduler
# import time for recording time for each epoch
import time

# import tensor flow
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables warning about not utilizing AVX AVX2
# set backend and dim ordering (updated for keras 2.5/ tf 2, keras is now within tensorflow)
K=tf.keras.backend
K.set_image_data_format('channels_last')


import matplotlib.pyplot as plt

# import Image for loading data
from PIL import Image
from skimage import io
from skimage.measure import block_reduce
from sklearn.metrics import roc_auc_score, roc_curve, auc, cohen_kappa_score, confusion_matrix, classification_report

#----------------------------------------------------------------------------------------------------------------------------------------------
def save_image(array, filename):
    """Save image to file
    Inputs: 
        array = 2D np.array containing mage data to be saved
        filename = name under which to save file, including file extension
    """
    image = Image.fromarray(array)
    image.save(filename)

def load_volume_from_file(volume_dims=(64,64,64), image_dims=None, 
				  image_filename=None, label_filename=None, 
				  coords=None, data_type='float64', offset=128):
  """Load a sub-volume of the 3D image. 
  These can either be generated randomly, or be taked from defined co-ordinates (z, x, y) within the image.
  Inputs: 
    volume_dims = shape of sub-volume, given as (z,x,y) (tuple of int, default (64,64,64))
    image_dims = shape of image to be accessed, (z,x,y) (tuple of int)
    image_filename = filename for np array of pre-processed image data (z,x,y)
    label_filename = filename for np array of labels, (z,x,y,c) where c is the number of chanels. Array should be int8. Optional.
    coords = coordinates for top left corner of sub-volume (tuple of int, if not specified will be randomly generated)
    data_type = data type of the image array (string, default 'float634')
    offset = number of bytes in file before pixel data begins (int, default 128)
  Outputs:
    volume = sub-section of the image array with dimensions equal to volume_dims
    labels_volume = corrosponding array of image labels (if labels provided)
  """

  #Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
  
  # Check for sensible volume dimensions
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_dims[i]:
      raise Exception('Volume dimensions out of range')
	
  if coords is not None:
    # check for sensible coordinates
    for i in range(3):
      if coords[i]<0 or coords[i]>(image_dims[i]-volume_dims[i]):
        raise Exception('Coordinates out of range')
  else:
    # generate random coordinates for upper left corner of volume
    coords = np.zeros(3)
    coords[0] = random.randint(0,(image_dims[0]-volume_dims[0]))
    coords[1] = random.randint(0,(image_dims[1]-volume_dims[1])) 
    coords[2] = random.randint(0,(image_dims[2]-volume_dims[2]))
  
  # Set number of bytes per pixel, depending on data_type  
  if data_type == 'float64' or data_type == 'int64':
	  pixel = 8
  elif data_type == 'float32' or data_type == 'int32':
	  pixel = 4
  elif data_type == 'int16':
	  pixel = 2
  elif data_type == 'int8' or data_type == 'bool':
	  pixel = 1
  else: raise Exception('Data type not supported')
	
  # Calculate y axis and z axis offset (number of bytes to skip to get to the next row)
  y_offset = image_dims[2]*pixel
  z_offset = image_dims[1]*image_dims[2]*pixel
  
  # Load data from file, one row at a time, using memmap
  volume=np.zeros(volume_dims)
  for z in range(volume_dims[0]):
    for x in range(volume_dims[1]):
        offset_zx = np.int64(offset) + np.int64(pixel*(coords[2])) + np.int64(y_offset)*np.int64(x+coords[1]) + np.int64(z_offset)*np.int64(z+coords[0])
        volume[z,x,:]=np.memmap(image_filename, dtype=data_type,mode='c',shape=(1,1,volume_dims[2]),
			 offset=offset_zx)
  
  # If labels_filename given, generate labels_volume using same coordinates
  if label_filename is not None:
      labels_volume = np.zeros(volume_dims)
      for z in range(volume_dims[0]):
          for x in range(volume_dims[1]):
              offset_zx = np.int64(offset) + np.int64(coords[2]) + np.int64(image_dims[2])*np.int64(x+coords[1]) + np.int64(image_dims[1])*np.int64(image_dims[2])*np.int64(z+coords[0])
              labels_volume[z,x,:]=np.memmap(label_filename, dtype='int8',mode='c',shape=(1,1,volume_dims[2]),
                         offset=offset_zx)
      return volume, labels_volume
  else:
      return volume


def load_volume(volume_dims=(64,64,64), image_stack=None, labels=None, coords=None):
  """Load a sub-volume of the 3D image. 
    These can either be generated randomly, or be taked from defined co-ordinates (z, x, y) within the image.
    Inputs: 
    	volume_dims = shape of sub-volume, given as (z,x,y), tuple of int
    	image_stack = 3D image, preprocessed and given as np array (z,x,y)
    	labels = np array of labels, (z,x,y,c) where c is the number of chanels. Should be binary and one hot encoded. Optional.
    	coords = coordinates for top left corner of sub-volume (if not specified, will be randomly generated)
    Outputs:
    	volume = sub-section of the image array with dimensions equal to volume_dims
    	labels_volume = corrosponding array of image labels (if labels provided)
    """
  image_dims = image_stack.shape #image_dims[0] = z, [1] = x, [2] = y
  
  #Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
  # Check for sensible volume dimensions
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_dims[i]:
      raise Exception('Volume dimensions out of range')
	
  if coords is not None:
    # check for sensible coordinates
    for i in range(3):
      if coords[i]<0 or coords[i]>(image_dims[i]-volume_dims[i]):
        raise Exception('Coordinates out of range')
  else:
    # generate random coordinates for upper left corner of volume
    coords = np.zeros(3)
    coords[0] = random.randint(0,(image_dims[0]-volume_dims[0]))
    coords[1] = random.randint(0,(image_dims[1]-volume_dims[1])) 
    coords[2] = random.randint(0,(image_dims[2]-volume_dims[2]))
  
  # Create volume from coordinates
  volume = image_stack[int(coords[0]):int(coords[0] + volume_dims[0]), int(coords[1]):int(coords[1] + volume_dims[1]), int(coords[2]):int(coords[2] + volume_dims[2])]
  
  if labels is not None:
    # Create corrsponding labels
    labels_volume = labels[int(coords[0]):int(coords[0] + volume_dims[0]), int(coords[1]):int(coords[1] + volume_dims[1]), int(coords[2]):int(coords[2] + volume_dims[2])]
    return volume, labels_volume
  else:
    return volume


# load batch of sub-volumes for training or label prediction
def load_batch(batch_size=1, volume_dims=(64,64,64), 
               image_stack=None, labels=None, 
               coords=None, n_classes=None, step_size=None): 
  """Load a batch of sub-volumes
    Inputs:
    	batch size = number of image sub-volumes per batch (int, default 1)
    	volume_dims = shape of sub-volume, given as (z,x,y), tuple of int
    	image_stack = 3D image, preprocessed and given as np array (z,x,y)
    	labels = np array of labels, (z,x,y,c) where c is the number of chanels. Should be binary and one hot encoded. Optional.
    	coords = coordinates for top left corner of sub-volume (if not specified, will be randomly generated)
    	step_size = if loading volumes with pre-specified coordinates, this specifies the pixel distance between consecutive volumes 
    			(equals volume_dims by default), (z,x,y) tuple of int
    Output:
    	img_batch = array of image sub-volumes in the format: (batch_size, img_depth, img_hight, img_width)"""
    
  # Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
  # Check for sensible volume dimensions
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_stack.shape[i]:
      raise Exception('Volume dimensions out of range')   
    
  if coords is not None:
    # Check for sensible coordinates
    if step_size is None: step_size = volume_dims
    for i in range(3):
      if coords[i]<0 or coords[i]>(image_stack.shape[i]-volume_dims[i]):
        raise Exception('Coordinates out of range')
    # Check if labels have been provided
    if labels is not None:
      img_batch = []
      labels_batch = []
      for z in range(batch_size):
        # Find coordinates
        tmp_coords = (coords[0]+ (step_size[0]*z), coords[1], coords[2]) # move one volume dimension along the z axis
        print('Loading image volume and labels with coordinates:{}'.format(tmp_coords))
        # Load sub-volume, image and labels
        volume, labels_volume = load_volume(volume_dims=volume_dims, coords=tmp_coords, image_stack=image_stack, labels=labels)
        # Reshape volume and append to batch
        volume = volume.reshape(volume_dims[0], volume_dims[1], volume_dims[2], 1)
        img_batch.append(volume)
        # One-hot-encoding labels and append to batch
        labels_volume = to_categorical(labels_volume, n_classes)
        labels_batch.append(labels_volume)
    else:
      img_batch = []
      for z in range(batch_size):			
        # Find coordinates
        tmp_coords = (coords[0]+ (step_size[0]*z), coords[1], coords[2]) # move one volume dimension along the z axis
        print('Loading image volume with coordinates:{}'.format(tmp_coords))
        # Load random sub-volume, image and labels
        volume = load_volume(volume_dims=volume_dims, coords=tmp_coords, image_stack=image_stack)
        # Reshape volume and append to batch
        volume = volume.reshape(volume_dims[0], volume_dims[1], volume_dims[2], 1)
        img_batch.append(volume)
  else:
    # Load volumes with randomly generated coordinates
    if labels is not None:
      img_batch = []
      labels_batch = []
      for z in range(batch_size):
        # Load sub-volume, image and labels
        volume, labels_volume = load_volume(volume_dims=volume_dims, image_stack=image_stack, labels=labels)
        # Reshape volume and append to batch
        volume = volume.reshape(volume_dims[0], volume_dims[1], volume_dims[2], 1)
        img_batch.append(volume)
        # One-hot-encoding labels and append to batch
        labels_volume = to_categorical(labels_volume, n_classes)
        labels_batch.append(labels_volume)
    else:
      img_batch = []
      for z in range(batch_size):
        # Load random sub-volume, image only
        volume = load_volume(volume_dims=volume_dims, image_stack=image_stack)
        # Reshape volume and append to batch
        volume = volume.reshape(volume_dims[0], volume_dims[1], volume_dims[2], 1)
        img_batch.append(volume)
 
	# Convert img_batch to np arrray
  img_batch = np.asarray(img_batch)
	# Return either img_batch or img_batch and labels_batch   
  if labels is not None:
    labels_batch = np.asarray(labels_batch)
    return img_batch, labels_batch
  else:
    return img_batch

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def weighted_crossentropy(y_true, y_pred, weights):
	"""Custom loss function - weighted to address class imbalance"""
	if weights is None: 
		return K.categorical_crossentropy(y_true, y_pred,) # No weighting
	else:
		weight_mask = y_true[...,0]*weights[0]
		for i in range(1,len(weights)):
			weight_mask += y_true[...,i]*weights[i]
		return K.categorical_crossentropy(y_true, y_pred,) * weight_mask



"""Custom metrics"""
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def dice(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    dice = 2*(P*R)/(P+R+K.epsilon())
    return dice

def jaccard(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3,4])
  union = K.sum(y_true,[1,2,3,4])+K.sum(y_pred,[1,2,3,4])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

#Tian's metrics, use when y_true/y_pred are np arrays rather than keras tensors
def precision_logical(y_true, y_pred):
	#true positive
	TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
	#false positive
	FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
	precision1=TP/(TP+FP)
	return precision1

def recall_logical(y_true, y_pred):
	#true positive
	TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
	#false negative
	FN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
	recall1=TP/(TP+FN)
	return recall1

"""Custom callbacks"""
class TimeHistory(Callback):
    # Record time taken to perform each epoch
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class TimedStopping(Callback):
#    From https://github.com/keras-team/keras/issues/1625
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

#---------------------------------------------------------------------------------------------------------------------------------------------------

def tUbeNet(n_classes=2, input_height=64, input_width=64, input_depth=64, 
            n_gpus=None, learning_rate=1e-3, loss=None, metrics=['accuracy']):
    """tUbeNet model
    Inputs:
        n_classes = number of classes (int, default 2)
        input_height = hight of input image (int, default 64)
        input_width = width of input image (int, default 64)
        input_depth = depth of input image (int, default 64)
        n_gpus = number of GPUs to train o, if not provided model will train on CPU (int, default None)
        learning_rate = learning rate (float, default 1e-3)
        loss = loss function, function or string
        metrics = training metrics, list of functions or strings
    Outputs:
        model = compiled model
        model_gpu = compiled multi-GPU model
    
    Adapted from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """

    inputs = Input((input_depth, input_height, input_width, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')(inputs)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')(activ1)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(activ1)
    drop1 = Dropout(0.25)(pool1)   
      
    conv2 = Conv3D(64, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(drop1)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ2)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(activ2)
    drop2 = Dropout(0.25)(pool2)
     

    conv3 = Conv3D(128, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(drop2)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ3)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(activ3)
    drop3 = Dropout(0.5)(pool3)
    

    conv4 = Conv3D(256, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(drop3)
    activ4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ4)
    activ4 = LeakyReLU(alpha=0.2)(conv4)			
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(activ4)
    drop4 = Dropout(0.5)(pool4)
    
  
    conv5 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(drop4)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ5)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(activ5)    
    drop5 = Dropout(0.5)(pool5)
    
      
    conv6 = Conv3D(1024, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(drop5)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ6)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    
#    # Global Feature Vector
#    _,d,h,w,n = activ6._shape_tuple()
#    ap = AveragePooling3D(pool_size=(d,h,w))(activ6)
#    fl = Flatten()(ap)
#    out = Dense(n,kernel_initializer=RandomNormal(stddev=0.02))(fl)
#    out = Reshape((1,1,1,n))(out)
#    out = Lambda(K.tile, arguments={'n':(1,d,h,w,1)})(out)
    

    up7 = concatenate([Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')(conv6), activ5], axis=4)    
    conv7 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(up7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)   

    up8 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')(activ7), activ4], axis=4)
    conv8 = Conv3D(256, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(up8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Conv3D(256, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    

    up9 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')(activ8), activ3], axis=4)
    conv9 = Conv3D(128, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(up9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)
    conv9 = Conv3D(128, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)

    up10 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')(activ9), activ2], axis=4)
    conv10 = Conv3D(64, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(up10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)
    conv10 = Conv3D(64, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)

    up11 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')(activ10), activ1], axis=4)
    conv11 = Conv3D(32, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(up11)
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    conv11 = Conv3D(32, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(activ11)    
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    
    conv12 = Conv3D(n_classes, (1, 1, 1), activation='softmax')(activ11)
        
    # create model on CPU
    if n_gpus is not None:
        with tf.device("/cpu:0"):	
    	    model = Model(inputs=[inputs], outputs=[conv12])
        
        # tell model to run on multiple gpus
        model_gpu = multi_gpu_model(model, gpus=n_gpus) 
        model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
        return model_gpu, model
    else:
        model = Model(inputs=[inputs], outputs=[conv12])
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)



def fine_tuning(model=None, n_classes=2, freeze_layers=0, n_gpus=None, 
                learning_rate=1e-5, loss=None, metrics=['accuracy']):
    """ Fine Tuning
    Replaces classifer layer and freezes shallow layers for fine tuning
    Inputs:
        model = ML model
        n_classes = number of classes (int, default 2)
        freeze_layers = number of layers to freeze for training (int, default 0)
        n_gpus = number of GPUs to train on, if undefined model will train on CPU (int, default None)
        learning_rate = learning rate (float, default 1e-5)
        loss = loss function, function or string
        metrics = training metrics, list of functions or strings
    Outputs:
        model = compiled model
        model_gpu = compiled multi-GPU model
    """

    # recover the output from the last layer in the model and use as input to new Classifer
    last = model.layers[-2].output
    classifier = Conv3D(n_classes, (1, 1, 1), activation='softmax', name='newClassifier')(last)
    
    # rename new classifier layer to avoid error caused by layer having the same name as first layer of base model
    
    model = Model(inputs=[model.input], outputs=[classifier])
    
    # freeze weights for selected layers
    for layer in model.layers[:freeze_layers]: layer.trainable = False
    
    if n_gpus is not None:       
        # tell model to run on multiple gpus
        model_gpu = multi_gpu_model(model, gpus=n_gpus) 
        model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
        return model_gpu, model
    else:
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
        return model
 
def piecewise_schedule(i, lr0, decay):
	""" Learning rate function 
    Updates learning rate at end epoch.
    Inputs:
        i = training epoch (int)
        lr0 = initial learning rate (float)
        decay = decay rate (float)
    """
	lr = lr0 * decay**(i)
	return lr

# train model on image_stack and corrosponding labels
# def train_model(model=None, model_gpu=None, 
#                 image_stack=None, labels=None, 
#                 image_test=None, labels_test=None, 
#                 volume_dims=(64,64,64), batch_size=2, n_rep=100, n_epochs=2,
#                 path=None, model_filename=None, output_filename=None):
#     """ Training 
#     Inputs:
#         model = ML model
#         model_gpu = model compiled on multiple GPUs
#         image_stack = np array of image data (z, x, y)
#         labels = np array of labels, (z, x, y, c) where c is number of classes
#         image_test = valdiation image data
#         labels_test = validation label data
#         voume_dims = sub-volume size to be passed to model ((z,x,y) int, default (64,64,64))
#         batch_size = number of image sub volumes per batch int, default 2)
#         n_rep = number of training iterations - each iteration is trained on a new batch of sub-voumes (int, default 100)
#         n_epochs = number of epochs per training iteration (int, default 2)
#         path = path for saving outputs and updated model
#         model_filename = filename for trained model
#         output_filename = filename for saving output graphs
#     """

#     print('Training model')
#     print('Number of epochs = {}'.format(n_epochs))
#     accuracy_list=[]   
#     precision_list=[]
#     recall_list=[] 
#     training_cycle_list=[] 
    
#     classes = np.unique(labels)
#     n_classes = len(classes)
#     print('Training with {} classes'.format(n_classes)) 
      
#     # Train for *n_rep* cycles of *n_epochs* epochs, loading a new batch of volumes every cycle
#     for i in range(n_rep):
# 	    print('Training cycle {}'.format(i))
      
#       # When loading a batch of volumes, check for the percentage of vessel pixels present and only train of volumes containing >0.01% vessels
#       # These numbers may need to be changed depending on the vessel density in the data being analysed
# 	    vessels_present = False
# 	    while vessels_present==False:
# 		    vol_batch, labels_batch = load_batch(batch_size=batch_size, volume_dims=volume_dims, 
#                                            n_classes=n_classes, image_stack=image_stack, labels=labels)
# 		    if 0.001<(np.count_nonzero(labels_batch[:,:,:,:,1])/labels_batch[:,:,:,:,1].size):
# 			    vessels_present = True
# 		    else:
# 			    del vol_batch, labels_batch
# 			    
# 	    # update learning rate decay with a piecewise contasnt decay rate of 0.1 every 5000 iterations
# 	    if i % 5000 == 0 and i != 0:
# 		    setLR(model_gpu, i, 'piecewise')
          
# 	    # fit model
# 	    model_gpu.fit(vol_batch, labels_batch, batch_size=batch_size, epochs=n_epochs, verbose=1)
# 	    
#         # calculate metrics for validation data (every 400 training iterations)
# 	    if i in range(6000,50001,400):
# 		    if image_test is not None:
# 			    training_cycle_list.append(i)
# 			    print('start prediction')
# 			    predict_segmentation(model_gpu=model_gpu, image_stack=image_test, labels=labels_test, 
#                                volume_dims=volume_dims, batch_size=batch_size, classes=classes,
#                                accuracy_list=accuracy_list, precision_list=precision_list, recall_list=recall_list, 
#                                save_output= False, binary_output=False, validation_output=True)
# 			    print('end prediction')
                
#         # save model every 1000 iterations
# 	    if i in range(1000,50001,1000):
# 		    save_model(model, path, model_filename)
# 		    
#     if output_filename is not None:
#         # plot accuracy
#         plt.plot(training_cycle_list,accuracy_list)
#         plt.title('Model accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Training iterations')
#         plt.savefig(path+output_filename+'_accuracy')
#         plt.show()
        
#         # plot precision    
#         plt.plot(training_cycle_list,precision_list)
#         plt.title('Model precision')
#         plt.ylabel('Precision')
#         plt.xlabel('Training iterations')
#         plt.savefig(path+output_filename+'_precision')
#         plt.show()
        
#         # plot recall    
#         plt.plot(training_cycle_list,recall_list)
#         plt.title('Model recall')
#         plt.ylabel('Recall')
#         plt.xlabel('Training iterations')
#         plt.savefig(path+output_filename+'_recall')
#         plt.show()   

#     return training_cycle_list, accuracy_list, precision_list, recall_list



# Get predicted segmentations (one hot encoded) for an image stack
def predict_segmentation(model_gpu=None, data_dir=None, 
                         volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), 
                         binary_output=True, save_output= True, prediction_filename = None, path=None): 
  """# Prediction
    Inputs:
        model_gpu = ML model
        data_dir = directory of data (DataDir object)
        volume_dims = sub-volume size to be passed to model ((z,x,y) int, default (64,64,64))
        batch_size = number of image sub volumes per batch int, default 2)
        overlap =
        classes =
        binary_output =
        save_output =
        filename = filename for saving outputs (string)
        path = path for saving outputs (string)
 
    """  
  n_classes = len(classes)
  for index in range(0,len(data_dir.list_IDs)):
      # Format volume_dims as (z,x,y)
      if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
        volume_dims = (volume_dims, volume_dims, volume_dims)
      elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
        volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
        
      # Check for sensible volume dimensions and step size
      for i in range(3):
        if volume_dims[i]<=0 or volume_dims[i]>data_dir.image_dims[index][i]:
          raise Exception('Volume dimensions out of range')
    	
      if overlap is None:
        overlap=0
      for i in range(3):
        if volume_dims[i]/2<overlap:
          raise Exception('overlap cannot be greater than half of the smallest volume dimensions')
      step_size=np.array(volume_dims)
      step_size-=overlap
            
      # Initialise seg_pred
      seg_pred = np.zeros((int(step_size[0]*batch_size), data_dir.image_dims[index][1], data_dir.image_dims[index][2]))
      
      # Initialise 'overlap_region' variables
      overlap_region_bottom = np.zeros((int(overlap*batch_size), data_dir.image_dims[index][1], data_dir.image_dims[index][2], n_classes))
      overlap_region_right = np.zeros((step_size[0]*batch_size, overlap, data_dir.image_dims[index][2], n_classes))
      overlap_region_back = 1
      
      for k in range (int((data_dir.image_dims[index][0]-volume_dims[0])/seg_pred.shape[0])+1):
        # find z coordinates for batch
        z = k*batch_size*step_size[0]
             
        # break if batch will go outside of range of image
        if (data_dir.image_dims[index][0]-z)<(batch_size*step_size[0]+overlap):
          break
    			
        for i in range (int(data_dir.image_dims[index][1]/step_size[1])):
          x = i*step_size[1] # x coordinate for batch
    	
          for j in range (int(data_dir.image_dims[index][2]/step_size[2])):
            y = j*step_size[2] # y coordinate for batch		
            
            # Create batch along z axis of image (axis 0 of image stack)
            vol = np.zeros((batch_size, *volume_dims))
            for n in range(batch_size):# Generate data sub-volume at coordinates, add to batch
                # Load data from file
                vol[n, 0:volume_dims[0],...] = load_volume_from_file(volume_dims=volume_dims, image_dims=data_dir.image_dims[index],
                                       image_filename=data_dir.image_filenames[index], label_filename=None, 
                                       coords=(z+n*volume_dims[0],x,y), data_type=data_dir.data_type[index], offset=128)						
    			# predict segmentation using model
            vol=vol.reshape(*vol.shape,1)
            vol_pred_ohe = model_gpu.predict(vol,verbose=1) 
            del vol
      
            # average overlapped region in z axis
            vol_pred_ohe_av_z = np.zeros((seg_pred.shape[0],vol_pred_ohe.shape[2],vol_pred_ohe.shape[3],vol_pred_ohe.shape[4]))
            for n in range(batch_size):
              # Define top overlapping region
              overlap_region_top = vol_pred_ohe[n,0:overlap,:,:,:]
              # If this is first iteration in z, average of overlapping region is just overal_region_top
              if z==0: overlap_region_av = overlap_region_top
              # Else, load the bottom overlapping region for the previous z coordinate with the same x and y coordinates
              else: 
                if (n-1)<0: overlap_region_av = (overlap_region_top+overlap_region_bottom[(batch_size-1):(batch_size-1)+overlap,x:x+vol_pred_ohe.shape[2],y:y+vol_pred_ohe.shape[3],:])/2
                else: overlap_region_av = (overlap_region_top+overlap_region_bottom[(n-1):(n-1)+overlap,x:x+vol_pred_ohe.shape[2],y:y+vol_pred_ohe.shape[3],:])/2 
              unique_region = vol_pred_ohe[n,overlap:step_size[0],:,:,:]
              vol_pred_ohe_av_z[(n)*step_size[0]:(n)*step_size[0]+overlap,:,:,:] = overlap_region_av
              vol_pred_ohe_av_z[(n)*step_size[0]+overlap:(n+1)*step_size[0],:,:,:] = unique_region
              overlap_region_bottom[n:n+overlap,x:x+vol_pred_ohe.shape[2],y:y+vol_pred_ohe.shape[3],:] = vol_pred_ohe[n,step_size[0]:step_size[0]+overlap,:,:,:] # Save bottom overlap region for next iteration
    
              del overlap_region_top, unique_region  
    	          
            del vol_pred_ohe
    #        # Append bottom overlap region if this is the last iteration in the z axis
    #        if k+1 >= int(((image_stack.shape[0]-volume_dims[0])/seg_pred.shape[0])+1):
    #          vol_pred_ohe_av_z = np.append(vol_pred_ohe_av_z,overlap_region_bottom, axis=0)
    
    	      
            # average overlapped region in x axis
            vol_pred_ohe_av_x = np.zeros((vol_pred_ohe_av_z.shape[0],step_size[1],vol_pred_ohe_av_z.shape[2],vol_pred_ohe_av_z.shape[3]))
            overlap_region_left = vol_pred_ohe_av_z[:,0:overlap,:,:]
            if x==0: overlap_region_av = overlap_region_left
            else: overlap_region_av = (overlap_region_left+overlap_region_right[:,:,y:y+vol_pred_ohe_av_z.shape[2],:])/2
            unique_region = vol_pred_ohe_av_z[:,overlap:step_size[1],:,:]
            vol_pred_ohe_av_x[:,0:overlap,:,:] = overlap_region_av
            vol_pred_ohe_av_x[:,overlap:step_size[1],:,:] = unique_region
            overlap_region_right[:,:,y:y+vol_pred_ohe_av_z.shape[2],:] = vol_pred_ohe_av_z[:,step_size[1]:step_size[1]+overlap,:,:] # Save right overlap region for next iteration
    
            # Append right overlap region if this is the last iteration in the x axis
    #        if i+1 >= int(image_stack.shape[1]/step_size[1]):
    #          vol_pred_ohe_av_x = np.append(vol_pred_ohe_av_x,overlap_region_right[:,:,y:y+vol_pred_ohe_av_z.shape[2],:], axis=1)
    #        del vol_pred_ohe_av_z, overlap_region_av, unique_region
            
            
            #average overlapped region in y axis
            vol_pred_ohe_av_y = np.zeros((vol_pred_ohe_av_x.shape[0], vol_pred_ohe_av_x.shape[1],step_size[2],vol_pred_ohe_av_x.shape[3]))
            overlap_region_front = vol_pred_ohe_av_x[:,:,0:overlap,:]
            if y==0: overlap_region_av = overlap_region_front
            else: overlap_region_av = (overlap_region_front+overlap_region_back[:,0:60,:,:])/2 
            unique_region = vol_pred_ohe_av_x[:,:,overlap:step_size[2],:]
            vol_pred_ohe_av_y[:,:,0:overlap,:] = overlap_region_av
            vol_pred_ohe_av_y[:,:,overlap:step_size[2],:] = unique_region
            overlap_region_back = vol_pred_ohe_av_x[:,:,step_size[2]:step_size[2]+overlap,:] # Save back overlap region for next iteration
            del vol_pred_ohe_av_x, overlap_region_av, unique_region
            # Append back overlap region if this is the last iteration in the y axis
    #        if j+1 >= int(image_stack.shape[2]/step_size[2]):
    #          vol_pred_ohe_av_y = np.append(vol_pred_ohe_av_y,overlap_region_back, axis=2)
    
            class_pred = np.argmax(vol_pred_ohe_av_y,axis=3) # find most probable class for each pixel
            vol_pred = np.zeros(vol_pred_ohe_av_y.shape[:-1])
            for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
                vol_pred[class_pred==i] = cls
                
            seg_pred[0:vol_pred_ohe_av_y.shape[0], x:(x+vol_pred_ohe_av_y.shape[1]), y:(y+vol_pred_ohe_av_y.shape[2])] = vol_pred[:,:,:]
    
        
            # if binary_output==True:
            #   # reverse one hot encoding
            #   class_pred = np.argmax(vol_pred_ohe_av_y,axis=3) # find most probable class for each pixel
            #   vol_pred = np.zeros(vol_pred_ohe_av_y.shape[:-1])
            #   for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
            #     vol_pred[class_pred==i] = cls
    			
            #   # add volume to seg_pred array
            #   seg_pred[0:vol_pred_ohe_av_y.shape[0], x:(x+vol_pred_ohe_av_y.shape[1]), y:(y+vol_pred_ohe_av_y.shape[2])] = vol_pred[:,:,:]
              
            # else:
            #   # add volume to seg_pred array
            #   seg_pred[0:vol_pred_ohe_av_y.shape[0], x:(x+vol_pred_ohe_av_y.shape[1]), y:(y+vol_pred_ohe_av_y.shape[2])] = vol_pred_ohe_av_y[:,:,:,1]
    
               
    	# save segmented images from this batch
        if save_output==True:
          filename=os.path.join(path, prediction_filename+'_'+str(data_dir.list_IDs[index]))
          np.save(filename, seg_pred)
					



def data_preprocessing(image_filename=None, label_filename=None, downsample_factor=1, pad_array=None):
	"""# Pre-processing
    Load data, downsample if neccessary, normalise and pad.
    Inputs:
        image_filename = image filename (string)
        label_filename = labels filename (string)
        downsample_factor = factor by which to downsample in x and y dimensions (int, default 1)
        pad_array = size to pad image to, should be able to be written as 2^n where n is an integer (int, default 1024)
    Outputs:
        img_pad = image data as an np.array, scaled between 0 and 1, downsampled and padded with zeros
        seg_pad = label data as an np.array, scaled between 0 and 1, downsampled and padded with zeros
        classes = list of classes present in labels
    """
   # Load image
	print('Loading images from '+str(image_filename))
	img=io.imread(image_filename)

	if len(img.shape)>3:
	  print('Image data has more than 3 dimensions. Cropping to first 3 dimensions')
	  img=img[:,:,:,0]

	# Downsampling
	if downsample_factor >1:
	  print('Downsampling by a factor of {}'.format(downsample_factor))
	  img=block_reduce(img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)
	  
	# Normalise 
	print('Rescaling data between 0 and 1')
	img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1
	
	# Seems that the CNN needs 2^n data dimensions (i.e. 64, 128, 256, etc.)
	# Set the images to 1024x1024 (2^10) arrays
	if pad_array is not None:
		print('Padding array')
		xpad=(pad_array-img.shape[1])//2
		ypad=(pad_array-img.shape[2])//2
	
		img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
		img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img
		img=img_pad
		print('Shape of padded image array: {}'.format(img_pad.shape))
	
	#Repeat for labels is present
	if label_filename is not None:
		print('Loading labels from '+str(label_filename))
		seg=io.imread(label_filename)
	
		# Downsampling
		if downsample_factor >1:
		  print('Downsampling by a factor of {}'.format(downsample_factor))
		  seg=block_reduce(seg, block_size=(1, downsample_factor, downsample_factor), func=np.max) #max instead of mean to maintain binary image  
		  
		# Normalise 
		#print('Rescaling data between 0 and 1')
		#seg = (seg-np.amin(seg))/(np.amax(seg)-np.amin(seg))
		
		# Pad
		if pad_array is not None:
		  print('Padding array')
		  seg_pad = np.zeros([seg.shape[0],pad_array,pad_array], dtype='float32')
		  seg_pad[0:seg.shape[0],xpad:seg.shape[1]+xpad,ypad:seg.shape[2]+ypad] = seg
		  seg=seg_pad
		
		# Find the number of unique classes in segmented training set
		classes = np.unique(seg)
		
		return img, seg, classes
	
	return img


def load_saved_model(model_path=None, filename=None,
                     learning_rate=1e-3, n_gpus=2, loss=None, metrics=['accuracy'],
                     freeze_layers=None, n_classes=2, fine_tuning=False):
	"""# Load Saved Model
    Inputs:
        model_path = path of model to be opened (string)
        filename = model filename (string)
        n_gpus = number of GPUs for multi GPU model
        fine_tuning = if 'True' model with be prepared for fine tuning with default settings (bool, default 'False')
        freese_layers = number of shallow layers that won't be trained if fine tuning (int, default none)
        n_classes = number of unique classes (int, default 2)
        loss = loss function to be used in training
        metrics = metrics to be monitored during training (default 'accuracy') 
        learning_rate = learning rate for training (float, default 1e-3)
    Outputs:
        model_gpu = multi GPU model
        model = model on CPU (required for saving)
    """
	mfile = os.path.join(model_path,filename+'.h5') # file containing weights
	jsonfile = os.path.join(model_path,filename+'.json') # file containing model template in json format
	print('Loading model')
	# open json
	json_file = open(jsonfile, 'r')
	# load model from json
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json, custom_objects={'tf':tf})
	# load weights into new model
	model.load_weights(mfile)
	if fine_tuning:
		model_gpu, model = fine_tuning(model=model, freeze_layers=freeze_layers, n_gpus=n_gpus, 
                                 learning_rate=learning_rate, loss=loss, metrics=metrics)
	else:
		model_gpu = multi_gpu_model(model, gpus=n_gpus) 
		model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
	print('Template model structure')
	model.summary()
	print('GPU model structure')
	model_gpu.summary()
	return model_gpu, model

# SAVE MODEL
def save_model(model, model_path, filename):
	"""# Save Model as .json and .h5 (weights)
    Inputs:
        model = model object
        model_path = path for model to be saved to (string)
        filename = model filename (string)
    """
	mfile_new = os.path.join(model_path, filename+'.h5') # file containing weights
	jsonfile_new = os.path.join(model_path, filename+'.json')
	print('Saving model')
	model_json = model.to_json()
	with open(jsonfile_new, "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights(mfile_new)

def roc_analysis(model=None, data_dir=None, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), save_prediction=False, prediction_filename=None): 
    optimal_thresholds = []
    recall = []
    precision = []
    
    for index in range(0,len(data_dir.list_IDs)):
        print('Analysing ROC on '+str(data_dir.list_IDs[index])+' data')
        y_pred_all = np.zeros(data_dir.image_dims[index])
        y_test_all = np.zeros(data_dir.image_dims[index])                       
        for k in range(math.ceil(data_dir.image_dims[index][0]/(volume_dims[0]*batch_size))):
            # find z coordinates for batch
            z = k*batch_size*volume_dims[0]   
            for i in range (int(data_dir.image_dims[index][1]/volume_dims[1])):
              x = i*volume_dims[1] # x coordinate for batch
              for j in range (int(data_dir.image_dims[index][2]/volume_dims[2])):
                y = j*volume_dims[2] # y coordinate for batch		
                print('Coordinates: ({},{},{})'.format(z,y,x))
                # initialise batch and temp volume dimensions
                X_test = np.zeros((batch_size, *volume_dims))
                y_test = np.zeros((batch_size, *volume_dims))
                volume_dims_temp = list(volume_dims)
                for n in range(batch_size):# Generate data sub-volume at coordinates, add to batch
                    overspill = (n+1)*volume_dims[0]-(data_dir.image_dims[index][0]-z)+1
                    if overspill>0 and overspill<volume_dims[0]:
                        # Reduce volume dimensions for batch that would overflow data
                        volume_dims_temp[0] = volume_dims_temp[0] - overspill
                    elif overspill>=volume_dims[0]:
                        # Do not load data for this batch
                        break
                    # Load data from file
                    X_test[n, 0:volume_dims_temp[0],...], y_test[n, 0:volume_dims_temp[0],...] = load_volume_from_file(volume_dims=volume_dims_temp, image_dims=data_dir.image_dims[index],
                                   image_filename=data_dir.image_filenames[index], label_filename=data_dir.label_filenames[index], 
                                   coords=(z+n*volume_dims[0],x,y), data_type=data_dir.data_type[index], offset=128)	

                # Run prediction
                X_test=X_test.reshape(*X_test.shape, 1)
                y_pred=model.predict(X_test,verbose=0)
               
                # Add postive class to stack
                for n in range(y_pred.shape[0]):
                    # Remove padded slices in z axis if necessary (ie. if volume_dims_temp has been updated to remove overspill)
                    if volume_dims_temp[0] != volume_dims[0]:
                        iz = np.where(X_test[n,...]!=0)[0] # find instances of non-zero values in X_test along axis 1
                        if len(iz)==0:
                            continue #if no non-zero values, continue to next iteration
                        y_test_crop = y_test[n, 0:max(iz)+1, ...] # use this to index y_test and y_pred
                        y_pred_crop = y_pred[n, 0:max(iz)+1, ...]
                        
                        y_pred_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_pred_crop.shape[0], x:x+y_pred_crop.shape[1], y:y+y_pred_crop.shape[2]]=y_pred_crop[...,1]
                        y_test_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_test_crop.shape[0], x:x+y_test_crop.shape[1], y:y+y_test_crop.shape[2]]=y_test_crop
        
                    else:                
                        y_pred_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_pred.shape[1], x:x+y_pred.shape[2], y:y+y_pred.shape[3]]=y_pred[n,...,1]
                        y_test_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_test.shape[1], x:x+y_test.shape[2], y:y+y_test.shape[3]]=y_test[n,...]
        
              
        # calculate false/true positive rates and area under curve
        fpr, tpr, thresholds = roc_curve(np.ravel(y_test_all), np.ravel(y_pred_all))
        area_under_curve = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        print('Optimal threshold: {}'.format(thresholds[optimal_idx]))
        optimal_thresholds.append(thresholds[optimal_idx])
        print('Recall at optimal threshold: {}'.format(tpr[optimal_idx]))
        recall.append(tpr[optimal_idx])
        print('Precision at optimal threshold: {}'.format(1-fpr[optimal_idx]))
        precision.append(1-fpr[optimal_idx])
        
        # Save predicted segmentation      
        if save_prediction:
            # threshold using optimal threshold
            #y_pred_all[y_pred_all > optimal_thresholds[index]] = 1 
            for im in range(y_pred_all.shape[0]):
                save_image(y_pred_all[im,:,:], prediction_filename+'_'+str(data_dir.list_IDs[index])+'_'+str(im+1)+'.tif')
                save_image(y_test_all[im,:,:], prediction_filename+'_'+str(data_dir.list_IDs[index])+'_'+str(im+1)+'true.tif')
            print('Predicted segmentation saved to {}'.format(prediction_filename))
                
        # Plot ROC 
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.5f)' % area_under_curve)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for '+str(data_dir.list_IDs[index]))
        plt.legend(loc="lower right")
        plt.show()
        fig.savefig(prediction_filename+'ROC_'+str(data_dir.list_IDs[index])+'.png')

    return optimal_thresholds, recall, precision

def multiclass_analysis(model=None, data_dir=None, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), save_prediction=False, prediction_filename=None, path=None): 
    
    for index in range(0,len(data_dir.list_IDs)):
        print('Analysing '+str(data_dir.list_IDs[index])+' data')
        y_pred_all = np.zeros((*data_dir.image_dims[index],len(classes)))
        y_test_all = np.zeros(data_dir.image_dims[index])                       
        for k in range(math.ceil(data_dir.image_dims[index][0]/(volume_dims[0]*batch_size))):
            # find z coordinates for batch
            z = k*batch_size*volume_dims[0]   
            for i in range (int(data_dir.image_dims[index][1]/volume_dims[1])):
              x = i*volume_dims[1] # x coordinate for batch
              for j in range (int(data_dir.image_dims[index][2]/volume_dims[2])):
                y = j*volume_dims[2] # y coordinate for batch		
                print('Coordinates: ({},{},{})'.format(z,y,x))
                # initialise batch and temp volume dimensions
                X_test = np.zeros((batch_size, *volume_dims))
                y_test = np.zeros((batch_size, *volume_dims))
                volume_dims_temp = list(volume_dims)
                for n in range(batch_size):# Generate data sub-volume at coordinates, add to batch
                    overspill = (n+1)*volume_dims[0]-(data_dir.image_dims[index][0]-z)+1
                    if overspill>0 and overspill<volume_dims[0]:
                        # Reduce volume dimensions for batch that would overflow data
                        volume_dims_temp[0] = volume_dims_temp[0] - overspill
                    elif overspill>=volume_dims[0]:
                        # Do not load data for this batch
                        break
                    # Load data from file
                    X_test[n, 0:volume_dims_temp[0],...], y_test[n, 0:volume_dims_temp[0],...] = load_volume_from_file(volume_dims=volume_dims_temp, image_dims=data_dir.image_dims[index],
                                   image_filename=data_dir.image_filenames[index], label_filename=data_dir.label_filenames[index], 
                                   coords=(z+n*volume_dims[0],x,y), data_type=data_dir.data_type[index], offset=128)	

                # Run prediction
                X_test=X_test.reshape(*X_test.shape, 1)
                y_pred=model.predict(X_test,verbose=0)
               
                # Add postive class to stack
                for n in range(y_pred.shape[0]):
                    # Remove padded slices in z axis if necessary (ie. if volume_dims_temp has been updated to remove overspill)
                    if volume_dims_temp[0] != volume_dims[0]:
                        iz = np.where(X_test[n,...]!=0)[0] # find instances of non-zero values in X_test along axis 1
                        if len(iz)==0:
                            continue #if no non-zero values, continue to next iteration
                        y_test_crop = y_test[n, 0:max(iz)+1, ...] # use this to index y_test and y_pred
                        y_pred_crop = y_pred[n, 0:max(iz)+1, ...]
                        
                        y_pred_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_pred_crop.shape[0], x:x+y_pred_crop.shape[1], y:y+y_pred_crop.shape[2],:]=y_pred_crop
                        y_test_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_test_crop.shape[0], x:x+y_test_crop.shape[1], y:y+y_test_crop.shape[2]]=y_test_crop
        
                    else:                
                        y_pred_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_pred.shape[1], x:x+y_pred.shape[2], y:y+y_pred.shape[3],:]=y_pred[n,...]
                        y_test_all[z+(n*volume_dims[0]):z+(n*volume_dims[0])+y_test.shape[1], x:x+y_test.shape[2], y:y+y_test.shape[3]]=y_test[n,...]
        
              
        # Reverse OHE
        y_pred_all = y_pred_all.argmax(axis=3)
        print(y_pred_all.shape)
        # print precision and recall
        class_report=classification_report(np.ravel(y_test_all), np.ravel(y_pred_all), digits=3)
        print(class_report)
        kappa=cohen_kappa_score(np.ravel(y_test_all), np.ravel(y_pred_all))
        print('Cohen kappa score: {}'.format(kappa))
        
        # Save predicted segmentation      
        if save_prediction:
 
            filename=os.path.join(path, prediction_filename+'_'+str(data_dir.list_IDs[index]))
            np.save(filename, y_pred_all)
            print('Predicted segmentation saved to {}'.format(prediction_filename))
                

    return kappa, class_report
