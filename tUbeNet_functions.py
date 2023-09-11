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
from functools import partial
from model import tUbeNet
import nibabel as nib

# import required objects and fuctions from keras
from tensorflow.keras.models import Model, model_from_json
# CNN layers
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, LeakyReLU, Dropout#, AveragePooling3D, Reshape, Flatten, Dense, Lambda
# utilities
from tensorflow.keras.utils import multi_gpu_model, to_categorical #np_utils
# opimiser
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
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

# set memory limit on gpu
physical_devices = tf.config.list_physical_devices('GPU')
try:
  for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import matplotlib.pyplot as plt

# import Image for loading data
from PIL import Image
from skimage import io
from skimage.measure import block_reduce
from sklearn.metrics import roc_auc_score, roc_curve, auc

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
        raise Exception('Coordinates out of range in dimension '+str(i))
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

"""Custom loss functions"""
def weighted_crossentropy(y_true, y_pred, weights):
	"""Custom loss function - weighted to address class imbalance"""
	weight_mask = y_true[...,0] * weights[0] + y_true[...,1] * weights[1]
	return K.categorical_crossentropy(y_true, y_pred,) * weight_mask

def DiceBCELoss(y_true, y_pred, smooth=1e-6):    
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1-dice(y_true, y_pred)
    Dice_BCE = (BCE + dice_loss)/2
    return Dice_BCE

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

# Use when y_true/y_pred are np arrays rather than keras tensors
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


def piecewise_schedule(i, lr0, decay):
	""" Learning rate function 
    Updates learning rate at end epoch.
    Inputs:
        i = training epoch (int)
        lr0 = initial learning rate (float)
        decay = decay rate (float)
    """
	lr = lr0 * decay**(i)
	tf.summary.scalar('learning rate', data=lr, step=i)
	return lr

#---------------------------------------------------------------------------------------------------------------------------------------------------
  

# Get predicted segmentations (one hot encoded) for an image stack
def predict_segmentation(model=None, data_dir=None, 
                         volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), 
                         binary_output=True, save_output=True, path=None): 
  """# Prediction
    Inputs:
        model = ML model
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

      # Calculate number of volumes that will fit into image dimentions
      k_max = math.ceil(data_dir.image_dims[index][0]/step_size[0])
      i_max = math.ceil(data_dir.image_dims[index][1]/step_size[1])
      j_max = math.ceil(data_dir.image_dims[index][2]/(step_size[2]*batch_size)+0.0001)
      
      # Initialise 'overlap_region' variables
      overlap_region_bottom = []
      overlap_region_right = [None]*j_max
      overlap_region_back = [None]*j_max
      for ind in range(j_max): overlap_region_back[ind]=[None]*i_max
      
      volume_dims_temp=list(volume_dims)
      overspill=list((0,0,0))
      
      for k in range(k_max):
        # find z coordinates for batch
        z = k*step_size[0]
        print('z=',z)
        # calculate if batch will go outside of range of image
        overspill[0]=volume_dims[0]-(data_dir.image_dims[index][0]-z)
        if overspill[0]>0 and overspill[0]<volume_dims[0]:
            volume_dims_temp[0]=volume_dims[0]-overspill[0]
        else: volume_dims_temp[0]=volume_dims[0]
        
        # Initialise seg_pred, where predictions will be stored until images can be saved
        seg_pred = np.zeros((volume_dims_temp[0], data_dir.image_dims[index][1], data_dir.image_dims[index][2]))
    			
        for i in range(i_max):
          x = i*step_size[1] # x coordinate for batch
          print('x=',x)
          # calculate if batch will go outside of range of image
          overspill[1]=volume_dims[1]-(data_dir.image_dims[index][1]-x)
          if overspill[1]>0 and overspill[1]<volume_dims[1]:
              volume_dims_temp[1]=volume_dims[1]-overspill[1]
          else: volume_dims_temp[1]=volume_dims[1]
    	
          for j in range(j_max):
            y = j*step_size[2]*batch_size # y coordinate for batch		
            print('y=',y)
            # Create batch along z axis of image (axis 0 of image stack)
            vol = np.zeros((batch_size, *volume_dims))
            for n in range(batch_size):# Generate data sub-volume at coordinates, add to batch
                # calculate if volume goes outside of range of data
                overspill[2] = volume_dims[2]-(data_dir.image_dims[index][2]-(y+n*step_size[2]))
                if overspill[2]>0 and overspill[2]<volume_dims[2]:
                    volume_dims_temp[2]=volume_dims[2]-overspill[2]
                elif overspill[2]>=volume_dims[2]:
                    break
                else: volume_dims_temp[2]=volume_dims[2]
                # Load volume from data
                vol[n, 0:volume_dims_temp[0], 0:volume_dims_temp[1], 0:volume_dims_temp[2]] = load_volume_from_file(volume_dims=volume_dims_temp, image_dims=data_dir.image_dims[index],
                                        image_filename=data_dir.image_filenames[index], label_filename=None, 
                                       coords=(z,x,y+n*step_size[2]), data_type=data_dir.data_type[index], offset=128) #only take first output						
    			# predict segmentation using model
            vol_pred_ohe = model.predict(vol,verbose=1) 
            #vol_pred_ohe=vol_pred_ohe[0] #Required if output is a tuple of the prediction array and 
            del vol
      
            # average overlapped region in y axis
            vol_pred_ohe_av_y = np.zeros((vol_pred_ohe.shape[1],vol_pred_ohe.shape[2],step_size[2]*batch_size,vol_pred_ohe.shape[4]))
            for n in range(batch_size):
                # Define top overlapping region
                overlap_region_top = vol_pred_ohe[n,:,:,0:overlap,:]
                # If this is first iteration in y, average of overlapping region is just overlap_region_top
                if y==0: overlap_region_av = overlap_region_top
                # Else, load the bottom overlapping region for the previous z coordinate with the same x and y coordinates
                else: overlap_region_av = (overlap_region_top+overlap_region_bottom)/2
                unique_region = vol_pred_ohe[n,:,:,overlap:step_size[2],:]
                vol_pred_ohe_av_y[:,:,(n)*step_size[2]:(n)*step_size[2]+overlap,:] = overlap_region_av
                vol_pred_ohe_av_y[:,:,(n)*step_size[2]+overlap:(n+1)*step_size[2],:] = unique_region
                overlap_region_bottom = vol_pred_ohe[n,:,:,step_size[2]:step_size[2]+overlap,:] # Save bottom overlap region for next iteration

                del overlap_region_top, overlap_region_av, unique_region  
    	          
            del vol_pred_ohe
    
    	      
            # average overlapped region in x axis
            vol_pred_ohe_av_x = np.zeros((vol_pred_ohe_av_y.shape[0],step_size[1],vol_pred_ohe_av_y.shape[2],vol_pred_ohe_av_y.shape[3]))
            overlap_region_left = vol_pred_ohe_av_y[:,0:overlap,:,:]
            if x==0: overlap_region_av = overlap_region_left
            else: overlap_region_av = (overlap_region_left+overlap_region_right[j])/2
            unique_region = vol_pred_ohe_av_y[:,overlap:step_size[1],:,:]
            vol_pred_ohe_av_x[:,0:overlap,:,:] = overlap_region_av
            vol_pred_ohe_av_x[:,overlap:step_size[1],:,:] = unique_region
            overlap_region_right[j]=vol_pred_ohe_av_y[:,step_size[1]:step_size[1]+overlap,:,:]
           
            del vol_pred_ohe_av_y, overlap_region_left, overlap_region_av, unique_region
            

            # average overlapped region in z axis
            vol_pred_ohe_av_z = np.zeros((step_size[0],vol_pred_ohe_av_x.shape[1],vol_pred_ohe_av_x.shape[2],vol_pred_ohe_av_x.shape[3]))
            overlap_region_front = vol_pred_ohe_av_x[0:overlap,:,:,:]
            if z==0: overlap_region_av = overlap_region_front
            else: overlap_region_av = (overlap_region_front+overlap_region_back[j][i])/2 
            unique_region = vol_pred_ohe_av_x[overlap:step_size[0],:,:,:]
            vol_pred_ohe_av_z[0:overlap,:,:,:] = overlap_region_av
            vol_pred_ohe_av_z[overlap:step_size[0],:,:,:] = unique_region
            overlap_region_back[j][i] = vol_pred_ohe_av_x[step_size[0]:step_size[0]+overlap,:,:,:] # Save back overlap region for next iteration
            del vol_pred_ohe_av_x, overlap_region_front, overlap_region_av, unique_region

        
            if binary_output==True:
              # reverse one hot encoding
              class_pred = np.argmax(vol_pred_ohe_av_z,axis=3) # find most probable class for each pixel
              vol_pred = np.zeros(vol_pred_ohe_av_z.shape[:-1])
              for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
                vol_pred[class_pred==i] = cls
    			
              # add volume to seg_pred array
              seg_pred[0:min(vol_pred_ohe_av_z.shape[0], volume_dims_temp[0]), x:min((x+vol_pred_ohe_av_z.shape[1]), seg_pred.shape[1]), y:min((y+vol_pred_ohe_av_z.shape[2]),seg_pred.shape[2])] = vol_pred[:volume_dims_temp[0],:volume_dims_temp[1],:(seg_pred.shape[2]-y)]
              
            else:
              # add volume to seg_pred array
              seg_pred[0:min(vol_pred_ohe_av_z.shape[0], volume_dims_temp[0]), x:min((x+vol_pred_ohe_av_z.shape[1]), seg_pred.shape[1]), y:min((y+vol_pred_ohe_av_z.shape[2]),seg_pred.shape[2])] = vol_pred_ohe_av_z[:volume_dims_temp[0],:volume_dims_temp[1],:(seg_pred.shape[2]-y),1]
    
               
    	# save segmented images from this batch
        if save_output==True:
          for im in range (seg_pred.shape[0]):
            filename = os.path.join(path,data_dir.list_IDs[index]+"_prediction_"+str(z+im+1)+".tif")
            save_image(seg_pred[im,:,:], filename)
					



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
	if image_filename.endswith(('.nii','.nii.gz')):
	  img=nib.load(image_filename).get_fdata
	else:
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
	try:
		img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1
	except: 
        # break image up into quarters and normalise one chunck at a time
		quarter=int(img.shape[0]/4)
		img_min = np.amin(img)
		denominator = np.amax(img)-img_min
		for i in range (3):
			img[i*quarter:(i+1)*quarter,:,:]=(img[i*quarter:(i+1)*quarter,:,:]-img_min)/denominator
		img[3*quarter:,:,:]=(img[3*quarter:,:,:]-img_min)/denominator
        
	# Pad image in x and y axis
# 	if pad_array is not None:
# 		print('Padding array')
# 		xpad=(pad_array-img.shape[1])//2
# 		ypad=(pad_array-img.shape[2])//2 	
# 		img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
# 		img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img
# 		img=img_pad
# 		print('Shape of padded image array: {}'.format(img_pad.shape))
	if pad_array is not None:
		print('Padding array')
        #Pad bottom of image stack with blank slices to make up to the minimum batch size for running prediciton on full dataset
		factor=(img.shape[0]//pad_array)+1
		padding = np.zeros([int((pad_array*factor)-img.shape[0]),img.shape[1], img.shape[2]], dtype='float32')
		img=np.concatenate((img, padding), axis=0)
		print('Shape of padded image array: {}'.format(img.shape))
	
	#Repeat for labels is present
	if label_filename is not None:
		print('Loading labels from '+str(label_filename))
		seg=io.imread(label_filename)
	
		# Downsampling
		if downsample_factor >1:
		  print('Downsampling by a factor of {}'.format(downsample_factor))
		  seg=block_reduce(seg, block_size=(1, downsample_factor, downsample_factor), func=np.max) #max instead of mean to maintain binary image  
		  
		# Normalise 
		print('Rescaling data between 0 and 1')
		seg = (seg-np.amin(seg))/(np.amax(seg)-np.amin(seg))
		
		# Pad
		if pad_array is not None:
		  print('Padding array')
		  #Pad bottom of image stack with blank slices to make up to the minimum batch size for running prediciton on full dataset
		  factor=(img.shape[0]//pad_array)+1
		  padding = np.zeros([int((pad_array*factor)-img.shape[0]),img.shape[1], img.shape[2]], dtype='float32')
		  img=np.concatenate((img, padding), axis=0)
		  print('Shape of padded image array: {}'.format(img.shape))
		
		# Find the number of unique classes in segmented training set
		classes = np.unique(seg)
		
		return img, seg, classes
	
	return img

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

def roc_analysis(model=None, data_dir=None, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), 
                 save_prediction=False, prediction_filename=None, binary_output=False): 
    optimal_thresholds = []
    recall = []
    precision = []
    volume_dims_temp=list(volume_dims)#convert from tuple to list for easier re-assignment of elements
    
    for index in range(0,len(data_dir.list_IDs)):
        print('Analysing ROC on '+str(data_dir.list_IDs[index])+' data')
        y_pred_all = np.zeros(data_dir.image_dims[index])
        y_test_all = np.zeros(data_dir.image_dims[index])                       
        for k in range (math.ceil(data_dir.image_dims[index][0]/volume_dims[0])):
            # find z coordinates for batch
            z = k*volume_dims[0]   
            for i in range (math.ceil(data_dir.image_dims[index][1]/volume_dims[1])):
              x = i*volume_dims[1] # x coordinate for batch
              for j in range(math.ceil(data_dir.image_dims[index][2]/(volume_dims[2]*batch_size))):
                y = j*batch_size*volume_dims[2] # y coordinate for batch		
                print('Coordinates: ({},{},{})'.format(z,x,y))
                # initialise batch and temp volume dimensions
                X_test = np.zeros((batch_size, *volume_dims))
                y_test = np.zeros((batch_size, *volume_dims))
                
                #Calculate if sub-volume will overflow the data volume, if yes: update temp volume to avoid out of bounds error
                overspill=np.array((1,1,1))
                overspill[0]=volume_dims[0]-(data_dir.image_dims[index][0]-z)+1
                overspill[1]=volume_dims[1]-(data_dir.image_dims[index][1]-x)+1
                for dim in range (2):
                    if overspill[dim]>0 and overspill[dim]<volume_dims[dim]:
                        volume_dims_temp[dim]=volume_dims[dim]-overspill[dim]
                    else: volume_dims_temp[dim]=volume_dims[dim]
                
                for n in range(batch_size):# Generate data sub-volume at coordinates, add to batch
                    overspill[2] = volume_dims[2]-(data_dir.image_dims[index][2]-(y+n*volume_dims[2]))+1
                    if overspill[2]>0 and overspill[2]<volume_dims[2]:
                        volume_dims_temp[2]=volume_dims[2]-overspill[2]
                    elif overspill[2]>=volume_dims[2]:
                        break #stop adding to this batch as it has gone out of range of data
                    else: volume_dims_temp[2]=volume_dims[2]

                    # Load data from file
                    X_test[n, 0:volume_dims_temp[0], 0:volume_dims_temp[1], 0:volume_dims_temp[2]], y_test[n,0:volume_dims_temp[0], 0:volume_dims_temp[1], 0:volume_dims_temp[2]] = load_volume_from_file(volume_dims=volume_dims_temp, image_dims=data_dir.image_dims[index],
                                   image_filename=data_dir.image_filenames[index], label_filename=data_dir.label_filenames[index], 
                                   coords=(z,x,y+n*volume_dims[2]), data_type=data_dir.data_type[index], offset=128)	

                # Run prediction
                X_test=X_test.reshape(*X_test.shape, 1)
                y_pred=model.predict(X_test,verbose=0)
               
                # Add postive class to stack
                for n in range(y_pred.shape[0]):
                    # Remove padding if necessary (ie. if volume_dims_temp has been updated to remove overspill)
                    if volume_dims_temp != volume_dims:
                        i_xyz = np.where(X_test[n,...]!=0) # find instances of non-zero values in X_test -> indicates real data present (not padding)
                        if len(i_xyz[0])==0:
                            continue
                        max_xyz=np.array((0,0,0))
                        for dim in range (3): max_xyz[dim]=max(i_xyz[dim])+1 # find max coordinate of real data in each axis
                        y_test_crop = y_test[n,0:max_xyz[0],0:max_xyz[1],0:max_xyz[2]] # use this to index y_test and y_pred, crop padded volume
                        y_pred_crop = y_pred[n,0:max_xyz[0],0:max_xyz[1],0:max_xyz[2],:]
                        
                        y_pred_all[z:z+y_pred_crop.shape[0], x:x+y_pred_crop.shape[1], y+(n*volume_dims[2]):y+(n*volume_dims[2])+y_pred_crop.shape[2]]=y_pred_crop[...,1]
                        y_test_all[z:z+y_test_crop.shape[0], x:x+y_test_crop.shape[1], y+(n*volume_dims[2]):y+(n*volume_dims[2])+y_test_crop.shape[2]]=y_test_crop
        
                    else:                
                        y_pred_all[z:z+y_pred.shape[1], x:x+y_pred.shape[2], y+(n*volume_dims[2]):y+(n*volume_dims[2])+y_pred.shape[3]]=y_pred[n,...,1]
                        y_test_all[z:z+y_test.shape[1], x:x+y_test.shape[2], y+(n*volume_dims[2]):y+(n*volume_dims[2])+y_test.shape[3]]=y_test[n,...]
        
              
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
        
        if binary_output:
            binary_pred = np.zeros(y_pred_all.shape)
            binary_pred[y_pred_all>=thresholds[optimal_idx]] = 1
            y_pred_all = binary_pred
        
        # Save predicted segmentation      
        if save_prediction:
            # threshold using optimal threshold
            #y_pred_all[y_pred_all > optimal_thresholds[index]] = 1 
            for im in range(y_pred_all.shape[0]):
                save_image(y_pred_all[im,:,:], os.path.join(prediction_filename,str(data_dir.list_IDs[index])+'_'+str(im+1)+'_pred.tif'))
                save_image(y_test_all[im,:,:], os.path.join(prediction_filename,str(data_dir.list_IDs[index])+'_'+str(im+1)+'_true.tif'))
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
        fig.savefig(os.path.join(prediction_filename,'ROC_'+str(data_dir.list_IDs[index])+'.png'))

    return optimal_thresholds, recall, precision
