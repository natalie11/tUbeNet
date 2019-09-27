# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import random

# import required objects and fuctions from keras
from keras.models import Model, model_from_json #load_model
# CNN layers
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, LeakyReLU, Dropout
# utilities
from keras.utils import multi_gpu_model, to_categorical #np_utils
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
import matplotlib.pyplot as plt

# import Image for loading data
from PIL import Image
from skimage import io
from skimage.measure import block_reduce

from functools import partial

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 1           	# factor by which images are downsampled in x and y dimensions 
pad_array = 2048	           	# size images are padded up to, to achieve n^2 x n^2 structure 
volume_dims = (64,128,128)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
n_epochs = 3			         	# number of epoch for training CNN
batch_size = 2		 	       	# batch size for training CNN
n_rep = 50001		         	 	# number of training cycle repetitions
use_saved_model = True	        	# use saved model structure and weights? Yes=True, No=False
save_model = False		        	# save model structure and weights? Yes=True, No=False
fine_tuning = False
class_weights = (1,5) 	        	# relative weighting of background to blood vessel classes
save_after = (5000,15000)       	# save intermediate softmax output after this many training cycles
binary_output = True 	         	# save as binary (True) or softmax (False)

# Paths and filenames
path = "F:\\PaediatricGlioma"
img_filename = os.path.join(path,"PaediatricGliomaCrop.tif")
seg_filename = os.path.join(path,"PaediatricGliomaCrop.tif")
whole_img_filename = os.path.join(path,"PaediatricGliomaCrop.tif")
model_path = "G:\\Vessel Segmentation\\saved_weights\\3D"
model_filename = 'HREM_Adam_50000x2epochs_excl0.1percentVessels_4xdownsampled_1-5weighting_FineTunedCT_3114cycles'
updated_model_filename = 'updated'
history_filename = 'history'


#----------------------------------------------------------------------------------------------------------------------------------------------
"""Define functions"""

# save image
def save_image(array, filename):
	image = Image.fromarray(array)
	image.save(filename)

"""Load a sub-volume of the 3D image. These can either be generated randomly, or be taked from defined co-ordinates (z, x, y) within the image."""

# load sub-volume from 3D image stack
def load_volume(volume_dims=(64,64,64), image_stack=None, coords=None, labels=None):
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
    # Create coorsponding labels
    labels_volume = labels[int(coords[0]):int(coords[0] + volume_dims[0]), int(coords[1]):int(coords[1] + volume_dims[1]), int(coords[2]):int(coords[2] + volume_dims[2])]
    return volume, labels_volume
  else:
    return volume

"""Load a batch of sub-volumes in the format: (batch_size, img_depth, img_hight, img_width)"""

# load batch of sub-volumes for training or label prediction
def load_batch(batch_size=1, volume_dims=(64,64,64), image_stack=None, coords=None, labels=None, n_classes=None, step_size=None):
  
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

"""Create custom loss function - weighted to address class imbalance"""

def weighted_crossentropy_Nat(y_true, y_pred, weights):
	weight_mask = y_true[...,0] * weights[0] + y_true[...,1] * weights[1]
	return K.categorical_crossentropy(y_true, y_pred,) * weight_mask

# create partial for  to pass to complier
custom_loss=partial(weighted_crossentropy_Nat, weights=class_weights)

"""Create custom metrics"""

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

#Tian metrics
def precision1(y_true, y_pred):
	#true positive
	TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
	#false positive
	FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
	precision1=TP/(TP+FP)
	return precision1
def recall1(y_true, y_pred):
	#true positive
	TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
	#false negative
	FN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
	recall1=TP/(TP+FN)
	return recall1

"""Create custom callbacks"""

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
                
time_callback = TimeHistory()		      
stop_time_callback = TimedStopping(seconds=18000, verbose=1) 
stop_loss_callback = EarlyStopping(monitor='val_loss', min_delta=0.000005, patience=50, mode='min')

"""Create tUbeNet model"""

def tUbeNet(n_classes=2, input_height=64, input_width=64, input_depth=64, n_gpu=2, learning_rate=1e-3, metrics=['accuracy']):

    """
    Adapted from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    """

    inputs = Input((input_height, input_width, input_depth, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation= 'linear', padding='same')(inputs)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation= 'linear', padding='same')(activ1)
    activ1 = LeakyReLU(alpha=0.2)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(activ1)
    drop1 = Dropout(0.25)(pool1)   
      
    conv2 = Conv3D(64, (3, 3, 3), activation='linear', padding='same')(drop1)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='linear', padding='same')(activ2)
    activ2 = LeakyReLU(alpha=0.2)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(activ2)
    drop2 = Dropout(0.25)(pool2)
     

    conv3 = Conv3D(128, (3, 3, 3), activation='linear', padding='same')(drop2)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='linear', padding='same')(activ3)
    activ3 = LeakyReLU(alpha=0.2)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(activ3)
    drop3 = Dropout(0.5)(pool3)
    

    conv4 = Conv3D(256, (3, 3, 3), activation='linear', padding='same')(drop3)
    activ4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='linear', padding='same')(activ4)
    activ4 = LeakyReLU(alpha=0.2)(conv4)			
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(activ4)
    drop4 = Dropout(0.5)(pool4)
    
  
    conv5 = Conv3D(512, (3, 3, 3), activation='linear', padding='same')(drop4)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='linear', padding='same')(activ5)
    activ5 = LeakyReLU(alpha=0.2)(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(activ5)    
    drop5 = Dropout(0.5)(pool5)
    
      
    conv6 = Conv3D(1024, (3, 3, 3), activation='linear', padding='same')(drop5)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv3D(512, (3, 3, 3), activation='linear', padding='same')(activ6)
    activ6 = LeakyReLU(alpha=0.2)(conv6)
    

    up7 = concatenate([Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same')(activ6), activ5], axis=4)    
    conv7 = Conv3D(512, (3, 3, 3), activation='linear', padding='same')(up7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv3D(512, (3, 3, 3), activation='linear', padding='same')(activ7)
    activ7 = LeakyReLU(alpha=0.2)(conv7)   

    up8 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(activ7), activ4], axis=4)
    conv8 = Conv3D(256, (3, 3, 3), activation='linear', padding='same')(up8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Conv3D(256, (3, 3, 3), activation='linear', padding='same')(activ8)
    activ8 = LeakyReLU(alpha=0.2)(conv8)
    

    up9 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(activ8), activ3], axis=4)
    conv9 = Conv3D(128, (3, 3, 3), activation='linear', padding='same')(up9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)
    conv9 = Conv3D(128, (3, 3, 3), activation='linear', padding='same')(activ9)
    activ9 = LeakyReLU(alpha=0.2)(conv9)

    up10 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(activ9), activ2], axis=4)
    conv10 = Conv3D(64, (3, 3, 3), activation='linear', padding='same')(up10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)
    conv10 = Conv3D(64, (3, 3, 3), activation='linear', padding='same')(activ10)
    activ10 = LeakyReLU(alpha=0.2)(conv10)

    up11 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(activ10), activ1], axis=4)
    conv11 = Conv3D(32, (3, 3, 3), activation='linear', padding='same')(up11)
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    conv11 = Conv3D(32, (3, 3, 3), activation='linear', padding='same')(activ11)    
    activ11 = LeakyReLU(alpha=0.2)(conv11)
    
    conv12 = Conv3D(n_classes, (1, 1, 1), activation='softmax')(activ11)

    # create model on CPU
    with tf.device("/cpu:0"):	
	    model = Model(inputs=[inputs], outputs=[conv12])
    
    # tell model to run on 2 gpus
    model_gpu = multi_gpu_model(model, gpus=n_gpu) 
    model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)
    return model_gpu, model

""" Fine Tuning - Define function to replace classifier and freeze shallow layers """

def fine_tuning(model=None, freeze_layers=0, n_gpu=2, learning_rate=1e-5, metrics=['accuracy']):

    
    # recover the output from the last layer in the model and use as input to new Classifer
    last = model.layers[-2].output
    classifier = Conv3D(n_classes, (1, 1, 1), activation='softmax', name='newClassifier')(last)
    
    # rename new classifier layer to avoid error caused by layer having the same name as first layer of base model
    
    model = Model(inputs=[model.input], outputs=[classifier])
    
    # freeze weights for selected layers
    for layer in model.layers[:freeze_layers]: layer.trainable = False
    
    model_gpu = multi_gpu_model(model, gpus=n_gpu)
    model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)
    return model_gpu, model

""" Learning rate function """
# 
def setLR(model_gpu, i, schedule):
	if schedule == 'piecewise':
		decay_rate = 0.9
		lr = K.get_value(model_gpu.optimizer.lr)
		K.set_value(model_gpu.optimizer.lr, lr * decay_rate)
		print("learning rate changed to {}".format(lr * decay_rate))
	elif schedule == 'exponential':
		lr = K.get_value(model_gpu.optimizer.lr)
		decay_rate = 0.1 ** (i / 50000)
		lr0 = 0.00005
		K.set_value(model_gpu.optimizer.lr, lr0 * decay_rate)
		print("decay rate changed to {}".format(decay_rate))
		print("learning rate changed to {}".format(lr0 * decay_rate))
	else:
		print('No schedule chosen. Learning rate not updated')

""" Training - Define training function """

# train model on image_stack and corrosponding labels
def train_model(model_gpu=None, image_stack=None, labels=None, image_test=None, labels_test=None, volume_dims=(64,64,64), batch_size=2, n_rep=100, n_epochs=2, classes=(0,1)):
    print('Training model')
    print('Number of epochs = {}'.format(n_epochs))
    accuracy_list=[]   
    precision_list=[]
    recall_list=[] 
    training_cycle_list=[] 
    
    
    n_classes = len(classes)
    print('Training with {} classes'.format(n_classes)) 
    
    # Train for *n_rep* cycles of *n_epochs* epochs, loading a new batch of volumes every cycle
    for i in range(n_rep):
	    print('Training cycle {}'.format(i))
      
      # When loading a batch of volumes, check for the percentage of vessel pixels present and only train of volumes containing >0.01% vessels
      # These numbers may need to be changed depending on the vessel density in the data being analysed
	    vessels_present = False
	    while vessels_present==False:
		    vol_batch, labels_batch = load_batch(batch_size=batch_size, volume_dims=volume_dims, n_classes=n_classes, image_stack=image_stack, labels=labels)
		    if 0.001<(np.count_nonzero(labels_batch[:,:,:,:,1])/labels_batch[:,:,:,:,1].size):
			    vessels_present = True
		    else:
			    del vol_batch, labels_batch
			    
         # update learning rate decay with a piecewise contasnt decay rate of 0.1 every 5000 iterations
	    if i % 5000 == 0 and i != 0:
		    setLR(model_gpu, i, 'piecewise')
          
	    # load data for validation
	    #val_batch, val_labels = load_batch(batch_size=batch_size, volume_dims=volume_dims, n_classes=n_classes, image_stack=image_stack, labels=labels)
      
	    # fit model
	    model_gpu.fit(vol_batch, labels_batch, batch_size=batch_size, epochs=n_epochs, verbose=1)

	    
	    if i in range(6000,50001,400):
		    training_cycle_list.append(i)
		    print('start prediction')
		    if i in save_after:
			    predict_segmentation(model_gpu=model_gpu, image_stack=image_test, labels=labels_test, volume_dims=volume_dims, batch_size=batch_size, classes=(0,1), binary_output=True, accuracy_list=accuracy_list,precision_list=precision_list,recall_list=recall_list, save_output= True, save_name=str(n_rep)+'training_cycles', curve_output= True)
			    #print('save_output= Ture')
		    else:
			    predict_segmentation(model_gpu=model_gpu, image_stack=image_test, labels=labels_test, volume_dims=volume_dims, batch_size=batch_size, classes=(0,1), binary_output=True, accuracy_list=accuracy_list,precision_list=precision_list,recall_list=recall_list, save_output= False, curve_output= True)
		    print('end prediction')
	    if i in range(1000,50001,1000):
		    mfile_new = os.path.join(model_path, updated_model_filename+str(i)+'.h5') # file containing weights
		    jsonfile_new = os.path.join(model_path, updated_model_filename+str(i)+'.json')
		    print('Saving model')
		    model_json = model.to_json()
		    with open(jsonfile_new, "w") as json_file:
			    json_file.write(model_json)
				# serialize weights to HDF5
		    model.save_weights(mfile_new)
		    
    # plot accuracy
    plt.plot(training_cycle_list,accuracy_list)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training cycles')
#    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\loss_and_accuracy\\accuracy')
    plt.show()
    
    # plot precision    
    plt.plot(training_cycle_list,precision_list)
    plt.title('Model precision')
    plt.ylabel('Precision')
    plt.xlabel('Training cycles')
#    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\loss_and_accuracy\\precision')
    plt.show()
    
    # plot recall    
    plt.plot(training_cycle_list,recall_list)
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Training cycles')
#    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\loss_and_accuracy\\recall')
    plt.show()   
# =============================================================================
#   # summarize history for loss
#     plt.plot(hist.history['loss'])
#     plt.plot(hist.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\loss_and_accuracy\\loss')
#     plt.show()
# =============================================================================
    
   
	    # save predicted segmentations at defined intervals during training
	    #if i in save_after:
		    #predict_segmentation(model_gpu=model_gpu, image_stack=image_stack, volume_dims=volume_dims, batch_size=batch_size, n_rep=i, n_epochs=n_epochs, n_classes=n_classes, binary_output=False)

"""# Prediction
Define prediction function 
"""

# Get predicted segmentations (one hot encoded) for an image stack
def predict_segmentation(model_gpu=None, image_stack=None, labels=None, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), binary_output=True, save_output= True, save_name = None, accuracy_list=None,precision_list=None,recall_list=None, validation_output= False):
  print('Using model to predict segmentation')
  
  # Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
    
  # Check for sensible volume dimensions and step size
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_stack.shape[i]:
      raise Exception('Volume dimensions out of range')
	
  if overlap is None:
    overlap=0
  for i in range(3):
    if volume_dims[i]/2<overlap:
      raise Exception('overlap cannot be greater than half of the smallest volume dimensions')
  step_size=np.array(volume_dims)
  step_size-=overlap
    
        
  # Initialise seg_pred
  seg_pred = np.zeros((int(step_size[0]*batch_size), image_stack.shape[1], image_stack.shape[2]))
  n_classes = len(classes)
  # Initialise 'overlap_region' variables
  overlap_region_bottom = np.zeros((int(overlap*batch_size), image_stack.shape[1], image_stack.shape[2], n_classes))
  overlap_region_right = np.zeros((step_size[0]*batch_size, overlap, image_stack.shape[2], n_classes))
  overlap_region_back = 1
  
  for k in range (int((image_stack.shape[0]-volume_dims[0])/seg_pred.shape[0])+1):
    # find z coordinates for batch
    z = k*batch_size*step_size[0]
         
    # break if batch will go outside of range of image
    if (image_stack.shape[0]-z)<(batch_size*step_size[0]+overlap):
      break
			
    for i in range (int(image_stack.shape[1]/step_size[1])):
      x = i*step_size[1] # x coordinate for batch
	
      for j in range (int(image_stack.shape[2]/step_size[2])):
        y = j*step_size[2] # y coordinate for batch		
        
    		# Create batch along z axis of image (axis 0 of image stack)
        vol = load_batch(batch_size=batch_size, volume_dims=volume_dims, coords=(z,x,y), n_classes=n_classes, image_stack=image_stack, step_size=step_size)
								
			# predict segmentation using model
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
        # Append bottom overlap region if this is the last iteration in the z axis
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
    
        del vol_pred_ohe_av_z, overlap_region_av, unique_region
        # Append right overlap region if this is the last iteration in the x axis
#        if i+1 >= int(image_stack.shape[1]/step_size[1]):
#          vol_pred_ohe_av_x = np.append(vol_pred_ohe_av_x,overlap_region_right, axis=1)
	  
        #average overlapped region in y axis
        vol_pred_ohe_av_y = np.zeros((vol_pred_ohe_av_x.shape[0], vol_pred_ohe_av_x.shape[1],step_size[2],vol_pred_ohe_av_x.shape[3]))
        overlap_region_front = vol_pred_ohe_av_x[:,:,0:overlap,:]
        if y==0: overlap_region_av = overlap_region_front
        else: overlap_region_av = (overlap_region_front+overlap_region_back)/2 
        unique_region = vol_pred_ohe_av_x[:,:,overlap:step_size[2],:]
        vol_pred_ohe_av_y[:,:,0:overlap,:] = overlap_region_av
        vol_pred_ohe_av_y[:,:,overlap:step_size[2],:] = unique_region
        overlap_region_back = vol_pred_ohe_av_x[:,:,step_size[2]:step_size[2]+overlap,:] # Save back overlap region for next iteration
    
        del vol_pred_ohe_av_x, overlap_region_av, unique_region
        # Append back overlap region if this is the last iteration in the y axis
#        if j+1 >= int(image_stack.shape[2]/step_size[2]):
#          vol_pred_ohe_av_y = np.append(vol_pred_ohe_av_y,overlap_region_back, axis=2)
    
        if binary_output==True:
          # reverse one hot encoding
          class_pred = np.argmax(vol_pred_ohe_av_y,axis=3) # find most probable class for each pixel
          vol_pred = np.zeros(vol_pred_ohe_av_y.shape[:-1])
          for i,cls in enumerate(classes): # for each pixel, set the values to that of the corrosponding class
            vol_pred[class_pred==i] = cls
			
          # add volume to seg_pred array
          seg_pred[0:vol_pred_ohe_av_y.shape[0], x:(x+vol_pred_ohe_av_y.shape[1]), y:(y+vol_pred_ohe_av_y.shape[2])] = vol_pred[:,:,:]
          
        else:
          # add volume to seg_pred array
          seg_pred[0:vol_pred_ohe_av_y.shape[0], x:(x+vol_pred_ohe_av_y.shape[1]), y:(y+vol_pred_ohe_av_y.shape[2])] = vol_pred_ohe_av_y[:,:,:,1]

    if validation_output==True: 
      if labels == None:
            print('No true labels supplies. Validation metrics cannot be calculated')
      else:	 
            accuracy=np.mean(seg_pred == labels)
            print('Accuracy of segmentation: {}'.format(accuracy))
            accuracy_list.append(accuracy)
            print('Accuracy list: {}'.format(accuracy_list))
            p=precision1(labels,seg_pred)
            precision_list.append(p)
            r=recall1(labels,seg_pred)
            recall_list.append(r)
            print('Precision of segmentation: {}'.format(precision_list))
            print('Recall of segmentation: {}'.format(recall_list))
            np.save('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\accuracy_list.npy', accuracy_list)
            np.save('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\precision_list.npy', precision_list)
            np.save('G:\\Vessel Segmentation\\liver_mag3.2_GFP_G5exp0.1_cy5_G4_exp0.015_thk1.72__2\\recall_list.npy', recall_list)    
           
	# save segmented images from this batch
    if save_output==True:
      for im in range (seg_pred.shape[0]):
        filename = os.path.join(path,str(z+im+1)+"_"+str(save_name)+".tif")
        save_image(seg_pred[im,:,:], filename)
					
  return accuracy_list,precision_list,recall_list

"""# Pre-processing
Load data, downsample if neccessary, normalise and pad.
"""

def data_preprocessing(img_filename=None, label_filename=None, downsample_factor=1, pad_array=1024):
	# Load image
	print('Loading images from '+str(img_filename))
	img=io.imread(img_filename)

	# Downsampling
	if downsample_factor >1:
	  print('Downsampling by a factor of {}'.format(downsample_factor))
	  img=block_reduce(img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)
	  
	# Normalise 
	print('Rescaling data between 0 and 1')
	img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1
	
	# Seems that the CNN needs 2^n data dimensions (i.e. 64, 128, 256, etc.)
	# Set the images to 1024x1024 (2^10) arrays
	print('Padding array')
	xpad=(pad_array-img.shape[1])//2
	ypad=(pad_array-img.shape[2])//2
	
	img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
	img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img
	del img
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
		print('Rescaling data between 0 and 1')
		seg = (seg-np.amin(seg))/(np.amax(seg)-np.amin(seg))
		
		# Pad
		print('Padding array')
		seg_pad = np.zeros([seg.shape[0],pad_array,pad_array], dtype='float32')
		seg_pad[0:seg.shape[0],xpad:seg.shape[1]+xpad,ypad:seg.shape[2]+ypad] = seg
		
		# Find the number of unique classes in segmented training set
		classes = np.unique(seg_pad)
		del seg
		
		return img_pad, seg_pad, classes
	
	return img_pad

img_pad, seg_pad, classes = data_preprocessing(image_filename=img_filename, label_filename=seg_filename, downsample_factor=downsample_factor, pad_array=pad_array)

"""# Load or Build Model
Load or build model, run training
"""

# LOAD / BUILD MODEL
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
	if fine_tuning:
		model_gpu, model = fine_tuning(model=model, freeze_layers=10, n_gpu=2, learning_rate=1e-5, metrics=['accuracy',precision,recall])
	else:
		model_gpu = multi_gpu_model(model, gpus=2) 
		model_gpu.compile(optimizer=Adam(lr=1e-3), loss=custom_loss, metrics=['accuracy', precision, recall])
else:   
	print('Building model')
	model_gpu, model = tUbeNet(n_classes=len(classes), input_height=volume_dims[0], input_width=volume_dims[1], input_depth=volume_dims[2], n_gpu=2, learning_rate=1e-5, metrics=['accuracy',precision,recall])
  
print('Template model structure')
model.summary()
print('GPU model structure')
model_gpu.summary()

"""Train and save model"""

#TRAIN
train_model(model_gpu=model_gpu, image_stack=img_pad, labels=seg_pad, volume_dims=volume_dims, batch_size=batch_size, n_rep=n_rep, n_epochs=n_epochs, n_classes=len(classes))

# SAVE MODEL
if save_model:
	mfile_new = os.path.join(model_path, updated_model_filename+'.h5') # file containing weights
	jsonfile_new = os.path.join(model_path, updated_model_filename+'.json')
	print('Saving model')
	model_json = model.to_json()
	with open(jsonfile_new, "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights(mfile_new)

"""# Predict Segmentation
Predict segmentation for entire dataset
"""

#PREDICT
print('Loading images from '+str(img_filename))
# Find the number of unique classes in segmented training set
classes = (0,1)
n_classes = len(classes)
whole_img=io.imread(whole_img_filename)
whole_img=whole_img[:,:,:,0]
whole_img=block_reduce(whole_img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)

xpad=(pad_array-whole_img.shape[1])//2
ypad=(pad_array-whole_img.shape[2])//2

whole_img = (whole_img-np.amin(whole_img))/(np.amax(whole_img)-np.amin(whole_img))
whole_img_pad = np.zeros([whole_img.shape[0],pad_array,pad_array], dtype='float32')
whole_img_pad[0:whole_img.shape[0],xpad:whole_img.shape[1]+xpad,ypad:whole_img.shape[2]+ypad] = whole_img
del whole_img
predict_segmentation(model_gpu=model_gpu, image_stack=whole_img_pad, volume_dims=volume_dims, batch_size=batch_size, n_rep=n_rep, n_epochs=n_epochs, n_classes=n_classes, binary_output=binary_output, overlap=4)