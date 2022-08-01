# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import numpy as np
import tUbeNet_functions_attn as tube
import random
import pickle
import os
join = os.path.join
#import datetime
#import json
import io
from matplotlib import pyplot as plt
from scipy.ndimage import rotate, zoom
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical #np_utils
#---------------------------------------------------------------------------------------------------------------------------------------------
class DataHeader:
    def __init__(self, ID=None, image_dims=(1024,1024,1024), image_filename=None, label_filename=None):
	    'Initialization' 
	    self.ID = ID
	    self.image_dims = image_dims
	    self.image_filename = image_filename
	    self.label_filename = label_filename
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

class DataDir:
	def __init__(self, list_IDs, image_dims=(1024,1024,1024), image_filenames=None, label_filenames=None, data_type='float64'):
	    'Initialization'    
	    self.image_dims = image_dims
	    self.image_filenames = image_filenames
	    self.label_filenames = label_filenames
	    self.list_IDs = list_IDs
	    self.data_type = data_type

class DataGenerator(Sequence):
	def __init__(self, data_dir, batch_size=32, volume_dims=(64,64,64), shuffle=True, n_classes=2, dataset_weighting=None, augment=False):
	    'Initialization'
	    self.volume_dims = volume_dims
	    self.batch_size = batch_size
	    self.shuffle = shuffle
	    self.data_dir = data_dir
	    self.on_epoch_end()
	    self.n_classes = n_classes
	    self.dataset_weighting = dataset_weighting
	    self.augment = augment
	    
	def __len__(self):
		'Denotes the number of batches per epoch'
		batches=0 
		for i in range(len(self.data_dir.list_IDs)):
		    batches_per_dataset = int(np.floor(np.prod(self.data_dir.image_dims[i])/np.prod(self.volume_dims)))
		    batches += batches_per_dataset
		return batches
	
	def __getitem__(self, index):
		'Generate one batch of data'
		# random.choices only available in python 3.6
		# randomly generate list of ID for batch, weighted according to given 'dataset_weighting' if not None
		if len(self.data_dir.list_IDs)>2:
		    list_IDs_temp = random.choices(self.data_dir.list_IDs, weights=self.dataset_weighting, k=self.batch_size)
		else: list_IDs_temp=[self.data_dir.list_IDs[0]]*self.batch_size
		# Generate data
		X, y = self.__data_generation(list_IDs_temp)
        
		if self.augment:
		    X, y = self._augmenation(X,y)

		return X, y
	    
	def on_epoch_end(self):
	    'Updates indexes after each epoch'
	    self.indexes = np.arange(len(self.data_dir.list_IDs))
	    if self.shuffle == True:
		    np.random.shuffle(self.indexes)
		    
	def __data_generation(self, list_IDs_temp):
	    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
	    # Initialization
	    X = np.empty((self.batch_size, *self.volume_dims))
	    y = np.empty((self.batch_size, *self.volume_dims))
	    for i, ID_temp in enumerate(list_IDs_temp):
		    index=self.data_dir.list_IDs.index(ID_temp)
                        
		    vessels_present=False
		    count=0
		    while vessels_present==False:
	    	     # Generate random coordinates within dataset
	    	     count = count+1
	    	     coords_temp=np.array([random.randint(0,(self.data_dir.image_dims[index][0]-self.volume_dims[0]-1)),
                    random.randint(0,(self.data_dir.image_dims[index][1]-self.volume_dims[1]-1)),
                    random.randint(0,(self.data_dir.image_dims[index][2]-self.volume_dims[2]-1))])
	    	     #print('coords_temp: {}'.format(coords_temp)) 
	    	     # Generate data sub-volume at coordinates, add to batch
	    	     X[i], y[i] = tube.load_volume_from_file(volume_dims=self.volume_dims, image_dims=self.data_dir.image_dims[index],
                           image_filename=self.data_dir.image_filenames[index], label_filename=self.data_dir.label_filenames[index], 
                           coords=coords_temp, data_type=self.data_dir.data_type[index], offset=128)	
	    	     if (np.count_nonzero(y[i][...,1])/y[i][...,1].size)>0.001 or count>5: #sub-volume must contain at least 0.1% vessels
	    	        vessels_present=True
                     
	    # Reshape to add depth of 1
	    X = X.reshape(*X.shape, 1)
		
	    return X, to_categorical(y, num_classes=self.n_classes)
    
	def _augmentation(self, X, y):
	    # Apply data augmentations to each image/label pair in batch
	    for i in range(self.batch_size):
		    #Rotate
		    angle = np.random.uniform(-30,30, size=1)
		    X[i] = rotate(X[i], float(angle), reshape=False, order=3, mode='reflect')
		    y[i] = rotate(y[i], float(angle), reshape=False, order=0, mode='reflect')
		    #Zoom
		    scale = np.random.uniform(0.75,1.25, size=1)
		    X[i] = zoom(X[i], float(scale), order=3, mode='reflect')
		    y[i] = zoom(y[i], float(scale), order=0, mode='reflect')
		    #Flip
		    #NB: do not flip in z axis due to asymmetric PSF in HREM data
		    axes = np.random.randint(4, size=1)
		    if axes=0:
	    	     #flip in x axis
	    	     X[i] = np.flip(X[i],1) 
	    	     y[i] = np.flip(y[i],1)
		    elif axes=1:
	    	     #flip in y axis
	    	     X[i] = np.flip(X[i],2) 
	    	     y[i] = np.flip(y[i],2)
		    elif axes=2:
	    	     #flip in x and y axis
	    	     X[i] = np.flip(X[i],(1,2)) 
	    	     y[i] = np.flip(y[i],(1,2)) 
		    #if axes=3, no flip
	    return X, y
    
    
class MetricDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,log_dir=None):
        super().__init__()
        self.log_dir = log_dir # directory where logs are saved
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs={}):
        # have tf log custom metrics and save to file
        print()
        with self.file_writer.as_default():
            for k,v in zip(logs.keys(),logs.values()):
                # iterate through monitored metrics (k) and values (v)
                tf.summary.scalar(k, v, step=epoch)

class ImageDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,generator,log_dir=None,index=0):
        super().__init__()
        self.log_dir = log_dir
        self.x = None
        self.y = None
        self.pred = None
        self.data_generator = generator
        self.index = index
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def plot_to_image(fig):
        # save image as png to memory, then convert to tensor (to allow display with tensorboard)
        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        #plt.savefig('F:\epoch'+str(epoch)+'_output.png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(),channels=4)
        image = tf.expand_dims(image,0)
        
        return image
    
    def on_epoch_end(self, epoch, logs={}):

        self.x, self.y = self.data_generator.__getitem__(self.index)
        self.pred = self.model.predict(self.x)

        sz = self.x.shape

        # Take centre slice
        ind = int(sz[1]/2.)

        im = self.x[0,ind,:,:,0].squeeze()        
        pred_im = np.argmax(self.pred[0,ind,:,:,:],axis=-1)
        pred_imTrue = np.argmax(self.y[0,ind,:,:,:],axis=-1)

        # Plot
        columns = 3
        rows = 1

        fsz = 5
        fig = plt.figure(figsize=(fsz*columns,fsz*rows))

        i = 1
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(im)
        ax.title.set_text('Image')
        plt.axis("off")

        i = 2
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_im)
        ax.title.set_text('Predicted labels')
        plt.axis("off")

        i = 3
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_imTrue)
        ax.title.set_text('Labels')
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        #plt.savefig('F:\epoch'+str(epoch)+'_output.png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(),channels=4)
        image = tf.expand_dims(image,0)
        
        return image

        with self.file_writer.as_default():
                tf.summary.image("Example subvolume",image,step=epoch)
