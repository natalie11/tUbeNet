# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import numpy as np
import math
import tUbeNet_functions as tube
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
	def __init__(self, data_dir, batch_size=32, volume_dims=(64,64,64), shuffle=True, n_classes=2, 
              dataset_weighting=None, augment=False, classifier=False):
	    'Initialization'
	    self.volume_dims = volume_dims
	    self.batch_size = batch_size
	    self.shuffle = shuffle
	    self.data_dir = data_dir
	    self.on_epoch_end()
	    self.n_classes = n_classes
	    self.dataset_weighting = dataset_weighting
	    self.augment = augment
	    self.classifier = classifier
	    
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
		if self.classifier: X, y = self.__data_generation_classifier(list_IDs_temp)
		else: 
		    X, y = self.__data_generation(list_IDs_temp)
		    if self.augment: #classifier data cannot be augmented by current function
		        self._augmentation(X,y)

		# Reshape to add depth of 1
		X = X.reshape(*X.shape, 1)

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
                     
	    return X, to_categorical(y, num_classes=self.n_classes)
    
	def __data_generation_classifier(self, list_IDs_temp):
	    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
	    # Initialization
	    X = np.empty((self.batch_size, *self.volume_dims))
	    y = np.zeros((self.batch_size))
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
	    	     X[i], y_temp = tube.load_volume_from_file(volume_dims=self.volume_dims, image_dims=self.data_dir.image_dims[index],
                           image_filename=self.data_dir.image_filenames[index], label_filename=self.data_dir.label_filenames[index], 
                           coords=coords_temp, data_type=self.data_dir.data_type[index], offset=128)	
	    	     if (np.count_nonzero(y_temp[...,1])/y_temp[...,1].size)>0.02: #sub-volume must contain at least 2% vessels to be labeled as vessel
	    	        y[i]=1 #label as containing vessels
	    	        vessels_present=True
	    	     elif count>5: vessels_present=True #generate next sample
	    print("Batch contains {} volumes with vessels".format(np.count_nonzero(y)))             
	    return X, to_categorical(y, num_classes=self.n_classes)
    
	def _augmentation(self, X, y):
	    # Apply data augmentations to each image/label pair in batch
	    for i in range(self.batch_size):
		    #Rotate
		    angle = np.random.uniform(-30,30, size=1)
		    X[i] = rotate(X[i], float(angle), reshape=False, order=3, mode='reflect')
		    y[i] = rotate(y[i], float(angle), reshape=False, order=0, mode='reflect')
		    #Zoom and crop
		    scale = np.random.uniform(1.0,1.25, size=1)
		    Xzoom = zoom(X[i], float(scale), order=3, mode='reflect')
		    yzoom = zoom(y[i], float(scale), order=0, mode='reflect')
		    (d,h,w)=X[i].shape
		    (dz,hz,wz)=Xzoom.shape
		    dz=int((dz-d)//2)
		    hz=int((hz-h)//2)
		    wz=int((wz-w)//2)
		    X[i]=Xzoom[dz:int(dz+d), hz:int(hz+h), wz:int(wz+w)]
		    y[i]=yzoom[dz:int(dz+d), hz:int(hz+h), wz:int(wz+w)]
		    #Flip
		    #NB: do not flip in z axis due to asymmetric PSF in HREM data
		    axes = np.random.randint(4, size=1)
		    if axes==0:
	    	        #flip in x axis
	    	        X[i] = np.flip(X[i],1) 
	    	        y[i] = np.flip(y[i],1)
		    elif axes==1:
	    	        #flip in y axis
	    	        X[i] = np.flip(X[i],2) 
	    	        y[i] = np.flip(y[i],2)
		    elif axes==2:
	    	        #flip in x and y axis
	    	        X[i] = np.flip(X[i],(1,2)) 
	    	        y[i] = np.flip(y[i],(1,2)) 
		    #if axes==3, no flip
	    return X, y
    
    
class MetricDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,log_dir=None):
        super().__init__()
        self.log_dir = log_dir # directory where logs are saved
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs={}):
        # have tf log custom metrics and save to file
        with self.file_writer.as_default():
            for k,v in zip(logs.keys(),logs.values()):
                # iterate through monitored metrics (k) and values (v)
                tf.summary.scalar(k, v, step=epoch)

class ImageDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator, log_dir=None, index=0):
        super().__init__()
        self.x = None
        self.y = None
        self.pred = None
        self.data_generator = generator #data generator
        self.log_dir = log_dir # directory where logs are saved
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.index=index

    def on_epoch_end(self, epoch, logs={}):
        self.x, self.y = self.data_generator.__getitem__(self.index)
        self.pred = self.model.predict(self.x)
        
        x_shape=self.x.shape
        z_centre = int(x_shape[1]/2)
        img = self.x[0,z_centre,:,:,:] #take centre slice in z-stack
        labels = np.reshape(np.argmax(self.y[0,z_centre,:,:,:], axis=-1),(x_shape[1],x_shape[2],1)) #reverse one hot encoding
        pred = np.reshape(np.argmax(self.pred[0,z_centre,:,:,:], axis=-1),(x_shape[1],x_shape[2],1)) #reverse one hot encoding
        img = tf.convert_to_tensor(img,dtype=tf.float32)
        labels = tf.convert_to_tensor(labels,dtype=tf.float32)
        pred = tf.convert_to_tensor(pred,dtype=tf.float32)
        with self.file_writer.as_default():
            tf.summary.image("Example output", [img, labels, pred], step=epoch)


class FilterDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,log_dir=None):
        super().__init__()
        self.log_dir = log_dir # directory where logs are saved
        self.file_writer = tf.summary.create_file_writer(log_dir)
        
    def find_grid_dims(self, n):
        # n = number of filters
        # Starting at sqrt(n), check if i is a factor, if yes use i and n/i as rows/columns respectively
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0: #NB is i==1, you have a prime number of filters..
                return (i, int(n / i))   
            
    def make_grid(self, n, filters):
        # n = number of filters
        (rows, columns)=self.find_grid_dims(n)
        # normalize filter between 0-1
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        cz = int(math.ceil(filters.shape[0]/2))
        fig = plt.figure()
        index=1
        for i in range(rows):
            for j in range(columns):
                plt.subplot(rows, columns, index)
                plt.xticks([]) #no ticks
                plt.yticks([])
                plt.grid(False)
                plt.imshow(filters[cz,:,:,0,index-1]) #plot central slice of 3D filter
                index=index+1
                
        return fig

    def plot_to_img(self, plot):            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(plot)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def on_epoch_end(self, epoch, logs={}):
        # visualise filters for conv1 layer
        layer=self.model.layers[1] #first block after input layer
        # get filter weights
        filters = layer.get_weights()[0] #first conv layer only
        n = filters.shape[-1] #number of filters
        plot = self.make_grid(n, filters)
        image = self.plot_to_img(plot)
            
        with self.file_writer.as_default():
            tf.summary.image("Convolution 1 filters from layer "+str(layer.name), image, step=epoch)
