# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import numpy as np
import math
import random
import pickle
import os
join = os.path.join

import io
from matplotlib import pyplot as plt
from scipy.ndimage import rotate, zoom
import tensorflow as tf
import dask.array as da

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
	def __init__(self, list_IDs, image_dims=(1024,1024,1024), image_filenames=None, label_filenames=None, data_type='float64', exclude_region=None):
	    'Initialization'    
	    self.image_dims = image_dims
	    self.image_filenames = image_filenames
	    self.label_filenames = label_filenames
	    self.list_IDs = list_IDs
	    self.data_type = data_type
	    self.exclude_region = exclude_region

class DataGenerator(Sequence):
	def __init__(self, data_dir, batch_size=32, volume_dims=(64,64,64), shuffle=True, n_classes=2, 
              dataset_weighting=None, augment=False, vessel_threshold=0.001, **kwargs):
	    'Initialization'
	    super().__init__(**kwargs) 
        
	    self.volume_dims = volume_dims
	    self.batch_size = batch_size
	    self.shuffle = shuffle
	    self.data_dir = data_dir
	    self.on_epoch_end()
	    self.n_classes = n_classes
	    self.dataset_weighting = dataset_weighting
	    self.augment = augment
	    self.vessel_threshold = vessel_threshold
        
        # Open zarr arrays
	    self._images = [da.from_zarr(p) for p in self.data_dir.image_filenames]
	    self._labels = [da.from_zarr(p) for p in self.data_dir.label_filenames]
	    
	def __len__(self):
		'Denotes the max number of batches per epoch'
		batches=0 
		for i in range(len(self.data_dir.list_IDs)):
		    batches_per_dataset = int(np.floor(np.prod(self.data_dir.image_dims[i])/np.prod(self.volume_dims)))
		    batches += batches_per_dataset
		return batches
	
	def __getitem__(self, index):
		'Generate one batch of data'
		# randomly generate list of IDs for batch, weighted according to given 'dataset_weighting' if not None
		if len(self.data_dir.list_IDs)>2:
		    list_IDs_temp = random.choices(self.data_dir.list_IDs, weights=self.dataset_weighting, k=self.batch_size)
		else: list_IDs_temp=[self.data_dir.list_IDs[0]]*self.batch_size
		# Generate data
		X, y = self.__data_generation(list_IDs_temp)
		if self.augment: 
		    self._augmentation(X,y)

		# Reshape to add depth of 1, one hot encode labels
		X = X.reshape(*X.shape, 1)
		y = to_categorical(y, num_classes=self.n_classes)
		return X, y
	    
	def on_epoch_end(self):
	    'Updates indexes after each epoch'
	    self.indexes = np.arange(len(self.data_dir.list_IDs))
	    if self.shuffle == True:
		    np.random.shuffle(self.indexes)
            
		    
	def random_coordinates(self, image_dims, exclude_region):
	    coords=np.zeros(3)
	    for ax in range(3):
		    coords[ax] = random.randint(0,(image_dims[ax]-self.volume_dims[ax]))
		    if exclude_region[ax] is not None:
	    	     exclude = range(exclude_region[ax][0]-self.volume_dims[ax], exclude_region[ax][1])
	    	     while coords[ax] in exclude: # if coordinate falls in excluded region, generate new coordinate
	    	        coords[ax] = random.randint(0,(image_dims[ax]-self.volume_dims[ax]))
                
	    return coords
		    
	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.volume_dims))
		y = np.empty((self.batch_size, *self.volume_dims))
		for i, ID_temp in enumerate(list_IDs_temp):
			index=self.data_dir.list_IDs.index(ID_temp)
            
			X_da = self._images[index]
			y_da = self._labels[index]
            
			vessels_present=False
			count=0
			while not vessels_present:
                #Generate random coordinates within dataset
				count+=1
				z0, x0, y0 = self.random_coordinates(self.data_dir.image_dims[index], 
                                                    self.data_dir.exclude_region[index])
				dz, dx, dy = self.volume_dims
                #Load labels at coordinates
				y_slice = y_da[z0:z0+dz, x0:x0+dx, y0:y0+dy]
				y_slice = y_slice.compute()  # brings just this sub-volume to RAM as np.array
                
                #Check fraction of pixels classed as vessel in labels before loading in image data
				frac = y_slice.astype(bool).mean()
				if frac>self.vessel_threshold or count>5: vessels_present=True
				if vessels_present:
					X_slice = X_da[z0:z0+dz, x0:x0+dx, y0:y0+dy]
					X_slice = X_slice.compute() 
                    
			X[i]=X_slice.astype(np.float32)
			y[i]=y_slice.astype(np.int32)
		return X, y
       
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
