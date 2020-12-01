# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import numpy as np
import tUbeNet_functions as tube
import random
import pickle
import os
join = os.path.join
import datetime
import json
import io
# kera utils
from matplotlib import pyplot as plt
import tensorflow as tf
Sequence = tf.keras.utils.Sequence 
to_categorical = tf.keras.utils.to_categorical #np_utils

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
    def __init__(self, data_dir, batch_size=32, volume_dims=(64,64,64), shuffle=True, n_classes=2, dataset_weighting=None):
	    'Initialization'
	    self.volume_dims = volume_dims
	    self.batch_size = batch_size
	    self.shuffle = shuffle
	    self.data_dir = data_dir
	    self.on_epoch_end()
	    self.n_classes = n_classes
	    self.dataset_weighting = dataset_weighting
	    
    def __len__(self):
        'Denotes the number of batches per epoch'
        batches = 0 
        for i in range(len(self.data_dir.list_IDs)):
            batches_per_dataset = int(np.floor(np.prod(self.data_dir.image_dims[i])/np.prod(self.volume_dims)))
            batches += batches_per_dataset
            return batches
        
    def __getitem__(self, index):
        'Generate one batch of data'
		# random.choices only available in python 3.6
		# randomly generate list of ID for batch, weighted according to given 'dataset_weighting' if not None
        list_IDs_temp = random.choices(self.data_dir.list_IDs, weights=self.dataset_weighting, k=self.batch_size)
		# Generate data
        X, y = self.__data_generation(list_IDs_temp)
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
            coords_temp=np.array([random.randint(0,(self.data_dir.image_dims[index][0]-self.volume_dims[0]-1)),
                                  random.randint(0,(self.data_dir.image_dims[index][1]-self.volume_dims[1]-1)),
                                  random.randint(0,(self.data_dir.image_dims[index][2]-self.volume_dims[2]-1))])
            
            # Generate data sub-volume at coordinates, add to batch
            X[i], y[i] = tube.load_volume_from_file(volume_dims=self.volume_dims, image_dims=self.data_dir.image_dims[index],
                                                image_filename=self.data_dir.image_filenames[index], label_filename=self.data_dir.label_filenames[index], 
                                                coords=coords_temp, data_type=self.data_dir.data_type[index], offset=128)
	    	  
	    	     
	    	    
	    	  	

                     
	    # Reshape to add depth of 1
        X = X.reshape(*X.shape, 1)

		
        return X, to_categorical(y, num_classes=self.n_classes)
    
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

# class ImageDisplayCallback(tf.keras.callbacks.Callback):

#     def __init__(self,generator,log_dir=None,index=0):
#         super().__init__()
#         self.log_dir = join(log_dir,'image')
#         self.x = None
#         self.y = None
#         self.pred = None
#         self.data_generator = generator
#         self.index = index
#         self.file_writer = tf.summary.create_file_writer(self.log_dir)

#     def plot_to_image(fig):
#         # save image as png to memory, then convert to tensor (to allow display with tensorboard)
#         buf = io.BytesIO()
#         plt.savefig(buf,format='png')
#         #plt.savefig('F:\epoch'+str(epoch)+'_output.png')
#         plt.close(fig)
#         buf.seek(0)
#         image = tf.image.decode_png(buf.getvalue(),channels=4)
#         image = tf.expand_dims(image,0)
        
#         return image
    
#     def on_batch_end(self, batch, logs={}):
    
#         if batch%20!=0:
#             return

#         # Plot
#         columns = 3
#         nexample = 3
#         rows = nexample * 2

#         fsz = 5
#         fig = plt.figure(figsize=(fsz*columns,fsz*rows))
        

#         ii = 0 # plot count
#         for i in [0,1]:
#             pred = self.model.predict(self.x[i])

#             for j in range(nexample):

#                 # Choose slice (most vessel pixels)
#                 predtrue = self.predTrue[i][j,:,:,:]
#                 zdim = predtrue.shape[0] 
#                 ind = np.argmax([np.sum(predtrue[k,:,:]) for k in range(zdim)])

#                 im = self.x[i][j,ind,:,:,0].squeeze()        
#                 pred_im = np.argmax(pred[j,ind,:,:,:],axis=-1)
#                 pred_imTrue = self.predTrue[i][j,ind,:,:].squeeze()   #np.argmax(self.y[0,ind,:,:,:],axis=-1)
            
#                 ii += 1
#                 ax = fig.add_subplot(rows, columns, ii)
#                 plt.imshow(im)
#                 ax.title.set_text(self.ids[i][j])
#                 plt.axis("off")

#                 ii += 1
#                 ax = fig.add_subplot(rows, columns, ii)
#                 plt.imshow(pred_im,vmin=0,vmax=1)
#                 ax.title.set_text('Predicted labels')
#                 plt.axis("off")

#                 ii += 1
#                 ax = fig.add_subplot(rows, columns, ii)
#                 plt.imshow(pred_imTrue,vmin=0,vmax=1)
#                 ax.title.set_text('Labels')
#                 plt.axis("off")

#         buf = io.BytesIO()
#         plt.savefig(buf,format='png')
#         #plt.savefig('F:\batch'+str(batch)+'_output.png')
#         plt.close(fig)
#         buf.seek(0)
#         image = tf.image.decode_png(buf.getvalue(),channels=4)
#         image = tf.expand_dims(image,0)

#         with self.file_writer.as_default():
#             tf.summary.image("Example subvolume",image,step=batch)
    
#from sklearn.metrics import roc_auc_score
#from keras.callbacks import Callback
#
#class roc_callback(Callback):
#    def __init__(self,data_dir,val_data_dir, test_generator):
#	    'Initialization'    
#	    self.data_dir = data_dir
#	    self.val_data_dir = val_data_dir
#
#    def on_train_begin(self, logs={}):
#        return
#
#    def on_train_end(self, logs={}):
#        return
#
#    def on_epoch_begin(self, epoch, logs={}):
#        return
#
#    def on_epoch_end(self, epoch, logs={}):
#        y_pred = self.model.predict_generator(self.test_generator)
#        roc = roc_auc_score(self.y, y_pred)
#        y_pred_val = self.model.predict(self.x_val)
#        roc_val = roc_auc_score(self.y_val, y_pred_val)
#        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
#        return
#
#    def on_batch_begin(self, batch, logs={}):
#        return
#
#    def on_batch_end(self, batch, logs={}):
#        return