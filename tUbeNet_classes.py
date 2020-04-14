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
# kera utils
from keras.utils import Sequence, to_categorical #np_utils

# set backend as tensor flow
from keras import backend as K
K.set_image_dim_ordering('tf')

#---------------------------------------------------------------------------------------------------------------------------------------------
class DataHeader:
    def __init__(self, modality=None, image_dims=(1024,1024,1024), image_filenames=None, label_filenames=None):
	    'Initialization' 
	    self.modality = modality
	    self.image_dims = image_dims
	    self.image_filenames = image_filenames
	    self.label_filenames = label_filenames
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
	def __init__(self, data_dir, batch_size=32, volume_dims=(64,64,64), shuffle=True, n_classes=2):
	    'Initialization'
	    self.volume_dims = volume_dims
	    self.batch_size = batch_size
	    self.shuffle = shuffle
	    self.data_dir = data_dir
	    self.on_epoch_end()
	    self.n_classes = n_classes
	    
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.data_dir.list_IDs) / self.batch_size))
	
	def __getitem__(self, index):
		'Generate one batch of data'
       
		list_IDs_temp = [] 
		for i in range(self.batch_size):
       # Randomly select dataset ID       
				ind = random.randint(0,100)
				if ind<10:
						ID_temp=2 #RSOM
				elif ind<40:
						ID_temp=0 #CT
				else:
						ID_temp=1 #HREM
				list_IDs_temp.append(self.data_dir.list_IDs[ID_temp])

		# Generate data
		#print('list IDs: {}'.format(list_IDs_temp))
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
		    index=np.where(self.data_dir.list_IDs == ID_temp)
		    index=index[0][0]
                        
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