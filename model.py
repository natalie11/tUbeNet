# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:21:04 2022

@author: Natal
"""
#Import libraries
import os
import numpy as np
from functools import partial

# import required objects and fuctions from keras
from tensorflow.keras.models import Model, model_from_json
# CNN layers
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, LeakyReLU, Dropout#, AveragePooling3D, Reshape, Flatten, Dense, Lambda
# utilities
from tensorflow.keras.utils import multi_gpu_model #np_utils
# opimiser
from tensorflow.keras.optimizers import Adam

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

tf.config.run_functions_eagerly(True)

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

def weighted_crossentropy(y_true, y_pred, weights):
	"""Custom loss function - weighted to address class imbalance"""
	weight_mask = y_true[...,0] * weights[0] + y_true[...,1] * weights[1]
	return K.categorical_crossentropy(y_true, y_pred,) * weight_mask

def DiceBCELoss(y_true, y_pred, smooth=1e-6):    
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1-dice(y_true, y_pred)
    Dice_BCE = (BCE + dice_loss)/2
    return Dice_BCE

class AttnBlock(tf.keras.layers.Layer):
	def __init__(self, channels=32):
		super(AttnBlock,self).__init__()
		self.Wq = Conv3D(channels, (3, 3, 3), padding='same', kernel_initializer='he_uniform')
		self.Wk = Conv3D(channels, (3, 3, 3), padding='same', kernel_initializer='he_uniform')
		self.map = Conv3D(channels, (1, 1, 1), activation= 'sigmoid', padding='same', kernel_initializer='he_uniform')
	def call (self, query, key):
		w_query=self.Wq(query)
		w_key=self.Wk(key)
		dot_prod=tf.matmul(w_query, w_key, transpose_b=True)
		attn_map=self.map(dot_prod)
		return attn_map*query
    
class EncodeBlock(tf.keras.layers.Layer):
	def __init__(self, channels=32, alpha=0.2, dropout=0.25):
		super(EncodeBlock,self).__init__()
		self.conv1 = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.conv2 = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.lrelu = LeakyReLU(alpha=alpha)
		self.pool = MaxPooling3D(pool_size=(2, 2, 2))
		self.dropout = Dropout(dropout)
	def call (self, x):
		conv1 = self.conv1(x)
		activ1 = self.lrelu(conv1)
		conv2 = self.conv2(activ1)
		activ2 = self.lrelu(conv2)
		pool = self.pool(activ2)
		drop = self.dropout(pool)
		return drop

class DecodeBlock(tf.keras.layers.Layer):
	def __init__(self, channels=32, alpha=0.2):
		super(DecodeBlock,self).__init__()
		self.transpose = Conv3DTranspose(channels, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')
		self.conv = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.lrelu = LeakyReLU(alpha=alpha)
		self.channels = channels
	def call (self, skip, x):
		attn = AttnBlock(channels=self.channels)(skip, x)
		transpose = self.transpose(attn)
		activ1 = self.lrelu(transpose)
		conv = self.conv(activ1)
		activ2 = self.lrelu(conv)
		return activ2
    
class tUbeNet(tf.keras.Model):   
    def __init__(self, n_classes=2, input_dims=(64,64,64), dropout=0.25, alpha=0.2):
        super(tUbeNet,self).__init__()
        self.n_classes=n_classes
        self.input_dims=input_dims
        self.dropout=dropout
        self.alpha=alpha
        
    def build(self):        
        inputs = Input((*self.input_dims, 1))
        
        block1 = EncodeBlock(channels=32, alpha=self.alpha, dropout=self.dropout)(inputs)
        block2 = EncodeBlock(channels=64, alpha=self.alpha, dropout=self.dropout)(block1)
        block3 = EncodeBlock(channels=128, alpha=self.alpha, dropout=self.dropout)(block2)
        block4 = EncodeBlock(channels=256, alpha=self.alpha, dropout=self.dropout)(block3)
        block5 = EncodeBlock(channels=512, alpha=self.alpha, dropout=self.dropout)(block4)
        
        block6 = Conv3D(1024, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(block5)
        block6 = LeakyReLU(alpha=self.alpha)(block6)
        block6 = Conv3D(512, (3, 3, 3), activation='linear', padding='same', kernel_initializer='he_uniform')(block6)
        block6 = LeakyReLU(alpha=self.alpha)(block6)
        
        upblock1 = DecodeBlock(channels=512, alpha=self.alpha)(block5, block6)
        upblock2 = DecodeBlock(channels=256, alpha=self.alpha)(block4, upblock1)
        upblock3 = DecodeBlock(channels=128, alpha=self.alpha)(block3, upblock2)
        upblock4 = DecodeBlock(channels=64, alpha=self.alpha)(block2, upblock3)
        upblock5 = DecodeBlock(channels=32, alpha=self.alpha)(block1, upblock4)
    
        out = Conv3D(self.n_classes, (1, 1, 1), activation='softmax')(upblock5)
        model = Model(inputs=[inputs], outputs=[out]) 
        
        return model
    
    def create(self, loss=None, class_weights=(1,1), learning_rate=1e-3, metrics=['accuracy']):
        if loss == 'weighted categorical crossentropy':
            custom_loss=partial(weighted_crossentropy, weights=class_weights)
            custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
            custom_loss.__module__ = weighted_crossentropy.__module__
        elif loss == 'DICE BCE':
            custom_loss=partial(DiceBCELoss,smooth=1e-6)
            custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
            custom_loss.__module__ = DiceBCELoss.__module__
        else:
            print('Loss not recognised, using categorical crossentropy')
            custom_loss='categorical_crossentropy'
            
        physical_devices = tf.config.list_physical_devices('GPU')
        n_gpus=len(physical_devices)
        if n_gpus >1:
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            print("Creating model on {} GPUs".format(n_gpus))
            with strategy.scope():	
    	           model = self.build()
    	           model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)
        else:
            model = self.build()
            model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)
        
        return model
    
    def load_weights(self, filename=None, loss=None, class_weights=(1,1), 
             learning_rate=1e-5, metrics=['accuracy'], freeze_layers=0, fine_tune=False):
        """ Fine Tuning
        Replaces classifer layer and freezes shallow layers for fine tuning
        Inputs:
        filename = path to file containing model weights
        freeze_layers = number of layers to freeze for training (int, default 0)
        learning_rate = learning rate (float, default 1e-5)
        loss = loss function, function or string
        metrics = training metrics, list of functions or strings
        Outputs:
        model = compiled model
        """
        
        physical_devices = tf.config.list_physical_devices('GPU')
        n_gpus=len(physical_devices)
        # create path for file containing weights
        if os.path.isfile(filename+'.h5'):
            mfile = filename+'.h5'
        elif os.path.isfile(filename+'.hdf5'):
            mfile = (filename+'.hdf5')
        else: print("No model weights file found")
        
        if loss == 'weighted categorical crossentropy':
            custom_loss=partial(weighted_crossentropy, weights=class_weights)
            custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
            custom_loss.__module__ = weighted_crossentropy.__module__
        elif loss == 'DICE BCE':
            custom_loss=partial(DiceBCELoss,smooth=1e-6)
            custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
            custom_loss.__module__ = DiceBCELoss.__module__
        else:
            print('Loss not recognised, using categorical crossentropy')
            custom_loss='categorical_crossentropy'
        
        
        if fine_tune:
            if n_gpus>1:
    	           strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    	           print("Creating model on {} GPUs".format(n_gpus))
    	           with strategy.scope():
    	                  model = self.build()
    	                  # load weights into new model
    	                  model.load_weights(mfile)
                          
    	                  # recover the output from the last layer in the model and use as input to new Classifer
    	                  last = model.layers[-2].output
    	                  classifier = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='newClassifier')(last)
    	                  model = Model(inputs=[model.input], outputs=[classifier])
                          # freeze weights for selected layers
    	                  for layer in model.layers[:freeze_layers]: layer.trainable = False
                          
    	                  model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)

            else:
    	           model = self.build()
    	           # load weights into new model
    	           model.load_weights(mfile)
                          
    	           # recover the output from the last layer in the model and use as input to new Classifer
    	           last = model.layers[-2].output
    	           classifier = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='newClassifier')(last)
    	           model = Model(inputs=[model.input], outputs=[classifier])
    	           # freeze weights for selected layers
    	           for layer in model.layers[:freeze_layers]: layer.trainable = False
                          
    	           model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss, metrics=metrics)

        else:
            if n_gpus>1:
    	           strategy = tf.distribute.MirroredStrategy()
    	           print("Creating model on {} GPUs".format(n_gpus))
    	           with strategy.scope():
    	                  model = self.build()
    	                  # load weights into new model
    	                  model.load_weights(mfile)
    	                  model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
            else:
    	           model = self.build()
    	           # load weights into new model
    	           model.load_weights(mfile)
    	           model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

        print('Model Summary')
        model.summary()
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        