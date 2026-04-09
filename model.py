# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:21:04 2022

@author: Natal
"""
#Import libraries
import os
from functools import partial
import tUbeNet_metrics as metrics

# import required objects and fuctions from keras
from tensorflow.keras.models import Model
# CNN layers
from tensorflow.keras.layers import (
    Input, concatenate, Conv3D, MaxPooling3D, 
    Conv3DTranspose, LeakyReLU, Dropout, Dense, Flatten, GroupNormalization)
# opimiser
from tensorflow.keras.optimizers import Adam

# import tensor flow
import tensorflow as tf

# set backend and dim ordering
K=tf.keras.backend
K.set_image_data_format('channels_last')

# set memory limit on gpu
physical_devices = tf.config.list_physical_devices('GPU')
try:
  for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
except:
  pass
tf.config.optimizer.set_experimental_options({"remapping": False})

"""Model blocks"""
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
	def __init__(self, channels=32, alpha=0.2, dropout=0.3):
		super(EncodeBlock,self).__init__()
		self.conv1 = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.conv2 = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.norm = GroupNormalization(groups=int(channels/4), axis=4)
		self.lrelu = LeakyReLU(negative_slope=alpha)
		self.pool = MaxPooling3D(pool_size=(2, 2, 2))
		self.dropout = Dropout(dropout)
	def call (self, x):
		conv1 = self.conv1(x)
		activ1 = self.lrelu(conv1)
		norm1 = self.norm(activ1)
		conv2 = self.conv2(norm1)
		activ2 = self.lrelu(conv2)
		norm2 = self.norm(activ2)
		pool = self.pool(norm2)
		drop = self.dropout(pool)
		return drop

class DecodeBlock(tf.keras.layers.Layer):
	def __init__(self, channels=32, alpha=0.2):
		super(DecodeBlock,self).__init__()
		self.transpose = Conv3DTranspose(channels, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_uniform')
		self.conv = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.attn = AttnBlock(channels=channels)
		self.norm = GroupNormalization(groups=int(channels/4), axis=4)
		self.lrelu = LeakyReLU(negative_slope=alpha)
		self.channels = channels
	def build(self, input_shape):
		super().build(input_shape)
	def call (self, skip, x, attention=False):
		if attention:
			attn = self.attn(skip, x)
		else:
			attn = concatenate([skip, x], axis=4)
		transpose = self.transpose(attn)
		activ1 = self.lrelu(transpose)
		norm1 = self.norm(activ1)
		conv = self.conv(norm1)
		activ2 = self.lrelu(conv)
		norm2 = self.norm(activ2)
		return norm2
    
class UBlock(tf.keras.layers.Layer):
	def __init__(self, channels=32, alpha=0.2):
		super(UBlock,self).__init__()     
		self.conv1 = Conv3D(channels, (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.conv2 = Conv3D(int(channels/2), (3, 3, 3), activation= 'linear', padding='same', kernel_initializer='he_uniform')
		self.norm = GroupNormalization(groups=int(channels/4), axis=4)
		self.lrelu = LeakyReLU(negative_slope=alpha)
	def call (self, x):
		conv1 = self.conv1(x)
		activ1 = self.lrelu(conv1)
		norm1 = self.norm(activ1)
		conv2 = self.conv2(norm1)
		activ2 = self.lrelu(conv2)
		return activ2
    
class EncoderOnlyOutput(tf.keras.layers.Layer):
    # Additional classifier layer for encoder-only model - takes output from UBlock and outputs class probabilities
    # Not currently used
	def __init__(self, channels=64, alpha=0.2):
		super(EncoderOnlyOutput,self).__init__()
		self.flatten = Flatten()
		self.dense1 = Dense(channels, activation='linear', kernel_initializer='he_uniform')
		self.dense2 = Dense(2, activation='softmax') #classifier
		self.lrelu = LeakyReLU(negative_slope=alpha)
	def call (self, x):
		flatten = self.flatten(x)
		dense1 = self.dense1(flatten)
		activ1 = self.lrelu(dense1)
		dense2 = self.dense2(activ1)
		return dense2

"""Build Model"""
class tUbeNet(tf.keras.Model):   
    def __init__(self, n_classes=2, input_dims=(64,64,64), dropout=0.3, alpha=0.2, attention=False, dual_output=False):
        super(tUbeNet,self).__init__()
        self.n_classes=n_classes
        self.input_dims=input_dims
        self.dropout=dropout
        self.alpha=alpha
        self.attention=attention
        self.dual_output=dual_output
        
    def build_model(self, encoder_only=False):        
        inputs = Input((*self.input_dims, 1))
             
        block1 = EncodeBlock(channels=32, alpha=self.alpha, dropout=self.dropout)(inputs)
        block2 = EncodeBlock(channels=64, alpha=self.alpha, dropout=self.dropout)(block1)
        block3 = EncodeBlock(channels=128, alpha=self.alpha, dropout=self.dropout)(block2)
        block4 = EncodeBlock(channels=256, alpha=self.alpha, dropout=self.dropout)(block3)
        block5 = EncodeBlock(channels=512, alpha=self.alpha, dropout=self.dropout)(block4)
        
        block6 = UBlock(channels=1024, alpha=self.alpha)(block5)
        
        if encoder_only:
            output = block6
            
        elif self.dual_output:
            # Decoder 1: Mask (vessel segmentation)
            upblock1_mask = DecodeBlock(channels=512, alpha=self.alpha)(block5, block6, attention=self.attention)
            upblock2_mask = DecodeBlock(channels=256, alpha=self.alpha)(block4, upblock1_mask, attention=self.attention)
            upblock3_mask = DecodeBlock(channels=128, alpha=self.alpha)(block3, upblock2_mask, attention=self.attention)
            upblock4_mask = DecodeBlock(channels=64, alpha=self.alpha)(block2, upblock3_mask, attention=self.attention)
            upblock5_mask = DecodeBlock(channels=32, alpha=self.alpha)(block1, upblock4_mask, attention=self.attention)
            
            output_mask = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='mask_output')(upblock5_mask)
            
            # Decoder 2: Skeleton (distance field)
            # Cross-decoder skip connections: incorporate mask decoder features
            upblock1_skel = DecodeBlock(channels=512, alpha=self.alpha)(block5, block6, attention=self.attention)
            upblock2_skel = DecodeBlock(channels=256, alpha=self.alpha)(
                concatenate([block4, upblock1_mask], axis=4), upblock1_skel, attention=self.attention)
            upblock3_skel = DecodeBlock(channels=128, alpha=self.alpha)(
                concatenate([block3, upblock2_mask], axis=4), upblock2_skel, attention=self.attention)
            upblock4_skel = DecodeBlock(channels=64, alpha=self.alpha)(
                concatenate([block2, upblock3_mask], axis=4), upblock3_skel, attention=self.attention)
            upblock5_skel = DecodeBlock(channels=32, alpha=self.alpha)(
                concatenate([block1, upblock4_mask], axis=4), upblock4_skel, attention=self.attention)
            
            # Output single channel for distance map (sigmoid for 0-1 range)
            output_skeleton = Conv3D(1, (1, 1, 1), activation='sigmoid', name='skeleton_output')(upblock5_skel)
            
            # Return list of outputs
            output = [output_mask, output_skeleton]
            
        else:
            upblock1 = DecodeBlock(channels=512, alpha=self.alpha)(block5, block6, attention=self.attention)
            upblock2 = DecodeBlock(channels=256, alpha=self.alpha)(block4, upblock1, attention=self.attention)
            upblock3 = DecodeBlock(channels=128, alpha=self.alpha)(block3, upblock2, attention=self.attention)
            upblock4 = DecodeBlock(channels=64, alpha=self.alpha)(block2, upblock3, attention=self.attention)
            upblock5 = DecodeBlock(channels=32, alpha=self.alpha)(block1, upblock4, attention=self.attention)
    
            output = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='output')(upblock5)
            
        model = Model(inputs=inputs, outputs=output) 
        return model
    
    def selectLoss(self, loss_name, class_weights=None):
        """select loss from custom losses"""
        
        if loss_name == 'WCCE':
            # Check class_weights are sensible - if not default to categorical crossentropy
            if class_weights is None:
                print("No class weights provided, using unweighted categorical crossentropy")
                custom_loss='categorical_crossentropy'
            elif len(class_weights)!=self.n_classes:
                print("Number of class weights does not match number of classes, using unweighted categorical crossentropy")
                custom_loss='categorical_crossentropy'
            else:
                custom_loss=partial(metrics.weighted_crossentropy, weights=class_weights)
                custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
                custom_loss.__module__ = metrics.weighted_crossentropy.__module__
        elif loss_name == 'DICE BCE':
            if self.n_classes==2:
                custom_loss=partial(metrics.diceBCELoss,smooth=1e-6)
                custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
                custom_loss.__module__ = metrics.diceBCELoss.__module__
            else:
                print("DICE BCE can only be used with binary labels. Using DICE CE instead.")
                custom_loss=partial(metrics.diceCELoss, ignore_background=True, smooth=1e-6)
                custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
                custom_loss.__module__ = metrics.diceCELoss.__module__
        elif loss_name == 'DICE CE':
            custom_loss=partial(metrics.diceCELoss, ignore_background=True, smooth=1e-6)
            custom_loss.__name__ = "custom_loss" #partial doesn't cope name or module attribute from function
            custom_loss.__module__ = metrics.diceCELoss.__module__
        elif loss_name == 'focal':
            custom_loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.2, gamma=5)
        else:
            print('Loss not recognised, using categorical crossentropy')
            custom_loss='categorical_crossentropy'   

        # If using dual-output model, return two losses - one for each output
        if self.dual_output:
            # Create learnable weighted loss
            skeleton_loss = partial(metrics.skeletonLoss, smooth=1e-6)
            skeleton_loss.__name__ = "skeleton_loss" #partial doesn't cope name or module
            skeleton_loss.__module__ = metrics.skeletonLoss.__module__
            return custom_loss, skeleton_loss
        else:
             return custom_loss
  
    def create(self, loss=None, class_weights=(1,1), learning_rate=1e-3, 
               metrics=['accuracy'], encoder_only=False):

        # Build and compile model, with or without dual-outputs
        def build_and_compile(self, loss, learning_rate, metrics, encoder_only):
            model = self.build_model(encoder_only=encoder_only)
            if self.dual_output:
                mask_loss, skeleton_loss = self.selectLoss(loss, class_weights)
                model.compile(optimizer=Adam(learning_rate=learning_rate), 
                              loss={'mask_output':mask_loss, 'skeleton_output': skeleton_loss},
                              metrics={'mask_output': metrics[0], 'skeleton_output': metrics[1]})
            else:
                custom_loss = self.selectLoss(loss, class_weights)
                model.compile(optimizer=Adam(learning_rate=learning_rate), 
                              loss=custom_loss, metrics=metrics)
            return model
        
        #Check for multiple GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        n_gpus=len(physical_devices)
        if n_gpus >1:
            strategy = tf.distribute.MirroredStrategy()
            print("Creating model on {} GPUs".format(n_gpus))
            with strategy.scope():	
                model = build_and_compile(self, loss, learning_rate, metrics, encoder_only)
        else:
            model = build_and_compile(self, loss, learning_rate, metrics, encoder_only)
            
        print('Model Summary')
        model.summary()        
        return model
    
    def load_weights_and_compile(self, filename=None, loss=None, class_weights=(1,1), 
             learning_rate=1e-5, metrics=['accuracy'], freeze_layers=6, fine_tune=False):
        """ 
        Inputs:
        filename = path to file containing model weights
        freeze_layers = number of layers to freeze for training (int, default 6 = all layers in encoder)
        learning_rate = learning rate (float, default 1e-5)
        loss = loss function, function or string
        metrics = training metrics, list of functions or strings
        fine_tune = replace classifier layer, bool
        Outputs:
        model = compiled model
        """
        
        physical_devices = tf.config.list_physical_devices('GPU')
        n_gpus=len(physical_devices)
        
        # create path for file containing weights
        if filename is None:
            raise ValueError("Model weights filename must be provided")
        if os.path.isfile(filename):
            mfile=filename
        elif os.path.isfile(filename+'.h5'):
            mfile = filename+'.h5'
        elif os.path.isfile(filename+'.hdf5'):
            mfile = (filename+'.hdf5')
        else: raise OSError("Could not locate model weights file at {}".format(filename))
        
        custom_loss=self.selectLoss(loss,class_weights)

        def check_metrics(metrics):
            if self.dual_output and not all(isinstance(m, (list, tuple)) for m in metrics):
                # If metrics does not contain sub-lists for each output, add RSME for skeleton output by default
                return [metrics, ['root_mean_squared_error']]
            return metrics
        
        def compile_model(model, custom_loss, learning_rate, metrics, fine_tune=False):
            if self.dual_output:
                # Assign different losses and metrics to each output, with correct layer names depending on whether fine-tuning or not
                if fine_tune:
                    mask_output_name='new_mask_output'
                    skeleton_output_name='new_skeleton_output'
                else:
                    mask_output_name='mask_output'
                    skeleton_output_name='skeleton_output'
                model.compile(optimizer=Adam(learning_rate=learning_rate), 
                              loss={mask_output_name:custom_loss[0], skeleton_output_name: custom_loss[1]},
                              metrics={mask_output_name: metrics[0], skeleton_output_name: metrics[1]})
            else:
                 model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss, metrics=metrics)
            return model
        
        def load_for_fine_tuning(mfile, freeze_layers, learning_rate, custom_loss, metrics):
            model = self.build_model()
            # load weights into new model
            try:
                model.load_weights(mfile)
            except Exception as e:
                if self.n_classes >2:
                    # Likely using the publically avaiable binary pre-trained weights for a multi-class model
                    print(f"ERROR LOADING WEIGHTS. This may be caused by a mismatch in the number of classes.\
                          \nAttempting to load weights into binary model. A new classifier with {self.n_classes} classes will be added.")
                    try:
                        # Build a base model with binary classifier - this will be replaced later
                        model = tUbeNet(n_classes=2, input_dims=self.input_dims, dropout=self.dropout, 
                                        alpha=self.alpha, attention=self.attention, dual_output=self.dual_output).build_model()
                        # Load weights from mfile
                        model.load_weights(mfile)
                    except: print(f"ERROR LOADING WEIGHTS. Ensure model weights file matches the model architecture. If loading weights from a single-output model to a dual-output architecture, use 'load_encoder_and_compile' function.\
                                  \nOriginal error: {e}") # If this doesn't work - revert to original error message
                else: print(f"ERROR LOADING WEIGHTS. Ensure model weights file matches the model architecture. If loading weights from a single-output model to a dual-output architecture, use 'load_encoder_and_compile' function.\
                                  \nOriginal error: {e}")

            # recover the output from the last layer in the model and use as input to new Classifer
            if self.dual_output:
                last_mask = model.get_layer('mask_output').input
                last_skeleton = model.get_layer('skeleton_output').input
                new_mask_classifier = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='new_mask_output')(last_mask)
                new_skeleton_classifier = Conv3D(1, (1, 1, 1), activation='sigmoid', name='new_skeleton_output')(last_skeleton)
                model = Model(inputs=[model.input], outputs=[new_mask_classifier, new_skeleton_classifier])
            else:
                last = model.layers[-2].output
                classifier = Conv3D(self.n_classes, (1, 1, 1), activation='softmax', name='newClassifier')(last)
                model = Model(inputs=[model.input], outputs=[classifier])

            # freeze weights for selected layers
            for layer in model.layers[:freeze_layers]: layer.trainable = False

            # Compile model
            metrics = check_metrics(metrics)
            model = compile_model(model, custom_loss, learning_rate, metrics)
            return model
        
        if fine_tune:
            if n_gpus>1:
                   strategy = tf.distribute.MirroredStrategy()
                   print("Creating model on {} GPUs".format(n_gpus))
                   with strategy.scope():
                        model = load_for_fine_tuning(mfile, freeze_layers, learning_rate, custom_loss, metrics)             
            else:
                   model = load_for_fine_tuning(mfile, freeze_layers, learning_rate, custom_loss, metrics)

        # If not fine-tuning, load weights into model without replacing classifier or freezing layers
        else:
            if n_gpus>1:
                   strategy = tf.distribute.MirroredStrategy()
                   print("Creating model on {} GPUs".format(n_gpus))
                   with strategy.scope():
                          model = self.build_model()
                          # load weights into new model
                          try:
                            model.load_weights(mfile)
                          except Exception as e: print(f"ERROR LOADING WEIGHTS. Ensure model weights file matches the model architecture (single- versus dual-output) and number of segmentation classes.\
                                                       \nIf loading weights for fine tuning, ensure fine_tune flag is present.\
                                                       \nOriginal error: {e}")
                          model = compile_model(model, custom_loss, learning_rate, check_metrics(metrics), fine_tune=False)
            else:
                   model = self.build_model()
                   # load weights into new model
                   try:
                       model.load_weights(mfile)
                   except Exception as e: print(f"ERROR LOADING WEIGHTS. Ensure model weights file matches the model architecture (single- versus dual-output) and number of segmentation classes.\
                                                \nIf loading weights for fine tuning, ensure fine_tune flag is present.\
                                                \nOriginal error: {e}")
                   model = compile_model(model, custom_loss, learning_rate, check_metrics(metrics), fine_tune=False)

        print('Model Summary')
        model.summary()
        return model
        
    def load_encoder_and_compile(self, filename=None, loss=None, class_weights=(1,1), 
             learning_rate=1e-5, metrics=['accuracy'], freeze_layers=6):
        """ 
        Inputs:
        filename = path to file containing model weights
        freeze_layers = number of layers to freeze for training (int, default 6 = all layers in encoder)
        learning_rate = learning rate (float, default 1e-5)
        loss = loss function, function or string
        metrics = training metrics, list of functions or strings
        Outputs:
        model = compiled model
        """
        
        physical_devices = tf.config.list_physical_devices('GPU')
        n_gpus=len(physical_devices)
        # create path for file containing weights
        if filename is None:
            raise ValueError("Model weights filename must be provided")
        if os.path.isfile(filename):
            mfile=filename
        elif os.path.isfile(filename+'.h5'):
            mfile = filename+'.h5'
        elif os.path.isfile(filename+'.hdf5'):
            mfile = (filename+'.hdf5')
        else: ("Could not locate model weights file at {}".format(filename))

        def build_load_compile(mfile, loss, class_weights, learning_rate, metrics, freeze_layers):
            # build and load weights into encoder only model
            encoder_model = self.build_model(encoder_only=True)
            encoder_model.load_weights(mfile)

            # build full model
            model = self.build_model()

            # transfer weights for all encoder layers
            for i, layer in enumerate(encoder_model.layers):
                print('Copying weights from', layer.name, 'to', model.layers[i].name)
                weights = layer.get_weights()
                model.layers[i].set_weights(weights)
                # freeze encoder layers if specified
                if i < freeze_layers:
                    model.layers[i].trainable = False
            
            # Compile model
            if self.dual_output:
                mask_loss, skeleton_loss = self.selectLoss(loss, class_weights)
                model.compile(optimizer=Adam(learning_rate=learning_rate), 
                              loss={'mask_output':mask_loss, 'skeleton_output': skeleton_loss},
                              metrics={'mask_output': metrics[0], 'skeleton_output': metrics[1]})
            else:
                custom_loss = self.selectLoss(loss, class_weights)
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss, metrics=metrics)

            return model
        
        if n_gpus>1:
            strategy = tf.distribute.MirroredStrategy()
            print("Creating model on {} GPUs".format(n_gpus))
            with strategy.scope():
                # build and load weights into encoder only model
                model = build_load_compile(mfile, loss, class_weights, learning_rate, metrics, freeze_layers)

        else:
            # build and load weights into encoder only model
            model = build_load_compile(mfile, loss, class_weights, learning_rate, metrics, freeze_layers)

        print('Model Summary')
        model.summary()
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        