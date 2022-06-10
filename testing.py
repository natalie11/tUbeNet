# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:50:19 2022

@author: Natal
"""
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




x=np.random.rand(1,8,8,8,1)
print(x.shape)

conv1 = Conv3D(32, (3,3,3), padding='same')(x)
print(conv1.shape)

pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
print(pool1.shape)

conv2 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(pool1)
print(conv2.shape)

