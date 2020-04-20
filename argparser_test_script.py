# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:33:06 2020

@author: Natal
"""
import numpy as np
from tUbeNet_classes import DataHeader, DataDir, DataGenerator
import pickle
import os

# Create data
data=np.zeros((17,10,10))
labels=np.ones((17,10,10))

modality='none'

output_name_data='C:/Users/Natal/Documents/CABI/data/zeros3'
output_name_labels='C:/Users/Natal/Documents/CABI/labels/ones3'
output_name_header='C:/Users/Natal/Documents/CABI/header/test3'

# Save data
output_file_data = output_name_data
output_file_labels = output_name_labels
np.save(output_file_data, data)
np.save(output_file_labels, labels)

# Save header
header = DataHeader(modality=modality, image_dims=data.shape, image_filename=output_file_data, 
               label_filename=output_file_labels)
header.save(output_name_header)

# Delete data and header
del data, labels, header

# Load header
headers=[] #make list

# add 5 headers to list
header_filenames=os.listdir('C:/Users/Natal/Documents/CABI/header')
for file in header_filenames:
    file=os.path.join('C:/Users/Natal/Documents/CABI/header',file )
    with open(file, "rb") as f:
        loaded_header = pickle.load(f)
    headers.append(loaded_header)
    
# Create empty data directory    
data_dir = DataDir([], image_dims=[], 
                   image_filenames=[], 
                   label_filenames=[], data_type=[])

# Fill directory
for header in headers:
    data_dir.list_IDs.append(header.modality)
    data_dir.image_dims.append(header.image_dims)
    data_dir.image_filenames.append(header.image_filename)
    data_dir.label_filenames.append(header.label_filename)
    data_dir.data_type.append('float32')

# Create Data Generator
params = {'batch_size': 2,
          'volume_dims': (64,64,64), 
          'n_classes': 2,
	       'shuffle': False}

training_generator=DataGenerator(data_dir, **params)