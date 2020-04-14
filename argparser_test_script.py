# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:33:06 2020

@author: Natal
"""
import os
import numpy as np
from tUbeNet_classes import DataHeader

data=np.zeros((10,10,10))
labels=np.ones((10,10,10))

modality='none'

output_name_data='C:/Users/Natal/Documents/CABI/data/zeros'
output_name_labels='C:/Users/Natal/Documents/CABI/labels/ones'
output_name_header='C:/Users/Natal/Documents/CABI/header/test'

output_file_data = output_name_data
output_file_labels = output_name_labels
np.save(output_file_data, data)
np.save(output_file_labels, labels)

header = DataHeader(modality=modality, image_dims=data.shape, image_filenames=output_file_data, 
               label_filenames=output_file_labels)
header.save(output_name_header)