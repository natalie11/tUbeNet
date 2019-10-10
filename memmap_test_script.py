# -*- coding: utf-8 -*-
"""
Script for testing memmap image loading.

Created on Mon Oct  7 13:51:22 2019

@author: Natal
"""

#Import libraries
import os
import numpy as np
import random
# import Image for loading data
from PIL import Image
from skimage import io
from skimage.measure import block_reduce

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""

# Paramters
downsample_factor = 1           	# factor by which images are downsampled in x and y dimensions 
pad_array = 1024	           	   # size images are padded up to, to achieve n^2 x n^2 structure 
volume_dims = (64,128,128)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 

# Paths and filenames
path = "F:\\Paired datasets\\CT"
img_filename = os.path.join(path,"LS_C1M3_yanan_4subsample.tif")
seg_filename = os.path.join(path,"LS_C1M3_yanan_segmentation.tif")

#----------------------------------------------------------------------------------------------------------------------------------------------
def data_preprocessing(image_filename=None, label_filename=None, downsample_factor=1, pad_array=1024):
	"""# Pre-processing
    Load data, downsample if neccessary, normalise and pad.
    Inputs:
        image_filename = image filename (string)
        label_filename = labels filename (string)
        downsample_factor = factor by which to downsample in x and y dimensions (int, default 1)
        pad_array = size to pad image to, should be able to be written as 2^n where n is an integer (int, default 1024)
    Outputs:
        img_pad = image data as an np.array, scaled between 0 and 1, downsampled and padded with zeros
        seg_pad = label data as an np.array, scaled between 0 and 1, downsampled and padded with zeros
        classes = list of classes present in labels
    """
   # Load image
	print('Loading images from '+str(image_filename))
	img=io.imread(image_filename)

	# Downsampling
	if downsample_factor >1:
	  print('Downsampling by a factor of {}'.format(downsample_factor))
	  img=block_reduce(img, block_size=(1, downsample_factor, downsample_factor), func=np.mean)
	  
	# Normalise 
	print('Rescaling data between 0 and 1')
	img = (img-np.amin(img))/(np.amax(img)-np.amin(img)) # Rescale between 0 and 1
	
	# Seems that the CNN needs 2^n data dimensions (i.e. 64, 128, 256, etc.)
	# Set the images to 1024x1024 (2^10) arrays
	print('Padding array')
	xpad=(pad_array-img.shape[1])//2
	ypad=(pad_array-img.shape[2])//2
	
	img_pad = np.zeros([img.shape[0],pad_array,pad_array], dtype='float32')
	img_pad[0:img.shape[0],xpad:img.shape[1]+xpad,ypad:img.shape[2]+ypad] = img
	del img
	print('Shape of padded image array: {}'.format(img_pad.shape))
	
	#Repeat for labels is present
	if label_filename is not None:
		print('Loading labels from '+str(label_filename))
		seg=io.imread(label_filename)
	
		# Downsampling
		if downsample_factor >1:
		  print('Downsampling by a factor of {}'.format(downsample_factor))
		  seg=block_reduce(seg, block_size=(1, downsample_factor, downsample_factor), func=np.max) #max instead of mean to maintain binary image  
		  
		# Normalise 
		print('Rescaling data between 0 and 1')
		seg = (seg-np.amin(seg))/(np.amax(seg)-np.amin(seg))
		
		# Pad
		print('Padding array')
		seg_pad = np.zeros([seg.shape[0],pad_array,pad_array], dtype='float32')
		seg_pad[0:seg.shape[0],xpad:seg.shape[1]+xpad,ypad:seg.shape[2]+ypad] = seg
		
		# Find the number of unique classes in segmented training set
		classes = np.unique(seg_pad)
		del seg
		
		return img_pad, seg_pad, classes
	
	return img_pad

def save_image(array, filename):
    """Save image to file
    Inputs: 
        array = 2D np.array containing mage data to be saved
        filename = name under which to save file, including file extension
    """
    image = Image.fromarray(array)
    image.save(filename)

def load_volume_from_file(volume_dims=(64,64,64), image_dims=None, 
				  image_filename=None, label_filename=None, 
				  coords=None, data_type='float64', offset=128):
  """Load a sub-volume of the 3D image. 
  These can either be generated randomly, or be taked from defined co-ordinates (z, x, y) within the image.
  Inputs: 
    volume_dims = shape of sub-volume, given as (z,x,y) (tuple of int, default (64,64,64))
    image_dims = shape of image to be accessed, (z,x,y) (tuple of int)
    image_filename = filename for np array of pre-processed image data (z,x,y)
    label_filename = filename for np array of labels, (z,x,y,c) where c is the number of chanels. Array should be int8. Optional.
    coords = coordinates for top left corner of sub-volume (tuple of int, if not specified will be randomly generated)
    data_type = data type of the image array (string, default 'float634')
    offset = number of bytes in file before pixel data begins (int, default 128)
  Outputs:
    volume = sub-section of the image array with dimensions equal to volume_dims
    labels_volume = corrosponding array of image labels (if labels provided)
  """

  #Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
  
  # Check for sensible volume dimensions
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_dims[i]:
      raise Exception('Volume dimensions out of range')
	
  if coords is not None:
    # check for sensible coordinates
    for i in range(3):
      if coords[i]<0 or coords[i]>(image_dims[i]-volume_dims[i]):
        raise Exception('Coordinates out of range')
  else:
    # generate random coordinates for upper left corner of volume
    coords = np.zeros(3)
    coords[0] = random.randint(0,(image_dims[0]-volume_dims[0]))
    coords[1] = random.randint(0,(image_dims[1]-volume_dims[1])) 
    coords[2] = random.randint(0,(image_dims[2]-volume_dims[2]))
  
  # Set number of bytes per pixel, depending on data_type  
  if data_type == 'float64' or data_type == 'int64':
	  pixel = 8
  elif data_type == 'float32' or data_type == 'int32':
	  pixel = 4
  elif data_type == 'int16':
	  pixel = 2
  elif data_type == 'int8' or data_type == 'bool':
	  pixel = 1
  else: raise Exception('Data type not supported')
	
  # Calculate y axis and z axis offset (number of bytes to skip to get to the next row)
  x_offset = image_dims[1]*pixel
  z_offset = image_dims[1]*image_dims[2]*pixel
  
  # Load data from file, one row at a time, using memmap
  volume=np.zeros(volume_dims)
  for z in range(volume_dims[0]):
    for x in range(volume_dims[1]):
        volume[z,x,:]=np.memmap(image_filename, dtype=data_type,mode='c',shape=(1,1,volume_dims[2]),
			 offset=(offset + pixel*(coords[2]) + x_offset*(x+coords[1]) + z_offset*(z+coords[0])))
  
  # If labels_filename given, generate labels_volume using same coordinates
  if label_filename is not None:
      labels_volume = np.zeros(volume_dims)
      for z in range(volume_dims[0]):
          for x in range(volume_dims[1]):
              labels_volume[z,x,:]=np.memmap(label_filename, dtype='int8',mode='c',shape=(1,1,volume_dims[2]),
                         offset=(offset + coords[2] + image_dims[1]*(x+coords[1]) + image_dims[1]*image_dims[2]*(z+coords[0])))
      return volume, labels_volume
  else:
      return volume

  


def load_volume(volume_dims=(64,64,64), image_stack=None, labels=None, coords=None):
  """Load a sub-volume of the 3D image. 
    These can either be generated randomly, or be taked from defined co-ordinates (z, x, y) within the image.
    Inputs: 
    	volume_dims = shape of sub-volume, given as (z,x,y), tuple of int
    	image_stack = 3D image, preprocessed and given as np array (z,x,y)
    	labels = np array of labels, (z,x,y,c) where c is the number of chanels. Should be binary and one hot encoded. Optional.
    	coords = coordinates for top left corner of sub-volume (if not specified, will be randomly generated)
    Outputs:
    	volume = sub-section of the image array with dimensions equal to volume_dims
    	labels_volume = corrosponding array of image labels (if labels provided)
    """
  image_dims = image_stack.shape #image_dims[0] = z, [1] = x, [2] = y
  
  #Format volume_dims as (z,x,y)
  if type(volume_dims) is int: # if only one dimension is given, assume volume is a cube
    volume_dims = (volume_dims, volume_dims, volume_dims)
  elif len(volume_dims)==2: # if two dimensions given, assume first dimension is depth
    volume_dims = (volume_dims[0], volume_dims[1], volume_dims[1])
  # Check for sensible volume dimensions
  for i in range(3):
    if volume_dims[i]<=0 or volume_dims[i]>image_dims[i]:
      raise Exception('Volume dimensions out of range')
	
  if coords is not None:
    # check for sensible coordinates
    for i in range(3):
      if coords[i]<0 or coords[i]>(image_dims[i]-volume_dims[i]):
        raise Exception('Coordinates out of range')
  else:
    # generate random coordinates for upper left corner of volume
    coords = np.zeros(3)
    coords[0] = random.randint(0,(image_dims[0]-volume_dims[0]))
    coords[1] = random.randint(0,(image_dims[1]-volume_dims[1])) 
    coords[2] = random.randint(0,(image_dims[2]-volume_dims[2]))
  
  # Create volume from coordinates
  volume = image_stack[int(coords[0]):int(coords[0] + volume_dims[0]), int(coords[1]):int(coords[1] + volume_dims[1]), int(coords[2]):int(coords[2] + volume_dims[2])]
  
  if labels is not None:
    # Create corrsponding labels
    labels_volume = labels[int(coords[0]):int(coords[0] + volume_dims[0]), int(coords[1]):int(coords[1] + volume_dims[1]), int(coords[2]):int(coords[2] + volume_dims[2])]
    return volume, labels_volume
  else:
    return volume

#----------------------------------------------------------------------------------------------------------------------------------------------

""" Load and Preprocess data """
img_pad, seg_pad, classes = data_preprocessing(image_filename=img_filename, label_filename=seg_filename, 
                                                    downsample_factor=downsample_factor, pad_array=pad_array)

"""
TESTING MEMMAP LOADING
"""
img_pad=img_pad.astype('float32')
seg_pad=seg_pad.astype('int8')
image_array=os.path.join(path,"image_array.npy")
label_array=os.path.join(path,"label_array.npy")
np.save(image_array,img_pad)
np.save(label_array,seg_pad)
img_subvol, labels_subvol = load_volume(volume_dims=volume_dims,image_stack=img_pad, labels=seg_pad, coords=(300,400,500))
for i in range(64):
    save_image(img_subvol[i,:,:], os.path.join(path,'load_volume_output_'+str(i)+'.tif'))
    save_image(labels_subvol[i,:,:], os.path.join(path,'load_volume_labels_output'+str(i)+'.tif'))

img_subvol_mem, labels_subvol_mem = load_volume_from_file(volume_dims=volume_dims, image_dims = (682,1024,1024),
                     image_filename=image_array, label_filename=label_array,
                     offset=128,coords=(300,400,500), data_type='float32')
for i in range(64):
    save_image(img_subvol_mem[i,:,:], os.path.join(path,'load_memmap_output_'+str(i)+'.tif'))
    save_image(labels_subvol_mem[i,:,:], os.path.join(path,'load_memmap_labels_output'+str(i)+'.tif'))