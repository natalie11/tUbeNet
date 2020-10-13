# tUbeNet
tUbeNet is a 3D CNN for semantic segmenting of vasculature from 3D grayscale medical images.

## How to use

### Dependancies
This code is compatible with python 3.7, Tensorflow 2.1.0 and Keras 2.2.4

### Preparing data
To prepare data you wish to run a prediction on, use the 'tubeNet_preprocessing.py' script. 

This script will:
* rescale pixel intensity values between 0 and 1
* convert image data to a float 32 numpy array 
* convert labels (if present) to int 8 numpy array
* downsample data (and labels) by a specified factor (optional)
* crop background (optional)
* pad data (optional)
* save data (and labels) and .npy files

To use this script you will need to set the following parameters:
* downsample_factor (factor by which images are downsampled in x and y dimensions, set to 1 if not downsampling)
* pad_array (size images are padded up to, set to None is not padding)
* val_fraction (fraction of data to hold back for validation)  
* path (path to directory containing data to be processed)
* image_filename (filename of image data)
* label_filename (filename of label data, set to None if not using labels)
* output_path (where .npy files will be saved)
* output_name (filename for output to be saved under)

### Running a prediction with the pre-trained model
Once your data has been prepared, you can use the pre-trained model to predict labels using the 'tubenet_multimodal.py' script.

### Fine-tuning the pre-trained model
To fine tune the model, run 'tubenet_multimodal.py' with the options use_saved_model and fine_tuning set to True.

### Training tUbeNet from scratch
To train from scratch, run 'tubenet_multimodal.py' with the options use_saved_model and fine_tuning set to False.

