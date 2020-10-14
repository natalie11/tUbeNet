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
* downsample_factor - factor by which images are downsampled in x and y dimensions, set to 1 if not downsampling
* pad_array - size images are padded up to, set to None is not padding
* val_fraction - fraction of data to hold back for validation
* path - path to directory containing data to be processed
* image_filename - filename of image data
* label_filename - filename of label data, set to None if not using labels
* output_path - where .npy files will be saved
* output_name - filename for output to be saved under

### Running a prediction with the pre-trained model
Once your data has been prepared, you can use the pre-trained model to predict labels using the 'tubenet_multimodal.py' script.
Paramters to set within script:

* volume_dims  - size of subvolume to be passed to CNN (z, x, y) 
* n_epochs - number of epochs for training CNN
* steps_per_epoch - total number of steps (batches of samples) to yield from generator before declaring one epoch finished
* batch_size - batch size 
* use_saved_model - use saved model structure and weights (True/False)
* save_model - save model structure and weights (True/False)
* fine_tuning - prepare model for fine tuning by replacing classifier and freezing shallow layers (True/False)
* class_weights - relative weighting of background to blood vessel classes (int, int)
* binary_output - save as binary image (True) or softmax output (False)
* n_classes - number of classes (optimised for 2 classes)
* dataset_weighting - weight larger datasets more highly

* path - path to directory containing processed training data in two sub-folders: 'data' and 'labels'
* val_path - path to directory containing processed validation data in two sub-folders: 'data' and 'labels'

* model_path - path to directory containing model (if using saved model), and where trained model will be saved
* model_filename - filename of exisiting model (not including extension), if not useing saved model, set to None
* updated_model_filename - filename under which the trained model will be saved

### Fine-tuning the pre-trained model
To fine tune the model, run 'tubenet_multimodal.py' with the options use_saved_model and fine_tuning set to True.

### Training tUbeNet from scratch
To train from scratch, run 'tubenet_multimodal.py' with the options use_saved_model and fine_tuning set to False.

