# tUbeNet
tUbeNet is a 3D CNN for semantic segmenting of vasculature from 3D grayscale medical images. tUbeNet was designed to be trained on varied data from different modalities, scales and pathologies, creating a generalisable base model that can be fine-tuned to specific tasks with minimal additional data (https://doi.org/10.1101/2023.07.24.550334).

If you are interested in using tUbeNet for your own data, please get in touch and we will happily share the base model weights - from there you just need to manually annotate a small portion of your data for fine-tuning. The original multi-modal training dataset is also available, along with manual labels, upon request. Contact via email: natalie.holroyd.16@ucl.ac.uk

![github_fig](https://github.com/natalie11/tUbeNet/assets/30265332/49dde486-2e54-41e1-98cc-f83f6f910688)

## How to use

### Dependancies
This code is compatible with python 3.8 and Tensorflow 2.3.0

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
* create a header file

To use this script you will need to set the following parameters:
* downsample_factor - factor by which images are downsampled in x and y dimensions, set to 1 if not downsampling
* pad_array - size images are padded up to in each dimension (z, x, y), set to None if not padding
* val_fraction - fraction of data to hold back for validation
* crop - set to 'True' to crop background containing no labelled vessels, otherwise ‘False’
* path - path to directory containing data to be processed
* image_filename - filename of image data
* label_filename - filename of label data, set to None if not using labels
* output_path - where .npy files will be saved
* output_name - an identifying name the processed data will be saved under

### Using the tubenet_multimodal script 
Once your data has been prepared, you can use the pre-trained model to predict labels using the 'tubenet_multimodal.py' script, fine-tune the model using your own data, or train a new model from scratch.
The parameters to set within script are as follows:

*Model parameters*
* volume_dims - size of subvolume to be passed to CNN (z, x, y) 
* n_epochs - number of epochs for training CNN
* steps_per_epoch - total number of steps (batches of samples) to yield from generator before declaring one epoch finished
* batch_size - batch size 
* class_weights - relative weighting of background to blood vessel classes, to account for imbalanced classes (int, int)
* n_classes - number of classes (optimised for 2 classes: background and vessel)
* dataset_weighting – biases the frequency with which the batch generator will pull for each dataset: weight larger datasets more highly to avoid overfitting to small datasets

*Run settings*
* use_saved_model - use saved model structure and weights (True/False)
* fine_tuning - prepare model for fine tuning by replacing classifier and freezing shallow layers (True/False)
* binary_output - save as binary image (True) or softmax output (False)
* save_model - save model structure and weights after training (True/False)
* prediction_only - set to True if running a prediction only, False if training

*Paths*
* path - path to directory containing header files for processed (this folder will be generated automatically by the preprocessing script)
* val_path - path to directory containing header files for processed validation data (optional, set to None if not using validation data)

* model_path - path to directory containing model (if using saved model), and where trained model will be saved
* model_filename - filename of existing model (not including extension). If not using saved model, set to None
* updated_model_filename - filename under which the trained model will be saved

* output_path - path were output images will be saved

#### Predicting a segmentation using a pre-trained model
Ensure the 'prediction_only' and 'use_saved_model' parameters are set to True. Set 'model_path' and 'model_filename' corresponding to the pre-trained model use wish to use. Set 'path' to point to the directory containing the header files for your pre-processed image data (produced by tubenet_preprocessing.py). 

Run the script and the segmented images will be saved as tifs in 'output_path'.

#### Fine-tuning the pre-trained model
To fine tune the model, run 'tubenet_multimodal.py' with the options 'use_saved_model' and 'fine_tuning' set to True.  Set 'model_path' and 'model_filename' corrosonding to the pre-trained model use wish to fine-tune. Set 'path' to point to the directory containing the header files for your pre-processed image data & labels (produced by tubenet_preprocessing.py). Make sure 'prediction_only' is set to False.

When running the script, you should be able to monitor the training progress (accuracy, precision, recall and DICE score) using tensorboard. If you have provided validation data, the script will run a prediction on this validation data and produce a ROC curve.

#### Training tUbeNet from scratch
To train from scratch, run 'tubenet_multimodal.py' with the options 'use_saved_model' and 'fine_tuning' set to False. Set 'path' to point to the directory containing the header files for your pre-processed image data & labels (produced by tubenet_preprocessing.py). Make sure 'prediction_only' is set to False. 

When running the script, you should be able to monitor the training progress (accuracy, precision, recall and DICE score) using tensorboard. If you have provided validation data, the script will run a prediction on this validation data and produce a ROC curve.
