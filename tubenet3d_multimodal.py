# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
from functools import partial
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator, DataHeader
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import argparse
import pickle

#----------------------------------------------------------------------------------------------------------------------------------------------
"""Set hard-coded parameters and file paths:"""
# Create argument parser
parser = argparse.ArugmentParser(description="tUbNet training script")

parser.add_argument("--volume_dims", help="dimensions of subsampling volume for use in training, three integers seperated by spaces (default %(default)s)",
                    type=int, nargs=3, default=(64, 64, 64))
parser.add_argument("--n_epochs", help="number of training epochs (default %(default)s)",
                    type=int, default=100)
parser.add_argument("--steps_per_epoch", help="number of steps per training epochs (default %(default)s)",
                    type=int, default=1000)
parser.add_argument("--batch_size", help="batch size (default %(default)s)",
                    type=int, default=2)
parser.add_argument("--class_weights", help="weighting of background class to vessel class, two integers seperated by spaces (default %(default)s)",
                    type=int, nargs=2, default=(1, 7))
parser.add_argument("--n_classes", help="number of classes (NOTE: model currently only supports binary classicifaction, ie n_classes=2)",
                    type=int, default=2)
parser.add_argument("--binary_output", help="presence of this flag indicates a binary output is desired, as opposed to softmax",
                    action="store_true")
parser.add_argument("--fine_tuning", help="include this flag if fine-tuning a pre-trained model",
                    action="store_true")

parser.add_argument("--data_dir", help="path to data directory containing dta header files",
                    type=str, required=True)
parser.add_argument("--validation_dir", help="path to validation data directory containing data header files",
                    type=str)
parser.add_argument("--output_dir", help="path to folder in which output images will be saved",
                    type=str)
parser.add_argument("--model_input", help="if using a previously saved model, provide the file path",
                    type=str)
parser.add_argument("--model_output", help="path to folder in which trained model will be saved",
                    type=str, required=True)

args=parser.parser_args()

# Paramters
volume_dims = args.volume_dims   	 	
n_epochs = args.n_epochs		     
steps_per_epoch = args.steps_per_epoch	      
batch_size = args.batch_size		 	       	   
class_weights = args.class_weights 	        	
n_classes= args.n_classes
binary_output = args.binary_output        	
fine_tuning = args.fine_tuning

# Training data
data_path = args.data_dir

# Validation data
if args.validation_dir:
    val_path = args.validation_dir
    test_header_filenames = os.listdir(val_path)
    
# Output data
output_path = args.output_dir

# Model
model_output = args.model_output
if args.model_input is not None:
    use_saved_model= True
    model_input = args.model_input



""" Parameters and paths - for use when not calling from command line """
## Paramters
#volume_dims = (64,64,64)    	 	# size of cube to be passed to CNN (z, x, y) in form (n^2 x n^2 x n^2) 
#n_epochs = 100			         	   # number of1 epoch for training CNN
#steps_per_epoch = 1000		      # total number of steps (batches of samples) to yield from generator before declaring one epoch finished
#batch_size = 2		 	       	   # batch size for training CNN
#use_saved_model = False	        	# use saved model structure and weights? Yes=True, No=False
#save_model = True		        	   # save model structure and weights? Yes=True, No=False
#fine_tuning = False               # prepare model for fine tuning by replacing classifier and freezing shallow layers
#class_weights = (1,7) 	        	# relative weighting of background to blood vessel classes
#binary_output = False	           	# save as binary (True) or softmax (False)
#n_classes=2
#
## Training data
#path = "F:\\Paired datasets\\train"
#image_filenames = os.listdir(os.path.join(path,"data"))
#label_filenames = os.listdir(os.path.join(path,"labels"))
#
## Validation data
#val_path = "F:\\Paired datasets\\test"
#X_test_filenames = os.listdir(os.path.join(val_path,"data"))
#y_test_filenames = os.listdir(os.path.join(val_path,"labels"))
#
## Model
#model_path = 'F:\\Paired datasets'
#model_filename = 'multimodal_cropped_100epochs_1000steps'
#updated_model_filename = 'multimodal_cropped_100epochs_1000steps_Feb2'
#output_filename = 'output'

#----------------------------------------------------------------------------------------------------------------------------------------------

# create partial for  to pass to complier
custom_loss=partial(tube.weighted_crossentropy, weights=class_weights)

# callbacks              
#time_callback = tube.TimeHistory()		      
#stop_time_callback = tube.TimedStopping(seconds=18000, verbose=1)

#""" Create data Directory """ OLD CODE!!
#for i in range(len(image_filenames)):
#    image_filenames[i]=os.path.join(path,'data\\'+image_filenames[i])
#    label_filenames[i]=os.path.join(path,'labels\\'+label_filenames[i])
#
#dt=np.dtype([('list_ID', object), ('image_filename', object), ('label_filename', object), 
#             ('image_dims', int,(3,)), ('data_type', object)])
#data = np.array([('CT', image_filenames[0], label_filenames[0], (489,667,544), 'float32'),
#              ('HREM', image_filenames[1], label_filenames[1], (312,3071,2223), 'float32'),
#              ('RSOM', image_filenames[2], label_filenames[2], (376,221,191), 'float32')], dtype=dt)
#
#data_dir = DataDir(data['list_ID'], image_dims=data['image_dims'], image_filenames=data['image_filename'], 
#                   label_filenames=data['label_filename'], data_type=data['data_type'])
 
""" Create data Directory """  
# Import data header
header_filenames=os.listdir(data_path)
headers = []
for file in header_filenames: #Iterate through header files
    file=os.path.join(data_path,file)
    with open(file, "rb") as f:
        data_header = pickle.load(f) # Unpickle DataHeader object
    headers.append(data_header) # Add to list of headers
    
# Create empty data directory    
data_dir = DataDir([], image_dims=[], 
                   image_filenames=[], 
                   label_filenames=[], 
                   data_type=[])

# Fill directory from headers
for header in headers:
    data_dir.list_IDs.append(header.modality)
    data_dir.image_dims.append(header.image_dims)
    data_dir.image_filenames.append(header.image_filename)
    data_dir.label_filenames.append(header.lable_filename)
    data_dir.data_type.append('float32')

""" Create Data Generator """
params = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
	       'shuffle': False}

training_generator=DataGenerator(data_dir, **params)

""" Load or Build Model """
if use_saved_model:
    model_gpu, model = tube.load_saved_model(filename=model_input,
                         learning_rate=1e-5, n_gpus=2, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision],
                         freeze_layers=10, fine_tuning=fine_tuning, n_classes=n_classes)
else:
    model_gpu, model = tube.tUbeNet(n_classes=n_classes, input_height=volume_dims[1], input_width=volume_dims[2], input_depth=volume_dims[0], 
                                    n_gpus=2, learning_rate=1e-5, loss=custom_loss, metrics=['accuracy', tube.recall, tube.precision])

""" Train and save model """
#TRAIN
schedule = partial(tube.piecewise_schedule, lr0=1e-5, decay=0.9)
filepath = os.path.join(model_output,"multimodal_checkpoint")
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
history=model_gpu.fit_generator(generator=training_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch, 
                                callbacks=[LearningRateScheduler(schedule), checkpoint])

# SAVE MODEL
tube.save_model(model, model_output, updated_model_filename)

""" Plot ROC """
# Create directory of validation data
for i in range(len(X_test_filenames)):
    X_test_filenames[i]=os.path.join(val_path,'data\\'+X_test_filenames[i])
    y_test_filenames[i]=os.path.join(val_path,'labels\\'+y_test_filenames[i])
    
val_data = np.array([('CT', X_test_filenames[0], y_test_filenames[0], (163,667,544), 'float32'),
              ('HREM', X_test_filenames[1], y_test_filenames[1], (104,3071,2223), 'float32'),
              ('RSOM', X_test_filenames[2], y_test_filenames[2], (125,221,191), 'float32')], dtype=dt)

val_data_dir = DataDir(val_data['list_ID'], image_dims=val_data['image_dims'], image_filenames=val_data['image_filename'], 
                   label_filenames=val_data['label_filename'], data_type=val_data['data_type'])


optimised_thresholds=tube.roc_analysis(model=model_gpu, data_dir=val_data_dir, volume_dims=(64,64,64), batch_size=2, overlap=None, classes=(0,1), save_prediction=True, prediction_filename='F:\Paired datasets\prediction')

#""" Predict Segmentation """
#whole_img_pad = tube.data_preprocessing(image_filename=whole_img_filename, downsample_factor=downsample_factor, pad_array=pad_array)
#tube.predict_segmentation(model_gpu=model_gpu, image_stack=whole_img_pad,
#                         volume_dims=volume_dims, batch_size=batch_size, overlap=4, classes=(0,1), 
#                         binary_output=True, save_output= True, prediction_filename = 'prediction', path=path)

""" Fine tuning """