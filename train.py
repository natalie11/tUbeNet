#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:21:26 2025

@author: natalie
"""

#Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Suppress info logs from tf 
import datetime
import argparse
from model import tUbeNet
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator, ImageDisplayCallback, MetricDisplayCallback, FilterDisplayCallback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tUbeNet_metrics import MacroDice
import tensorflow as tf

def main(args):
    """Set parameters and file paths:"""
    # Model paramters
    volume_dims = args.volume_dims    	         
    n_epochs = args.n_epochs			      
    steps_per_epoch = args.steps_per_epoch
    batch_size = args.batch_size
    dataset_weighting = args.dataset_weighting
    loss = args.loss
    lr0 = args.lr0
    class_weights = args.class_weights
    n_classes = args.n_classes #TO DO expand to handel multi-class case
    
    # Training and prediction options
    fine_tune = args.fine_tune  
    augment = args.no_augment
    attention = args.attention
    train_skeleton = args.train_skeleton
    
    """ Paths and filenames """
    # Training data
    data_headers = args.data_headers
    
    # Validation data
    val_headers = args.val_headers
    
    # Model
    model_path = args.model_path
    model_weights_file =  args.model_weights_file 

    # Image output
    output_path = args.output_path
    
    #----------------------------------------------------------------------------------------------------------------------------------------------
    """ Create Data Directory"""
    # Load data headers into a list
    header_filenames=[os.path.join(data_headers, f) for f in os.listdir(data_headers) if os.path.isfile(os.path.join(data_headers, f))]
    
    # Create data directory from headers 
    data_dir = DataDir.from_header(header_filenames, data_type='float32')
    
    """ Create Data Generator """
    # Check if skeletons are available
    skeleton_available = any(skel is not None for skel in data_dir.skeleton_filenames)
    if train_skeleton and not skeleton_available:
        print("WARNING: --train_skeleton requested but no skeleton data found in headers. Training without skeleton branch.")
        train_skeleton = False
    elif not train_skeleton and skeleton_available:
        print("NOTE: Skeleton data is available but --train_skeleton not requested. Skeletons will not be used for training.")
    
    params = {'batch_size': batch_size,
              'volume_dims': volume_dims, 
              'n_classes': n_classes,
              'dataset_weighting': dataset_weighting,
              'augment': augment,
              'skeleton_available': skeleton_available and train_skeleton,
    	       'shuffle': False}
    
    data_generator = DataGenerator(data_dir, **params)
    
    """ Load or Build Model """
    tubenet = tUbeNet(n_classes=n_classes, input_dims=volume_dims, attention=attention, 
                      dual_output=(skeleton_available and train_skeleton))
    
    if skeleton_available and train_skeleton:
        metrics_list = [['accuracy', 'recall', 'precision', MacroDice(n_classes, ignore_background=True)],['root_mean_squared_error', 'mean_absolute_error']]
    else:
        metrics_list = ['accuracy', 'recall', 'precision', MacroDice(n_classes, ignore_background=True)]

    if model_weights_file is not None:
        # Load exisiting model with or without fine tuning adjustment (fine tuning -> classifier replaced and first 2 blocks frozen)
        if not os.path.isfile(model_weights_file):
            if os.path.isfile(os.path.join(model_path, model_weights_file)):
                model_weights_file=os.path.join(model_path, model_weights_file)
            else:
                raise FileNotFoundError("Could not locate model weights file at {}".format(model_weights_file))
        
        model = tubenet.load_weights(filename=model_weights_file, 
                                     loss=loss, 
                                     class_weights=class_weights, 
                                     learning_rate=lr0, 
                                     metrics=metrics_list,
                                     freeze_layers=6, fine_tune=fine_tune)
    
    else:
        model = tubenet.create(learning_rate=lr0, 
                               loss=loss, 
                               class_weights=class_weights, 
                               metrics=metrics_list)

    
    """ Train and save model """
    
    # Create folder for log files
    date = datetime.datetime.now()
    filepath = os.path.join(model_path,"{}_model_checkpoint.weights.h5".format(date.strftime("%d%m%y")))
    log_dir = os.path.join(model_path,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Define callbacks
    if val_headers is not None:
        monitored_metric='val_loss'
    else:
        monitored_metric='loss'
    checkpoint = ModelCheckpoint(filepath, monitor=monitored_metric, verbose=1, save_weights_only=True, save_best_only=True, mode='auto')
    tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True)
    imageCallback = ImageDisplayCallback(data_generator,log_dir=os.path.join(log_dir,'images')) 
    filterCallback = FilterDisplayCallback(log_dir=os.path.join(log_dir,'filters')) #experimental
    metricCallback = MetricDisplayCallback(log_dir=log_dir)
        
    # Create directory of validation data
    if val_headers is not None:
        # Import data header
        header_filenames=[os.path.join(val_headers, f) for f in os.listdir(val_headers) if os.path.isfile(os.path.join(val_headers, f))]

        # Create empty data directory    
        val_dir = DataDir.from_header(header_filenames, data_type='float32')
            
        vparams = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': None,
          'augment': False,
    	  'shuffle': False,
          'skeleton_available': skeleton_available and train_skeleton}
        
        val_generator=DataGenerator(val_dir, **vparams)
        
        # TRAIN with validation
        history=model.fit(data_generator, validation_data=val_generator,
                          validation_steps=5, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                          callbacks=[checkpoint, tbCallback, imageCallback, filterCallback, metricCallback])
    
    else:
        # TRAIN without validation
        history=model.fit(data_generator, epochs=n_epochs, 
                          steps_per_epoch=steps_per_epoch,
                          callbacks=[checkpoint, tbCallback, imageCallback, filterCallback, metricCallback])
       
    # SAVE MODEL
    model.save_weights(os.path.join(model_path,"{}_trained_model.weights.h5".format(date.strftime("%d%m%y"))))
    
    """ Plot ROC """
    # Evaluate model on validation data
    if val_headers is not None:
        validation_metrics = tube.roc_analysis(model, val_dir, 
                                          volume_dims=volume_dims,
                                          n_classes=n_classes, 
                                          output_path=output_path) 

def parse_dims(values):
    """Parse volume dimensions: allow either one int (isotropic) or three ints (anisotropic)."""
    if len(values) == 1:
        return (values[0], values[0], values[0])
    elif len(values) == 3:
        return tuple(values)
    else:
        raise argparse.ArgumentTypeError(
            "volume_dims must be either a single value (e.g. --volume_dims 64) "
            "or three values (e.g. --volume_dims 64 64 32).")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TubeNet model.")

    # Data paths
    parser.add_argument("--data_headers", type=str, required=True,
                        help="Path to directory containing training header files.")
    parser.add_argument("--val_headers", type=str, default=None,
                        help="Path to directory containing validation header files (optional).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory where trained model will be saved.")
    parser.add_argument("--model_weights_file", type=str, default=None,
                        help="Filename for pre-trained model weights (ending .h5 or .weights.h5). "
                        "If unset, the model will be trained from scratch.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where predictions/analysis outputs will be saved.")

    # Model parameters
    parser.add_argument("--volume_dims", type=int, nargs="+", default=[64, 64, 64],
                        help="Volume dimensions passed to CNN. Provide 1 value (isotropic) or 3 values (anisotropic).")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Number of batches generated per epoch.")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size.")
    parser.add_argument("--dataset_weighting", type=float, nargs="+", default=None,
                        help="Relative weighting when pulling training data from multiple datasets.")
    parser.add_argument("--loss", type=str, default="DICE BCE",
                        choices=["DICE BCE", "focal", "WCCE", "DICE CE"],
                        help="Loss function.")
    parser.add_argument("--lr0", type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument("--class_weights", type=float, nargs='+', default=None,
                        help="Relative class weights given as a list (e.g. background, vessels -> (0, 1)).")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes to predict. Ensure this is the same for all data included in training.")
    parser.add_argument("--no_augment", action="store_false",
                        help="Disable data augmentation.")
    parser.add_argument("--attention", action="store_true",
                        help="Enable attention mechanism in model (experimental).")

    # Training options
    parser.add_argument("--fine_tune", action="store_true",
                        help="Enable fine-tuning by freezing shallow layers.")

    # Skeleton training options
    parser.add_argument("--train_skeleton", action="store_true",
                        help="Enable dual-task training with skeleton prediction. Requires skeleton data in headers.")

    args = parser.parse_args()
    args.volume_dims = parse_dims(args.volume_dims)
    main(args)
