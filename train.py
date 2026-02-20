#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:21:26 2025

@author: natalie
"""

#Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Suppress info logs from tf 
import pickle
import datetime
import argparse
from model import tUbeNet
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir, DataGenerator, ImageDisplayCallback, MetricDisplayCallback, FilterDisplayCallback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tUbeNet_metrics import MacroDice, create_dual_task_loss
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
    binary_output = args.binary_output
    augment = args.no_augment
    attention = args.attention
    train_skeleton = args.train_skeleton
    skeleton_loss = args.skeleton_loss
    
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
    header_filenames=[f for f in os.listdir(data_headers) if os.path.isfile(os.path.join(data_headers, f))]
    headers = []
    try:
        for file in header_filenames: #Iterate through header files
            file=os.path.join(data_headers,file)
            with open(file, "rb") as f:
                data_header = pickle.load(f) # Unpickle DataHeader object
            headers.append(data_header) # Add to list of headers
    except FileNotFoundError: print("Unable to load data header files from {data_headers}") 
    
    # Create empty data directory    
    data_dir = DataDir([], image_dims=[], 
                       image_filenames=[], 
                       label_filenames=[],
                       skeleton_filenames=[],
                       data_type=[], exclude_region=[])
    
    # Fill directory from headers
    for header in headers:
        data_dir.list_IDs.append(header.ID)
        data_dir.image_dims.append(header.image_dims)
        data_dir.image_filenames.append(header.image_filename)
        data_dir.label_filenames.append(header.label_filename)
        data_dir.skeleton_filenames.append(getattr(header, 'skeleton_filename', None))  # Get skeleton if available
        data_dir.data_type.append('float32')
        data_dir.exclude_region.append((None,None,None)) #region to be left out of training for use as validation data (under development)

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
    
    if model_weights_file is not None:
        # Load exisiting model with or without fine tuning adjustment (fine tuning -> classifier replaced and first 2 blocks frozen)
        if not os.path.isfile(model_weights_file):
            if os.path.isfile(os.path.join(model_path, model_weights_file)):
                model_weights_file=os.path.join(model_path, model_weights_file)
            else:
                raise FileNotFoundError("Could not locate model weights file at {}".format(model_weights_file))
        
        if skeleton_available and train_skeleton:
            print("ERROR: Loading weights for dual-task model not yet implemented. Please train from scratch with --train_skeleton.")
            raise NotImplementedError("load_weights for dual-task models needs implementation")
        else:
            model = tubenet.load_weights(filename=os.path.join(model_path,model_weights_file), 
                                         loss=loss, 
                                         class_weights=class_weights, 
                                         learning_rate=lr0, 
                                         metrics=['accuracy', 'recall', 'precision', MacroDice(n_classes)],
                                         freeze_layers=6, fine_tune=fine_tune)
    
    else:
        if skeleton_available and train_skeleton:
            # Create dual-task model with learnable weighted loss
            print("Creating dual-task model for vessel segmentation + skeleton prediction...")
            model = tubenet.build_model(encoder_only=False)
            
            # Create learnable weighted loss
            dual_loss = create_dual_task_loss(
                mask_loss=skeleton_loss,
                class_weights=class_weights,
                init_log_weight_mask=-0.69,  # w1 ≈ 0.5
                init_log_weight_skel=-1.2,   # w2 ≈ 0.3
                init_log_weight_reg=-1.6     # w3 ≈ 0.2
            )
            
            # Compile model with dual loss
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr0),
                loss=dual_loss,
                metrics=['accuracy']
            )
        else:
            model = tubenet.create(learning_rate=lr0, 
                                   loss=loss, 
                                   class_weights=class_weights, 
                                   metrics=['accuracy', 'recall', 'precision', MacroDice(n_classes)])

    
    
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
    checkpoint = ModelCheckpoint(filepath, monitor=monitored_metric, verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True)
    imageCallback = ImageDisplayCallback(data_generator,log_dir=os.path.join(log_dir,'images')) 
    filterCallback = FilterDisplayCallback(log_dir=os.path.join(log_dir,'filters')) #experimental
    metricCallback = MetricDisplayCallback(log_dir=log_dir)
        
    # Create directory of validation data
    if val_headers is not None:
        # Import data header
        header_filenames=[f for f in os.listdir(val_headers) if os.path.isfile(os.path.join(val_headers, f))]
        headers = []
        try:
            for file in header_filenames: #Iterate through header files
                file=os.path.join(val_headers,file)
                with open(file, "rb") as f:
                    val_header = pickle.load(f) # Unpickle DataHeader object
                headers.append(val_header) # Add to list of headers
        except FileNotFoundError: print("Unable to load data header files from {val_headers}") 
            
        # Create empty data directory    
        val_dir = DataDir([], image_dims=[], 
                           image_filenames=[], 
                           label_filenames=[], 
                           data_type=[], exclude_region=[])
        
        # Fill directory from headers
        for header in headers:
            val_dir.list_IDs.append(header.ID)
            val_dir.image_dims.append(header.image_dims)
            val_dir.image_filenames.append(header.image_filename)
            val_dir.label_filenames.append(header.label_filename)
            val_dir.data_type.append('float32')
            val_dir.exclude_region.append((None,None,None))
       
    
        vparams = {'batch_size': batch_size,
          'volume_dims': volume_dims, 
          'n_classes': n_classes,
          'dataset_weighting': None,
          'augment': False,
    	       'shuffle': False}
        
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
    parser.add_argument("--binary_output", action="store_true",
                        help="Save predictions as binary image instead of softmax.")
    
    # Skeleton training options
    parser.add_argument("--train_skeleton", action="store_true",
                        help="Enable dual-task training with skeleton prediction. Requires skeleton data in headers.")
    parser.add_argument("--skeleton_loss", type=str, default="dice_bce",
                        choices=["dice_bce", "dice_ce", "wcce"],
                        help="Loss function for vessel mask in dual-task training.")

    args = parser.parse_args()
    args.volume_dims = parse_dims(args.volume_dims)
    main(args)
