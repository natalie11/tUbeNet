#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:56:27 2025

@author: natalie
"""

#Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Suppress info logs from tf 
import pickle
from model import tUbeNet
import tUbeNet_functions as tube
from tUbeNet_classes import DataDir
import argparse

def main(args):
    """Set arameters and file paths:"""
    # Paramters
    volume_dims = args.volume_dims
    overlap = args.overlap
    n_classes = args.n_classes #TO DO expand to handel multi-class case
    
    prob_output = args.prob_output
    attention = args.attention

    data_headers = args.data_headers
    model_path = args.model_path
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
    except IndexError: print("Unable to load data header files from {data_headers}") 
    
    # Create empty data directory    
    data_dir = DataDir([], image_dims=[], 
                       image_filenames=[], 
                       label_filenames=[], 
                       data_type=[], exclude_region=[])
    
    # Fill directory from headers
    for header in headers:
        data_dir.list_IDs.append(header.ID)
        data_dir.image_dims.append(header.image_dims)
        data_dir.image_filenames.append(header.image_filename)
        data_dir.label_filenames.append(header.label_filename)
        data_dir.data_type.append('float32')
        data_dir.exclude_region.append((None,None,None)) #region to be left out of training for use as validation data (under development)
    
    
    """ Load Model """
    tubenet = tUbeNet(n_classes=n_classes, input_dims=volume_dims, attention=attention)
    
    # Load exisiting model 
    model = tubenet.load_weights(filename=model_path, loss='DICE BCE')
    
    """ Plot ROC """
    # Evaluate model on data
    validation_metrics = tube.roc_analysis(model, data_dir, 
                                          volume_dims=volume_dims,
                                          n_classes=n_classes, 
                                          overlap=overlap,
                                          output_path=output_path,
                                          prob_output=prob_output) 

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
    parser = argparse.ArgumentParser(description="Evaluate TubeNet model on paired data.")

    parser.add_argument("--data_headers", type=str, required=True,
                        help="Path to directory containing preprocessed header files.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.h5 file).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where predictions will be saved.")

    parser.add_argument("--volume_dims", type=int, nargs="+", default=[64, 64, 64],
                        help="Volume dimensions passed to CNN. Provide 1 value (isotropic) "
                             "or 3 values (anisotropic). E.g. --volume_dims 64 OR --volume_dims 32 64 64")
    parser.add_argument("--overlap", type=int, nargs="+", default=None,
                        help="Overlap between patches during inference. Provide 1 value (isotropic) "
                             "or 3 values (anisotropic). E.g. --overlap 32 OR --volume_dims 16 32 32. "
                             "Defaults to half of volume_dims.")
    parser.add_argument("--prob_output", action="store_true",
                        help="Save predictions as softmax probabilities.")
    parser.add_argument("--attention", action="store_true",
                        help="Use this flag if loading a tubenet model built with attention blocks") 
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes to predict. Ensure this is the same for all data included in testing.")

    args = parser.parse_args()
    args.volume_dims = parse_dims(args.volume_dims)
    if args.overlap: args.overlap = parse_dims(args.overlap) #Parse if not None
    main(args)