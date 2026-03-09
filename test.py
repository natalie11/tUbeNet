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
    predict_skeleton = args.predict_skeleton

    data_headers = args.data_headers
    model_path = args.model_path
    output_path = args.output_path
    
    #----------------------------------------------------------------------------------------------------------------------------------------------
    """ Create Data Directory"""
    # Load data headers into a list
    header_filenames=[os.path.join(data_headers, f) for f in os.listdir(data_headers) if os.path.isfile(os.path.join(data_headers, f))]
    
    # Create data directory from headers 
    data_dir = DataDir.from_header(header_filenames, data_type='float32')
    
    """ Load Model """
    tubenet = tUbeNet(n_classes=n_classes, input_dims=volume_dims, attention=attention, dual_output=predict_skeleton)
    
    # Load exisiting model weights 
    model = tubenet.load_weights(filename=model_path, loss='DICE BCE')
    
    """ Plot ROC """
    # Evaluate model on data
    validation_metrics = tube.roc_analysis(model, data_dir, 
                                          volume_dims=volume_dims,
                                          n_classes=n_classes, 
                                          overlap=overlap,
                                          output_path=output_path,
                                          prob_output=prob_output,
                                          predict_skeleton=predict_skeleton) 

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
    parser.add_argument("--predict_skeleton", action="store_true",
                        help="Predict skeleton in addition to segmentation.")

    args = parser.parse_args()
    args.volume_dims = parse_dims(args.volume_dims)
    if args.overlap: args.overlap = parse_dims(args.overlap) #Parse if not None
    main(args)