#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:22:26 2025

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
    """Set parameters and file paths:"""
    # Paramters
    volume_dims = args.volume_dims
    overlap = args.overlap
    n_classes = 2 #TO DO expand to handel multi-class case
    
    binary_output = args.binary_output
    preview = args.preview
    attention = args.attention

    data_headers = args.data_headers
    model_path = args.model_path
    output_path = args.output_path
    tiff_path = args.tiff_path
    
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
        data_dir.label_filenames.append(None) #Labels not required for prediction
        data_dir.data_type.append('float32')
        data_dir.exclude_region.append((None,None,None)) #region to be left out of training for use as validation data (under development)
        
    
    """ Load Model """
    # Initialise model
    tubenet = tUbeNet(n_classes=n_classes, input_dims=volume_dims, attention=attention)
    # Load weights
    model = tubenet.load_weights(filename=model_path, loss='DICE BCE')
    
    # If undefined set overlap to half volume_dims
    if not overlap:
        overlap = (volume_dims[0]//2,volume_dims[1]//2,volume_dims[2]//2)
    
    """Predict segmentation"""
    for i in data_dir.image_filenames:
        # Isolate image filename
        image_directory, image_filename = os.path.split(i.replace('\\','/'))
        print("Begining Inference on {image_filename}")
        
        # Create output filenames
        dask_name = os.path.join(output_path,str(image_filename)+"_segmentation")
        if tiff_path: tiff_name=os.path.join(tiff_path,str(image_filename)+"_segmentation.tiff")
        else: tiff_name = None
        tube.predict_segmentation_dask(
            model,
            i,                 
            dask_name,                  
            volume_dims=volume_dims,   
            overlap=overlap,       
            n_classes=n_classes,
            export_bigtiff=tiff_name,
            preview=preview,          
            binary_output=binary_output, 
            prob_channel=1,   
        )

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
    parser = argparse.ArgumentParser(description="Run prediction using TubeNet model.")

    parser.add_argument("--data_headers", type=str, required=True,
                        help="Path to directory containing preprocessed header files.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.h5 file).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where predictions will be saved.")
    parser.add_argument("--tiff_path", type=str, default=None,
                        help="Optional path to save TIFF output in addition to Zarr.")

    parser.add_argument("--volume_dims", type=int, nargs="+", default=[64, 64, 64],
                        help="Volume dimensions passed to CNN. Provide 1 value (isotropic) "
                             "or 3 values (anisotropic). E.g. --volume_dims 64 OR --volume_dims 32 64 64")
    parser.add_argument("--overlap", type=int, nargs="+", default=None,
                        help="Overlap between patches during inference. Provide 1 value (isotropic) "
                             "or 3 values (anisotropic). E.g. --overlap 32 OR --volume_dims 16 32 32. "
                             "Defaults to half of volume_dims.")
    parser.add_argument("--binary_output", action="store_true",
                        help="Save predictions as binary image. Otherwise, the softmax output will be saved.")
    parser.add_argument("--preview", action="store_true",
                        help="Display preview of predicted segmentation during inference.")
    parser.add_argument("--attention", action="store_true",
                        help="Use this flag if loading a tubenet model built with attention blocks") 

    args = parser.parse_args()
    args.volume_dims = parse_dims(args.volume_dims)
    if args.overlap: args.overlap = parse_dims(args.overlap) #Parse if not None
    main(args)