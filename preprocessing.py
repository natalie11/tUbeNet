 # -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data (and optional labels) and convert into zarr format with data header


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube
import argparse

def list_image_files(directory):
    if os.path.isdir(directory):
        # Add all file paths of image_paths
        image_filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    elif os.path.isfile(directory):
        # If file is given, process this file only
        image_directory, image_filenames = os.path.split(directory.replace('\\','/'))
        image_filenames = [image_filenames]
    else: return None, None
    
    return image_directory, image_filenames

def main(args):
    #----------------------------------------------------------------------------------------------------------------------------------------------
    """Set hard-coded parameters and file paths:"""
    
    # Paramters
    chunks = tuple(args.chunks)       # chunk size for saving zarr files equal to chunk size used by model
    val_fraction = args.val_fraction  # fraction of data to use for validation
    crop = args.crop                  # crop images if there are large sections of background containing no vessels

    image_directory = args.image_directory
    label_directory = args.label_directory
    output_path = args.output_path
        
    #----------------------------------------------------------------------------------------------------------------------------------------------
    # Create list of image files
    image_directory, image_filenames = list_image_files(image_directory) 
    if image_directory==None: raise ValueError('Image directory could not be found')
    
    # Create list of label files
    if label_directory is not None:
        label_directory, label_filenames = list_image_files(label_directory)
        if label_directory==None: 
            raise ValueError('A label directory was provided but could not be found. Set label_directory to None if not using labels.')
        assert len(image_filenames)==len(label_filenames), "Expected same number of image and label files. Set label_directory to None if not using labels."
    else:
        label_filenames = [None]*len(image_filenames)
         
    # Process and save each dataset in directory
    for image_filename, label_filename in zip(image_filenames, label_filenames):
        # Set names and paths
        output_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(image_directory, image_filename)
        if label_filename is not None: 
            label_path = os.path.join(label_directory, label_filename)
        else: label_path = None
            
        # Run preprocessing
        data, labels, classes = tube.data_preprocessing(image_path=image_path, 
                                                        label_path=label_path)
        
        # Set data type
        data = data.astype('float32')
        if labels is not None:
            labels = labels.astype('int16')
        
        # Crop
        if crop and labels is not None:
            labels, data = tube.crop_from_labels(labels, data)
        
        # Split into test and train
        if val_fraction > 0 and labels is not None:
            
            n_training_imgs = int(data.shape[0]-np.floor(data.shape[0]*val_fraction))
            
            train_data = data[0:n_training_imgs,...]
            train_labels = labels[0:n_training_imgs,...]
            
            test_data = data[n_training_imgs:,...]
            test_labels = labels[n_training_imgs:,...]
            
            # Create folders
            train_folder = os.path.join(output_path,"train")
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            train_name = str(output_name)+"_train"
            
            test_folder = os.path.join(output_path,"test")
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            test_name = str(output_name)+"_test"
            
            # Save train data
            train_name = str(output_name)+"_train"
            
            train_path, train_header = tube.save_as_zarr_array(train_data, labels=train_labels, 
                                                               output_path=train_folder, 
                                                               output_name=train_name, 
                                                               chunks=chunks)
            print("Processed training data and header files saved to "+str(train_path))
            
            # Save test data
            test_path, test_header = tube.save_as_zarr_array(test_data, labels=test_labels, 
                                                               output_path=test_folder, 
                                                               output_name=test_name, 
                                                               chunks=chunks)
            print("Processed test data and header files saved to "+str(test_path))
            
        else:
            save_path, save_header = tube.save_as_zarr_array(data, labels=labels, 
                                                               output_path=output_path, 
                                                               output_name=output_name, 
                                                               chunks=chunks)
            print("Processed data and header files saved to "+str(save_path))

def parse_chunks(values):
    if len(values) == 1:
        return (values[0], values[0], values[0])
    elif len(values) == 3:
        return tuple(values)
    else:
        raise argparse.ArgumentTypeError(
            "Chunks must be either a single value (e.g. --chunks 64) "
            "or three values (e.g. --chunks 64 64 32).")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image and label datasets for TubeNet.")
    parser.add_argument("--image_directory", type=str, required=True,
                        help="Path to image file or directory")
    parser.add_argument("--label_directory", type=str, default=None,
                        help="Path to label file or directory. Set to None if not using labels.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where processed data will be saved")
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 64, 64],
                        help="Chunk size for saving zarr files. "
                             "Provide 1 value (isotropic) or 3 values (anisotropic). "
                             "E.g. --chunks 64 OR --chunks 64 64 32")
    parser.add_argument("--val_fraction", type=float, default=0.0,
                        help="Fraction of data to use for validation (0-1)")
    parser.add_argument("--crop", action='store_true',
                        help="Enable cropping if there are large background sections with no vessels")
                               

    args = parser.parse_args() 
    args.chunks = parse_chunks(args.chunks) #create tuple of values for chunk dimensions
    
    main(args)