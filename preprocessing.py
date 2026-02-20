 # -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data (and optional labels) and convert into zarr format with data header


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import tUbeNet_functions as tube
import argparse

def main(args):
    #----------------------------------------------------------------------------------------------------------------------------------------------
    """Set hard-coded parameters and file paths:"""
    
    # Paramters
    chunks = tuple(args.chunks)       # chunk size for saving zarr files equal to chunk size used by model
    val_fraction = args.val_fraction  # fraction of data to use for validation
    crop = args.crop                  # crop images if there are large sections of background containing no vessels
    skeleton_sigma = args.skeleton_sigma  # distance field decay parameter
    generate_skeleton = args.generate_skeleton  # whether to auto-generate skeleton if not provided

    image_directory = args.image_directory
    label_directory = args.label_directory
    skeleton_directory = args.skeleton_directory
    output_path = args.output_path
        
    #----------------------------------------------------------------------------------------------------------------------------------------------
    # Create list of image files
    image_directory, image_filenames = tube.list_image_files(image_directory) 
    if image_directory==None: raise ValueError('Image directory could not be found')
    
    # Create list of label files if provided, else create list of 'None'
    if label_directory is not None:
        label_directory, label_filenames = tube.list_image_files(label_directory)
        if label_directory==None: 
            raise ValueError('A label directory was provided but could not be found. Set label_directory to None if not using labels.')
        assert len(image_filenames)==len(label_filenames), "Expected same number of image and label files. Set label_directory to None if not using labels."
    else:
        label_filenames = [None]*len(image_filenames)
    
    # Create list of skeleton files if provided, else create list of 'None'
    if skeleton_directory is not None:
        skeleton_directory, skeleton_filenames = tube.list_image_files(skeleton_directory)
        if skeleton_directory==None:
            raise ValueError('A skeleton directory was provided but could not be found. Set skeleton_directory to None if not using skeletons.')
        assert len(image_filenames)==len(skeleton_filenames), "Expected same number of image and skeleton files."
    else:
        skeleton_filenames = [None]*len(image_filenames)
         
    # Process and save each dataset in directory
    print("Image files:")
    print(*image_filenames, sep="\n")
    if label_directory is not None:
        print("Label files:")
        print(*label_filenames, sep="\n")
    if skeleton_directory is not None:
        print("Skeleton files:")
        print(*skeleton_filenames, sep="\n")   
    
    for image_filename, label_filename, skeleton_filename in zip(image_filenames, label_filenames, skeleton_filenames):
        # Set names and paths
        output_name = os.path.splitext(image_filename)[0] #Take name from image_filneame, without extension
        image_path = os.path.join(image_directory, image_filename)
        # Set label and skeleton paths if provided, else set to None
        if label_filename is not None: 
            label_path = os.path.join(label_directory, label_filename)
        else: label_path = None
        if skeleton_filename is not None:
            skeleton_path = os.path.join(skeleton_directory, skeleton_filename)
        else:
            skeleton_path = None
            
        # Run preprocessing - conversiton to dask array, normalisation, remapping of label values if required 
        data, labels, skeleton = tube.data_preprocessing(image_path=image_path, 
                                                        label_path=label_path,
                                                        skeleton_path=skeleton_path,
                                                        chunks=chunks)

        if labels is not None:

            # Crop is enabled
            if crop: 
                data, labels, skeleton = tube.crop_from_labels(data, labels, skeleton=skeleton)
      
            # Generate skeleton if one is not provided and generate_skeleton true
            if skeleton is None and generate_skeleton:
                print(f"Generating skeleton from binary mask for {output_name}...")
                skeleton = tube.generate_skeleton(labels, chunks=chunks)

            if skeleton is not None:
                # Create distance field from skeleton
                # this helps the model learn by providing a smoother, larger target 
                print(f"Creating distance field from skeleton for {output_name} with sigma={skeleton_sigma}...")
                skeleton = tube.compute_distance_field(skeleton, labels, sigma=skeleton_sigma, chunks=chunks)

        # Split into test and train is val_fraction is provided, then save
        if val_fraction > 0 and labels is not None:
            
            training, testing = tube.split_train_test(data, labels, val_fraction, skeleton=skeleton)

            # Create folders
            train_folder = os.path.join(output_path,"train")
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            train_name = str(output_name)+"_train"
            
            test_folder = os.path.join(output_path,"test")
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            test_name = str(output_name)+"_test"


            train_path, train_header = tube.save_as_zarr_array(training[0], 
                                                                labels=training[1], 
                                                                skeleton=training[2],
                                                                output_path=train_folder,
                                                                output_name=train_name,
                                                                chunks=chunks)
            print("Processed training data and header files saved to "+str(train_path))

            test_path, test_header = tube.save_as_zarr_array(testing[0], 
                                                                labels=testing[1], 
                                                                skeleton=testing[2],
                                                                output_path=test_folder,
                                                                output_name=test_name,
                                                                chunks=chunks)
            print("Processed test data and header files saved to "+str(test_path))

        else:
            # No splitting, just save full dataset
            save_path, save_header = tube.save_as_zarr_array(data, 
                                                             labels=labels, 
                                                             skeleton=skeleton,
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
    parser.add_argument("--skeleton_directory", type=str, default=None,
                        help="Path to skeleton file or directory (optional). If not provided and --generate_skeleton is set, skeletons will be generated from labels.")
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
    parser.add_argument("--generate_skeleton", action='store_true',
                        help="Enable automatic skeleton generation from binary mask using skimage skeletonization")
    parser.add_argument("--skeleton_sigma", type=float, default=2.0,
                        help="Decay parameter (sigma) for exponential distance field: exp(-distance/sigma). "
                             "Smaller values = sharper decay. Default: 2.0")
                               

    args = parser.parse_args() 
    args.chunks = parse_chunks(args.chunks) #create tuple of values for chunk dimensions
    
    main(args)