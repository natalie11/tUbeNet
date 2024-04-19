 # -*- coding: utf-8 -*-
"""tUbeNet 3D 
Data Preprocessing script: load image data (and optional labels) and convert into numpy arrays with data header


Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
import tUbeNet_functions as tube
from tUbeNet_classes import DataHeader
import argparse

def preprocess_image( 
    downsample_factor = 1, 
    pad_array = (None, None, None),
    val_fraction = 0, 
    crop = False,
    image_path = '',
    label_path = '',
    image_filename = '',
    label_filename = '',
    output_path = '',
    output_name = '',
    iterative = None
    ):

    #----------------------------------------------------------------------------------------------------------------------------------------------
    # Define path for data & labels if necessary
    print(image_path,image_filename)
    image_filename = os.path.join(image_path, image_filename)
    if label_filename!='':
        if label_path=='':
            label_path = image_path
        label_filename = os.path.join(label_path, label_filename)

    if label_filename!='':
        data, labels, classes = tube.data_preprocessing(image_filename=image_filename, label_filename=label_filename,
                                               downsample_factor=downsample_factor, pad_array=pad_array)
    else:
        data = tube.data_preprocessing(image_filename=image_filename, downsample_factor=downsample_factor, pad_array=pad_array)
        
    # Set data type
    data = data.astype('float32')
    if label_filename!='':
        labels = labels.astype('int8')

    # Crop
    if crop:
        iz, ix, iy = np.where(labels[...]!=0) # find instances of non-zero values in X_test along axis 1
        labels = labels[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1] # use this to index y_test and y_pred
        data = data[min(iz):max(iz)+1, min(ix):max(ix)+1, min(iy):max(iy)+1]
        print(data.shape)


    # Split into test and train
    if val_fraction > 0:
        n_training_imgs = int(data.shape[0]-np.floor(data.shape[0]*val_fraction))
        
        train_data = data[0:n_training_imgs,...]
        train_labels = labels[0:n_training_imgs,...]
        
        test_data = data[n_training_imgs:,...]
        test_labels = labels[n_training_imgs:,...]
        
        # Save as numpy arrays
        # Create folders
        train_folder = os.path.join(output_path,"train")
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        test_folder = os.path.join(output_path,"test")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        
        # Save train data
        train_name=os.path.join(train_folder,str(output_name)+"_train")
        header_folder=os.path.join(train_folder, "headers")
        if not os.path.exists(header_folder):
            os.makedirs(header_folder)
        header_name=os.path.join(header_folder,str(output_name)+"_train_header")
        
        np.save(train_name,train_data)
        np.save(str(train_name)+"_labels",train_labels)
        header = DataHeader(ID=output_name, image_dims=train_labels.shape, image_filename=train_name, 
                            label_filename=str(train_name)+"_labels")
        header.save(header_name)
        print("Processed training data and header files saved to "+str(train_folder))
        
        # Save test data
        test_name=os.path.join(test_folder,str(output_name)+"_test")
        header_folder=os.path.join(test_folder, "headers")
        if not os.path.exists(header_folder):
            os.makedirs(header_folder)
        header_name=os.path.join(header_folder,str(output_name)+"_test_header")
        
        np.save(test_name,test_data)
        np.save(str(test_name)+"_labels",test_labels)
        header = DataHeader(ID=output_name, image_dims=test_labels.shape, image_filename=test_name, 
                            label_filename=str(test_name)+"_labels")
        header.save(header_name)
        print("Processed test data and header files saved to "+str(test_folder))
        
    else:
        header_folder=os.path.join(output_path, "headers")
        if not os.path.exists(header_folder):
            os.makedirs(header_folder)
        header_name=os.path.join(header_folder,str(output_name)+"_header")
        
        # Save data as numpy array
        np.save(os.path.join(output_path, output_name), data)
        
        if label_filename is not None: 
            # Save labels as numpy array
            np.save(os.path.join(output_path, str(output_name)+"_labels"), labels)
            header = DataHeader(ID=output_name, image_dims=labels.shape, image_filename=os.path.join(output_path, output_name),
                                label_filename=os.path.join(output_path, str(output_name)+"_labels"))
            header.save(header_name)
        else:
            # Save header with label_filename=None
            header = DataHeader(ID=output_name, image_dims=data.shape, image_filename=os.path.join(output_path, output_name),
                                label_filename=None)
            header.save(str(header_name)+'_header')
        
        print("Processed data and header files saved to "+str(output_path))
 
def iterative_preprocessing(args):
 
    im_dir = args['image_path'] # os.path.join(args['path'],args['image_filename'])
    image_files = os.listdir(im_dir)
    
    label_dir = args['label_path'] #os.path.join(args['path'],args['label_filename'])
    label_files = os.listdir(label_dir)
    
    for f in image_files:
        image_file_spl,ext = os.path.splitext(f)
        if ext in ['.tif','.tiff','.jpeg','.jpg','.nii','.bmp']:
            # Remove ID (assumed to be an identifier after the final '_' in the filename (e.g. _image, _label, _vessel, _seg, etc.)
            image_file_spl_noid = image_file_spl.split('_')
            image_file_spl_noid = '_'.join(image_file_spl_noid[:-1])
            # Look for that identifier in the label folder
            mtch = -1
            try:
                mtch = ['_'.join(x.split('_')[:-1]) for x in label_files].index(image_file_spl_noid)
            except Exception as e:
                print(e)
                
            if mtch>=0:
                import copy
                argsCur = copy.copy(args)

                argsCur['image_filename'] = f
                argsCur['label_filename'] = label_files[mtch]
                argsCur['output_name'] = image_file_spl_noid
                
                preprocess_image(**argsCur)

def main():
 
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    
    parser.add_argument('--downsample_factor', default=1, type=int)
    parser.add_argument('--pad_array', default=(None,None,None),type=tuple)
    parser.add_argument('--val_fraction', default=0,type=float)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--image_path', default='',type=str)
    parser.add_argument('--label_path', default='',type=str)
    parser.add_argument('--image_filename', default='',type=str)
    parser.add_argument('--label_filename', default='',type=str)
    parser.add_argument('--output_path', default='',type=str)
    parser.add_argument('--output_name', default='',type=str)
    parser.add_argument('--iterative', action='store_false')
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.iterative==True:
        iterative_preprocessing(vars(args))
    else:
        preprocess_image(**vars(args))
 
if __name__=='__main__':

    """
    python tUbeNet_preprocessing.py --image_path /mnt/data1/VANGAN/vamp/tiffs/composite --label_path /mnt/data1/VANGAN/vamp/tiffs/vessel --output_path /mnt/data2/vamp/tubenet

    """

    main()
       
