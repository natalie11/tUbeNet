import numpy as np
import argparse
import os
import pickle
from tUbeNet.tUbeNet_classes import DataHeader

"""
Helper script to change the location of data referenced in header files 
(e.g. if data is movbed to a new location)
"""

def relocate_data_in_header(header_file,new_image_dir=None,new_label_dir=None,check_exists=True,remove_if_doesnt_exist=False):

    with open(header_file,'rb') as f:
        data = pickle.load(f)

    if new_image_dir is not None:
        new_path = os.path.join(new_image_dir,os.path.basename(data.image_filename))
        if check_exists and not os.path.exists(new_path+'.npy'):
            print(f'{new_path} does not exist!')
            if remove_if_doesnt_exist:
                print(f'Removing {new_path}')
                os.remove(header_file)
                return
        else:
            #print(f'CHANGED (IMAGE): {data.image_filename} to {new_path}')
            data.image_filename = os.path.join(new_path)
    if new_label_dir is not None:
        new_path = os.path.join(new_label_dir,os.path.basename(data.label_filename))
        if check_exists and not os.path.exists(new_path+'.npy'):
            print(f'{new_path} does not exist!')
            if remove_if_doesnt_exist:
                print(f'Removing {new_path}')
                os.remove(header_file)
                return
        else:
            #print(f'CHANGED (LABEL): {data.label_filename} to {new_path}')
            data.label_filename = os.path.join(new_path)
    data.save(header_file)
        
def iterative_relocate_data_in_header(header_dir,new_image_dir=None,new_label_dir=None):

    files = os.listdir(header_dir)
    
    for hf in files:
        header_file = os.path.join(header_dir,hf)
        relocate_data_in_header(header_file,new_image_dir=new_image_dir,new_label_dir=new_label_dir)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Relocate data in header files")
    
    parser.add_argument('--header_dir', default='', type=str)
    parser.add_argument('--new_image_dir', default='',type=str)
    parser.add_argument('--new_label_dir', default='',type=str)
    args = vars(parser.parse_args())
    
    iterative_relocate_data_in_header(args['header_dir'],args['new_image_dir'],args['new_label_dir'])
