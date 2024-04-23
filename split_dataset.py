import numpy as np
import argparse
import os
import pickle
from tUbeNet.tUbeNet_classes import DataHeader
import shutil

"""
Helper script to change the location of data referenced in header files 
(e.g. if data is movbed to a new location)
"""
        
def split_data(header_dir,new_dir,fraction=0.5):

    files = os.listdir(header_dir)
    
    n = int(len(files) * fraction)

    # Use random.sample to select items
    selected_files = np.random.choice(files, size=n, replace=False)
    
    new_header_dir = os.path.join(new_dir,'headers')
    os.makedirs(new_header_dir, exist_ok=True)
    
    breakpoint()
    
    for hf in selected_files:
        header_file = os.path.join(header_dir,hf)
        with open(header_file,'rb') as f:
            data = pickle.load(f)

        new_im_file = os.path.join(new_dir,os.path.basename(data.image_filename))
        shutil.move(data.image_filename+'.npy', new_im_file+'.npy')
        new_label_file = os.path.join(new_dir,os.path.basename(data.label_filename))
        shutil.move(data.label_filename+'.npy', new_label_file+'.npy')
        
        data.image_filename = new_im_file
        data.label_filename = new_label_file
        
        os.remove(os.path.join(header_dir,hf))
        new_header_file = os.path.join(new_header_dir,os.path.basename(hf))
        data.save(new_header_file)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Split data")
    
    parser.add_argument('--dir', default='', type=str)
    parser.add_argument('--new_dir', default='',type=str)
    parser.add_argument('--fraction', default=0.5,type=float)
    args = vars(parser.parse_args())
    
    split_data(args['dir'],args['new_dir'],args['fraction'])
