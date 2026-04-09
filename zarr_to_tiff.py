# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:18:35 2026

@author: Natal
"""
import os
import numpy as np
import tifffile as tiff
import dask.array as da

image_path = "c:/Users/Natal/Documents/CCM/code_testing/skeleton/preprocessed/test/2Photon_murineOlfactoryBulbLectin_subvolume500x500x79-val"
output_path = 'C:/Users/Natal/Documents/CCM/code_testing/skeleton/prediction'
name = "example_test"

img = da.from_zarr(image_path)

if len(img.shape)>3:
    # Save each channel independantly
    for c in range(img.shape[-1]):
        tiff_name=os.path.join(output_path,name+"_channel_"+str(c)+".tiff")
        with tiff.TiffWriter(tiff_name, bigtiff=True) as tw:
            for z in range(img.shape[0]):
                img_slice = np.array(img[z, :, :, c])  # bring one 2D slice to RAM
                tw.write(img_slice, photometric="minisblack", metadata=None)               
else:
    tiff_name=os.path.join(output_path,name+".tiff")
    with tiff.TiffWriter(tiff_name, bigtiff=True) as tw:
        for z in range(img.shape[0]):
            img_slice = np.array(img[z, :, :])  # bring one 2D slice to RAM
            tw.write(img_slice, photometric="minisblack", metadata=None)   