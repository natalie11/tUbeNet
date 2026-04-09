# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:18:35 2026

@author: Natal
"""
import os
import numpy as np
import tifffile as tiff
import dask.array as da

image_path = '/media/natalie/Vessel Datasets/Collabs/Gaspar_fly/predictions/Predictions_3class_model/DkepulauanaF1_abdomencrop1_prediction/labels'
output_path = '/media/natalie/Vessel Datasets/Collabs/Gaspar_fly/predictions/Predictions_3class_model'
name = "DkepulauanaF1_labels_pred"

img = da.from_zarr(image_path)
tiff_name=os.path.join(output_path,name+".tiff")

if len(img.shape)>3:
    # Save multi-channel imgage, reorder dims to ZCYX
    img=np.moveaxis(img, -1, 1)
    tiff.imwrite(tiff_name, img, metadata={"axes": "ZCYX"}, imagej=True, bigtiff=True)             
else:
    with tiff.TiffWriter(tiff_name, bigtiff=True) as tw:
        for z in range(img.shape[0]):
            img_slice = np.array(img[z, :, :])  # bring one 2D slice to RAM
            tw.write(img_slice, photometric="minisblack", metadata=None)   