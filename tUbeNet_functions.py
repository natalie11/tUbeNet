# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
from skimage import io
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import dask.array as da
import zarr
from tqdm import tqdm
from scipy.signal.windows import general_hamming
import tifffile as tiff

# import tensor flow
import tensorflow as tf

# set backend and dim ordering
K=tf.keras.backend
K.set_image_data_format('channels_last')

# set memory limit on gpu
physical_devices = tf.config.list_physical_devices('GPU')
try:
  for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
except:
  pass
tf.config.optimizer.set_experimental_options({
    "remapping": False
})

#---------------------------INFERENCE------------------------------------------------------------------------------------------------------------------------
				
def predict_segmentation_dask(
    model,
    image_path,                 # e.g. "/path/dataset.zarr/image"
    out_store,                  # e.g. "/path/dataset_pred.zarr" (folder will be created)
    volume_dims=(64, 64, 64),   # (Z,X,Y)
    overlap=(16, 16, 16),       # (Z,X,Y) overlap (must be < volume_dims)
    n_classes=2,                # softmax classes produced by model
    export_bigtiff=None,        # e.g. "/path/dataset_pred.tif" to export 3D TIFF (optional)
    preview=False,              # Preview segmentation for every slab of subvolumes processed in the z axis
):
    """
    Sliding-window inference with smooth blending.
    Writes two Zarr datasets on disk during accumulation: 'sum' and 'wsum'.
    Final result is written as 'labels' and 'softmax'.
    Optionally, writes a BigTIFF 3D volume without holding everything in RAM.
    """

    # Open image using Dask array and check dimensions
    img = da.from_zarr(image_path) # shape (Z,X,Y) or (Z,X,Y,1)
    if img.ndim == 4 and img.shape[-1] == 1:
        img = img[..., 0]
    assert img.ndim == 3, "Expected (Z,X,Y) image"
    
    # Make all dimensions int
    Z, X, Y = map(int, img.shape)
    stride = np.array(volume_dims, dtype="int32")-np.array(overlap, dtype="int32")
    
    # Check for sensible overlap dimensions
    if any(stride<0):
        raise ValueError("overlap must be less than volume_dims on each axis")
        
    def auto_pad(img, volume_dims, stride):
        img_shape = np.array(img.shape)
        volume_dims = np.array(volume_dims)
        
        pad_widths = []
        new_shape = []
        
        for shape_i, dim_i, stride_i in zip(img_shape, volume_dims, stride):
            # Number of strides needed to cover full image volume
            target_size = int(np.ceil((shape_i-dim_i)/stride_i)*stride_i+dim_i)

            total_pad = target_size-shape_i
            
            # Pad must be at least half volume_dims to avoid boundary artefact
            half = dim_i//2
            before = max(half, total_pad//2) # pad on either side of image
            after = max(total_pad-before, half)
            
            pad_widths.append((before, after)) # (Before, After) in each dimension
            new_shape.append(shape_i + before + after)
        
        padded = da.pad(img, pad_widths, mode='reflect')
        #print(f"Padded from {img.shape} to {tuple(new_shape)}") #Debugging
        return padded, pad_widths
       
    # Pad image to avoid boundary effects and allow patches to cover whole image
    img, pad_widths = auto_pad(img, volume_dims, stride)

    # Prepare output Zarr stores
    # Accumulates weighted sum of softmax outputs, and summed weights from hann filter (for normalising)
    root = zarr.open(out_store, mode="w")
    sum_arr = root.create_dataset("sum", shape=(*img.shape, n_classes), chunks=(*volume_dims, n_classes),
                                  dtype="float32")
    wsum_arr = root.create_dataset("wsum", shape=(*img.shape, 1), chunks=(*volume_dims, 1),
                                   dtype="float32")

    # Compute Hann window for blending
    wz, wx, wy = general_hamming(volume_dims[0],0.75), general_hamming(volume_dims[1],0.75), general_hamming(volume_dims[2],0.75)
    w_patch = wz[:, None, None] * wx[None, :, None] * wy[None, None, :]
    w_patch /= (w_patch.max() + 1e-8) # Normalise to max 1
    w_patch = w_patch.astype(np.float32)[...,None] # (Z,X,Y,1) 

    # Compute sliding window coordinates
    windows = da.lib.stride_tricks.sliding_window_view(img, volume_dims)[::stride[0], 
                                    ::stride[1], ::stride[2]]
    #print("windows shape:", windows.shape) #debugging

    # Total patches for progress bar
    total_patches = windows.shape[0]*windows.shape[1]*windows.shape[2]

    # Preview
    def plot_preview(original, pred, z, out_store):
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original, cmap="gray")
        axs[0].set_title(f"Input z={z}")
        axs[0].axis("off")

        axs[1].imshow(pred, cmap="viridis")
        axs[1].set_title(f"Prediction z={z}")
        axs[1].axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(out_store,'preview_z'+str(z)+'.png'))
        

    # Inference step - iterate through windows and blend with weighted sum
    step_i = 0
    with tqdm(total=total_patches, desc="Inference", unit="patch") as pbar:
        for zi in range(windows.shape[0]):
            # Define position within image (z-axis)
            z0 = zi * stride[0]
            z1 = z0 + volume_dims[0]
            
            for xi in range(windows.shape[1]):
                # Define position within image (x-axis)
                x0 = xi * stride[1]
                x1 = x0 + volume_dims[1]
                
                for yi in range(windows.shape[2]):
                    # Define position within image (y-axis)
                    y0 = yi * stride[2]
                    y1 = y0 + volume_dims[2]
                    
                    # Read patch (compute only this slice)
                    patch = windows[zi, xi, yi].compute().astype(np.float32, copy=False)
                    patch = patch[None,...,None] # Reshape to (1,Z,X,Y,C)

                    # Predict softmax probability (batch of 1)
                    pred = model.predict(patch, verbose=0)
                    pred = pred[0] # pred shape: (1,Z,X,Y,C) -> (Z,X,Y,C)

                    # Add weighted prediciton and weighs to accumlators in correct positions                    
                    sum_arr[z0:z1, x0:x1, y0:y1, :] += pred * w_patch
                    wsum_arr[z0:z1, x0:x1, y0:y1, :] += w_patch
                    
                    # Update progress bar
                    step_i += 1
                    pbar.update(1)
       
            if preview and z0>0 and z1<img.shape[0]: # Produce preview if flag present AND this is not the first/last slab (avoids printing padded region) 
                # Normalize current accum/weights to preview
                z_mid_slice = z0 + (volume_dims[0] // 2)
                # Read single slice
                preview_sum = sum_arr[z_mid_slice, :, :, :].astype(np.float32)
                preview_w = wsum_arr[z_mid_slice, :, :, :].astype(np.float32)
                
                # Normalise to accumulated weights, avoid division by zero. preview_pred shape is (X Y C).
                preview_pred = np.where(preview_w > 0, preview_sum/ np.maximum(preview_w, 1e-8), 0.0)
                
                # If n_classes>2 reverse OHE. otherwise print softmax output for foreground class
                if n_classes >2:
                    preview_pred = np.argmax(preview_pred, axis=-1)
                else:
                    preview_pred = preview_pred[...,1]
                    
                # Remove padding. Preview_pred shape now (X, Y)
                preview_pred = preview_pred[pad_widths[1][0]:img.shape[1]-pad_widths[1][1],
                                  pad_widths[2][0]:img.shape[2]-pad_widths[2][1]]
                
                # Also read the corresponding input slice
                orig_slice = img[z_mid_slice, :, :].compute()
                orig_slice = orig_slice[pad_widths[1][0]:img.shape[1]-pad_widths[1][1],
                                  pad_widths[2][0]:img.shape[2]-pad_widths[2][1]] # Remove padding
                plot_preview(orig_slice, preview_pred, z_mid_slice, out_store)
                
                
    # Crop outputs
    sum_arr = sum_arr[pad_widths[0][0]:img.shape[0]-pad_widths[0][1],
                      pad_widths[1][0]:img.shape[1]-pad_widths[1][1],
                      pad_widths[2][0]:img.shape[2]-pad_widths[2][1]]
    wsum_arr = wsum_arr[pad_widths[0][0]:img.shape[0]-pad_widths[0][1],
                      pad_widths[1][0]:img.shape[1]-pad_widths[1][1],
                      pad_widths[2][0]:img.shape[2]-pad_widths[2][1]]

    # Normalize and write final zarr
    # Create output store
    labels = root.create_dataset("labels", shape=(Z, X, Y), chunks=volume_dims, dtype="uint8")
    softmax = root.create_dataset("softmax", shape=(Z, X, Y, n_classes), chunks=(*volume_dims, n_classes), dtype="float32")
    
    # Process chunk-by-chunk to avoid OOM
    for zi in tqdm(range(windows.shape[0]), desc="Normalising and saving"):
        z0 = zi * stride[0]
        z1 = z0 + volume_dims[0]
        
        # load a slab of weighted sums and weights
        slab_sum = sum_arr[z0:z1, :, :, :]        # (vz,X,Y,C)
        slab_w   = wsum_arr[z0:z1, :, :, 0:1]     # (vz,X,Y,1)
        slab_sum = np.array(slab_sum)             # bring to RAM slab only
        slab_w   = np.array(slab_w)
        probs = np.where(slab_w > 0, slab_sum / np.maximum(slab_w, 1e-8), 0.0)  # (vz,X,Y,C)
        
        softmax[z0:z1, :, :, :]  =  probs      
        labels[z0:z1, :, :] = np.argmax(probs, axis=-1).astype(np.uint8) 

    # Delete sum_arr and wsum_arr now that we're finished with them
    del root["sum"]
    del root["wsum"]

    # Optional: export BigTIFF 3D, slice-by-slice to handle very large images
    if export_bigtiff:
        with tiff.TiffWriter(export_bigtiff, bigtiff=True) as tw:
            for z in tqdm(range(Z), desc="Export BigTIFF", unit="slice"):
                seg_slice = np.array(labels[z, :, :])  # bring one 2D slice to RAM
                tw.write(seg_slice, photometric="minisblack", metadata=None)

    return softmax, os.path.join(out_store, "segmentation")

#-----------------------PREPROCESSING FUNCTIONS--------------------------------------------------------------------------------------------------------------
def data_preprocessing(image_path=None, label_path=None, chunks='auto'):
    """# Pre-processing
    Load data, downsample if neccessary, normalise and pad.
    Inputs:
    image_path = path to image data (string)
    label_path = path to labels (string)
    Outputs:
    img_pad = image data as an np.array, scaled between 0 and 1
    seg_pad = label data as an np.array, scaled between 0 and 1
    classes = list of classes present in labels
    """
    
    # Load image
    print('Loading images from '+str(image_path))
    img=io.imread(image_path)
    img=da.from_array(img,chunks=chunks)
    print('Size '+str(img.shape))

    if len(img.shape)==4:
        print('Image data has dimensions. Cropping to first 3 dimensions')
        img=img[:,:,:,0]
    assert img.ndim == 3, "Expected (Z,X,Y) image"
    
    # Normalise 
    print('Rescaling data between 0 and 1')
    img_min = da.min(img)
    denominator = da.nanmax(img)-img_min 
    img = (img-img_min)/denominator # Rescale between 0 and 1
    
    # Set data type
    img = img.astype('float32')
    
	#Repeat for labels is present
    if label_path is not None:
        print('Loading labels from '+str(label_path))
        seg = io.imread(label_path)
        seg = da.from_array(seg,chunks=chunks)

        # Find the number of unique classes in segmented training set
        classes = da.unique(seg).compute()
        
        assert len(classes)<51, "Over 50 unqiue classes identified - check labels file is correct."
        print('Labels processed with classes '+str(classes))
        
        # # TO DO - add check for non-integer classes
        # # Map classes to integer label values starting at 0
        # mapping = {old: new for new, old in enumerate(classes)}
        # seg_mapped = da.zeros_like(seg, dtype=np.int32)
        # for old, new in mapping.items():
        #     seg_mapped[seg==old] = new
        
        seg = seg.astype('int16')            
           		
        return img, seg

    return img, None

def list_image_files(directory):
    """ Lists files in a directory. 
    If given a path to a single file - splits the directory path from the filename.
    Returns directory path and list of filename."""
    if os.path.isdir(directory):
        # Add all file paths of image_paths
        image_filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    elif os.path.isfile(directory):
        # If file is given, process this file only
        image_directory, image_filenames = os.path.split(directory.replace('\\','/'))
        image_filenames = [image_filenames]
    else:
        return None, None
    
    return image_directory, image_filenames

def crop_from_labels(labels, data):
    """Crops 3D dask array based on labels.
    Returns a 3D dask array.""" 
    iz, ix, iy = da.nonzero(labels) # find instances of non-zero values in X_test along axis 1
    labels = labels[da.min(iz):da.max(iz)+1, da.min(ix):da.max(ix)+1, da.min(iy):da.max(iy)+1] # use this to index data and labels
    data = data[da.min(iz):da.max(iz)+1, da.min(ix):da.max(ix)+1, da.min(iy):da.max(iy)+1]
    print("Cropped to {}".format(data.shape))
    
    return labels, data

def split_train_test(labels, data, val_fraction):
    """"Splits paired images and labels into training and validation sets given a validation fraction.
    Takes data and labels as dask arrays."""
    n_training_imgs = int(data.shape[0]-da.floor(data.shape[0]*val_fraction))
            
    train_data = data[0:n_training_imgs,...]
    train_labels = labels[0:n_training_imgs,...]
            
    test_data = data[n_training_imgs:,...]
    test_labels = labels[n_training_imgs:,...]
    
    return train_data, train_labels, test_data, test_labels
            
def save_as_zarr_array(data, labels=None, output_path=None, output_name=None, chunks=(64,64,64)):
    """"Data (and optionally labels) are saved in chunked zarr format.
    A data header is created to record image shape, ID and path for data/labels."""
    # Create header folder if does not exist
    header_folder=os.path.join(output_path, "headers")
    if not os.path.exists(header_folder):
        os.makedirs(header_folder)
    header_name=os.path.join(header_folder,str(output_name)+"_header")
    
    # Rechunk dask array and save as zarr
    data = data.rechunk(chunks)
    data.to_zarr(os.path.join(output_path, output_name))
    
    from tUbeNet_classes import DataHeader
    
    # Repeat of labels if present
    if labels is not None: 
        # Rechunk dask array and save as zarr
        labels = labels.rechunk(chunks)
        labels.to_zarr(os.path.join(output_path, str(output_name)+"_labels"))
        
        # Save data header for easy reading in
        header = DataHeader(ID=output_name, image_dims=labels.shape, 
                            image_filename=os.path.join(output_path, output_name),
                            label_filename=os.path.join(output_path, str(output_name)+"_labels"))
        header.save(header_name)
    else:
        # Save data header for easy reading in, with label_filename=None
        header = DataHeader(ID=output_name, image_dims=data.shape, 
                            image_filename=os.path.join(output_path, output_name),
                            label_filename=None)
        header.save(header_name)
        
    return output_path, header_name

#---------------------------EVALUATION----------------------------------------------------------------------

def roc_analysis(model, data_dir, volume_dims=(64,64,64), 
                 overlap=None, n_classes=2, 
                 output_path=None): 
    """"Plots ROC Curve and Precision-Recall Curve for paired 
    ground truth labels and non-thresholded predictions (e.g. softmax output).
    Calculates DICE score, Area under ROC, Average Precision and optimal threshold.
    Optionally saves tiff image of predicted labels."""
    optimal_thresholds = []
    recall = []
    precision = []
    average_precision = []
    
    if not overlap:
        overlap = (volume_dims[0]//2,volume_dims[1]//2,volume_dims[2]//2)

    for index in range(0,len(data_dir.list_IDs)):
        print('Evaluating model on '+str(data_dir.list_IDs[index])+' data')

        # Build output name from image filename and output path     
        dask_name = os.path.join(output_path, str(data_dir.list_IDs[index])+"_prediction")
        tiff_name = str(dask_name)+".tif"

        # Predict segmentation
        y_pred, zarr_path = predict_segmentation_dask(
            model,
            data_dir.image_filenames[index],                 
            dask_name,              
            volume_dims=volume_dims,   
            overlap=overlap,       
            n_classes=n_classes,
            preview=False  
        )
        
        y_pred = da.array(y_pred)
                
        # Create 1D numpy array of predicted output (softmax)
        y_pred1D = da.ravel(y_pred).astype(np.float32)
        
        # Create 1D numpy array of true labels
        y_test = da.from_zarr(data_dir.label_filenames[index])
        y_test1D = da.ravel(y_test).astype(np.float32)
        
        if n_classes==2:
            """Calculate binary metrics"""
            # ROC Curve and area under curve
            fpr, tpr, _ = roc_curve(y_test1D, y_pred1D, pos_label=1)
            area_under_curve = auc(fpr, tpr)
            
            # Plot ROC 
            fig = plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                    lw=2, label='ROC curve (area = %0.5f)' % area_under_curve)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for '+str(data_dir.list_IDs[index]))
            plt.legend(loc="lower right")
            fig.savefig(os.path.join(output_path,'ROC_'+str(data_dir.list_IDs[index])+'.png'))
            
            # Precision-Recall Curve      
            # Report and log DICE and average precision
            p, r, thresholds = precision_recall_curve(y_test1D, y_pred1D, pos_label=1)
            ap = average_precision_score(np.asarray(y_test1D), np.asarray(y_pred1D))
            average_precision.append(ap)
            
            fig = plt.figure()
            plt.plot(r, p, color='darkorange',
                    lw=2, label='PR curve (AP = %0.5f)' % ap)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision Recall curve for '+str(data_dir.list_IDs[index]))
            plt.legend(loc="lower right")
            fig.savefig(os.path.join(output_path,'PRCurve_'+str(data_dir.list_IDs[index])+'.png'))
            
            f1 = 2*p*r/(p+r)
            optimal_idx = np.argmax(f1) # Find threshold to maximise DICE
            
            print('Optimal threshold: {}'.format(thresholds[optimal_idx]))
            optimal_thresholds.append(thresholds[optimal_idx])
            print('Recall at optimal threshold: {}'.format(r[optimal_idx]))
            recall.append(r[optimal_idx])
            print('Precision at optimal threshold: {}'.format(p[optimal_idx]))
            precision.append(p[optimal_idx])
            print('DICE Score: {}'.format(f1[optimal_idx]))
            print('Average Precision Score: {}'.format(ap))
                    
            # Convert to binary with optimal threshold
            if binary_output:
                y_pred[y_pred>thresholds[optimal_idx]]=1
                y_pred[y_pred<1]=0

        # else:
        #     """Multi-class metrics - need to change predicition function to output one-hot-encoded segmentation """
        #     classes = da.unique(y_test)
        #     p, r, f1, s = precision_recall_fscore_support(y_test1D, y_pred1D, labels=np.array(classes), 
        #                                           average=None, zero_division=np.nan)     
        #     table = [p, r, f1, s]
        #     df = pd.DataFrame(table, columns = np.array(classes), index=['Precision', 'Recall', 'F1 Score', 'Support'])
        #     print(df)

        # Save as tiff              
        with tiff.TiffWriter(tiff_name, bigtiff=True) as tw:
            for z in tqdm(range(y_pred.shape[0]), desc="Export BigTIFF", unit="slice"):
                seg_slice = np.array(y_pred[z, :, :])  # bring one 2D slice to RAM
                tw.write(seg_slice, photometric="minisblack", metadata=None)
        print('Predicted segmentation saved to {}'.format(tiff_name))

    return optimal_thresholds, recall, precision, average_precision
