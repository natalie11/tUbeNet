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
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.util import apply_parallel
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

#-----------------------PREPROCESSING FUNCTIONS--------------------------------------------------------------------------------------------------------------

def fix_label_format(seg, chunks):
    """ Finds unique classes in mutli-channel segmentation files, 
    and combines into a single 3D array, where each class has a unique pixel value.
    E.g. (Z,X,Y,2) where thefirst channel has pixel values 0 and 1, and the second channel has 
    pixel values 0, 1, 2 becomes (Z,X,Y) with pixel values 0, 1, 2, 3"""
    
    print("Merging channels into a single class map")
    assert len(seg.shape)==4, "Expected (Z, X, Y) or (Z, X, Y, C) segmentation"
    
    seg = da.from_array(seg) # Create Dask Array with automatic chunk size
    channel_info = {} # Record info about unique classes in each channel

    for c in range(seg.shape[-1]):
        values = da.unique(seg[..., c]).compute()
        assert len(values)<51, "Over 50 unqiue classes identified - check labels file is correct."
        channel_info[c] = values
        print(f"Channel {c} classes: {values}")

    merged = da.zeros(seg.shape[:-1], chunks=chunks, dtype="int16")
    next_class = 1 # Start re-labelling classes starting at 1
    
    for c, values in channel_info.items():
        for v in values:
            if v == 0: # Skip background class in each channel
                continue

            mask = seg[..., c] == v
            # Check for overlapping classes
            overlap = da.logical_and(mask, merged!=0)
            if da.any(overlap).compute():
                tot = da.sum(overlap).compute()
                print(f"WARNING: Overlap detected for channel {c}, value {v} ({tot} pixels). Previous label overwritten.")

            merged = da.where(mask, next_class, merged)
            next_class += 1
    
    print("Labels file now has shape "+str(merged.shape)+" with classes "+str(da.unique(merged).compute())+".")
    
    return merged

def data_preprocessing(image_path=None, label_path=None, skeleton_path=None, chunks='auto'):
    """# Pre-processing
    Load data, downsample if neccessary, normalise and pad.
    Inputs:
    image_path = path to image data (string)
    label_path = path to labels (string)
    skeleton_path = path to skeleton data (string)
    Outputs:
    img = image data as an dask array, scaled between 0 and 1, float32
    seg = label data as an dask array, scaled between 0 and 1, int16
    skeleton = skeleton data as dask array, scaled between 0 and 1, int16
    """
    
    # Load image
    print('Loading images from '+str(image_path))
    img=io.imread(image_path)
    img=da.from_array(img,chunks=chunks)
    print('Size '+str(img.shape))

    if len(img.shape)==4:
        print('Image data has 4 dimensions. Cropping to first 3 dimensions')
        img=img[:,:,:,0]
    assert img.ndim == 3, "Expected (Z,X,Y) image"
    
    # Normalise 
    print('Rescaling data between 0 and 1')
    img_min = da.min(img)
    denominator = da.nanmax(img)-img_min 
    img = (img-img_min)/denominator # Rescale between 0 and 1
    
    # Set data type
    img = img.astype('float32')
    seg = None
    skeleton = None

	#Repeat for labels is present
    if label_path is not None:
        print('Loading labels from '+str(label_path))
        seg = io.imread(label_path)
        
        # Find the number of unique classes in segmented training set
        if len(seg.shape)>3:
            # Assume classes are saved as different channels
            seg = fix_label_format(seg, chunks)
        else:
            seg = da.from_array(seg,chunks=chunks)
            classes = da.unique(seg).compute()
            
            assert len(classes)<51, "Over 50 unqiue classes identified - check labels file is correct."
            print('Labels processed with classes '+str(classes))
            
            # Check that classes are consecutive integers starting at 0
            classes=np.sort(classes)
            is_consecutive = (classes[0] == 0 and np.array_equal(classes, np.arange(len(classes))))
    
            # If not - map classes to integer label values starting at 0
            if not is_consecutive:
                mapping = {old: new for new, old in enumerate(classes)}
                seg_mapped = da.zeros_like(seg, dtype=np.int32)
                for old, new in mapping.items():
                    seg_mapped[seg==old] = new
                seg = seg_mapped
            
            seg = seg.astype('uint16')            
           		
    if skeleton_path is not None:
        print('Loading skeleton from '+str(skeleton_path))
        skeleton = io.imread(skeleton_path)
        skeleton = da.from_array(skeleton, chunks=chunks)

        if len(skeleton.shape)==4:
            print('Skeleton data has 4 dimensions. Cropping to first 3 dimensions')
            skeleton=skeleton[:,:,:,0]
        assert skeleton.ndim == 3, "Expected (Z,X,Y) skeleton data"

        if da.unique(skeleton).compute().max()>1 or len(da.unique(skeleton).compute())>2:
            print("Warning: Skeleton data has values greater than 1 or non-binary values. Converting to binary mask with threshold 0.5.")
            skeleton = da.where(skeleton>=0.5,1,0)
        skeleton = skeleton.astype('uint8')

    return img, seg, skeleton

def list_image_files(directory):
    """ Lists files in a directory. 
    This function is used to handle either a directory of files, or a single file.
    If given a path to a single file - splits the directory path from the filename.
    Returns directory path and list of filename."""
    if os.path.isdir(directory):
        # Add all file paths of image_paths
        image_filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        image_filenames = sorted(image_filenames) # Sort alphabetically
    elif os.path.isfile(directory):
        # If file is given, process this file only
        directory, image_filenames = os.path.split(directory.replace('\\','/'))
        image_filenames = [image_filenames]
    else:
        return None, None
    
    return directory, image_filenames

def crop_from_labels(data, labels, skeleton=None):
    """Crops 3D dask array based on labels.
    Returns a 3D dask array.""" 
    iz, ix, iy = da.nonzero(labels) # find instances of non-zero values in X_test along axis 1
    zmin, xmin, ymin = da.min(iz), da.min(ix), da.min(iy)
    zmax, xmax, ymax = da.max(iz), da.max(ix), da.max(iy)

    labels = labels[zmin:zmax+1, xmin:xmax+1, ymin:ymax+1] # use this to index data and labels
    data = data[zmin:zmax+1, xmin:xmax+1, ymin:ymax+1]
    if skeleton is not None:
        skeleton = skeleton[zmin:zmax+1, xmin:xmax+1, ymin:ymax+1]
    print("Cropped to {}".format(data.shape))
    
    return labels, data, skeleton

def split_train_test(data, labels, val_fraction, skeleton=None):
    """"Splits paired images and labels into training and validation sets given a validation fraction.
    Optionally splits skeletons if provided.
    Takes data/labels/skeleton as dask arrays.
    Outputs lists of training and testing [data, labels, skeletons (if provided)] as dask arrays."""
    n_training_imgs = int(data.shape[0]-da.floor(data.shape[0]*val_fraction))
    training = []
    testing = []
            
    training.append(data[0:n_training_imgs,...])
    testing.append(data[n_training_imgs:,...])

    training.append(labels[0:n_training_imgs,...])
    testing.append(labels[n_training_imgs:,...])

    if skeleton is not None:
        training.append(skeleton[0:n_training_imgs,...])
        testing.append(skeleton[n_training_imgs:,...])
    else:
        training.append(None)
        testing.append(None)

    return training, testing
            
def save_as_zarr_array(data, labels=None, skeleton=None, output_path=None, output_name=None, chunks=(64,64,64)):
    """"Data (and optionally labels, skeleton) are saved in chunked zarr format.
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
    
    labels_filename = None
    skeleton_filename = None
    
    # Repeat of labels if present
    if labels is not None: 
        # Rechunk dask array and save as zarr
        labels = labels.rechunk(chunks)
        labels.to_zarr(os.path.join(output_path, str(output_name)+"_labels"))
        labels_filename = os.path.join(output_path, str(output_name)+"_labels")
        
    # Repeat of skeleton if present
    if skeleton is not None:
        # Rechunk dask array and save as zarr
        skeleton = skeleton.rechunk(chunks)
        skeleton.to_zarr(os.path.join(output_path, str(output_name)+"_skeleton"))
        skeleton_filename = os.path.join(output_path, str(output_name)+"_skeleton")

    # Save data header for easy reading in, with label_filename=None
    header = DataHeader(ID=output_name, image_dims=data.shape, 
                            image_filename=os.path.join(output_path, output_name),
                            label_filename=labels_filename,
                            skeleton_filename=skeleton_filename)
    header.save(header_name)
        
    return output_path, header_name

def generate_skeleton(labels, chunks=(64,64,64)):
    """ Generate skeleton from binary vessel mask using 3D skeletonization.
    Inputs:
        labels: binary dask array with values 0 and 1
    Outputs:
        skeleton: 3D array with skeletonized vessels (same shape as labels)
    """
    # Use apply_parallel to apply skimage skeletonization across chunks
    # Compute=False causes apply_parallel to return a dask array
    overlap = (chunks[0]//2, chunks[1]//2, chunks[2]//2) # Overlap of half chunk size to avoid boundary artefacts
    skeleton = apply_parallel(skeletonize, labels, chunks=chunks, depth=overlap, 
                              mode='nearest', compute=False, dtype='bool')
        
    return skeleton.astype(np.uint8)

def compute_distance_field(skeleton, binary_mask, sigma=2.0, chunks=(64,64,64)):
    """
    Compute distance field from skeleton with exponential decay.
    Distance values decay exponentially from skeleton (value 1) to 5 pixels away (value ~0).
    
    Inputs:
        skeleton: 3D binary array (skeleton centerlines)
        binary_mask: 3D binary array (vessel mask to constrain field)
        sigma: decay parameter for exponential function (higher = wider falloff)
        normalize: If True, output is scaled to [0, 1]; if False, raw exponential decay
    
    Outputs:
        distance_field: 3D array with values decaying from 1 to 0, same shape as skeleton
    """

    # Invert skeleton, as distance_transform_edt computes distance to nearest non-zero voxel
    skel_inverted = da.logical_not(skeleton.astype(bool))

    # apply distance_transform_edt on overlapping chunks, calculates euclidean distance from skeleton
    # returns 0 at skeleton voxels, increasing away from skeleton
    overlap = (chunks[0]//2, chunks[1]//2, chunks[2]//2)
    distance_to_skeleton = apply_parallel(distance_transform_edt, skel_inverted, chunks=chunks,
                                          depth=overlap, mode='nearest', compute=False, dtype=np.float64)
    
    # Apply exponential decay: exp(-distance/sigma)
    # This gives 1.0 at skeleton, decaying to ~0 at distance >> sigma
    assert sigma > 0, "Sigma must be positive for exponential decay"
    distance_field = da.exp(-distance_to_skeleton / sigma)
    
    # Constrain to vessel mask (set to 0 outside vessels)
    distance_field = da.where(binary_mask, distance_field, 0.0)
    distance_field = distance_field.astype(np.float32)
    
    return distance_field

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

#---------------------------EVALUATION----------------------------------------------------------------------

def roc_analysis(model, data_dir, volume_dims=(64,64,64), 
                 overlap=None, n_classes=2, ignore_background=True,
                 output_path=None, prob_output=True): 
    """"
    Plots ROC Curve and Precision-Recall Curve for paired ground truth labels 
    and non-thresholded predictions (e.g. softmax output).
    Calculates DICE score, Area under ROC, Average Precision and optimal threshold.
    Optionally saves tiff image of predicted labels.
    
    model - trained model to be evaluated
    data_dir - direcory of image data and corrosponding labels
    volume_dims and overlap - parameters to pass to predict_segmentation_dask
    n_classes - number of unquie classes in data labels including backgroud
    ignore_background - when true, metrics are no calculated for class 0
    ouput_path - location to save plots and predicted segmentation
    prob_output - if true, the softmax probabilities will be saved, as opposed to the labels, allowing for custom thresholding
    """
    optimal_thresholds = []
    recall = []
    precision = []
    average_precision = []
    
    if not overlap:
        overlap = (volume_dims[0]//2,volume_dims[1]//2,volume_dims[2]//2)

    for index in range(0,len(data_dir.list_IDs)):
        print('Evaluating model on '+str(data_dir.list_IDs[index])+' data')
        
        # Add to lists 
        optimal_thresholds.append([])
        recall.append([])
        precision.append([])
        average_precision.append([])

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
        y_test = da.from_zarr(data_dir.label_filenames[index])
        
        # """Multi-class metrics - need to change predicition function to output one-hot-encoded segmentation """
        # classes = da.unique(y_test)
        # p, r, f1, s = precision_recall_fscore_support(y_test1D, y_pred1D, labels=np.array(classes), 
        #                                       average=None, zero_division=np.nan)     
        # table = [p, r, f1, s]
        # df = pd.DataFrame(table, columns = np.array(classes), index=['Precision', 'Recall', 'F1 Score', 'Support'])
        # print(df)
               
        # Skip background class if ignore_bakcground is true
        start = 0
        if ignore_background: start = 1
        
        # Loop through classes
        for c in range(start, n_classes): 
            
            print("Class {}".format(c))
            # Create 1D numpy array of predicted output (softmax)
            y_pred1D = da.ravel(y_pred[...,c]).astype(np.float32)
            
            # Create 1D numpy array of true labels for class
            y_test_binary = da.where(y_test==c,1,0)
            y_test1D = da.ravel(y_test_binary).astype(np.float32)
            del y_test_binary
            
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
            plt.title('Receiver operating characteristic for '+str(data_dir.list_IDs[index])+' class '+str(c))
            plt.legend(loc="lower right")
            fig.savefig(os.path.join(output_path,'ROC_'+str(data_dir.list_IDs[index])+'_class_'+str(c)+'.png'))
            
            # Precision-Recall Curve      
            # Report and log DICE and average precision
            p, r, thresholds = precision_recall_curve(y_test1D, y_pred1D, pos_label=1)
            ap = average_precision_score(np.asarray(y_test1D), np.asarray(y_pred1D))
                        
            fig = plt.figure()
            plt.plot(r, p, color='darkorange',
                    lw=2, label='PR curve (AP = %0.5f)' % ap)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision Recall curve for '+str(data_dir.list_IDs[index])+' class '+str(c))
            plt.legend(loc="lower right")
            fig.savefig(os.path.join(output_path,'PRCurve_'+str(data_dir.list_IDs[index])+'_class_'+str(c)+'.png'))
            
            f1 = 2*p*r/(p+r+1e-6)
            optimal_idx = np.argmax(f1) # Find threshold to maximise DICE
            
            # Print metrics and add to lists
            print('Optimal threshold: {}'.format(thresholds[optimal_idx]))
            optimal_thresholds[index].append(thresholds[optimal_idx])
            print('Recall at optimal threshold: {}'.format(r[optimal_idx]))
            recall[index].append(r[optimal_idx])
            print('Precision at optimal threshold: {}'.format(p[optimal_idx]))
            precision[index].append(p[optimal_idx])
            print('DICE Score: {}'.format(f1[optimal_idx]))
            print('Average Precision Score: {}'.format(ap))
            average_precision[index].append(ap)
                    
            
        # Save as tiff 
        if prob_output: 
            # Reorder ZXYC to ZCXY to allow saving with imwrite
            y_pred=np.moveaxis(y_pred, -1, 1)        
            tiff.imwrite(tiff_name, y_pred, metadata={"axes": "ZCYX"}, imagej=True, bigtiff=True)
            print('Predicted segmentation saved to {}'.format(tiff_name))
        else:
            # Reverse one hot encoding using optimal thresholds 
            # !!To Do!! These thresholds should be defined on training data as opposed to test/validation
            # Add threshold optimisation in training and pass to validation analysis ROC?
            # Or always use simple argmax? This requires extra class weight tuning to prevent under-represented classes being missed 
            for c in range(start, n_classes):
                y_pred[..., c] = np.where(y_pred[..., c]>optimal_thresholds[index][c-start], 1, 0)
            y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8) 
            tiff.imwrite(tiff_name, y_pred, metadata={"axes": "ZYX"}, imagej=True, bigtiff=True)
            print('Predicted segmentation saved to {}'.format(tiff_name))

    return optimal_thresholds, recall, precision, average_precision
