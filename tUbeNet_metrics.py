# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""

#Import libraries
import os
import numpy as np
from skimage import io
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import dask.array as da
import zarr
from tqdm import tqdm
import tifffile as tiff
import pandas as pd

# import tensor flow
import tensorflow as tf

# set backend and dim ordering
K=tf.keras.backend
K.set_image_data_format('channels_last')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""Custom metrics"""
# Use when y_true/ y_pred are keras tensors - for passing to model
def precision(y_true, y_pred, pos_class=1):
    true_positives = K.sum(K.round(K.clip(y_true[...,pos_class] * y_pred[...,pos_class], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,pos_class], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred, pos_class=1):
    true_positives = K.sum(K.round(K.clip(y_true[...,pos_class] * y_pred[...,pos_class], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,pos_class], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def dice(y_true, y_pred, pos_class=1, smooth=1e-6):
    y_pred = tf.cast(y_pred, tf.float32) # Change tensor dtype 
    y_true = tf.cast(y_true, tf.float32)
    P = precision(y_true, y_pred, pos_class=pos_class)
    R = recall(y_true, y_pred, pos_class=pos_class)
    dice = (2*(P*R)+smooth)/(P+R+smooth)
    return dice

def softDice(y_true, y_pred, smooth=1e-6):
    axes = tuple(range(len(y_pred.shape)-1)) #get axes to reduce along 
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2*intersection+smooth)/(denominator+smooth)
    return dice

class MacroDice(tf.keras.metrics.Metric):
    """
    Macro-averaged DICE metric for mulit-class segmentation.
    
    Inputs:
      - y_true one-hot encoded (B, ..., C)
      - y_pred probabilities (softmax) (B, ..., C)
    
    ignore_background=True means DICE score for backgorund class (0) will not be included in average.
    """
    def __init__(self, n_classes, ignore_background=False, smooth=1e-6, name='macro_dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
        
        if ignore_background:
            self.class_ids = tf.range(1, n_classes)
        else:
            self.class_ids = tf.range(0, n_classes)
        
        # Accumulators
        self.intersection = self.add_weight(
            name="intersection",
            shape=(n_classes,),
            initializer="zeros"
        )
        self.denominator = self.add_weight(
            name="denominator",
            shape=(n_classes,),
            initializer="zeros"
        )
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculated per batch
        y_pred = tf.cast(y_pred, tf.float32) # Change tensor dtype 
        y_true = tf.cast(y_true, tf.float32)
        
        # Set axes over which to sum (exclude class axis)
        axes = tuple(range(len(y_pred.shape)-1))
        
        # Calc soft intersection and sum
        # Use softmax prob rather than predicted label to smooth results 
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
        
        self.intersection.assign_add(intersection)
        self.denominator.assign_add(denominator)
    
    def result(self):
        # Calculated per epoch
        dice_per_class = (2.0 * self.intersection + self.smooth)/ (self.denominator + self.smooth)
        dice_included = tf.gather(dice_per_class, self.class_ids)
        
        # Mean DICE across included classes 
        return tf.reduce_mean(dice_included)
    
    def reset_states(self):
        # Reset at start of epoch
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.union.assign(tf.zeros_like(self.denominator))

#-----------------------------------------------------------------------------------------------------------------------
"""Custom Losses"""
def weighted_crossentropy(y_true, y_pred, weights):
	"""Custom loss function - weighted to address class imbalance"""
	weights = tf.convert_to_tensor(weights, dtype=y_pred.dtype)
	weights = tf.reshape(weights, [1, 1, 1, 1, -1]) #reshape to match num dims in y_true

	# Create per-voxel weight mask
	weight_mask = y_true * weights
	weight_mask = tf.reduce_sum(weight_mask, axis=-1)
    
	return tf.keras.losses.categorical_crossentropy(y_true, y_pred,) * weight_mask

def diceCELoss(y_true, y_pred, smooth=1e-6):    
    """Custom loss function - calculates categorical crossentropy and mean soft dice across all classes"""   
    y_pred = tf.cast(y_pred, tf.float32) # Change tensor dtype 
    y_true = tf.cast(y_true, tf.float32)
    
    CE = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    dice_classwise = softDice(y_true, y_pred, smooth=smooth)
    meanDice = tf.reduce_mean(dice_classwise)

    dice_CE = (CE +(1-meanDice))/2
    return dice_CE

def diceBCELoss(y_true, y_pred, smooth=1e-6): 
    """Custom loss function - calculates categorical crossentropy and mean soft dice across all classes""" 
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1-dice(y_true, y_pred, smooth=smooth)
    dice_BCE = (BCE + dice_loss)/2
    return dice_BCE

#-----------------------------------------------------------------------------------------------------------------------
"""ROC Analysis"""
def class_roc_analysis(y_pred, y_test, pos_class=1, 
                       output_name=None, 
                       output_path=None): 
    """"Plots ROC Curve and Precision-Recall Curve for paired 
    ground truth labels and non-thresholded predictions (e.g. softmax output).
    Calculates DICE score, Area under ROC, Average Precision and optimal threshold.
    Optionally saves tiff image of predicted labels.
    Input - y_pred and y_true must be OHE dask arrays
    Output - segmentation metrics for pos_class"""
  
    # Create 1D numpy array of predicted output (softmax)
    y_pred1D = da.ravel(y_pred).astype(np.float32)
    
    # Create 1D numpy array of true labels
    y_test1D = da.ravel(y_test).astype(np.float32)
    
    
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
    plt.title('Receiver operating characteristic for '+str(output_name)+' class '+str(pos_class))
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(output_path,'ROC_'+str(output_name)+'_class_'+str(pos_class)+'.png'))
    
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
    plt.title('Precision Recall curve - '+str(output_name)+' class '+str(pos_class))
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(output_path,'PRCurve_'++str(output_name)+'_class_'+str(pos_class)+'.png'))
    
    f1 = 2*p*r/(p+r)
    optimal_idx = np.argmax(f1) # Find threshold to maximise DICE
    
    # Assign and print metrics at optimal thershold 
    optimal_threshold=thresholds[optimal_idx]
    print('Optimal threshold: {}'.format(optimal_threshold))
    recall=r[optimal_idx]
    print('Recall at optimal threshold: {}'.format(recall))
    precision=p[optimal_idx]
    print('Precision at optimal threshold: {}'.format(precision))
    dice=f1[optimal_idx]
    print('DICE Score: {}'.format(dice))
    print('Average Precision Score: {}'.format(ap))
            
    return optimal_threshold, recall, precision, ap