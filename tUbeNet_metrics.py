# -*- coding: utf-8 -*-
"""tUbeNet 3D 
U-Net based CNN for vessel segmentation

Developed by Natalie Holroyd (UCL)
"""
#Import libraries
from functools import partial

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

def softDice(y_true, y_pred, smooth=1e-6, ignore_background=False):
    "Calculates DICE score for each class using softmax probabilities (not thresholded)"
    axes = tuple(range(len(y_pred.shape)-1)) #reduce along all axes except class
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes) # True positives
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes) # 2xTP + FP + FN
    dice = (2*intersection+smooth)/(denominator+smooth)
    
    if ignore_background:
        dice = dice[..., 1:]  # Exclude background class (index 0)
    
    return dice

def skeleton_dice(y_true, y_pred, smooth=1e-6):
    """Calculates DICE score for skeleton prediction from sigmoid output"""
    y_pred = tf.cast(y_pred, tf.float32) # Change tensor dtype 
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred) # True positives
    denominator = tf.reduce_sum(y_true + y_pred) # 2xTP + FP + FN
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
        self.denominator.assign(tf.zeros_like(self.denominator))

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

def diceCELoss(y_true, y_pred, ignore_background=False, smooth=1e-6):    
    """Custom loss function - calculates categorical crossentropy and mean soft dice across all classes"""   
    y_pred = tf.cast(y_pred, tf.float32) # Change tensor dtype 
    y_true = tf.cast(y_true, tf.float32)
    
    CE = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    dice_classwise = softDice(y_true, y_pred, smooth=smooth, ignore_background=ignore_background)
    meanDice = tf.reduce_mean(dice_classwise)

    dice_CE = (CE + (1-meanDice))/2
    return dice_CE

def diceBCELoss(y_true, y_pred, smooth=1e-6): 
    """Custom loss function - calculates binary crossentropy and soft dice""" 
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1-dice(y_true, y_pred, smooth=smooth)
    dice_BCE = (BCE + dice_loss)/2
    return dice_BCE

def skeletonLoss(y_true, y_pred, smooth=1e-6):
    """Custom loss function combining MSE and 'soft dice' between sigmoid skeleton prediction and distance field of skeleton."""
    MSE=tf.keras.losses.mse(y_true, y_pred)

    """Notes on soft dice score rationale: 
    'Soft' dice is calculated between skeleton distance field and sigmoid skeleton prediction (0-1) to provide a smooth output.
    Both are masked to remove values below 0.5, which correspond to voxels >~2 voxels from the skeleton in the distance field. 
    This avoids the issue of low value predictions adding to a high cumulative 'false positive' rate far from the skeleton.
    E.g. if background voxels in the prediciton all have values ~0.1 this would sum to a high soft FP count
    despite not being considered 'positive' predictions when thresholded. Due to the sparsity of skeleton voxels, this is essential
    to provide a meaningful measure of skeleton similarity not dominated by the large number of background voxels."""
    # Mask to remove values below 0.5
    y_mask = tf.math.greater(y_true, tf.constant(0.5, dtype=y_true.dtype))
    y_masked = tf.math.multiply(y_true, tf.cast(y_mask, tf.float32))

    y_pred_mask = tf.greater(y_pred, tf.constant(0.5, dtype=y_pred.dtype))
    y_pred_masked = tf.math.multiply(y_pred, tf.cast(y_pred_mask, tf.float32))

    # Calculate DICE score between masked skeleton prediction and masked skeleton distance field
    intersection = tf.reduce_sum(y_masked * y_pred_masked) # True positives
    denominator = tf.reduce_sum(y_masked + y_pred_masked) # 2xTP + FP + FN
    dice = (2*intersection+smooth)/(denominator+smooth)

    # Weight losses - balance between MSE and DICE
    # MSE is a typically ~2 orders of magnitude smaller than (1-dice)so weighting is higher
    w_mse = 100
    w_dice = 1

    skelDice_MSE = w_mse*MSE + w_dice*(1-dice)
    return skelDice_MSE
