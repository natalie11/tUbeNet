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
"""Custom Losses for Skeleton Prediction"""

def skeletonSoftDice(y_true, y_pred, smooth=1e-6):
    """Soft DICE loss for skeleton/distance field prediction
    
    Inputs:
        y_true: Ground truth distance field (continuous 0-1)
        y_pred: Predicted distance field (continuous 0-1)
        smooth: Small epsilon for numerical stability
    
    Outputs:
        dice: Mean soft DICE loss across batch
    """
    axes = tuple(range(len(y_pred.shape)-1)) # get axes to reduce along 
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2*intersection+smooth)/(denominator+smooth)
    return 1 - tf.reduce_mean(dice)  # Return loss (1 - dice)


def skeletonOverlapRegularizer(y_mask_pred, y_skeleton_pred, smooth=1e-6):
    """Regularizer to ensure predicted skeleton stays within predicted mask
    
    Encourages skeleton predictions to overlap with vessel mask predictions.
    Uses cross-entropy between skeleton and binary mask to penalize skeleton 
    predictions outside the mask.
    
    Inputs:
        y_mask_pred: Predicted mask probabilities (softmax output, shape: B, Z, X, Y, n_classes)
        y_skeleton_pred: Predicted skeleton distances (sigmoid output, shape: B, Z, X, Y, 1)
        smooth: Small epsilon for numerical stability
    
    Outputs:
        loss: Regularization loss (lower is better when skeleton overlaps with mask)
    """
    # Extract vessel probability from mask (typically class 1 in binary case)
    # If using softmax, take the positive class probability
    if len(y_mask_pred.shape) == 5:
        vessel_prob = y_mask_pred[..., 1]  # Take class 1 (vessel) probability
    else:
        vessel_prob = y_mask_pred
    
    # Squeeze skeleton to match vessel_prob shape
    skeleton_pred = tf.squeeze(y_skeleton_pred, axis=-1)
    
    # Penalize skeleton predictions where vessel probability is low
    # Use inverse of vessel probability as weight (penalize where vessels are unlikely)
    penalty = skeleton_pred * (1 - vessel_prob)
    
    # Mean penalty across spatial dimensions
    loss = tf.reduce_mean(penalty)
    return loss


class LearnableWeightedLoss(tf.keras.losses.Loss):
    """Custom loss with learnable weights for multi-task learning
    
    Combines mask and skeleton losses with learnable temperature-scaled weights.
    Weights are learned during training via a learnable parameter.
    
    Inputs:
        mask_loss_fn: Loss function for mask prediction
        skeleton_loss_fn: Loss function for skeleton prediction
        regularizer_fn: Regularization function
        init_log_weight_mask: Initial log-scale weight for mask loss (default log(0.5) ≈ -0.69)
        init_log_weight_skel: Initial log-scale weight for skeleton loss (default log(0.3) ≈ -1.2)
        init_log_weight_reg: Initial log-scale weight for regularizer (default log(0.2) ≈ -1.6)
    """
    def __init__(self, mask_loss_fn, skeleton_loss_fn, regularizer_fn,
                 init_log_weight_mask=-0.69, init_log_weight_skel=-1.2, init_log_weight_reg=-1.6,
                 name='learnable_weighted_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_loss_fn = mask_loss_fn
        self.skeleton_loss_fn = skeleton_loss_fn
        self.regularizer_fn = regularizer_fn
        
        # Initialize learnable log-space weights (log-space for numerical stability)
        self.log_weight_mask = tf.Variable(
            initial_value=init_log_weight_mask, 
            trainable=True, 
            name="log_weight_mask"
        )
        self.log_weight_skel = tf.Variable(
            initial_value=init_log_weight_skel, 
            trainable=True, 
            name="log_weight_skel"
        )
        self.log_weight_reg = tf.Variable(
            initial_value=init_log_weight_reg, 
            trainable=True, 
            name="log_weight_reg"
        )
    
    def call(self, y_true, y_pred):
        """
        Inputs:
            y_true: list [y_true_mask, y_true_skeleton] where:
                - y_true_mask: One-hot encoded vessel mask (B, Z, X, Y, n_classes)
                - y_true_skeleton: Binary or distance skeleton (B, Z, X, Y, 1)
            y_pred: list [y_pred_mask, y_pred_skeleton] where:
                - y_pred_mask: Softmax vessel prediction (B, Z, X, Y, n_classes)
                - y_pred_skeleton: Sigmoid skeleton prediction (B, Z, X, Y, 1)
        
        Returns:
            Combined weighted loss
        """
        y_true_mask, y_true_skeleton = y_true
        y_pred_mask, y_pred_skeleton = y_pred
        
        # Compute individual losses
        loss_mask = self.mask_loss_fn(y_true_mask, y_pred_mask)
        loss_skeleton = self.skeleton_loss_fn(y_true_skeleton, y_pred_skeleton)
        loss_regularizer = self.regularizer_fn(y_pred_mask, y_pred_skeleton)
        
        # Convert log-space weights to linear space and normalize
        weight_mask = tf.exp(self.log_weight_mask)
        weight_skel = tf.exp(self.log_weight_skel)
        weight_reg = tf.exp(self.log_weight_reg)
        
        total_weight = weight_mask + weight_skel + weight_reg
        weight_mask = weight_mask / total_weight
        weight_skel = weight_skel / total_weight
        weight_reg = weight_reg / total_weight
        
        # Combine losses
        combined_loss = (weight_mask * loss_mask + 
                         weight_skel * loss_skeleton + 
                         weight_reg * loss_regularizer)
        
        return combined_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mask_loss_fn': self.mask_loss_fn,
            'skeleton_loss_fn': self.skeleton_loss_fn,
            'regularizer_fn': self.regularizer_fn,
        })
        return config


def create_dual_task_loss(mask_loss='dice_bce', class_weights=None, 
                          init_log_weight_mask=-0.69, init_log_weight_skel=-1.2, 
                          init_log_weight_reg=-1.6):
    """
    Factory function to create a dual-task loss with learnable weights
    
    Inputs:
        mask_loss: Name of loss function for mask ('dice_bce', 'dice_ce', 'wcce')
        class_weights: Class weights for weighted losses (None or tuple)
        init_log_weight_mask: Initial log-scale weight for mask loss
        init_log_weight_skel: Initial log-scale weight for skeleton loss
        init_log_weight_reg: Initial log-scale weight for regularizer
    
    Outputs:
        loss_fn: Compiled LearnableWeightedLoss instance
    """
    # Select mask loss function
    if mask_loss.lower() == 'dice_bce':
        mask_loss_fn = partial(diceBCELoss, smooth=1e-6)
    elif mask_loss.lower() == 'dice_ce':
        mask_loss_fn = partial(diceCELoss, smooth=1e-6)
    elif mask_loss.lower() == 'wcce':
        if class_weights is None:
            raise ValueError("class_weights required for weighted crossentropy")
        mask_loss_fn = partial(weighted_crossentropy, weights=class_weights)
    else:
        raise ValueError(f"Unknown mask_loss: {mask_loss}")
    
    # Skeleton loss is always soft DICE
    skeleton_loss_fn = partial(skeletonSoftDice, smooth=1e-6)
    
    # Regularizer function
    regularizer_fn = skeletonOverlapRegularizer
    
    # Create learnable loss
    loss = LearnableWeightedLoss(
        mask_loss_fn=mask_loss_fn,
        skeleton_loss_fn=skeleton_loss_fn,
        regularizer_fn=regularizer_fn,
        init_log_weight_mask=init_log_weight_mask,
        init_log_weight_skel=init_log_weight_skel,
        init_log_weight_reg=init_log_weight_reg
    )
    
    return loss

