[![DOI](https://zenodo.org/badge/187050295.svg)](https://doi.org/10.5281/zenodo.15683547) [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbiomethods%2Fbpaf087-blue)](https://doi.org/10.1093/biomethods/bpaf087) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# tUbeNet
tUbeNet is a 3D convolutional neural network (CNN) for semantic segmentation of vasculature from 3D grayscale medical images. It was trained on varied data across different modalities, scales and pathologies, to create a generalisable foundation model, which can be fine-tuned to new images with a minimal additional training ([Paper here](https://doi.org/10.1093/biomethods/bpaf087)).

* Download pretrained weights [here](https://doi.org/10.5522/04/25498603.v2).
* The original training/test data can be found [here](https://doi.org/10.5522/04/25715604.v1).
* Contact: natalie.holroyd.16@ucl.ac.uk for questions, troubleshooting and tips!

![github_fig](https://github.com/natalie11/tUbeNet/assets/30265332/49dde486-2e54-41e1-98cc-f83f6f910688)

## Installation
tUbetnet uses Python 3.11 and Tensorflow 2.20. You can create an environment for running tUbenet using **pip** or **conda**.

### Option 1: Conda
First install anaconda or miniconda following the instructions [here](https://www.anaconda.com/docs/getting-started/anaconda/install).
You can then create a new virtual environment using the .yml file included in this repository by running this command in your command prompt.
```
# Create environment from YAML
conda env create -f tubenet_env.yml
# Activate environment
conda activate tubenet
```

### Option 2: Pip
Create a virtual environment using venv and then install all the required libraries using pip.
```
# Create Environment
python -m venv '\path\to\environment'
# Activate Environment
'\path\to\environment\Scripts\activate.bat'
# Install requirements
pip install -r requirements.txt
```

**Note on GPU usage:** tUbenet has been tested with CUDA 12.9 and cudnn 9.3 (pinned in requirements.txt and tubenet_env.yml). These versions are compatible with Nvidia GPUs with the Pascal microachritecture (e.g. GeForce GTX 10 series) and newer. GPU users will need a Nvidia driver >=525.60.13. On memory usage: the pre-trained model was trained on two 8 GB Nvidia GeForce GTX 1080 GPUs, but tUbeNet is also compatible with single GPU training. Peak memory usage was measured at 5.49 GB when training on a single GPU. Inference time was 222 ms per 64x64x64 volume when run on a single GPU.

**Note for Windows/ MacOS users:** TensorFlow no longer supports GPU usage on Windows or Mac. You can still run tUbnet with CPU only - just make sure you DELETE the Nvidia packages from requirements.txt / tubenet_env.yml before installing. Or see [tensorflow's website](https://www.tensorflow.org/install/pip#windows-wsl2) for instructions on using Windows Subsystem for Linux (WSL) to allow GPU utilization on a Windows machine. 

## How to use

### Workflow

tUbeNet is organized into four callable scripts:

* preprocessing.py → Prepares raw data into model-ready Zarr format.
* train.py → Train a new model or fine-tune the pretrained one.
* test.py → Evaluate a trained model on labeled data (with ROC analysis).
* predict.py → Run inference on unlabeled data and save segmentations.

Small volumes of OCT-A imaging data (OCTA-Data.tif) and paired manual labels (OCTA-Labels.tif) are provided to enable quick testing of the model to confirm successful installation.

### Preparing data
This step converts raw image volumes (.tif/.nii) (and optional binary labels) into Zarr format with header files that can be read by the train/test/predict scripts. You can run this script on an individual image or a folder of images.

The zarr format allows individual chunks of an image to be read from the disk, making processing training and inference much more memory efficient. You can set the chunck size yourself (as below) or use the default size of 64 x 64 x 64 pixels. This script can also optionally crop your images based on the labels provided - creating a subvolume that contains all the labelled vessels while trimming image regions devoid of vessels. Finally, using 'val_fraction' you can optionally chose a proportion of each image volume to reserve for validation. 

With labels (and optional validation data split, cropping):
```
python preprocessing.py \
    --image_directory '\path\to\images' \
    --label_directory '\path\to\labels' \
    --output_path 'path\to\processed' \
    --chunks 64 \ 
    --val_fraction 0.2 \ 
    --crop 
```

Without labels (prediction only):
```
python preprocessing.py \
    --image_directory '\path\to\images' \
    --output_path 'path\to\processed' \
    --chunks 64
```

Key arguments:

```--image_directory``` → Path to raw images (file or folder).

```--label_directory``` → Path to labels (optional).

```--output_path``` → Where processed Zarr + header files are saved.

```--chunks``` → Patch size for saving (default: 64 64 64).

```--val_fraction``` → Split fraction for validation (0–1).

```--crop``` → Crop background regions without vessels.

### Training and Fine-tuning
Run train.py to train from scratch or fine-tune a pretrained model. Training can be run with out without validation data. During training, batches of image subvolumes (64x64x64 pixels) with be generated - the steps_per_epoch argument sets the number of batches generated per training epoch. By providing pre-trained model weights and using the '--fine_tuning' flag, you can fine tune our existing model to your own data. Updated model weights will be saved to the model path provided. Predicted labels and evaluation metrics for the validation data (Receiver Operating Characteristic Curve and Precision Recall Curve - only if validation data was provided) will be saved to the provided output path. 

Train from scratch:
```
python train.py \
    --data_headers 'path\to\train\headers' \
    --val_headers 'path\to\test\headers' \
    --model_path 'path\to\model_output' \
    --output_path 'path\to\prediction' \
    --n_epochs 100 \
    --steps_per_epoch 200 \
    --batch_size 6 \
    --loss "DICE BCE" \
    --lr0 0.0005 
```

Fine-tune a pretrained model:
```
python train.py \
    --data_headers 'path\to\train\headers' \
    --val_headers 'path\to\test\headers' \
    --model_path 'path\to\model_output' \
    --model_weights_file 'path\pretrained_model.weights.h5' \
    --output_path 'path\to\prediction' \
    --n_epochs 50 \
    --steps_per_epoch 200 \
    --fine_tune 
```

Key arguments:

```--data_headers``` → Folder containing headers for training data (generated from preprocessing).

```--val_headers``` → Folder containing headers for validation data (optional).

```--model_path``` → Where models and logs are saved.

```--model_weights_file``` → Pretrained model weights (optional).

```--n_epochs```, ```--steps_per_epoch```, ```--batch_size``` → Epochs, batches per epoch and batch size respectively.

```--loss``` → Loss function - chose from "DICE BCE" (recommended), "focal", "WCCE" (Weighted Categorical CrossEntropy). See the tubenet preprint for details on loss functions.

```--lr0``` → Initial learning rate.

```--class_weights``` → Weights of background to vessels - only relevant when using Weighted Categorical CrossEntropy loss (WCCE).

```--fine_tune``` → Enables fine-tuning by frezzing the first 2 encoding blocks and replacing the classifier layer.

```--volume_dims``` → Input patch size (default: 64 64 64).

```--attention``` → Enable attention blocks in place of skips (experimental).

#### Monitoring training
Training logs can be viewed in TensorBoard using ```tensorboard --logdir path\to\model_output\logs```.

### Testing

Use test.py to evaluate a trained model on labeled test data. This will generate ROC and Precision Recall Curve graphs, as well as labelled images in tiff and Zarr format.

```
python test.py \
    --data_headers 'path\to\data\headers' \
    --model_path 'path\pretrained_model.weights.h5' \
    --output_path 'path\to\prediction' \
    --volume_dims 64 \
    --overlap 32 \
    --binary_output
```

Key arguments:

```--data_headers``` → Folder containing headers for test data (generated from preprocessing).

```--model_weights_file``` → Trained model weights

```--output_path``` → Folder where predictions and evaluation outputs will be saved

```--volume_dims``` → The size of image chunks passed to the model for inference (default: 64 64 64) Note: this sould match the chunk size passed to the model during training

```--overlap``` → Labels are predicted on overlapping image chunks and averaged (to avoid boundary artefacts). The overlap should be approximately half of the chunk volume, but can be reduced (to speed up inference time) or increased as desired.

```--binary_output``` → Use this flag to save label predictions as binary images. Otherwise, the softmax output from the final model layer with be saved.

### Predicting on Unlabelled Data

Use predict.py for running inference (label predicition) on new data without labels. Predicted labels will be saved in zarr format, and optionally as 3D tiff images. Use the --binary_output flag to save label predictions as binary images. Otherwise, the softmax output from the final model layer with be saved (values between 0 and 1, with values closer to 1 implying higher likelyhood of the pixel belonging to a vessel). The softmax output is often be useful for identifying areas of the image that the model is struggling to classify, and allows you to set your own threshold for classifying vessles.

```
python predict.py \
    --data_headers 'path\to\data\headers' \
    --model_path 'path\pretrained_model.weights.h5' \
    --output_path 'path\to\prediction' \
    --tiff_path 'path\to\tiff_outputs' \
    --volume_dims 64 64 64 \
    --overlap 32 32 32 \
    --binary_output \
    --preview
```

Key arguments:

```--data_headers``` → Folder containing headers for data (generated from preprocessing).

```--model_weights_file``` → Trained model weights

```--output_path``` → Folder where predictions will be saved in zarr format

```--tiff_path``` → Folder where predictions will be saved as 3D tiff images (optional)

```--volume_dims``` → The size of image chunks passed to the model for inference (default: 64 64 64) Note: this sould match the chunk size passed to the model during training

```--overlap``` → Labels are predicted on overlapping image chunks and averaged (to avoid boundary artefacts). The overlap should be approximately half of the chunk volume, but can be reduced (to speed up inference time) or increased as desired.

```--binary_output``` → Use this flag to save label predictions as binary images. Otherwise, the softmax output from the final model layer with be saved.
Zarr segmentations in --output_path.

```--preview``` → Use this flag to save prediction previews at regular intervals throughout inference. This is useful for checking the the model prediction is sensible without having to wait for the entire image to be processed.

## Citing
If you use this model in any published work, please cite our [paper](https://doi.org/10.1093/biomethods/bpaf087).
