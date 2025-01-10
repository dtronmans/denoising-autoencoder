<em>This repository is part of my thesis on detecting ovarian cancer using deep learning approaches, and I will soon
make all repositories related to this project public.</em>

# Denoising autoencoder to remove annotations embedded in medical images

Medical scans sometimes come with annotations drawn by the radiologist or doctor. These annotations can act as
confounders for a deep learning method that aims to classify or segment these images.

For example, if radiologists draw arrows that point to benign tumors, a standard CNN will learn to recognize the arrows
as indicators of a benign tumor. The arrows act as confounders, and the classifier will predict "benign" when it sees
the arrows, even if the tumor is malignant.

## How it works

### Training process

Given a dataset that has clean images (medical scans without annotations) and noisy images (medical scans with
annotations), artificially noised/annotated images are generated from the clean images to resemble the noised images. From this, you now have a dataset that has pairs of clean and noised images, and the network learns to go from the noised image back to the clean image.

To generate the dataset of artificially noised images, you need your custom drawing function that adds annotations to the clean images in your dataset. The custom drawing function will of course depend on the nature of your images and the annotations.

<img src="./media/process.png"/>

### Some examples

<table>
    <tr>
        <th>Original image</th>
        <th>Inferred (removed annotations)</th>
    </tr>
    <tr>
        <td><img src="./media/result_one_before.png"/></td>
        <td><img src="media/result_one_after.png"/></td>
    </tr>
</table>

The network relies on a custom weighted loss function that I defined in losses.py. Since annotations are a very small part of the image, 

## Installation

Make a Conda environment (this has been developed and tested with Python 3.12.7) and run <em>pip install -r
requirements.txt</em>

## Usage

Dataset structure: folder with "clean" and "annotated" subdirectories.
The image names in both directories should be the same (corresponding image pairs).

Training: python -m src.train

Inference:

## Config parameters

### Training parameters:

<ul>
    <li><em>loss_alpha:</em></li> Weighted loss term for annotated (foreground) parts of the image
    <li><em>loss_beta:</em></li> Weighted loss term for background parts of the image
    <li><em>resize_size:</em></li> Dimensions to resize the image
    <li><em>epochs:</em></li> Number of epochs
    <li><em>lr:</em></li> learning rate (optimizer is SGD with momentum)
    <li><em>batch_size:</em></li> batch size
    <li><em>val_split:</em></li> Between 0 and 1, ratio of images in the validation set
    <li><em>Architecture:</em></li> Either Autoencoder, or AutoencoderWithSkipConnections. Note that some higher-frequency information from the original image could be lost if you opt for Autoencoder instead of AutoencoderWithSkipConnections.
</ul>