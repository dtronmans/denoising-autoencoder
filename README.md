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

Visualize curves during training: in a separate console, run <em>tensorboard --logdir=runs</em> and navigate to localhost:6006 in a browser window.

Inference: python -m src.inference

For inference, make sure to change the dataset path and the model path in src/infer.py.

## Config parameters

### Training parameters:

<ul>
    <li><em>loss_alpha: </em>Weighted loss term for annotated (foreground) parts of the image</li>
    <li><em>loss_beta: </em>Weighted loss term for background parts of the image</li> 
    <li><em>resize_size: </em>Dimensions to resize the image</li> 
    <li><em>epochs: </em>Number of epochs</li> 
    <li><em>lr: </em>learning rate (optimizer is SGD with momentum)</li> 
    <li><em>batch_size: </em>batch size</li>
    <li><em>val_split: </em>Between 0 and 1, ratio of images in the validation set</li>
    <li><em>Architecture: </em>Either Autoencoder, or AutoencoderWithSkipConnections. Note that some higher-frequency information from the original image could be lost if you opt for Autoencoder instead of AutoencoderWithSkipConnections.</li>
</ul>