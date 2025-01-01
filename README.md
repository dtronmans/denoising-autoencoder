# Denoising autoencoder to remove annotations embedded in medical images

Medical scans sometimes come with annotations drawn by the radiologist or doctor. These annotations can act as confounders for a deep learning method that aims to classify or segment these images.

For example, if radiologists draw arrows that point to benign tumors, a standard CNN will learn to recognize the arrows as indicators of a benign tumor. The arrows act as confounders.

## Example

Below are two examples of images with annotations removed using a trained denoising autoencoder:



Image from MMOTU with example of removing annotation

## How it works

The network is a standard U-Net that learns to map from 

## Usage



## Config parameters


## Inference