# Tensorflow VGG16 for AlzEYEmer
First download vgg16.npy from https://drive.google.com/drive/folders/0Bynv9fOtHYAJOHl0c2NKLXVxN28 

This is a project for Hackathon 2016 summer at Brown

## Getting Started
Before using the repo, download the vgg_pretrained weight 

## Introduction 
Try to use deep learning technique to identify dangerous locations for Alzheimer's patients and provide corresponding instructions.

## Built With

* Python 2.7

## How to use 
Take an image and our functions will return the location of this image:   
```
dic = {0:'basement',1:'bathroom',2:'bedroom',3:'dining_room',4:'kitchen',5:'living_room', 6:'street_building'}
```
```
import alz_1
alz_1.vgg_cl(image)
NOTE: this image should be a matrix(gray scale is accepted)
```

## Extra
This model is trained from part of SUN database


