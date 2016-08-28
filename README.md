# Tensorflow VGG16 for AlzEYEmer
First download vgg16.npy from https://drive.google.com/drive/folders/0Bynv9fOtHYAJOHl0c2NKLXVxN28
# Usage: take an image and return the location of this image:     dic={0:'basement',1:'bathroom',2:'bedroom',3:'dining_room',4:'kitchen',5:'living_room', 6:'street_building'}

# How to use 
import alz_1
alz_1.vgg_cl(image)
NOTE: this image should be a matrix(gray scale is accepted)
##Extra
This model is trained from part of SUN database
