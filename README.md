# OT1_DeepLearning

This repository was made for a project of machine learning at INSA Lyon. It consist of creating a neural network to find the colors in the pictures. And what's the coordinate of the balls in the picture.


## Prerequisites 
You need to have the pictures and the ground truth in the `data` folder with a subdirectory `train` and in this tolder you need two folder, one named `train` and another one named `val`. For this project we use 80% all the data to train the model and 20% to test it.

## Instructions

To run the program you can execute the python file named `net.py`, this file while call the `dataset_det.py` which will load all the data. The data are a set of pictures with ground truth.

The code is made for running on cuda GPU, if you haven't cuda cards, you have to remove every `.to("cuda")` and it will run on your CPU.

The command:

 ```
 python net.py
 ```
 Will start traning the model with the data.

 You can ajust some parameters, like the learning rate or the down sample.
