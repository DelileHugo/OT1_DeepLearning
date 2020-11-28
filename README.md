# OT1_DeepLearning

This repository was made for a project of machine learning at INSA Lyon. It consist of creatong a neural network to find the colors in the pictures. And what's the coordinate of the balls in the picture.


## Prerequisites 
You need to have the pictures and the ground truth in the `data`.

## Instructions

To run the program you can execute the python file named `net.py`, this file while call the `dataset_det.py` which will load all the data. The data are a set of pictures with ground truth.

The code is made for running on cuda GPU, if you havn't cuda cards, you have to remove every `.to("cuda")` and it will run on your CPU.

The command:

 ```
 python net.py
 ```
 Will start traning the model with the data.

 You can ajust some parameters, like the learning rate or the down sample.
