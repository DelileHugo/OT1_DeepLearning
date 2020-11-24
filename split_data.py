#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:18:55 2020

@author: edward
"""

import os, shutil
import glob
import random

random.seed(0)

os.makedirs("data\\train\\train\\", exist_ok=True)
os.makedirs("data\\train\\val\\", exist_ok=True)

images = sorted(glob.glob("data\\train\\*.jpg"))
labels = sorted(glob.glob("data\\train\\*.npy"))
files = list(zip(images, labels))

random.shuffle(files)
split = 0.2

total_train = int(len(files)*(1.0-split))

for image_file, label_file in files[:total_train]:
    shutil.move(image_file, image_file.replace("train\\", "train\\train\\"))
    shutil.move(label_file, label_file.replace("train\\", "train\\train\\"))
for image_file, label_file in files[total_train:]:
    shutil.move(image_file, image_file.replace("train\\", "train\\val\\"))
    shutil.move(label_file, label_file.replace("train\\", "train\\val\\"))
