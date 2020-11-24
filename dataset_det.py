import numpy as np
from skimage import io
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

class Balls_CF_Detection(Dataset):
    def __init__(self, dir, cache=False):
    
        if cache:    
            self.files = [io.imread(f) for f in sorted(glob.glob(f"{dir}*.jpg"))]
            self.labels = [np.load(f) for f in sorted(glob.glob(f"{dir}*.npy"))]
        else: 
            self.files = sorted(glob.glob(f"{dir}*.jpg"))
            self.labels = sorted(glob.glob(f"{dir}*.npy"))
        
        self.dir = dir
        self.cache = cache

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        if self.cache:
            img = self.files[index]    
        else:   
            img = io.imread(self.files[index])
        
        img = np.asarray(img)
        img = img.astype(np.float32)
        
        # Dims in: x, y, color
        # should be: color, x, y
        img = np.transpose(img, (2,0,1))
        img = torch.tensor(img)

        # Load presence and bounding boxes and split it up
        if self.cache:
            p_bb = self.labels[index]
        else:   
            p_bb = np.load(self.labels[index])     
        
        p  = p_bb[:,0]
        bb = p_bb[:,1:5]
        
        # You can change this to 2,4 ..etc if you want to train more quickly
        # But may affect performance
        DOWN_SAMPLE = 4
        
        return img[:,::DOWN_SAMPLE,::DOWN_SAMPLE], p, bb

    # Return the dataset size
    def __len__(self):
        return len(self.files)
        
if __name__ == "__main__":
    # train_dataset = Balls_CF_Detection ("../mini_balls/train", 20999,
    #     transforms.Normalize([128, 128, 128], [50, 50, 50]))
    train_dataset = Balls_CF_Detection("data\\train\\train\\")

    img,p,b = train_dataset.__getitem__(42)

    print ("Presence:")
    print (p)

    print ("Pose:")
    print (b)
