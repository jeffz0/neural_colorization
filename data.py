import torch
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

class colordata(Dataset):
    def __init__(self, basedir, listdir, shape=(32,32), obs_num=100, split='train'):

        self.img_fns = []
        
        self.basedir = basedir
        with open('%s/list.%s.vae.txt' % (os.path.join(basedir, listdir), split), 'r') as ftr:
            for img_fn in ftr:
                self.img_fns.append(img_fn.strip('\n'))

        self.img_num = len(self.img_fns)
        self.shape = shape
        self.obs_num = obs_num # Number of observations
        
        # Create mapping from (x,y) coordinates to positional encodings
        self.x_enc, self.y_enc = self.create_position_encodings(size=shape[0])
        
        # Create list of all possible (x,y) coordinates
        self.x_coords = np.hstack([np.array(list(range(self.shape[0]))).reshape(1,-1).T]*self.shape[0]).reshape(-1,)
        self.y_coords = np.vstack([np.array(list(range(self.shape[1]))).reshape(1,-1).T]*self.shape[1]).reshape(-1,)

    def __len__(self):
        return self.img_num
 
    def __getitem__(self, idx):
        color_ab = np.zeros((2, self.shape[0], self.shape[1]), dtype='f')
        recon_const = np.zeros((1, self.shape[0], self.shape[1]), dtype='f')

        img_large = cv2.imread(os.path.join(self.basedir,self.img_fns[idx]))
        if(self.shape is not None):
            img = cv2.resize(img_large, (self.shape[0], self.shape[1]))

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert to lab color space

        img_lab = ((img_lab*2.)/255.)-1. #normalizing

        recon_const[0, :, :] = img_lab[..., 0] # gray image

        color_ab[0, :, :] = img_lab[..., 1].reshape(1, self.shape[0], self.shape[1])
        color_ab[1, :, :] = img_lab[..., 2].reshape(1, self.shape[0], self.shape[1])

        indices = list(range(self.shape[0]*self.shape[1]))
        np.random.shuffle(indices)
        
        # Select obs_num number of coordinates for encoder
        x_coords_obs = self.x_coords[indices[:self.obs_num]]
        y_coords_obs = self.y_coords[indices[:self.obs_num]]
        # Create the positional encoding of the coordinates
        x_coords_obs_enc = self.x_enc[x_coords_obs]
        y_coords_obs_enc = self.y_enc[y_coords_obs]
        color_obs = color_ab[:,x_coords_obs, y_coords_obs]
        coords_obs = np.hstack((x_coords_obs_enc,y_coords_obs_enc,color_obs.T))

        # Select remaining number of coordinates for prediction decoder
        x_coords_pred = self.x_coords[indices[self.obs_num:]]
        y_coords_pred = self.y_coords[indices[self.obs_num:]]
        # Create the positional encoding of the coordinates
        x_coords_pred_enc = self.x_enc[x_coords_pred]
        y_coords_pred_enc = self.y_enc[y_coords_pred]
        coords_pred = np.hstack((x_coords_pred_enc,y_coords_pred_enc))
        pred_gt = color_ab[:,x_coords_pred, y_coords_pred].T

        return color_ab, recon_const, coords_obs, coords_pred, pred_gt, (x_coords_obs, y_coords_obs), (x_coords_pred, y_coords_pred)

    def create_position_encodings(self, size=32):
        H, W, C = size, size, 3

        L = 10 # parameter for size of encoding

        x_linspace = (np.linspace(0, W-1, W)/W)*2 -1 
        y_linspace = (np.linspace(0, H-1, H)/H)*2 -1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []

        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el 
            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

        x_el = np.array(x_el).T
        x_el_hf = np.array(x_el_hf).T
        y_el = np.array(y_el).T
        y_el_hf = np.array(y_el_hf).T

        return np.hstack((x_el, x_el_hf)), np.hstack((y_el, y_el_hf))

