import torch
import cv2
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader

class colordata(Dataset):
    def __init__(self, basedir, listdir, shape=(32,32), obs_num=400, split='train'):

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
        self.x_enc = torch.from_numpy(self.x_enc)
        self.y_enc = torch.from_numpy(self.y_enc)
        
        # Create list of all possible (x,y) coordinates
        #self.x_coords = np.hstack([np.array(list(range(self.shape[0]))).reshape(1,-1).T]*self.shape[0]).reshape(-1,)
        #self.y_coords = np.vstack([np.array(list(range(self.shape[1]))).reshape(1,-1).T]*self.shape[1]).reshape(-1,)
        self.x_coords, self.y_coords = torch.meshgrid(torch.arange(self.shape[0]), torch.arange(self.shape[1]))
        self.x_coords = self.x_coords.contiguous().view(-1)
        self.y_coords = self.y_coords.contiguous().view(-1)

    def __len__(self):
        return self.img_num
 
    def __getitem__(self, idx):
        #color_ab = torch.zeros(2, self.shape[0], self.shape[1])
        #recon_const = torch.zeros(1, self.shape[0], self.shape[1])

        img_large = cv2.imread(os.path.join(self.basedir, self.img_fns[idx]))
        if self.shape is not None:
            img = cv2.resize(img_large, (self.shape[0], self.shape[1]))

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert to lab color space

        img_lab = ((img_lab * 2.) / 255.) - 1. #normalizing
        img_lab = torch.from_numpy(img_lab)

        #recon_const[0, :, :] = img_lab[..., 0] # gray image

        #color_ab[0, :, :] = img_lab[..., 1]
        #color_ab[1, :, :] = img_lab[..., 2]
        color_ab = torch.stack(
            (img_lab[..., 1], img_lab[..., 2]), dim=0)

        #indices = list(range(self.shape[0]*self.shape[1]))
        #np.random.shuffle(indices)
        
        return color_ab, img_lab[..., 0].unsqueeze(0), (self.x_coords, self.y_coords), (self.x_enc, self.y_enc)

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


class RandSamplePerBatchCollator(object):
    def __init__(self, high=32*32, minimum_obs_num=64, maximum_obs_num=400):
        self.high = high
        self.minimum_obs_num = minimum_obs_num
        self.maximum_obs_num = maximum_obs_num
        
    def __call__(self, batch):
        obs_num = torch.randint(self.minimum_obs_num, self.maximum_obs_num, (1, )).item()
        obs_indices = torch.randint(0, self.high, (obs_num, ))

        x_coords_obs, y_coords_obs  = [], []
        x_coords_pred, y_coords_pred = [], []
        coords_obs, coords_pred, color_abs, recon_consts = [], [], [], []
        pred_gts = []
        for (color_ab, recon_const, (x_coords, y_coords),
             (x_enc, y_enc)) in batch:
            # L and AB channels
            color_abs.append(color_ab)
            recon_consts.append(recon_const)

            # Observed data
            x_coord_obs = x_coords[obs_indices]
            y_coord_obs = y_coords[obs_indices]
            x_coord_obs_enc = x_enc[x_coord_obs]
            y_coord_obs_enc = y_enc[y_coord_obs]
            gray_obs = recon_const[:, x_coord_obs, y_coord_obs]
            color_obs = color_ab[:, x_coord_obs, y_coord_obs]
            coord_obs = torch.cat(
                (x_coord_obs_enc, y_coord_obs_enc, gray_obs.t(), color_obs.t()), dim=-1)

            x_coords_obs.append(x_coord_obs)
            y_coords_obs.append(y_coord_obs)
            coords_obs.append(coord_obs)

            # Predicted data
            x_coords_pred.append(x_coords)
            y_coords_pred.append(y_coords)
#             coords_pred.append(torch.cat(
#                 (x_enc[x_coords], y_enc[y_coords]), dim=-1))
            gray_pred = recon_const[:, x_coords, y_coords]
            coords_pred.append(torch.cat(
                (x_enc[x_coords], y_enc[y_coords], gray_pred.t()), dim=-1))
            pred_gt = color_ab[:, x_coords, y_coords]
            pred_gts.append(pred_gt.t())

        color_abs = torch.stack(color_abs, dim=0)
        recon_consts = torch.stack(recon_consts, dim=0)
        x_coords_obs = torch.stack(x_coords_obs, dim=0)
        y_coords_obs = torch.stack(y_coords_obs, dim=0)
        x_coords_pred = torch.stack(x_coords_pred, dim=0)
        y_coords_pred = torch.stack(y_coords_pred, dim=0)
        coords_obs = torch.stack(coords_obs, dim=0)
        coords_pred = torch.stack(coords_pred, dim=0)
        pred_gts = torch.stack(pred_gts, dim=0)

        return (color_abs, recon_consts, coords_obs, coords_pred, pred_gts,
                (x_coords_obs, y_coords_obs), (x_coords_pred, y_coords_pred))


if __name__ == "__main__":
    out_dir = 'output/lfw/'
    basedir = 'pytorch_divcolor'
    listdir = 'data/imglist/lfw/'

    data_train = colordata(shape = (32,32), basedir=basedir,
                           listdir=listdir, obs_num = 100, split='train')

    train_loader = DataLoader(dataset=data_train, num_workers=1,
                              batch_size=32, shuffle=True, drop_last=True, 
                              collate_fn=RandSamplePerBatchCollator(maximum_obs_num=100))

    data_test = colordata(shape = (32,32), basedir=basedir,
        listdir=listdir, obs_num = 100, split='test')

    test_loader = DataLoader(dataset=data_test, num_workers=1,
                             batch_size=32, shuffle=True, drop_last=True,
                             collate_fn=RandSamplePerBatchCollator(maximum_obs_num=100))

    for i, (color_c, gray_c, obs, pred, pred_gt,
            (x_coords_obs, y_coords_obs), 
            (x_coords_pred, y_coords_pred)) in enumerate(train_loader):
        print("Color channels (AB) image:")
        print("    ", color_c.shape)
        print("Gray channel (L) image:")
        print("    ", gray_c.shape)
        print("Number of observations coordinates with position encoding (40-dim) and color label (2-dim):")
        print("    ", obs.shape)
        print("Number of predictions coordinates with position encoding (40-dim):")
        print("    ", pred.shape)
        print("Number of predictions coordinates with gt color label(2-dim):")
        print("    ", pred_gt.shape)
        print("(x,y) coordinates of observations: ")
        print("    ", (x_coords_obs.shape, y_coords_obs.shape))
        print("(x,y) coordinates of predictions: ")
        print("    ", (x_coords_pred.shape, y_coords_pred.shape))
        break

