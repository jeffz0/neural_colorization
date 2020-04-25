from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import collections

import cv2
import glob
import math
import numpy as np
import os

from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import visualize
from data import colordata
from CNP import get_encoder, get_decoder


parser = argparse.ArgumentParser(description='Neural Processes Colorizations')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--obs_num', type=int, default=800, 
                    help='number of observations per example (default: 800)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--vis_interval', type=int, default=100, 
                    help='how many iterations before visualizing output and losses')
parser.add_argument('--name', type=str, default="exp", 
                    help='name of model, models and outputs are saved under this model name')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = args.device

def train(encoder, decoder, train_loader, train_output_dir):
    _loss, _loss_a, _loss_b = 0., 0., 0.
    for i, (color_c, gray_c, obs, preds, preds_gt, (x_coords_obs, y_coords_obs), (x_coords_pred, y_coords_pred)) in enumerate(train_loader):
        optimizer.zero_grad()
        obs, preds, preds_gt = obs.to(device), preds.to(device), preds_gt.to(device)

        representation = encoder(obs.float())
        representation_expand = representation.unsqueeze(1).expand(-1,preds.shape[1],512)
        # Concatenate prediction coordinates with decoder output
        decoder_input = torch.cat((preds.float(), representation_expand),2) # bs x _ x (512+40)
        dist_a, mu_a, sigma_a, dist_b, mu_b, sigma_b = decoder(decoder_input)
        log_prob_a = dist_a.log_prob(preds_gt[:,:,0:1])
        log_prob_b = dist_b.log_prob(preds_gt[:,:,1:2])

        loss = -log_prob_a.mean() - log_prob_b.mean()
        loss.backward()
        optimizer.step()
        _loss += loss.item()
        _loss_a += log_prob_a.mean().item()
        _loss_b += log_prob_b.mean().item()
        
        if i % args.vis_interval == 0:
            print("Iteration ", i)
            print("Loss: ", _loss/(i+1))
            print("Loss a: ", _loss_a/(i+1))
            print("Loss b: ", _loss_b/(i+1))

            visualize(gray_c, color_c, x_coords_pred, y_coords_pred, mu_a, mu_b, 
                      out_dir = train_output_dir + str(epoch) + "_" + str(i) + ".jpg")
            
    print("Final Training Loss Iteration ", i)
    print("Loss: ", _loss/(i+1))
    print("Loss a: ", _loss_a/(i+1))
    print("Loss b: ", _loss_b/(i+1))

    visualize(gray_c, color_c, x_coords_pred, y_coords_pred, mu_a, mu_b, 
              out_dir = train_output_dir + str(epoch) + "_" + str(i) + ".jpg")

    
def test(encoder, decoder, test_loader, test_output_dir):
    encoder.eval()
    decoder.eval()
    _loss, _loss_a, _loss_b = 0., 0., 0.
    with torch.no_grad():
        for i, (color_c, gray_c, obs, preds, preds_gt, (x_coords_obs, y_coords_obs), (x_coords_pred, y_coords_pred)) in enumerate(test_loader):
            obs, preds, preds_gt = obs.to(device), preds.to(device), preds_gt.to(device)

            representation = encoder(obs.float())
            representation_expand = representation.unsqueeze(1).expand(-1,preds.shape[1],512)
            # Concatenate prediction coordinates with decoder output
            decoder_input = torch.cat((preds.float(), representation_expand),2) # bs x _ x (512+40)
            dist_a, mu_a, sigma_a, dist_b, mu_b, sigma_b = decoder(decoder_input)
            log_prob_a = dist_a.log_prob(preds_gt[:,:,0:1])
            log_prob_b = dist_b.log_prob(preds_gt[:,:,1:2])

            loss = -log_prob_a.mean() - log_prob_b.mean()
            _loss += loss.item()
            _loss_a += log_prob_a.mean().item()
            _loss_b += log_prob_b.mean().item()
            visualize(gray_c, color_c, x_coords_pred, y_coords_pred, mu_a, mu_b, 
                      out_dir = test_output_dir + str(epoch) + "_" + str(i) + ".jpg")
        
    print("Validation Loss ")
    print("Loss: ", _loss/(i+1))
    print("Loss a: ", _loss_a/(i+1))
    print("Loss b: ", _loss_b/(i+1))
    
    encoder.train()
    decoder.train()
    
if __name__ == "__main__":
    basedir = 'pytorch_divcolor'
    listdir = 'data/imglist/lfw/'

    epochs = args.epochs
    batch_size = args.batch_size
    obs_num = args.obs_num
    shape = (32,32)
    lr = args.lr
    model_name = args.name
    
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    output_dir = os.path.join("outputs",model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("Results and models will be saved in ", output_dir)
        
    model_output_dir = os.path.join("outputs",model_name,"model/")
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)    
    train_output_dir = os.path.join("outputs",model_name,"train/")
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
    test_output_dir = os.path.join("outputs",model_name,"test/")
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)

    data_train = colordata(\
        shape = shape, \
        basedir=basedir,\
        listdir=listdir,\
        obs_num = obs_num,\
        split='train')

    train_loader = DataLoader(dataset=data_train, num_workers=3,
                             batch_size=batch_size, shuffle=True, drop_last=True)

    data_test = colordata(\
        shape = shape, \
        basedir=basedir,\
        listdir=listdir,\
        obs_num = obs_num,\
        split='test')

    test_loader = DataLoader(dataset=data_test, num_workers=3,
                             batch_size=batch_size, shuffle=False, drop_last=True)


    encoder_output_sizes = [128, 256, 512, 512]
    decoder_output_sizes = [512, 512, 256, 128, 4]

    encoder = get_encoder(42, encoder_output_sizes).to(device)
    decoder = get_decoder(512+41, decoder_output_sizes).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in range(epochs):
        print("Epoch ", epoch)
        train(encoder, decoder, train_loader, train_output_dir)
        test(encoder, decoder, train_loader, test_output_dir)
        torch.save(encoder.state_dict(), model_output_dir + 'encoder_'+ str(epoch) +'.pth')
        torch.save(decoder.state_dict(), model_output_dir + 'decoder_'+ str(epoch) +'.pth')
        print("Saved models")
        print("===")
