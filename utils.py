import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn.functional as F

def get_gradient_loss(mu_a, mu_b, preds_gt, device=0):
    x = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]).float().to(device)

    y = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]).float().to(device)

    mu_reshape = torch.cat((mu_a.reshape(-1, 1, 32,32),mu_b.reshape(-1, 1, 32,32)),1)
    preds_gt_reshape = preds_gt.reshape(-1,2,32,32).float()

    x = x.view((1,1,3,3)).expand((1,2,3,3))
    G_x_pred = F.conv2d(mu_reshape, x)
    G_x_gt = F.conv2d(preds_gt_reshape.to(device) , x)

    y = y.view((1,1,3,3)).expand((1,2,3,3))
    G_y_pred = F.conv2d(mu_reshape, y)
    G_y_gt = F.conv2d(preds_gt_reshape.to(device), y)

    criterion = torch.nn.MSELoss()
    gradient_loss = criterion(G_x_pred, G_x_gt) + criterion(G_y_pred, G_y_gt)

    return gradient_loss

def get_l2_loss(mu_a, mu_b, preds_gt):
    mu_reshape = torch.cat((mu_a.reshape(-1, 1, 32,32),mu_b.reshape(-1, 1, 32,32)),1)
    preds_gt_reshape = preds_gt.reshape(-1,2,32,32).float()

    criterion = torch.nn.MSELoss()
    l2_loss = criterion(mu_reshape, preds_gt_reshape)
    return l2_loss

def print_statement(loss, log_loss, gradient_loss = None, l2_loss = None,):
    print_str = "Loss: {}, log loss: {}".format(round(loss,5), round(log_loss,5))
    if l2_loss is not None:
        print_str += ", l2 loss: {}".format(round(l2_loss,5))
    if gradient_loss is not None:
        print_str += ", grad loss: {}".format(round(gradient_loss,5))
    print(print_str)
    
def decodeimg(img_enc):
    img_dec = (((img_enc+1.)*1.)/2.)*255.
    img_dec[img_dec < 0.] = 0.
    img_dec[img_dec > 255.] = 255.
    return cv2.resize(np.uint8(img_dec), (32,32))

def decodepixels(pixel_enc):
    img_dec = (((pixel_enc+1.)*1.)/2.)*255.
    img_dec[img_dec < 0.] = 0.
    img_dec[img_dec > 255.] = 255.
    return np.uint8(img_dec)

def visualize(gray_c, color_c, x_coords_obs, y_coords_obs, x_coords_pred, y_coords_pred, mu_a, mu_b, image = 0, out_dir=None):
#     image = image # image index to visualize from mini-batch
    img_dim = gray_c[0].shape[1]
    
    img_lab = np.zeros((img_dim,img_dim,3), dtype='uint8') 
    img_lab[:,:,0] = decodeimg(gray_c[image].cpu().numpy().reshape((img_dim, img_dim)))
    img_lab[:,:,1] = decodeimg(color_c[image][0].cpu().numpy().reshape((img_dim, img_dim)))
    img_lab[:,:,2] = decodeimg(color_c[image][1].cpu().numpy().reshape((img_dim, img_dim)))

    gt = img_lab.copy()

    img_lab[x_coords_pred[image],y_coords_pred[image],1:2] = decodepixels(mu_a[image].detach().cpu().numpy())
    img_lab[x_coords_pred[image],y_coords_pred[image],2:3] = decodepixels(mu_b[image].detach().cpu().numpy())

    orig_img = cv2.cvtColor(gt, cv2.COLOR_LAB2RGB)
    output_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    obs_img = np.zeros((img_dim,img_dim,3), dtype='uint8')
    obs_img[x_coords_obs, y_coords_obs] = orig_img[x_coords_obs, y_coords_obs]

    plt.subplot(131)
    plt.axis('off'), plt.title("Observation Image")
    plt.imshow(obs_img)
    plt.subplot(132)
    plt.axis('off'), plt.title("Output")
    plt.imshow(output_img)
    plt.subplot(133)
    plt.axis('off'), plt.title("Original GT Image")
    plt.imshow(orig_img)
    if out_dir is not None:
        plt.savefig(out_dir)
    plt.show()
    