import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


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

def visualize(gray_c, color_c, x_coords_pred, y_coords_pred, mu_a, mu_b, image = 0, out_dir=None):
#     image = image # image index to visualize from mini-batch
    img_dim = gray_c[0].shape[1]
    
    img_lab = np.zeros((img_dim,img_dim,3), dtype='uint8') 
    img_lab[:,:,0] = decodeimg(gray_c[image].cpu().numpy().reshape((img_dim, img_dim)))
    img_lab[:,:,1] = decodeimg(color_c[image][0].cpu().numpy().reshape((img_dim, img_dim)))
    img_lab[:,:,2] = decodeimg(color_c[image][1].cpu().numpy().reshape((img_dim, img_dim)))

    gt = img_lab.copy()
    obs = img_lab

    img_lab[x_coords_pred[image],y_coords_pred[image],1:2] = decodepixels(mu_a[image].detach().cpu().numpy())
    img_lab[x_coords_pred[image],y_coords_pred[image],2:3] = decodepixels(mu_b[image].detach().cpu().numpy())

    orig_img = cv2.cvtColor(gt, cv2.COLOR_LAB2RGB)
    output_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    obs_img = cv2.cvtColor(obs, cv2.COLOR_LAB2RGB)

    obs_img[x_coords_pred[image], y_coords_pred[image]]=0

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
    