from h5py import File
import os
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import ball, dilation, erosion
import matplotlib.pyplot as plt

def make_gt(input_filepath, result_folder, key, dilate = 13, erode = 2):
    with File(input_filepath, mode = 'r') as f:
        gt = f[key][:]
    gt[gt>0]= 1


    kernel= np.array([1, -2, 1])
    boundaries_x = convolve(gt, kernel[:, None, None], mode = 'reflect')
    boundaries_y = convolve(gt, kernel[None, :, None], mode='reflect')
    boundaries_z = convolve(gt, kernel[None, None, :], mode='reflect')
    boundaries = boundaries_x + boundaries_y + boundaries_z
    #gt_cropped = gt[:,830:900,0:400]

    with File(os.path.join(result_folder, 'mem_gt.h5'), mode = 'w') as f:
        f.create_dataset('data', data = boundaries, compression = 'gzip')

    dilated = dilation((gt>0), selem = ball(dilate))

    eroded = erosion((gt>0), selem = ball(erode))

    #mem_gt_mask = np.bitwise_xor(dilated, eroded)
    mem_gt_mask = np.bitwise_xor(dilated, eroded)
    #mem_gt_mask[mem_gt_mask<0]=0


    with File(os.path.join(result_folder, 'mem_gt_mask.h5'), mode = 'w') as f:
        f.create_dataset('data', data = mem_gt_mask, compression='gzip')