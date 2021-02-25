
from h5py import File
import os
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import ball, dilation, erosion


def make_gt(input_filepath, result_folder, key, dilate=10, erode=2):
    """
    If you want to load a MIB *.model file use key='mibModel'

    :param input_filepath: hdf5 container
    :param result_folder: folder for resuts
    :param key: path within file
    :param dilate:
    :param erode:
    :return:
    """

    with File(input_filepath, mode='r') as f:
        gt = f[key][:]

    # Convolve with edge detection
    kernel = np.array([1, -2, 1])
    boundaries_x = convolve(gt, kernel[:, None, None], mode='reflect')
    boundaries_y = convolve(gt, kernel[None, :, None], mode='reflect')
    boundaries_z = convolve(gt, kernel[None, None, :], mode='reflect')
    boundaries = boundaries_x + boundaries_y + boundaries_z
    boundaries[boundaries > 0] = 1

    # Write to file
    with File(os.path.join(result_folder, 'mem_gt.h5'), mode='w') as f:
        f.create_dataset('data', data=boundaries, compression='gzip')

    # TODO: generate the mask by dilating the membrane prediction and eroding the binarized object map
    dilated = dilation(boundaries, selem=ball(dilate))
    eroded = erosion((gt > 0).astype('uint8') * 255, selem=ball(erode))

    ...

    mem_gt_mask = ...

    # Write to file
    with File(os.path.join(result_folder, 'mem_gt_mask.h5'), mode='w') as f:
        f.create_dataset('data', data=mem_gt_mask, compression='gzip')


