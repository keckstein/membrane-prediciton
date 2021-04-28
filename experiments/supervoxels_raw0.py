import sys
sys.path.append('/g/schwab/eckstein/code/membrane-prediciton/membranet_segmentation/scripts/')
import os
import h5py
from run_supervoxels import run

source_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_c013_0_result0150.h5'
prediction = h5py.File(f'{source_filepath}', mode = 'r')['data'][0:96,0:1000,0:1000]

with h5py.File(f'/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_c013_0_result0150_crop.h5', mode='w') as f:
    f.create_dataset('data', data=prediction, compression='gzip')

target_folder = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/supervoxels/'
source_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_c013_0_result0150_crop.h5'
target_filepath = f'{target_folder}c013_0_result0150_crop.h5'
source_in_file = 'data'

#if not os.path.exists(target_folder:
    #os.makedirs(target_folder)

run(source_filepath,target_filepath,source_in_file,verbose = False)

