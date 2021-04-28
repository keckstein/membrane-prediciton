import sys
sys.path.append('/g/schwab/eckstein/code/membrane-prediciton/membranet_segmentation/scripts/')
import os
from run_supervoxels import run

target_folder = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/supervoxels/'
source_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_raw_1_result0150.h5'
target_filepath = f'{target_folder}raw_1_result0150_3.h5'
source_in_file = 'data'

#if not os.path.exists(target_folder:
    #os.makedirs(target_folder)

run(source_filepath,target_filepath,source_in_file,verbose = False)

