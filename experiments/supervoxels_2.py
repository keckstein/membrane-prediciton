import sys
sys.path.append('/g/schwab/eckstein/code/membrane-prediciton/membranet_segmentation/scripts/')
import os
from run_supervoxels import run

target_folder = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/supervoxels/'
source_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_32_c013_result0150.h5'
target_filepath = f'{target_folder}c013_result0150.h5'
source_in_file = 'data'

run(source_filepath,target_filepath,source_in_file,verbose = False)

