import sys
sys.path.append('/g/schwab/eckstein/code/membrane-prediciton/membranet_segmentation/scripts/')
import os
from run_supervoxels import run


source_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/predictions/reshaped_prediction_32_c013_result_0150.h5'
target_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/cluster/piled_unet_14_run2_new_gt/supervoxels/c013_result_0150.h5'
source_in_file = 'data'
if not target_filepath:
    os.makedirs(target_filepath)

run(source_filepath,target_filepath,source_in_file,verbose = False)

