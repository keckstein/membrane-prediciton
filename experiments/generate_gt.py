from functions_classes.gt_generation import make_gt
import h5py
import numpy as np
input_filepath = '/g/schwab/eckstein/gt/original_raw_gt_mask/MIB Tomo Annotation.model'
result_folder = '/g/schwab/eckstein/gt/data_dilated_mask_20/'

make_gt(input_filepath, result_folder, key = 'mibModel', dilate = 20, erode = 2)

#gt = h5py.File('/g/schwab/eckstein/gt/data_dilated_mask/mem_gt_mask.h5', mode = 'r')

#gt = np.array(gt['data'][:])
#gt = gt.astype('uint8')
#gt = gt.astype('uint8')
#print(gt.shape)
#print(type(gt))

#with h5py.File('/g/schwab/eckstein/gt/data_dilated_mask/mem_gt_mask_int.h5', mode ='w') as f:
    #f.create_dataset('data', data = gt, compression = 'gzip')












