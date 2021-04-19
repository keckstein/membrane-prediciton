from functions_classes.gt_generation_fully_annotated_block_3 import make_gt_block
input_filepath = '/g/schwab/eckstein/gt/raw_gt_mask_new_annotation/MIB Tomo Annotation 3.model'
input_filepath_mask = '/g/schwab/eckstein/gt/raw_gt_mask_new_annotation/GT_block_3_mask.model'
result_folder = '/g/schwab/eckstein/gt/raw_gt_mask_new_annotation/block_3/no_mask_golgi/erode_5/'

make_gt_block(input_filepath, input_filepath_mask, result_folder, key = 'mibModel', erode = 5)

#gt = h5py.File('/g/schwab/eckstein/gt/data_dilated_mask/mem_gt_mask.h5', mode = 'r')

#gt = np.array(gt['data'][:])
#gt = gt.astype('uint8')
#gt = gt.astype('uint8')
#print(gt.shape)
#print(type(gt))

#with h5py.File('/g/schwab/eckstein/gt/data_dilated_mask/mem_gt_mask_int.h5', mode ='w') as f:
    #f.create_dataset('data', data = gt, compression = 'gzip')












