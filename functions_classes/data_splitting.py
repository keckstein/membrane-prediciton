import h5py


filepath_raw_channels = '/g/schwab/eckstein/gt/original_raw_gt_mask/raw_image.h5'
filepath_gt_channels = '/g/schwab/eckstein/gt/raw_gt_mask_new_annotation/whole_cube/mem_gt.h5'
filepath_mask = '/g/schwab/eckstein/gt/raw_gt_mask_new_annotation/whole_cube/mem_gt_mask_int.h5'

#result_folder_raw ='/g/schwab/eckstein/gt/data_split/'
result_folder_gt = '/g/schwab/eckstein/gt/data_split/gt_split_annotation_2/'
result_folder_mask = '/g/schwab/eckstein/gt/data_split/mask_split_annotation_2/'

raw_data = h5py.File(filepath_raw_channels, 'r')['data']
gt_data = h5py.File(filepath_gt_channels, 'r')['data']
mask_data = h5py.File(filepath_mask, 'r')['data']

raw_data_list =[]
gt_data_list =[]
mask_data_list =[]

"""raw_data_list += [raw_data[0:91]]
raw_data_list += [raw_data[91:162]]
raw_data_list += [raw_data[162:243]]
raw_data_list += [raw_data[243:334]]
raw_data_list += [raw_data[334:430]]

"""
gt_data_list += [gt_data[0:91]]
gt_data_list += [gt_data[91:162]]
gt_data_list += [gt_data[162:243]]
gt_data_list += [gt_data[243:334]]
gt_data_list += [gt_data[334:430]]

mask_data_list += [mask_data[0:91]]
mask_data_list += [mask_data[91:162]]
mask_data_list += [mask_data[162:243]]
mask_data_list += [mask_data[243:334]]
mask_data_list += [mask_data[334:430]]

i=0
"""for i in range(0,len(raw_data_list)):
    with h5py.File(f'{result_folder_raw}raw_{i}.h5', mode='w') as f:
        f.create_dataset('data', data=raw_data_list[i], compression='gzip')

"""

for i in range(0,len(gt_data_list)):
    with h5py.File(f'{result_folder_gt}gt_{i}.h5', mode='w') as f:
        f.create_dataset('data', data=gt_data_list[i], compression='gzip')


for i in range(0,len(mask_data_list)):
    with h5py.File(f'{result_folder_mask}mask_{i}.h5', mode='w') as f:
        f.create_dataset('data', data=mask_data_list[i], compression='gzip')
