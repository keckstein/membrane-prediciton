import torch as t
from torchinfo import summary
# from network_fib_data import network
# from experiments.network_fib_data import network
from functions_classes.class_network import network
import h5py
from predict_model import predict_model_from_h5_parallel_generator
import sys

sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/pytorch_tools/')

model = network()
print(summary(model, (1, 1, 64, 64, 64)))

model_filepath = '/g/schwab/eckstein/code/models/unet3d_fib_sem/result0588.h5'

model.load_state_dict(t.load(model_filepath))

result_folder = '/g/schwab/eckstein/code/models/unet3d_fib_sem/result_prediction_0588.h5'

aug_dict_preprocessing = dict(smooth_output_sigma=0)

filepath_val = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/val_raw_512.h5'
#val = h5py.File(filepath_val, 'r')['data'][:]

im_list = ['/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/val_raw_512.h5']

with t.no_grad():
    for filepath in im_list:
        with h5py.File(filepath, mode ='r') as f:
            area_size = list(f['data'].shape)
            print(area_size)
            channels = [[f['data'][:]]]
            print(channels[0][0].shape)

        predict_model_from_h5_parallel_generator(
            model=model,
            results_filepath=result_folder,
            raw_channels=channels,
            spacing=(32, 32, 32),
            area_size=area_size,
            target_shape=(64, 64, 64),
            num_result_channels=1,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            n_workers=8,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            full_dataset_shape=None,
            write_in_and_out=False
        )
