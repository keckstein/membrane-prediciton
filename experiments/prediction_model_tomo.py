import torch as t
from torchinfo import summary
# from network_fib_data import network
# from experiments.network_fib_data import network
from functions_classes.class_network import network
import h5py
from predict_model import predict_model_from_h5_parallel_generator
import sys
import os

sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/')
from pytorch_tools.piled_unets import PiledUnet
model = PiledUnet(n_nets = 3,
                    in_channels=1,
                    out_channels=[1,1,1],
                    filter_sizes_down =(
                        ((8, 16), (16, 32), (32, 64)),
                        ((8, 16), (16, 32), (32, 64)),
                        ((8, 16), (16, 32), (32, 64))
                    ),
                    filter_sizes_bottleneck=(
                        (64, 128),
                        (64, 128),
                        (64, 128)
                    ),
                    filter_sizes_up = (
                        ((64, 64), (32, 32), (16, 16)),
                        ((64, 64), (32, 32), (16, 16)),
                        ((64, 64), (32, 32), (16, 16))
                    ),
                    batch_norm=True,
                    output_activation='sigmoid',
                    predict = True
                   )

print(summary(model, (1, 1, 64, 64, 64)))

model_filepath = '/g/schwab/eckstein/code/models/unet3d_tomo/piled_unet_1/result0283.h5'

model.load_state_dict(t.load(model_filepath))

result_folder = '/g/schwab/eckstein/code/models/unet3d_tomo/piled_unet_1/'
raw_data_folder = '/g/schwab/eckstein/gt/data_split/raw_split'


aug_dict_preprocessing = dict(smooth_output_sigma=0)

#val = h5py.File(filepath_val, 'r')['data'][:]

im_list = ['raw_1.h5','raw_3.h5']

with t.no_grad():
    for filename in im_list:
        with h5py.File(os.path.join(raw_data_folder,filename), mode ='r') as f:
            area_size = list(f['data'].shape)
            channels = [[f['data'][:]]]

        predict_model_from_h5_parallel_generator(
            model=model,
            results_filepath= f'{result_folder}result_prediction_0283_{filename}',
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
