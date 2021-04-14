import numpy as np
import torch
import torch.nn as nn
# from torch import cat
import torch.optim as optim
import sys

# Choose the correct repo path
# sys.path.append('/home/eckstein/code/pytorch_membrane_net/')
sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/')
from pytorch_tools.piled_unets import PiledUnet
from pytorch_tools.data_generation import parallel_data_generator
from pytorch_tools.losses import CombinedLosses
import h5py
# import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
import datetime
from pytorch_tools.losses import WeightMatrixWeightedBCE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose correct data location
# filepath_raw_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image.h5'
# filepath_gt_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt.h5'
# filepath_raw_channels_test = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image_test_crop.h5'
# ilepath_gt_channels_test ='/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt_test_crop.h5'
# filepath_raw_channels = '/g/schwab/eckstein/gt/raw_image.h5'
# filepath_gt_channels = '/g/schwab/eckstein/gt/mem_gt.h5'
# filepath_mask = '/g/schwab/eckstein/gt/mem_gt_mask.h5'
input_filepath_data = '/scratch/eckstein/input_data/gt/data_split/'
# filepath_raw_1= f'{input_filepath_data}raw_split/raw_0.h5'
# filepath_raw_2=f'{input_filepath_data}raw_split/raw_1.h5'
filepath_raw_3 = f'{input_filepath_data}raw_split/raw_2.h5'
# filepath_raw_4=f'{input_filepath_data}raw_split/raw_3.h5'
# filepath_raw_5=f'{input_filepath_data}raw_split/raw_4.h5'

# filepath_gt_1=f'{input_filepath_data}gt_split/gt_0.h5'
# filepath_gt_2=f'{input_filepath_data}gt_split/gt_1.h5'
filepath_gt_3 = f'/scratch/eckstein/input_data/gt/raw_gt_mask_new_annotation/block_3/erode_13/mem_gt_block_3.h5'
# filepath_gt_4=f'{input_filepath_data}gt_split/gt_3.h5'
# filepath_gt_5=f'{input_filepath_data}gt_split/gt_4.h5'

# filepath_mask_1=f'{input_filepath_data}mask_split_dilate_13/mask_0.h5'
# filepath_mask_2=f'{input_filepath_data}mask_split_dilate_13/mask_1.h5'
filepath_mask_3 = f'/scratch/eckstein/input_data/gt/raw_gt_mask_new_annotation/block_3/no_mask_golgi/erode_3/mem_gt_mask_block_3.h5'
# filepath_mask_4=f'{input_filepath_data}mask_split_dilate_13/mask_3.h5'
# filepath_mask_5=f'{input_filepath_data}mask_split_dilate_13/mask_4.h5'


# raw_1 = h5py.File(filepath_raw_1, 'r')['data'][:]
# raw_2 = h5py.File(filepath_raw_2, 'r')['data'][:]
raw_3 = h5py.File(filepath_raw_3, 'r')['data'][:]
# raw_4 = h5py.File(filepath_raw_4, 'r')['data'][:]
# raw_5 = h5py.File(filepath_raw_5, 'r')['data'][:]

# gt_1 = h5py.File(filepath_gt_1, 'r')['data'][:]*255
# gt_2 = h5py.File(filepath_gt_2, 'r')['data'][:]*255
gt_3 = h5py.File(filepath_gt_3, 'r')['data'][:] * 255
# gt_4 = h5py.File(filepath_gt_4, 'r')['data'][:]*255
# gt_5 = h5py.File(filepath_gt_5, 'r')['data'][:]*255

# mask_1 = h5py.File(filepath_mask_1, 'r')['data'][:]*255
# mask_2 = h5py.File(filepath_mask_2, 'r')['data'][:]*255
mask_3 = h5py.File(filepath_mask_3, 'r')['data'][:] * 255
# mask_4 = h5py.File(filepath_mask_4, 'r')['data'][:]*255
# mask_5 = h5py.File(filepath_mask_5, 'r')['data'][:]*255


# print(f'raw1.shape = {raw_1.shape}')
# print(f'raw2.shape = {raw_2.shape}')
# print(f'raw3.shape = {raw_3.shape}')
# print(f'raw4.shape = {raw_4.shape}')
# print(f'raw5.shape = {raw_5.shape}')

train_gen = parallel_data_generator(
    raw_channels=[[raw_3[:, :, 357:772]]],
    gt_channels=[[gt_3[:, :, 357:772], mask_3[:, :, 357:772]]],
    spacing=(64, 64, 64),  # (32, 32, 32),  For testing, I increased the grid spacing, speeds things up for now
    area_size=[(raw_3[:, :, 357:772]).shape],
    # Can now be a tuple of a shape for each input volume        areas_and_spacings=None,
    target_shape=(64, 64, 64),
    gt_target_shape=(64, 64, 64),
    gt_target_channels=None,
    stop_after_epoch=False,
    aug_dict=dict(
        rotation_range=180,
        shear_range=20,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        depth_flip=True,
        noise_var_range=1e-1,  # test
        random_smooth_range=[0,0],
        smooth_output_sigma=0,
        displace_slices_range=0,
        fill_mode='reflect',
        cval=0,
        brightness_range=64,  # test
        contrast_range=(0.9, 1.2), # test
        transpose = False
    ),
    transform_ratio=0.9,
    batch_size=1,
    shuffle=True,
    add_pad_mask=False,
    noise_load_dict=None,
    n_workers=2,
    n_workers_noise=1,
    noise_on_channels=None,
    yield_epoch_info=True
)

val_gen = parallel_data_generator(
    raw_channels=[[raw_3[:, :, 0:357]]],
    gt_channels=[[gt_3[:, :, 0:357], mask_3[:, :, 0:357]]],
    spacing=(64, 64, 64),
    area_size=[(raw_3[:, :, 0:357]).shape],
    target_shape=(64, 64, 64),
    gt_target_shape=(64, 64, 64),
    stop_after_epoch=False,
    aug_dict=dict(smooth_output_sigma=0),
    transform_ratio=0.,
    batch_size=1,
    shuffle=False,
    add_pad_mask=False,
    n_workers=2,
    gt_target_channels=None,
    yield_epoch_info=True
)

# model
"""network = PiledUnet(n_nets = 3,
                    in_channels=1,
                    out_channels=[1,1,1],
                    filter_sizes_down =(
                        ((4, 8), (8, 16), (16, 32)),
                        ((8, 16), (16, 32), (32, 64)),
                        ((32, 64), (64, 128), (128, 256))
                    ),
                    filter_sizes_bottleneck=(
                        (32, 64),
                        (64, 128),
                        (256, 512)
                    ),
                    filter_sizes_up = (
                        ((32, 32), (16, 16), (8, 8)),
                        ((64, 64), (32, 32), (16, 16)),
                        ((256, 256), (128, 128), (64, 64))
                    ),
                    batch_norm=None,
                    output_activation='sigmoid',
                    predict = False)"""

network = PiledUnet(n_nets=3,
                    in_channels=1,
                    out_channels=[1, 1, 1],
                    filter_sizes_down=(
                        ((8, 16), (16, 32), (32, 64)),
                        ((8, 16), (16, 32), (32, 64)),
                        ((8, 16), (16, 32), (32, 64))
                    ),
                    filter_sizes_bottleneck=(
                        (64, 128),
                        (64, 128),
                        (64, 128)
                    ),
                    filter_sizes_up=(
                        ((64, 64), (32, 32), (16, 16)),
                        ((64, 64), (32, 32), (16, 16)),
                        ((64, 64), (32, 32), (16, 16))
                    ),
                    batch_norm=True,
                    output_activation='sigmoid',
                    )

network.to(device)
# set model to train mode
network.train()

# tensorboard
# example_input = torch.rand(1, 1, 64, 64, 64)
writer = SummaryWriter(
    '/g/schwab/eckstein/scripts/tensorboard/piled_unet_20_run1' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# writer.add_graph(network, example_input, verbose=True)  # graph with network structure, verbose = True prints result
# writer.flush()

# optimizer
optimizer = optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-7)
# define loss function

loss = CombinedLosses(losses=(
    WeightMatrixWeightedBCE(((0.2, 0.8),), weigh_with_matrix_sum=False),
    WeightMatrixWeightedBCE(((0.4, 0.6),), weigh_with_matrix_sum=False),
    WeightMatrixWeightedBCE(((0.5, 0.5),), weigh_with_matrix_sum=False)),
    y_pred_channels=(np.s_[:1], np.s_[1:2], np.s_[2:3]),
    y_true_channels=(np.s_[:], np.s_[:], np.s_[:]),
    weigh_losses=np.array([0.2, 0.3, 0.5])
)

sum_train_loss = 0
best_val_loss = None

i = 0
# training loop
for x, y, epoch, n, loe in train_gen:
    # network.train()
    # in your training loop:
    # optimizer.zero_grad()  # zero the gradient buffers
    network.train()
    optimizer.zero_grad()
    x = torch.tensor(np.moveaxis(x, 4, 1), dtype=torch.float32).to(device)
    y = torch.tensor(np.moveaxis(y, 4, 1), dtype=torch.float32).to(device)

    if y[0, 1, :].cpu().detach().numpy().max():
        i += 1

        # with h5py.File(f'/g/schwab/eckstein/train_data/x_iteration{epoch}_{n}.h5', mode='w') as f:
        # f.create_dataset('data', data=x[0][0], compression='gzip')

        # with h5py.File(f'/g/schwab/eckstein/train_data/y_iteration{epoch}_{n}.h5', mode='w') as f:
        # f.create_dataset('data', data=y[0][0], compression='gzip')

        # with h5py.File(f'/g/schwab/eckstein/train_data/mask_iteration{epoch}_{n}.h5', mode='w') as f:
        # f.create_dataset('data', data=y[0][1], compression='gzip')

        output = network(x)

        train_loss = loss(output, y)

        sum_train_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()

        print('Train loss for iteration: ', train_loss)
        print('Total train loss divided by number of iterations:', (sum_train_loss / i))
        print(f'Current epoch: {epoch}')
        print(f'Iteration within epoch: {n}')
        print(f'Is last iteration of this epoch: {loe}')
        # print(f'x.shape = {x.shape}')
        # print(f'y.shape = {y.shape}')

    # validation
    if loe:
        # plot train loss for epoch
        train_loss_epoch = (sum_train_loss / i)
        # train_loss_epoch = (sum_train_loss / (n+1))
        writer.add_scalar('train_loss', train_loss_epoch, epoch)
        writer.flush()
        print('Train loss for epoch: ', train_loss_epoch)
        sum_train_loss = 0
        i = 0

        with torch.no_grad():
            network.eval()
            sum_loss = 0
            val_loss = 0
            acc = 0
            val_acc = 0
            val_output = 0
            j = 0
            for x_val, y_val, val_epoch, val_n, val_loe in val_gen:
                x_val = torch.tensor(np.moveaxis(x_val, 4, 1), dtype=torch.float32).to(device)
                val_output = network(x_val)
                y_val = torch.tensor(np.moveaxis(y_val, 4, 1), dtype=torch.float32).to(device)

                if y_val[0, 1, :].cpu().detach().numpy().max():
                    j += 1
                    # with h5py.File(f'/g/schwab/eckstein/val_output/y_val_iteration{epoch}{val_n}.h5', mode ='w') as f:
                    # f.create_dataset('data', data = y_val[0][0], compression ='gzip' )

                    # with h5py.File(f'/g/schwab/eckstein/val_output/val_mask_iteration{epoch}{val_n}.h5', mode ='w') as f:
                    # f.create_dataset('data', data = y_val[0][1], compression ='gzip' )

                    # with h5py.File(f'/g/schwab/eckstein/val_output/val_output_iteration{epoch}{val_n}.h5', mode ='w') as f:
                    # f.create_dataset('data', data = val_output[0][0], compression ='gzip' )

                    # compute loss
                    validation_loss = loss(val_output, y_val)

                    # print(f'Validation loss for iteration {j}: ', validation_loss)
                    sum_loss += validation_loss.item()
                    # print('Total validation loss divided by number of iterations:', (sum_loss / j))

                    # compute accuracy
                    total_n = 262144
                    correct_n = torch.count_nonzero(((val_output[0, 0, :] > 0.5) == (y_val[0, 0, :] == 1)))

                    # correct_n = torch.sum(((val_output[0,0,:] > 0.5) == (y_val[0,0,:] == 1))).float()

                    # print('Correctly predicted: ', correct_n)
                    acc += (correct_n.item() / total_n)
                # print(correct_n.item() / total_n)
                # print(acc)

                # compute validation loss
                # if not val_loe:
                #   continue

                if val_loe:

                    print(val_n + 1)
                    print(val_epoch)
                    val_loss = sum_loss / j
                    print('Validation loss: ', val_loss)
                    # compute accuracy
                    val_acc = acc / j
                    print('Validation accuracy: ', val_acc)

                    writer.add_scalar('val_accuracy', val_acc, val_epoch)
                    writer.flush()
                    writer.add_scalar('val_loss', val_loss, val_epoch)
                    writer.flush()
                    val_acc = 0
                    acc = 0
                    # save model if val_loss is improved
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(network.state_dict(), f'/scratch/eckstein/models/piled_unet_20_run1/result{epoch:04d}.h5')
                    break

writer.close()

# if __name__ == "__main__":
# input = torch.rand((1,1,21,21,21))
# model = network()
# print(model(input))
