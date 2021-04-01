import numpy as np
import torch
import torch.nn as nn
# from torch import cat
import torch.optim as optim
import sys

# Choose the correct repo path
# sys.path.append('/Users/katharinaeckstein/Documents/source_code/pytorch_membrane_net/pytorch_tools/')
sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/pytorch_tools/')
from data_generation import parallel_data_generator

import h5py
# import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
import datetime
from functions_classes.loss_function import WeightMatrixWeightedBCELoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose correct data location
# filepath_raw_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image.h5'
# filepath_gt_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt.h5'
# filepath_raw_channels_test = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image_test_crop.h5'
# ilepath_gt_channels_test ='/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt_test_crop.h5'
#filepath_raw_channels = '/g/schwab/eckstein/gt/raw_image.h5'
#filepath_gt_channels = '/g/schwab/eckstein/gt/mem_gt.h5'
#filepath_mask = '/g/schwab/eckstein/gt/mem_gt_mask.h5'

filepath_raw_1='/g/schwab/eckstein/gt/data_split/raw_split/raw_0.h5'
filepath_raw_2='/g/schwab/eckstein/gt/data_split/raw_split/raw_1.h5'
filepath_raw_3='/g/schwab/eckstein/gt/data_split/raw_split/raw_2.h5'
filepath_raw_4='/g/schwab/eckstein/gt/data_split/raw_split/raw_3.h5'
filepath_raw_5='/g/schwab/eckstein/gt/data_split/raw_split/raw_4.h5'

filepath_gt_1='/g/schwab/eckstein/gt/data_split/gt_split/gt_0.h5'
filepath_gt_2='/g/schwab/eckstein/gt/data_split/gt_split/gt_1.h5'
filepath_gt_3='/g/schwab/eckstein/gt/data_split/gt_split/gt_2.h5'
filepath_gt_4='/g/schwab/eckstein/gt/data_split/gt_split/gt_3.h5'
filepath_gt_5='/g/schwab/eckstein/gt/data_split/gt_split/gt_4.h5'

filepath_gt_1='/g/schwab/eckstein/gt/data_split/gt_split/gt_0.h5'
filepath_gt_2='/g/schwab/eckstein/gt/data_split/gt_split/gt_1.h5'
filepath_gt_3='/g/schwab/eckstein/gt/data_split/gt_split/gt_2.h5'
filepath_gt_4='/g/schwab/eckstein/gt/data_split/gt_split/gt_3.h5'
filepath_gt_5='/g/schwab/eckstein/gt/data_split/gt_split/gt_4.h5'

filepath_mask_1='/g/schwab/eckstein/gt/data_split/mask_split/mask_0.h5'
filepath_mask_2='/g/schwab/eckstein/gt/data_split/mask_split/mask_1.h5'
filepath_mask_3='/g/schwab/eckstein/gt/data_split/mask_split/mask_2.h5'
filepath_mask_4='/g/schwab/eckstein/gt/data_split/mask_split/mask_3.h5'
filepath_mask_5='/g/schwab/eckstein/gt/data_split/mask_split/mask_4.h5'



raw_1 = h5py.File(filepath_raw_1, 'r')['data'][:]
raw_2 = h5py.File(filepath_raw_2, 'r')['data'][:]
raw_3 = h5py.File(filepath_raw_3, 'r')['data'][:]
raw_4 = h5py.File(filepath_raw_4, 'r')['data'][:]
raw_5 = h5py.File(filepath_raw_5, 'r')['data'][:]

gt_1 = h5py.File(filepath_gt_1, 'r')['data'][:]*255
gt_2 = h5py.File(filepath_gt_2, 'r')['data'][:]*255
gt_3 = h5py.File(filepath_gt_3, 'r')['data'][:]*255
gt_4 = h5py.File(filepath_gt_4, 'r')['data'][:]*255
gt_5 = h5py.File(filepath_gt_5, 'r')['data'][:]*255

mask_1 = h5py.File(filepath_mask_1, 'r')['data'][:]*255
mask_2 = h5py.File(filepath_mask_2, 'r')['data'][:]*255
mask_3 = h5py.File(filepath_mask_3, 'r')['data'][:]*255
mask_4 = h5py.File(filepath_mask_4, 'r')['data'][:]*255
mask_5 = h5py.File(filepath_mask_5, 'r')['data'][:]*255



print(f'raw1.shape = {raw_1.shape}')
print(f'raw2.shape = {raw_2.shape}')
print(f'raw3.shape = {raw_3.shape}')
print(f'raw4.shape = {raw_4.shape}')
print(f'raw5.shape = {raw_5.shape}')

train_gen = parallel_data_generator(
    raw_channels =[[raw_1],[raw_3],[raw_5]],
    gt_channels =[[gt_1, mask_1],[gt_3, mask_3],[gt_5, mask_5]],
    spacing=(64, 64, 64),  # (32, 32, 32),  For testing, I increased the grid spacing, speeds things up for now
    area_size=[raw_1.shape,raw_3.shape, raw_5.shape],
    # Can now be a tuple of a shape for each input volume        areas_and_spacings=None,
    target_shape=(64, 64, 64),
    gt_target_shape=(64, 64, 64),
    gt_target_channels=None,
    stop_after_epoch=False,
    aug_dict=dict(
        rotation_range=180,  # Angle in degrees
        shear_range=20,  # Angle in degrees
        zoom_range=[0.8, 1.2],  # [0.75, 1.5]
        horizontal_flip=True,
        vertical_flip=True,
        depth_flip=True,
        noise_var_range=1e-1,
        random_smooth_range=[0.6, 1.5],
        smooth_output_sigma=0,
        displace_slices_range=2,
        fill_mode='reflect',
        cval=0,
        brightness_range=92,
        contrast_range=(0.5, 2),
        transpose=True
    ),
    transform_ratio=0.9,
    batch_size=1,
    shuffle=True,
    add_pad_mask=False,
    noise_load_dict=None,
    n_workers=8,
    n_workers_noise=1,
    noise_on_channels=None,
    yield_epoch_info=True
)

val_gen = parallel_data_generator(
    raw_channels=[[raw_2],[raw_4]],
    gt_channels=[[gt_2, mask_2],[gt_4, mask_4]],
    spacing=(64, 64, 64),
    area_size= [raw_2.shape,raw_4.shape],
    target_shape=(64, 64, 64),
    gt_target_shape=(64, 64, 64),
    stop_after_epoch=False,
    aug_dict=dict(smooth_output_sigma=0),
    transform_ratio=0.,
    batch_size=1,
    shuffle=False,
    add_pad_mask=False,
    n_workers=8,
    gt_target_channels=None,
    yield_epoch_info=True
)


def conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    return conv


def double_down_conv(in_channels, out_channels, out_channels_2):
    conv_double_down = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels_2, kernel_size=3, padding=1),
        nn.BatchNorm3d(num_features=out_channels_2),
        nn.ReLU(inplace=True)
    )
    return conv_double_down


def double_up_conv(in_channels, out_channels):
    conv_double_up = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_double_up


class network(nn.Module):

    def __init__(self):
        super(network, self).__init__()
        # downsampling
        self.down_conv_1_2 = double_down_conv(1, 32, 64)
        # self.down_conv_2 = conv(32,64)
        self.max_pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_3_4 = double_down_conv(64, 64, 128)
        # self.down_conv_4 = conv(64, 128)
        self.max_pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_5_6 = double_down_conv(128, 128, 256)
        self.max_pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_7_8 = double_down_conv(256, 256, 512)

        # upsampling
        self.up_trans_1 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.up_conv_1_2 = double_up_conv(768, 256)
        # self.up_conv_2 = conv(64, 64)
        self.up_trans_2 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.up_conv_3_4 = double_up_conv(384, 128)
        self.up_trans_3 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.up_conv_5_6 = double_up_conv(192, 64)

        self.out = nn.Conv3d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            padding=0
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        # down
        x1 = self.down_conv_1_2(input)
        # print(x1.size())

        x2 = self.max_pool_1(x1)
        # print(x2.size())

        x3 = self.down_conv_3_4(x2)
        # print(x3.size())

        x4 = self.max_pool_2(x3)
        # print(x4.size())

        x5 = self.down_conv_5_6(x4)
        # print(x5.size())

        x6 = self.max_pool_3(x5)
        # print(x6.size())

        x7 = self.down_conv_7_8(x6)
        # print(x7.size())

        # up
        x8 = self.up_trans_1(x7)
        # print(x8.size())

        x9 = torch.cat([x8, x5], 1)
        # print(x9.size())

        x10 = self.up_conv_1_2(x9)
        # print(x10.size())

        x11 = self.up_trans_2(x10)
        # print(x11.size())

        x12 = torch.cat([x11, x3], 1)
        # print(x12.size())

        x13 = self.up_conv_3_4(x12)
        # print(x13.size())

        x14 = self.up_trans_3(x13)
        # print(x14.size())

        x15 = torch.cat([x14, x1], 1)
        # print(x15.size())

        x16 = self.up_conv_5_6(x15)
        # print(x16.size())

        x17 = self.out(x16)
        # print(x17.size())
        x18 = self.output_activation(x17)
        return x18


# model
network = network()
network.to(device)
# set model to train mode
network.train()

#tensorboard
#example_input = torch.rand(1, 1, 64, 64, 64)
writer = SummaryWriter('runs/figures/val_train_changed_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#writer.add_graph(network, example_input, verbose=True)  # graph with network structure, verbose = True prints result
#writer.flush()

# optimizer
optimizer = optim.Adam(network.parameters(), lr=0.001)
# define loss function
loss = WeightMatrixWeightedBCELoss([[0.5,0.5]])



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
            j=0
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
                        torch.save(network.state_dict(), f'/g/schwab/eckstein/code/models/unet3d_tomo/val_train_changed/result{epoch:04d}.h5')
                    break

writer.close()


# if __name__ == "__main__":
# input = torch.rand((1,1,21,21,21))
# model = network()
# print(model(input))
