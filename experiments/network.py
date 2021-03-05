import numpy as np
import torch
import torch.nn as nn
from torch import cat
import torch.optim as optim
import sys
# Choose the correct repo path
sys.path.append('/Users/katharinaeckstein/Documents/source_code/pytorch_membrane_net/pytorch_tools/')
#sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/pytorch_tools/')
from data_generation import parallel_data_generator
import h5py

# Choose correct data location
filepath_raw_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image.h5'
filepath_gt_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt.h5'
#filepath_raw_channels_test = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image_test_crop.h5'
#ilepath_gt_channels_test ='/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt_test_crop.h5'
#filepath_raw_channels = '/g/schwab/Eckstein/gt/raw_image.h5'
#filepath_gt_channels = '/g/schwab/Eckstein/gt/mem_gt.h5'

raw_channels = [[h5py.File(filepath_raw_channels, 'r')['data']]]
# FIXME: The raw data's x and y axis are still swapped, should be fixed on the side of the data, then the swapaxes command becomes obsolete
gt_channels = [[h5py.File(filepath_gt_channels, 'r')['data']]]
print(f'raw.shape = {raw_channels[0][0].shape}')
print(f'gt.shape = {gt_channels[0][0].shape}')


train_gen = parallel_data_generator(
    raw_channels,
    gt_channels,
    spacing=(128, 128, 128),  # (32, 32, 32),  For testing, I increased the grid spacing, speeds things up for now
    area_size=raw_channels[0][0].shape,  # Can now be a tuple of a shape for each input volume
    areas_and_spacings=None,
    target_shape=(64, 64, 64),
    gt_target_shape=(64, 64, 64),
    gt_target_channels=None,
    stop_after_epoch=True,
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
    batch_size=2,
    shuffle=True,
    add_pad_mask=False,
    noise_load_dict=None,
    n_workers=1,
    n_workers_noise=1,
    noise_on_channels=None,
    yield_epoch_info=True
)


def conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm3d(num_features = out_channels),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:,:, delta:tensor_size - delta,delta:tensor_size - delta,delta:tensor_size - delta]

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        #downsampling
        self.down_conv_1 = conv(1, 32)
        self.down_conv_2 = conv(32,64)
        self.max_pool_1 = nn.MaxPool3d(kernel_size = 2, stride =2)
        self.down_conv_3 = conv(64,64)
        self.down_conv_4 = conv(64, 128)

        #upsampling
        self.up_trans_1 = nn.ConvTranspose3d(in_channels = 128, out_channels = 128, kernel_size =2, stride=2, padding =1)
        self.up_conv_5 = conv(192, 64)
        self.up_conv_6 = conv(64, 64)

        self.out = nn.Conv3d(
                in_channels = 64,
                out_channels=1,
                kernel_size=1,
                padding =1
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        x1 = self.down_conv_1(input)
        print(x1.size())
        x2 = self.down_conv_2(x1)
        print(x2.size())
        x3 = self.max_pool_1(x2)
        print(x3.size())
        x4 = self.down_conv_3(x3)
        print(x4.size())
        x5 = self.down_conv_4(x4)
        print(x5.size())
        x6 = self.up_trans_1(x5)
        print(x6.size())
        x2 = crop_img(x2, x6)
        x7 = self.up_conv_5(torch.cat([x6,x2],1))
        print(x7.size())
        x8 = self.up_conv_6(x7)
        print(x8.size())

        x9 = self.out(x8)
        print(x9.size())
        x9 = self.output_activation(x9)
        return x9


#Optimizer
network = network()
network.train()
optimizer = optim.Adam(network.parameters(), lr=0.001)
#optimizer = optim.SGD(network.parameters(), lr=0.01)

n_epochs =5
for x, y, epoch, n, loe in train_gen:
# in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    x = torch.tensor(np.moveaxis(x, 4, 1), dtype=torch.float32)
    y = torch.tensor(np.moveaxis(y, 4, 1), dtype=torch.float32)
    output = network(x)
    loss = nn.BCELoss()
    loss = loss(output, y)
    loss.backward()
    optimizer.step()

    if epoch+1 == n_epochs:
        break


    print(f'Current epoch: {epoch}')
    print(f'Iteration within epoch: {n}')
    print(f'Is last iteration of this epoch: {loe}')
    print(f'x.shape = {x.shape}')
    print(f'y.shape = {y.shape}')
    break




#if __name__ == "__main__":
#input = torch.rand((1,1,21,21,21))
#model = network()
#print(model(input))
