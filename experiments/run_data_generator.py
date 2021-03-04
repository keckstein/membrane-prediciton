import sys
# Choose the correct repo path
# sys.path.append('/Users/katharinaeckstein/Documents/EMBL/Files/pytorch_membrane_net/pytorch_tools/')
sys.path.append('/g/schwab/hennies/src/github/pytorch_membrane_net/pytorch_tools/')
from data_generation import parallel_data_generator
import h5py

# Choose correct data location
# filepath_raw_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/raw_image.h5'
# filepath_gt_channels = '/Users/katharinaeckstein/Documents/EMBL/Files/mem_gt.h5'
filepath_raw_channels = '/g/schwab/Eckstein/gt/raw_image.h5'
filepath_gt_channels = '/g/schwab/Eckstein/gt/mem_gt.h5'

raw_channels = [[h5py.File(filepath_raw_channels, 'r')['data']]]
# FIXME: The raw data's x and y axis are still swapped, should be fixed on the side of the data, then the swapaxes command becomes obsolete
gt_channels = [[h5py.File(filepath_gt_channels, 'r')['data'][:].swapaxes(1, 2)]]
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

for x, y, epoch, n, loe in train_gen:

    print(f'Current epoch: {epoch}')
    print(f'Iteration within epoch: {n}')
    print(f'Is last iteration of this epoch: {loe}')
    print(f'x.shape = {x.shape}')
    print(f'y.shape = {y.shape}')
    break
