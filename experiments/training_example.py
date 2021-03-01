import sys

sys.path.append('/g/schwab/hennies/src/cnns_for_image_segmentation/cnns_for_image_segmentation/')

from pytorch.pytorch_tools.data_generation import parallel_data_generator
import os
from h5py import File
from pytorch.pytorch_tools.piled_unets import PiledUnet
from pytorch.pytorch_tools.losses import WeightMatrixWeightedBCE, CombinedLosses
# from pytorch.pytorch_tools.training import cb_save_model, cb_run_model_on_data, Trainer
from pytorch.pytorch_tools.train import Trainer
from pytorch.pytorch_tools.train import CBValLoss, CBValLossWriter, CBTrainLossWriter, CBSaveModel
from pytorch.pytorch_tools.train import CBPredict, CBValLossDetectLocalMin, CBLoeWriter
import torch as t

from torchsummary import summary
import numpy as np

n_workers = 3

experiment_name = 'unet3d_200417_10a_fov64_con_in2_dec3_transpose_zjit2_mask_gt_boundary'
results_folder = os.path.join(
    '/g/schwab/hennies/phd_project/image_analysis/autoseg/cnn_3d_devel',
    'unet3d_200417_fixed_zjitter_bug',
    experiment_name
)

if True:

    raw_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    raw_filepaths = [
        [
            os.path.join(raw_path, 'raw.h5'),
        ],
    ]
    gt_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    gt_filepaths = [
        [
            os.path.join(gt_path, 'gt_mem.h5'),
            os.path.join(gt_path, 'gt_mask_organelle_insides_erosion3_boudary3.h5')
        ],
    ]
    raw_channels = []
    for volumes in raw_filepaths:
        raws_data = []
        for chid, channel in enumerate(volumes):
            if chid == 1:
                # Specifically only load last channel of the membrane prediction
                raws_data.append(File(channel, 'r')['data'][..., -1])
            else:
                raws_data.append(File(channel, 'r')['data'][:])
        raw_channels.append(raws_data)
    gt_channels = [[File(channel, 'r')['data'][:] for channel in volumes] for volumes in gt_filepaths]

    val_raw_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    val_raw_filepaths = [
        [
            os.path.join(val_raw_path, 'val_raw_512.h5'),
        ],
        [
            os.path.join(
                '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/psp_200107_00_ds_20141002_hela_wt_xyz8nm_as_multiple_scales/step0_datasets/psp0_200108_02_select_test_and_val_cubes',
                'val0_x1390_y742_z345_pad.h5'

            )
        ]
    ]
    val_gt_path = '/g/schwab/hennies/phd_project/image_analysis/psp_full_experiments/boundary_raw_and_gt/'
    val_gt_filepaths = [
        [
            os.path.join(val_gt_path, 'val_gt_mem.h5'),
            os.path.join(val_gt_path, 'val_gt_mask_organelle_insides_erosion3.h5')
        ]
    ]
    val_raw_channels = []
    for volumes in val_raw_filepaths:
        val_raws_data = []
        for chid, channel in enumerate(volumes):
            if chid == 1:
                # Specifically only load last channel of the membrane prediction
                val_raws_data.append(File(channel, 'r')['data'][..., -1])
            else:
                val_raws_data.append(File(channel, 'r')['data'][:])
        val_raw_channels.append(val_raws_data)
    val_gt_channels = [[File(channel, 'r')['data'][:] for channel in volumes] for volumes in val_gt_filepaths]

if True:
    pre_smoothing = 0

    data_gen_args = dict(
        rotation_range=180,  # Angle in degrees
        shear_range=20,  # Angle in degrees
        zoom_range=[0.8, 1.2],  # [0.75, 1.5]
        horizontal_flip=True,
        vertical_flip=True,
        depth_flip=True,
        noise_var_range=1e-1,
        random_smooth_range=[0.6, 1.5],
        smooth_output_sigma=pre_smoothing,
        displace_slices_range=2,
        fill_mode='reflect',
        cval=0,
        brightness_range=92,
        contrast_range=dict(
            increase_ratio=(2 - 1) / (3 - 1),
            increase=2,
            decrease=3
        ),
        transpose=False
    )

    aug_dict_preprocessing = dict(
        smooth_output_sigma=pre_smoothing
    )

    train_gen = parallel_data_generator(
        raw_channels=raw_channels,
        gt_channels=gt_channels,
        spacing=(32, 32, 32),
        area_size=(32, 512, 512),
        # area_size=(32, 64, 64),
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        stop_after_epoch=False,
        aug_dict=data_gen_args,
        transform_ratio=0.9,
        batch_size=2,
        shuffle=True,
        add_pad_mask=False,
        n_workers=n_workers,
        noise_load_dict=None,
        gt_target_channels=None,
        areas_and_spacings=None,
        n_workers_noise=n_workers,
        noise_on_channels=None,
        yield_epoch_info=True
    )

    val_gen = parallel_data_generator(
        raw_channels=val_raw_channels[:1],
        gt_channels=val_gt_channels,
        spacing=(64, 64, 64),
        area_size=(256, 256, 256),
        target_shape=(64, 64, 64),
        gt_target_shape=(64, 64, 64),
        stop_after_epoch=False,
        aug_dict=aug_dict_preprocessing,
        transform_ratio=0.,
        batch_size=1,
        shuffle=False,
        add_pad_mask=False,
        n_workers=n_workers,
        gt_target_channels=None,
        yield_epoch_info=True
    )

model = PiledUnet(
    n_nets=3,
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
    output_activation='sigmoid'
)

model.cuda()
summary(model, (1, 96, 96, 96))

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# val_loss_func = WeightMatrixWeightedBCE(((0.3, 0.7),), weigh_with_matrix_sum=True)
# val_channels = np.s_[:, -1:, :]
min_save_epoch = 16

loss_func = CombinedLosses(
    losses=(
        WeightMatrixWeightedBCE(((0.1, 0.9),), weigh_with_matrix_sum=False),
        WeightMatrixWeightedBCE(((0.2, 0.8),), weigh_with_matrix_sum=False),
        WeightMatrixWeightedBCE(((0.3, 0.7),), weigh_with_matrix_sum=False)),
    y_pred_channels=(np.s_[:1], np.s_[1:2], np.s_[2:3]),
    y_true_channels=(np.s_[:], np.s_[:], np.s_[:]),
    weigh_losses=np.array([0.2, 0.3, 0.5])
)

trainer = Trainer(
    model,
    train_gen,
    n_epochs=200,
    loss_func=loss_func,
    optimizer=t.optim.Adam(model.parameters(),
                           lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-7),
    callbacks=[
        CBValLoss(
            loss_func, val_gen, y_channels=np.s_[:]
        ),
        CBValLossDetectLocalMin(
            field_of_view=8
        ),
        CBSaveModel(
            os.path.join(results_folder, 'model_{epoch:04d}.h5'),
            on_improvement=True,
            on_trainer_attribute='model_improved_local_min',
            min_epoch=min_save_epoch
        ),
        CBTrainLossWriter(os.path.join(results_folder, 'tb')),
        CBValLossWriter(os.path.join(results_folder, 'tb')),
        CBPredict(
            results_filepath=os.path.join(results_folder, 'improved_result1_{epoch:04d}.h5'),
            raw_channels=val_raw_channels[:1],
            spacing=(32, 32, 32),
            area_size=(64, 256, 256),
            target_shape=(64, 64, 64),
            num_result_channels=3,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            full_dataset_shape=None,
            n_workers=n_workers,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            on_improvement=True,
            on_trainer_attribute='model_improved_local_min',
            accept_existing_result=False,
            min_epoch=min_save_epoch
        ),
        CBPredict(
            results_filepath=os.path.join(results_folder, 'improved_result2_{epoch:04d}.h5'),
            raw_channels=val_raw_channels[1:],
            spacing=(32, 32, 32),
            area_size=(64, 256, 256),
            target_shape=(64, 64, 64),
            num_result_channels=3,
            smooth_output_sigma=aug_dict_preprocessing['smooth_output_sigma'],
            full_dataset_shape=None,
            n_workers=n_workers,
            compute_empty_volumes=True,
            thresh=None,
            write_at_area=False,
            offset=None,
            on_improvement=True,
            on_trainer_attribute='model_improved_local_min',
            accept_existing_result=False,
            min_epoch=min_save_epoch
        )
    ],
    each_iter_callbacks=[]
)

trainer.train()