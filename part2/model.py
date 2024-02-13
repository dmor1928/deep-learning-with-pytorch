import math

import torch
from torch import nn as nn

from util.unet import UNet
from util.logconf import logging

import torch.nn.functional as F
import random

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=conv_channels,
            kernel_size=3, padding=1,
            bias=True,  # Adds learnable bias to output
        )
        self.relu1 = nn.ReLU(inplace=True)  # Performs operation in-place
        self.conv2 = nn.Conv3d(
            in_channels=conv_channels, out_channels=conv_channels,
            kernel_size=3, padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)  # Defined separately since sometimes it can make errors
        self.maxpool = nn.MaxPool3d(2,2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)  # Batch-norm (single val per coord) aids training + prevents saturation

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(2 * conv_channels, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        # Channels progressively increase to progressively learn more complex details/ features of nodules

        self.head_linear = nn.Linear(1152, 2)  # 1125 = 8 * 8 channels * (2x3x3 images)
        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        block_out = self.tail_batchnorm(input_batch)

        block_out = self.block1(block_out)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0), -1)  # block_out.size(0)= batch size, flatten each data in batch

        linear_output = self.head_linear(conv_flat)  # Cross entropy (for training) needs linear output
        softmax_output = self.head_softmax(linear_output)  # softmax (for classification) gives (0-1) probability

        return linear_output, softmax_output

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:  # Initialise w's (linear and conv only since others don't have w)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:  # For modules with bias
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)  # clipped to prevent vanishing/ saturating


class UNetWrapper(nn.Module):
    # - Inputs smaller slices of the CT scan
    #   . Inputting entire 3D scan will be memory-inefficient
    # - But would likely still need context from adjacent Z layers
    # - Adjacent slices inputted as separate channels
    #   . The model will have to relearn the adjacency rltnship between channels
    #   . Should not be too difficult since it manages to learn it for multichannel coloured images
    # - Slice thickness not given, potentially causing issues. Therefore, present data with different Z thicknesses
    def __init__(self, **kwargs):  # **kwargs = in_channels, n_classes, depth, wf, padding, batch_norm, up_mode
        super().__init__()

        self.input_batch_norm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)  # initialise UNet
        self.final = nn.Sigmoid()

        self._init_weights()  # Our custom weight initialisation (line 62)

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batch_norm(input_batch)  # Batch-normalise the input
        un_output = self.unet(bn_output)  # run UNet
        fn_output = self.final(un_output)  # Restrict unconstrained output to [0, 1]
        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()  # Flip, offset, scale and rotate applied in build function
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        # input_g.shape[0] = number of slices. Expand transform_t to 7 slices in 3D, so that it applies to all slices
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g, affine_t,
                                          padding_mode='border', align_corners=False)  # Apply affine_t to input_g
        augmented_label_g = F.grid_sample(label_g.to(torch.float32), affine_t,
                                          padding_mode='border', align_corners=False)  # Apply affine_t to label_g

        # augmented_label_g = augmented_label_g > 0.5  # grid_sample returns fractional vals, > 0.5 converts back to Bool

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise  # Zeroes if false, nonzero if true

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5  # grid_sample returns fractional vals, > 0.5 converts back to Bool

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)  # eye produced 3x3 identity matrix

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:  # 50% chance of flipping, across each axis x, y of image
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)  # Uniform random from [-1, 1]
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1) # Uniform random from [-1, 1]
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2  # Rotate by random angle between 0 and 2π
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)
            rotation_t = torch.tensor([  # 2D rotation transformation matrix
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t  # Applies to rotation_t to transform_t

        return transform_t

