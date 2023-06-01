# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torch.utils import checkpoint

__all__ = [
    "RepVGG",
    "RepVGGBlock",
    "reg_vgg_a0", "reg_vgg_a1", "reg_vgg_a2",
    "reg_vgg_b0", "reg_vgg_b1", "reg_vgg_b1g2", "reg_vgg_b1g4",
    "reg_vgg_b2", "reg_vgg_b2g2", "reg_vgg_b2g4",
    "reg_vgg_b3", "reg_vgg_b3g2", "reg_vgg_b3g4",
    "convert_inference_model",
]

OPTIONAL_GROUPWISE_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
G2_MAP = {x: 2 for x in OPTIONAL_GROUPWISE_LAYERS}
G4_MAP = {x: 4 for x in OPTIONAL_GROUPWISE_LAYERS}


class RepVGG(nn.Module):

    def __init__(
            self,
            num_blocks,
            num_classes=1000,
            width_multiplier=None,
            override_groups_map=None,
            inference_mode=False,
            use_checkpoint=False
    ) -> None:
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.inference_mode = inference_mode
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_checkpoint = use_checkpoint

        self.cur_layer_idx = 1

        self.in_channels = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(3, self.in_channels, 3, 2, 1, inference_mode=self.inference_mode)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(self.in_channels, planes, 3, stride, 1, groups=cur_groups, inference_mode=self.inference_mode))
            self.in_channels = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.stage0(x)

        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RepVGGBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            inference_mode=False,
    ) -> None:
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.inference_mode = inference_mode

        self.relu = nn.ReLU(True)

        if inference_mode:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential()
            self.rbr_dense.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
            self.rbr_dense.add_module("bn", nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential()
            self.rbr_1x1.add_module("conv", nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False))
            self.rbr_1x1.add_module("bn", nn.BatchNorm2d(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.relu(self.rbr_reparam(x))

        if self.rbr_identity is None:
            rbr_out = 0
        else:
            rbr_out = self.rbr_identity(x)

        x = self.relu(self.rbr_dense(x) + self.rbr_1x1(x) + rbr_out)

        return x

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.

        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)

        if kernel1x1 is None:
            kernel1x1 = 0
        else:
            kernel1x1 = F_torch.pad(kernel1x1, [1, 1, 1, 1])

        return kernel3x3 + kernel1x1 + kernel_id, bias3x3 + bias1x1 + bias_id

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_inference_mode(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.inference_mode = True


def reg_vgg_a0(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, **kwargs)


def reg_vgg_a1(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


def reg_vgg_a2(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, **kwargs)


def reg_vgg_b0(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


def reg_vgg_b1(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=None, **kwargs)


def reg_vgg_b1g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=G2_MAP, **kwargs)


def reg_vgg_b1g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=G4_MAP, **kwargs)


def reg_vgg_b2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, **kwargs)


def reg_vgg_b2g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=G2_MAP, **kwargs)


def reg_vgg_b2g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=G4_MAP, **kwargs)


def reg_vgg_b3(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=None, **kwargs)


def reg_vgg_b3g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=G2_MAP, **kwargs)


def reg_vgg_b3g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=G4_MAP, **kwargs)


def convert_inference_model(model: nn.Module, model_save_path: str = None):
    model = copy.deepcopy(model)

    for module in model.modules():
        if hasattr(module, "switch_inference_mode"):
            module.switch_inference_mode()

    if model_save_path is not None:
        torch.save({"state_dict": model.state_dict()}, model_save_path)

    return model
