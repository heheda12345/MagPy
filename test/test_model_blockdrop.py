import os
if 'LD_PRELOAD' in os.environ:
    del os.environ['LD_PRELOAD']
import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS, ALL_MISS

import torch
import torch.nn as nn
import random
import math
import torch.nn.functional as F
from frontend.dynamic import add_branch_rewrite_pc
from frontend.control_flow import if_stmt
from frontend.utils import SetConfig


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


#--------------------------------------------------------------------------------------------------#
class FlatResNet(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy):

        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                action = policy[:, t].contiguous()
                residual = self.ds[segment](x) if b == 0 else x

                # early termination if all actions in the batch are zero
                if action.data.sum() == 0:
                    x = residual
                    t += 1
                    continue

                action_mask = action.float().view(-1, 1, 1, 1)
                fx = F.relu(residual + self.blocks[segment][b](x))
                x = fx * action_mask + residual * (1 - action_mask)
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                if policy[t] == 1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_full(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Smaller Flattened Resnet, tailored for CIFAR
class FlatResNet32(FlatResNet):

    def __init__(self, block, layers, num_classes=10):
        super(FlatResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks,
                  stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block,
                                          filt_size,
                                          num_blocks,
                                          stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion,
                                     stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


def get_input(batch_size):
    return torch.randn(batch_size, 3, 32,
                       32).cuda(), torch.randint(0, 2, (batch_size, 15)).cuda()


@pytest.mark.model
def test_blockdrop_full(caplog):
    reset()
    with torch.no_grad():
        model = FlatResNet32(BasicBlock, [5, 5, 5]).cuda()
        model.eval()
        batch_size = 2
        inp = torch.randn(batch_size, 3, 32, 32).cuda()
        expect_result = model.forward_full(inp)
        compiled = compile(model.forward_full)
        run_and_check(compiled, [MISS] * 20, 1, caplog, expect_result, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect_result, inp)


@pytest.mark.model
def test_blockdrop_dyn(caplog):
    reset()
    with torch.no_grad():
        with SetConfig({"backend": "eager"}):
            model = FlatResNet32(BasicBlock, [5, 5, 5]).cuda()
            model.eval()
            batch_size = 2
            inp = torch.randn(batch_size, 3, 32, 32, device="cuda")
            policy = torch.randint(0, 2, (batch_size, 15), device="cuda")
            expect_result = model(inp, policy)
            add_branch_rewrite_pc(get_next_frame_id(), 51)
            compiled = compile(model)
            run_and_check(compiled, [ALL_MISS], 1, caplog, expect_result, inp,
                          policy)
            run_and_check(compiled, [HIT], 1, caplog, expect_result, inp,
                          policy)
