import pytest
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from collections import OrderedDict
import torchvision.transforms as transforms
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction


def conv_bn(in_planes, out_planes, kernel_size, stride=1, padding=0):
    """convolution with batchnorm, relu"""
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False), nn.BatchNorm2d(out_planes), nn.ReLU())


BIPRECISION = True

NUM_BITS = 8

NUM_BITS_GRAD = 8

NUM_BITS_WEIGHT = 8


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls,
                ctx,
                input,
                num_bits=8,
                min_value=None,
                max_value=None,
                stochastic=False,
                inplace=False,
                enforce_true_zero=False,
                num_chunks=None,
                out_half=False):
        num_chunks = num_chunks = input.shape[
            0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)
        if max_value is None:
            max_value = y.max(-1)[0].mean(-1)
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        qmin = 0.0
        qmax = 2.0**num_bits - 1.0
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-08)
        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.0
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)
        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()
        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)
        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


def quantize(x,
             num_bits=8,
             min_value=None,
             max_value=None,
             num_chunks=None,
             stochastic=False,
             inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value,
                                   num_chunks, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(min_value *
                                                      (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(max_value *
                                                      (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input,
                        self.num_bits,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        num_chunks=16)


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls,
                ctx,
                input,
                num_bits=8,
                min_value=None,
                max_value=None,
                stochastic=True,
                inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value,
                                             ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


def quantize_grad(x,
                  num_bits=8,
                  min_value=None,
                  max_value=None,
                  stochastic=True,
                  inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value,
                                       stochastic, inplace)


def conv2d_biprec(input,
                  weight,
                  bias=None,
                  stride=1,
                  padding=0,
                  dilation=1,
                  groups=1,
                  num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias, stride, padding, dilation,
                    groups)
    out2 = F.conv2d(input, weight.detach(),
                    bias.detach() if bias is not None else None, stride,
                    padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 num_bits=8,
                 num_bits_weight=None,
                 num_bits_grad=None,
                 biprecision=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight,
                           num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding,
                              self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = conv2d_biprec(qinput,
                                   qweight,
                                   qbias,
                                   self.stride,
                                   self.padding,
                                   self.dilation,
                                   self.groups,
                                   num_bits_grad=self.num_bits_grad)
        return output


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(),
                    bias.detach() if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 num_bits=8,
                 num_bits_weight=None,
                 num_bits_grad=None,
                 biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight,
                           num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


def depBatchNorm2d(exists, *kargs, **kwargs):
    if exists:
        return nn.BatchNorm2d(*kargs, **kwargs)
    else:
        return lambda x: x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 batch_norm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=not batch_norm,
                               groups=32)
        self.bn2 = depBatchNorm2d(batch_norm, planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * 2,
                               kernel_size=1,
                               bias=not batch_norm)
        self.bn3 = depBatchNorm2d(batch_norm, planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        num_bits=NUM_BITS,
                        num_bits_weight=NUM_BITS_WEIGHT,
                        num_bits_grad=NUM_BITS_GRAD),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ResNet_imagenet(ResNet):

    def __init__(self,
                 num_classes=1000,
                 block=Bottleneck,
                 layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = QConv2d(3,
                             64,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False,
                             num_bits=NUM_BITS,
                             num_bits_weight=NUM_BITS_WEIGHT,
                             num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = QLinear(512 * block.expansion,
                          num_classes,
                          num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT,
                          num_bits_grad=NUM_BITS_GRAD)
        init_model(self)
        self.regime = [{
            'epoch': 0,
            'optimizer': 'SGD',
            'lr': 0.1,
            'weight_decay': 0.0001,
            'momentum': 0.9
        }, {
            'epoch': 30,
            'lr': 0.01
        }, {
            'epoch': 60,
            'lr': 0.001,
            'weight_decay': 0
        }, {
            'epoch': 90,
            'lr': 0.0001
        }]


def get_model():
    # ResNeXt_imagenet has lower speedup
    return ResNet_imagenet().cuda()


def get_input(batch_size):
    return (torch.randn(batch_size, 3, 224, 224).cuda(),), {}


from frontend.compile import compile, reset
from common.checker import assert_equal, run_and_check_cache, run_and_check, HIT, MISS, ALL_MISS


@pytest.mark.model
def test_model_quantized(caplog):
    reset()
    with torch.no_grad():
        model = get_model().eval()
        input_args, input_kwargs = get_input(1)
        expect = model(*input_args, **input_kwargs)
        compiled = compile(model)
        run_and_check(compiled, [ALL_MISS], 1, caplog, expect, *input_args,
                      **input_kwargs)
        run_and_check(compiled, [HIT], 1, caplog, expect, *input_args,
                      **input_kwargs)
