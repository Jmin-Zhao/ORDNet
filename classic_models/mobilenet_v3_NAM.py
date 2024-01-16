
from typing import Callable, List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

from attention.NAM import Att

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    它确保所有层的通道号可被 8 整除
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, activation_layer: Optional[Callable[..., nn.Module]] = None):
        kernel_size1 = (kernel_size - 1) * dilation + 1
        padding = (kernel_size1 - 1) // 2
        # dilation = kernel_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,  stride=stride,
                                                         padding=padding, dilation=dilation,  groups=groups, bias=False),
                                                         norm_layer(out_planes),
                                                         activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:   # 倒残差结构
    def __init__(self, input_c: int, kernel: int, expanded_c: int, out_c: int, use_se: bool, activation: str, stride: int, dilation: int,  width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,  cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer,dilation=cnf.dilation))
        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel, stride=cnf.stride, groups=cnf.expanded_c,
                                        norm_layer=norm_layer, activation_layer=activation_layer,dilation=cnf.dilation))

        # 加入注意力机制

        if cnf.use_se:
            # layers.append(SqueezeExcitation(cnf.expanded_c))  # SE
            # layers.append(CoordAtt(cnf.expanded_c, cnf.expanded_c))  # CA
            # layers.append(eca_layer(cnf.expanded_c))  # ECA
            layers.append(Att(cnf.expanded_c)) # NAM
        # project
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))
        # layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer,activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):

    # self指的是实例Instance本身，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self
    # 在python中创建类后，通常会创建一个 __init__()方法，这个方法会在创建类的实例的时候自动执行

    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int,  num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()     # MobileNetV3类继承nn.Module，super(MobileNetV3, self).__init__()就是对继承自父类nn.Module的属性进行初始化。而且是用nn.Module的初始化方法来初始化继承的属性。

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)      # batch normalization的目的是使一批（Batch）featuremap满足均值为0，方差为1的分布规律

        layers: List[nn.Module] = []



        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, firstconv_output_c, kernel_size=3, stride=2, norm_layer=norm_layer,  activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c, lastconv_output_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        # nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        # nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1)
                                        nn.Linear(last_channel, num_classes))
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor):
        return self._forward_impl(x)



"""
Constructs a large MobileNetV3 architecture from "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>. 
weights_link:
https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
Args:
    num_classes (int): number of classes
    reduced_tail (bool): If True, reduces the channel counts of all feature layers
        between C4 and C5 by 2. It is used to reduce the channel redundancy in the
        backbone for Detection and Segmentation.
"""
def mobilenet_v3_large(num_classes: int = 1000, reduced_tail: bool = False):

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1
    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, 1),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


""" 
weights_link:
https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth 
"""
def mobilenet_v3_nam(num_classes: int = 1000, reduced_tail: bool = False):

    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)   # partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, 1),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 576 // reduce_divider, True, "HS", 1, 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)




