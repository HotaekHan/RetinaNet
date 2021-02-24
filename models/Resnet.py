'''RetinaFPN in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.hub import load_state_dict_from_url

from models.SEBlock import SELayer

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _bn_function_factory(norm, conv, relu):
    def bn_function(inputs):
        output = conv(relu(norm(inputs)))
        return output

    return bn_function


class Bottleneck_pre_active(nn.Module):
    expansion = 4

    def __init__(self, in_planes, bottle_planes, stride=1, use_se=False, is_efficient=False):
        super(Bottleneck_pre_active, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn_function1 = _bn_function_factory(self.bn1, self.conv1, self.relu)

        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.bn_function2 = _bn_function_factory(self.bn2, self.conv2, self.relu)

        self.conv3 = nn.Conv2d(bottle_planes, self.expansion*bottle_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottle_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*bottle_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, self.expansion*bottle_planes, kernel_size=1, stride=stride, bias=False)
            )

        self.use_se = use_se
        if self.use_se:
            self.se_block = SELayer(n_channel=self.out_channels, reduction=16)

        self.efficient = is_efficient

    def forward(self, x):
        if self.efficient:
            out = cp.checkpoint(self.bn_function1, x)
            out = cp.checkpoint(self.bn_function2, out)
        else:
            out = self.bn_function1(x)
            out = self.bn_function2(out)

        out = self.conv3(self.relu(self.bn3(out)))

        if self.use_se:
            out = self.se_block(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, bottle_planes, stride=1, groups=1, base_width=64, dilation=1, use_se=False):
        super(Bottleneck, self).__init__()
        # ResNeXt: aggregated features in residual transformation.
        width = int(bottle_planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, self.expansion*bottle_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*bottle_planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*bottle_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*bottle_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*bottle_planes)
            )

        self.use_se = use_se
        if self.use_se:
            self.se_block = SELayer(n_channel=self.expansion*bottle_planes, reduction=16)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_se:
            out = self.se_block(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64, dilation=1, use_se=False):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        self.use_se = use_se
        if self.use_se:
            self.se_block = SELayer(n_channel=planes, reduction=16)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = self.se_block(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out


class ResFPN(nn.Module):
    def __init__(self, block, num_blocks, groups, width_per_group, dilation, use_se):
        super(ResFPN, self).__init__()
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = dilation

        self.in_planes = 64
        self.output_dims = 256

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, bottle_planes=64, num_blocks=num_blocks[0], stride=1, use_se=use_se)
        self.layer2 = self._make_layer(block, bottle_planes=128, num_blocks=num_blocks[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(block, bottle_planes=256, num_blocks=num_blocks[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(block, bottle_planes=512, num_blocks=num_blocks[3], stride=2, use_se=use_se)
        self.conv6 = nn.Conv2d(block.expansion * 512, self.output_dims, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(block.expansion * 512, self.output_dims, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(block.expansion * 256, self.output_dims, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(block.expansion * 128, self.output_dims, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, bottle_planes, num_blocks, stride, use_se):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, bottle_planes, stride, self.groups, self.base_width, self.dilation,
                                use_se))
            self.in_planes = bottle_planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, height, width = y.size()
        return nn.functional.interpolate(input=x, size=(height, width), mode='nearest') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7



def ResFPN18(is_pretrained=True, use_se=False):
    fpn = ResFPN(BasicBlock, [2, 2, 2, 2], groups=1, width_per_group=64, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNet18 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn


def ResFPN50(is_pretrained=True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 6, 3], groups=1, width_per_group=64, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNet50 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def ResFPN101(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 23, 3], groups=1, width_per_group=64, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNet101 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def ResFPN152(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 8, 36, 3], groups=1, width_per_group=64, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNet152 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnet152'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def ResNextFPN50(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNeXt50 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def ResNextFPN101(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained ResNeXt101 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def WideResFPN50(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 6, 3], groups=1, width_per_group=64*2, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained WideResNet50 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn

def WideResFPN101(is_pretrained = True, use_se=False):
    fpn = ResFPN(Bottleneck, [3, 4, 23, 3], groups=1, width_per_group=64*2, dilation=1, use_se=use_se)

    if is_pretrained is False:
        for m in fpn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    else:
        print('Loading pretrained WideResNet101 model with ImageNet..')
        state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'], progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        missing_keys = fpn.load_state_dict(state_dict, strict=False)
        print(missing_keys)

    return fpn


def test():
    net = WideResFPN101(is_pretrained=True)

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    fms = net((torch.randn(1, 3, 320, 320)))
    for fm in fms:
        print(fm.size())


if __name__ == '__main__':
    test()
