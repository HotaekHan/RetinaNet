import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResFPN18, ResFPN50, ResFPN101
# from models.Densenet import DenseFPN62, DenseFPN102
# from models.ShuffleNetV2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0

import math


class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_anchors, basenet, is_pretrained_base=False):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_head_layers = 4

        if basenet == 'Res18':
            if is_pretrained_base is True:
                self.base_networks = ResFPN18(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN18(is_pretrained=False, use_se=False)
        elif basenet == 'Res50':
            if is_pretrained_base is True:
                self.base_networks = ResFPN50(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN50(is_pretrained=False, use_se=False)
        elif basenet == 'Res101':
            if is_pretrained_base is True:
                self.base_networks = ResFPN101(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN101(is_pretrained=False, use_se=False)
        elif basenet == 'Res152':
            if is_pretrained_base is True:
                self.base_networks = ResFPN152(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResFPN152(is_pretrained=False, use_se=False)
        elif basenet == 'ResNeXt50':
            if is_pretrained_base is True:
                self.base_networks = ResNextFPN50(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResNextFPN50(is_pretrained=False, use_se=False)
        elif basenet == 'ResNeXt101':
            if is_pretrained_base is True:
                self.base_networks = ResNextFPN101(is_pretrained=True, use_se=False)
            else:
                self.base_networks = ResNextFPN101(is_pretrained=False, use_se=False)
        # elif basenet == 'Dense62':
        #     self.base_networks = DenseFPN62(use_se=False, efficient=False)
        # elif basenet == 'Dense102':
        #     self.base_networks = DenseFPN102(use_se=False, efficient=False)
        # elif basenet == 'ShuffleV2_x0_5':
        #     if is_pretrained_base is True:
        #         self.base_networks = shufflenet_v2_x0_5(pretrained=True, progress=True)
        #     else:
        #         self.base_networks = shufflenet_v2_x0_5(pretrained=False, progress=True)
        # elif basenet == 'ShuffleV2_x1_0':
        #     if is_pretrained_base is True:
        #         self.base_networks = shufflenet_v2_x1_0(pretrained=True, progress=True)
        #     else:
        #         self.base_networks = shufflenet_v2_x1_0(pretrained=False, progress=True)
        # elif basenet == 'ShuffleV2_x1_5':
        #     self.base_networks = shufflenet_v2_x1_5(pretrained=False, progress=True)
        # elif basenet == 'ShuffleV2_x2_0':
        #     self.base_networks = shufflenet_v2_x2_0(pretrained=False, progress=True)
        else:
            raise ValueError('not supported base network.')

        self.conv1_mask = nn.Conv2d(self.base_networks.output_dims, 256, kernel_size=3, padding=2, stride=1,
                                    dilation=2, bias=False)
        self.conv1_bn_mask = nn.BatchNorm2d(256)
        self.deconv_mask = nn.ConvTranspose2d(256, 2, kernel_size=8, padding=2, stride=4, groups=2)
        self.conv2_mask = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.conv2_bn_mask = nn.BatchNorm2d(2)
        self.conv3_mask = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2)
        self.softmax_mask = nn.Softmax2d()

        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.base_networks(x)

        mask = F.relu(self.conv1_bn_mask(self.conv1_mask(p3)))
        mask = self.deconv_mask(mask)
        mask = F.relu(self.conv2_bn_mask(self.conv2_mask(mask)))
        mask = self.conv3_mask(mask)

        attention = self.softmax_mask(mask)
        attention = attention[:, 1:2, :, :]
        attention = F.avg_pool2d(attention, kernel_size=4, stride=4, ceil_mode=True)

        attention = F.avg_pool2d(attention, kernel_size=2, stride=2, ceil_mode=True)
        masked_p3 = attention * p3
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2, ceil_mode=True)
        masked_p4 = attention * p4
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2, ceil_mode=True)
        masked_p5 = attention * p5

        fms = [masked_p3, masked_p4, masked_p5, p6, p7]

        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1), mask

    def _make_head(self, out_planes):
        layers = []
        for _ in range(self.num_head_layers):
            layers.append(nn.Conv2d(self.base_networks.output_dims, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


def load_model(num_classes, num_anchors, basenet='Res50', is_pretrained_base=True, do_freeze=False):
    retinanet = RetinaNet(num_classes, num_anchors, basenet, is_pretrained_base)

    # head initialize
    for m in retinanet.loc_head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    for m in retinanet.cls_head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # biased initialize for classification head layer.
    pi = 0.01
    nn.init.constant_(retinanet.cls_head[-1].bias, -math.log((1 - pi) / pi))

    # freeze layer
    if do_freeze is True:
        for name, layer in retinanet.base_networks.named_children():
            if name == 'conv1' or name == 'layer1':
                for param in layer.parameters():
                    param.requires_grad = False

    return retinanet


def test():
    net = load_model(9, 9, basenet='Res50', is_pretrained_base=False)

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    base_num_parameters = 0.
    for param in net.base_networks.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        base_num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))
    print("num. of basenet parameters : " + str(base_num_parameters))

    loc_preds, cls_preds, attention = net(torch.randn(1, 3, 320, 320))
    print(loc_preds.size())
    print(cls_preds.size())
    print(attention.size())
    loc_grads = torch.randn(loc_preds.size())
    cls_grads = torch.randn(cls_preds.size())
    att_grads = torch.randn(attention.size())
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)
    attention.backward(att_grads)


if __name__ == '__main__':
    test()
