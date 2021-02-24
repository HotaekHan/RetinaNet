import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from models.SEBlock import SELayer


def _bn_function_factory(norm, conv, relu):
    def bn_function(inputs):
        output = conv(relu(norm(inputs)))
        return output

    return bn_function


class DenseBlock_B(nn.Module):
    expansion = 4

    def __init__(self, kernel_size, input_channels, output_channels, efficient, use_se):
        super(DenseBlock_B, self).__init__()

        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.efficient = efficient

        self.batch_norm1 = nn.BatchNorm2d(num_features=self.input_channels, momentum=0.01)
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels * self.expansion,
                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn_function = _bn_function_factory(self.batch_norm1, self.conv1, self.relu1)

        self.batch_norm2 = nn.BatchNorm2d(num_features=self.output_channels * self.expansion, momentum=0.01)
        self.conv2 = nn.Conv2d(in_channels=self.output_channels * self.expansion, out_channels=self.output_channels,
                               kernel_size=self.kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_se = use_se
        if self.use_se:
            self.se_block = SELayer(n_channel=self.output_channels, reduction=5)

    def forward(self, inputs):
        if self.efficient:
            out = cp.checkpoint(self.bn_function, inputs)
        else:
            out = self.bn_function(inputs)
        out = self.conv2(self.relu2(self.batch_norm2(out)))

        if self.use_se:
            out = self.se_block(out)

        out = torch.cat((inputs, out), dim=1)

        return out

class TransitionBlock(nn.Module):
    def __init__(self, input_channels, strides, padding, theta=1.0):
        super(TransitionBlock, self).__init__()

        self.input_channels = input_channels
        output_channels = int(input_channels * theta)
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding

        self.batch_norm1 = nn.BatchNorm2d(num_features=self.input_channels, momentum=0.01)
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=self.strides, padding=self.padding)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.conv1(self.relu1(self.batch_norm1(inputs)))
        out = self.pool1(out)

        return out

class DenseNet_BC(nn.Module):
    def __init__(self, num_dense_blocks, k=40, theta=1.0, efficient=False, use_se=False):
        super(DenseNet_BC, self).__init__()
        self.k = k
        self.theta = theta
        self.efficient = efficient
        self.use_se = use_se
        self.output_dims = 256

        out_channels = 2 * self.k
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        num_blocks = num_dense_blocks[0]
        self.dense1 = self._make_blocks(DenseBlock_B, num_blocks,
                                        kernel_size=(3, 3), input_channels=out_channels, output_channels=self.k,
                                        efficient=self.efficient, use_se=self.use_se)
        block1_out_channels = out_channels + (num_blocks * self.k)
        self.transition1 = TransitionBlock(input_channels=block1_out_channels, strides=(1, 1), padding=(1, 1), theta=self.theta)
        block1_out_channels = int(block1_out_channels * self.theta)

        num_blocks = num_dense_blocks[1]
        self.dense2 = self._make_blocks(DenseBlock_B, num_blocks,
                                        kernel_size=(3, 3), input_channels=block1_out_channels, output_channels=self.k,
                                        efficient=self.efficient, use_se=self.use_se)
        block2_out_channels = block1_out_channels + (num_blocks * self.k)
        self.transition2 = TransitionBlock(input_channels=block2_out_channels, strides=(2, 2), padding=(1, 1), theta=self.theta)
        block2_out_channels = int(block2_out_channels * self.theta)

        num_blocks = num_dense_blocks[2]
        self.dense3 = self._make_blocks(DenseBlock_B, num_blocks,
                                        kernel_size=(3, 3), input_channels=block2_out_channels, output_channels=self.k,
                                        efficient=self.efficient, use_se=self.use_se)
        block3_out_channels = block2_out_channels + (num_blocks * self.k)
        self.transition3 = TransitionBlock(input_channels=block3_out_channels, strides=(2, 2), padding=(1, 1), theta=self.theta)
        block3_out_channels = int(block3_out_channels * self.theta)

        num_blocks = num_dense_blocks[3]
        self.dense4 = self._make_blocks(DenseBlock_B, num_blocks,
                                        kernel_size=(3, 3), input_channels=block3_out_channels, output_channels=self.k,
                                        efficient=self.efficient, use_se=self.use_se)
        block4_out_channels = block3_out_channels + (num_blocks * self.k)
        self.transition4 = TransitionBlock(input_channels=block4_out_channels, strides=(2, 2), padding=(1, 1), theta=self.theta)
        block4_out_channels = int(block4_out_channels * self.theta)

        self.conv6 = nn.Conv2d(block4_out_channels, self.output_dims, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(block4_out_channels, self.output_dims, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(block3_out_channels, self.output_dims, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(block2_out_channels, self.output_dims, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(self.output_dims, self.output_dims, kernel_size=3, stride=1, padding=1)

    def _make_blocks(self, block, num_block, kernel_size, input_channels, output_channels, efficient, use_se):
        blocks = list()

        for iter_block in range(1, num_block+1):
            blocks.append(block(kernel_size, input_channels, output_channels, efficient, use_se))
            input_channels = input_channels + output_channels
        return nn.Sequential(*blocks)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, height, width = y.size()
        return nn.functional.interpolate(input=x, size=(height, width), mode='nearest') + y

    def forward(self, inputs):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(inputs)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.dense1(c1)
        c2 = self.transition1(c2)
        c3 = self.dense2(c2)
        c3 = self.transition2(c3)
        c4 = self.dense3(c3)
        c4 = self.transition3(c4)
        c5 = self.dense4(c4)
        c5 = self.transition4(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7


def DenseFPN62(use_se=False, efficient=False):
    densenet = DenseNet_BC(num_dense_blocks=[3, 6, 12, 8], k=40, theta=1.0, efficient=efficient, use_se=use_se)

    for m in densenet.modules():
        if isinstance(m, nn.Conv2d):
           nn.init.normal_(m.weight, mean=0, std=0.01)
           if m.bias is not None:
               nn.init.constant_(m.bias, 0)
    return densenet


def DenseFPN102(use_se=False, efficient=False):
    densenet = DenseNet_BC(num_dense_blocks=[3, 6, 24, 16], k=40, theta=1.0, efficient=efficient, use_se=use_se)

    for m in densenet.modules():
        if isinstance(m, nn.Conv2d):
           nn.init.normal_(m.weight, mean=0, std=0.01)
           if m.bias is not None:
               nn.init.constant_(m.bias, 0)
    return densenet

def DenseFPN201(use_se=False, efficient=False):
    densenet = DenseNet_BC(num_dense_blocks=[6, 12, 48, 32], k=40, theta=1.0, efficient=efficient, use_se=use_se)

    for m in densenet.modules():
        if isinstance(m, nn.Conv2d):
           nn.init.normal_(m.weight, mean=0, std=0.01)
           if m.bias is not None:
               nn.init.constant_(m.bias, 0)
    return densenet

def DenseFPN264(use_se=False, efficient=False):
    densenet = DenseNet_BC(num_dense_blocks=[6, 12, 64, 48], k=40, theta=1.0, efficient=efficient, use_se=use_se)

    for m in densenet.modules():
        if isinstance(m, nn.Conv2d):
           nn.init.normal_(m.weight, mean=0, std=0.01)
           if m.bias is not None:
               nn.init.constant_(m.bias, 0)
    return densenet


def test():
    net = DenseFPN102(use_se=True, efficient=True)
    net = net.cuda()

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    tmp = torch.randn(1, 3, 640, 640)
    tmp = tmp.cuda()

    fms = net(tmp)
    for fm in fms:
        print(fm.size())


if __name__ == '__main__':
    test()