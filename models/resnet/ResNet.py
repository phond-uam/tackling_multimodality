# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:53:14 2021

@author: Michel Frising
         michel.frising@uam.es
Inspired from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn

# BasicBlock and BottleneckBlock have matching signatures so we can use _makeLayers on both of them

class BasicBlock(nn.Module):
    """
    Consists of a 3x3 conv, relu act, 3x3 conv and skip connection
    """
    expansion = 1
    def __init__(       
        self,
        in_channels,
        out_channels,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 1,
        dilation = 1,
        norm_layer = None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    """
    consists of a 1x1 conv, 3x3 conv, 1x1 conv to match channels again. relu and
    batchnorm after each layer.
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    expansion = 4
    def __init__(
        self,
        in_channels,
        out_channels,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, groups=groups, padding=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv1d(width, out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            in_dim,
            out_dim,
            zero_init_residual = False,
            groups = 1,
            width_per_group = 64,
            replace_stride_with_dilation = None,
            norm_layer = None,
        ):
        super(ResNet,self).__init__()
        if block=='BasicBlock':
            block = BasicBlock
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_dim, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.adaptavgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*block.expansion, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
  
    def forward(self, x):
        x = self.conv1(x)
        # print("conv1: ", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool1(x)

        x = self.layer1(x)
        # print("layer1: ", x.shape)
        x = self.layer2(x)
        # print("layer2: ", x.shape)
        x = self.layer3(x)
        # print("layer3: ", x.shape)
        x = self.layer4(x)
        # print("layer4: ", x.shape)

        x = self.adaptavgpool(x)
        # print("avgpool: ", x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
  
    def _make_layer(
            self,
            block,
            out_channels,
            blocks,
            stride = 1,
            dilate = False,
        ):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.in_channels != out_channels*block.expansion:
                # if stride > 1 we need an auxiliar downsampling conv to atch the 
                # channel dims
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride),
                    norm_layer(out_channels*block.expansion),
                )
    
            layers = []
            layers.append(
                block(
                    self.in_channels, out_channels, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
                )
            )
            self.in_channels = out_channels*block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.in_channels,
                        out_channels,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )
    
            return nn.Sequential(*layers)    