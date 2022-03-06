import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, inplanes, bn_d=0.1):
        super(CAM, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.pool = nn.MaxPool2d(7, 1, 3)
        self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                                kernel_size=1, stride=1)
        self.squeeze_bn = nn.BatchNorm2d(inplanes // 16, momentum=self.bn_d)
        self.relu = nn.ReLU(inplace=True)
        self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                                kernel_size=1, stride=1)
        self.unsqueeze_bn = nn.BatchNorm2d(inplanes, momentum=self.bn_d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 7x7 pooling
        y = self.pool(x)
        # squeezing and relu
        y = self.relu(self.squeeze_bn(self.squeeze(y)))
        # unsqueezing
        y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
        # attention
        return y * x


class FireUp(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                expand1x1_planes, expand3x3_planes, bn_d, stride):
        super(FireUp, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
        if self.stride == 2:
            self.upconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                            kernel_size=[1, 4], stride=[1, 2],
                                            padding=[0, 1])
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                    kernel_size=1)
            self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
            self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                    kernel_size=3, padding=1)
            self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

    def forward(self, x):
        x = self.activation(self.squeeze_bn(self.squeeze(x)))
        if self.stride == 2:
            x = self.activation(self.upconv(x))
        return torch.cat([
            self.activation(self.expand1x1_bn(self.expand1x1(x))),
            self.activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                expand1x1_planes, expand3x3_planes, bn_d=0.1):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

    def forward(self, x):
        x = self.activation(self.squeeze_bn(self.squeeze(x)))
        return torch.cat([
            self.activation(self.expand1x1_bn(self.expand1x1(x))),
            self.activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)


class RangeNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.bn_d = 0.1
        self.input_depth = 5
        self.strides = [2,2,2,2]
        self.out_dim = cfg.MODEL.VFE.OUT_CHANNEL
        self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3,
                                          stride=[1, self.strides[0]],
                                          padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                CAM(64))
        self.conv1b = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=1,
                                          stride=1, padding=0),
                                nn.BatchNorm2d(64, momentum=self.bn_d))
        self.conv2 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[1]],
                                             padding=1),
                                Fire(64, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d),
                                Fire(128, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d))
        self.conv3 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[2]],
                                             padding=1),
                                Fire(128, 32, 128, 128, bn_d=self.bn_d),
                                Fire(256, 32, 128, 128, bn_d=self.bn_d))
        self.conv4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                               stride=[1, self.strides[3]],
                                               padding=1),
                                Fire(256, 48, 192, 192, bn_d=self.bn_d),
                                Fire(384, 48, 192, 192, bn_d=self.bn_d),
                                Fire(384, 64, 256, 256, bn_d=self.bn_d),
                                Fire(512, 64, 256, 256, bn_d=self.bn_d))

        self.upconv1 = FireUp(512, 64, 128, 128, bn_d=self.bn_d,
                                stride=self.strides[0])
        self.upconv2 = FireUp(256, 32, 64, 64, bn_d=self.bn_d,
                                stride=self.strides[1])
        self.upconv3 = FireUp(128, 16, 32, 32, bn_d=self.bn_d,
                                stride=self.strides[2])
        self.upconv4 = FireUp(64, 16, 32, 32, bn_d=self.bn_d,
                                stride=self.strides[3])

        self.FC = nn.Conv2d(64,self.out_dim,kernel_size=1)
    
    def forward(self, x):
        # TODO: skip with detach? maybe not
        # Convolutional Encoder
        skip_1 = self.conv1b(x).detach()    # b x 64 x 64 x 1024
        skip_2 = self.conv1a(x)                  # b x 64 x 64 x 512

        skip_3 = self.conv2(skip_2)
        skip_2 = skip_2.detach()

        skip_4 = self.conv3(skip_3)
        skip_3 = skip_3.detach()

        code = self.conv4(skip_4)
        skip_4 = skip_4.detach()

        # Convolutional Decoder
        out = self.upconv1(code) + skip_4
        out = self.upconv2(out) + skip_3
        out = self.upconv3(out) + skip_2
        out = self.upconv4(out) + skip_1

        out = self.FC(out)

        return out