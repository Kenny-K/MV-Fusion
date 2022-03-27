import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor
from torch_geometric.utils import softmax


class RangeConv(MessagePassing):
    '''
    Pixel-wise Range-weighted convolution
    Y_(i+m)(j+n) = X_(i+m)(j+n) * K_mn * (D_ij-D_(i+m)(j+n))^-1
    '''
    def __init__(self, 
              aggr = "add", 
              flow: str = "source_to_target", 
              node_dim: int = 0,
              kernel_size = 3,
              H = 64,
              W = 1024
              ):
        super().__init__(aggr, flow, node_dim)
        if isinstance(kernel_size, int):
          self.kernel_size = [kernel_size, kernel_size]
        else:
          self.kernel_size = kernel_size

        self.H = H
        self.W = W
        self.edge_index, self.local_index, self.center_index, self.vec_edge_index = self.prepare_range_conv(H,W,kernel_size)

        self.kernel = nn.parameter.Parameter(torch.randn(self.kernel_size[0],self.kernel_size[1]), requires_grad=True)

    def prepare_range_conv(self, H, W, kernel_size):
        '''
            depth_image := 1 x H x W (pooled)

            output:
                range_feature := 1 x H x W
        '''
        grid = np.indices((H,W)).reshape(2,H*W)
        edge_block = np.concatenate((np.expand_dims(np.arange(H*W),axis=0), grid), axis=0)  # [3, H*W] (pix_index, h, w)
        edge_index = []
        local_block = np.zeros(shape=(2,H*W), dtype=np.int32)
        local_index = []
        h,w = kernel_size // 2, kernel_size // 2
        for step_h in range(-h,h+1):
            for step_w in range(-w,w+1):
                temp = np.copy(edge_block)
                local_temp = np.copy(local_block)
                if step_h != 0:
                    temp[1,:] += step_h
                    local_temp[0,:] += step_h
                if step_w != 0:
                    temp[2,:] += step_w
                    local_temp[1,:] += step_w
                edge_index.append(temp)
                local_index.append(local_temp)
        grid = torch.from_numpy(grid)
        edge_index = torch.from_numpy(np.concatenate(edge_index,axis=1))
        local_index = torch.from_numpy(np.concatenate(local_index,axis=1))

        # mask out index out of range
        mask_lb = edge_index[1,:] >= 0
        edge_index = edge_index[:,mask_lb]
        local_index = local_index[:,mask_lb]
        mask_ub = edge_index[1,:] <= H-1
        edge_index = edge_index[:,mask_ub]
        local_index = local_index[:,mask_ub]

        mask_lb = edge_index[2,:] >= 0
        edge_index = edge_index[:,mask_lb]
        local_index = local_index[:,mask_lb]
        mask_ub = edge_index[2,:] <= W-1
        edge_index = edge_index[:,mask_ub]
        local_index = local_index[:,mask_ub]

        # vectorize center pixel index 
        center_idx = grid[:,edge_index[0,:]]
        # vectorize edge index: reshape edges from (pix_index, h, w) to (vectorized, pix_index), but still [3,N] to [2,N]
        vec_edge_index = torch.zeros(2, edge_index.shape[1])
        vec_edge_index[0,:] = self.W * edge_index[1,:] + edge_index[2,:]
        vec_edge_index[1,:] = edge_index[0,:]
       
        return edge_index.long().cuda(), local_index.long().cuda(), center_idx.long().cuda(), vec_edge_index.long().cuda()


    def forward(self, x, pooled_depth_image):
        '''
          x := vectorized range_image  pix_num x C ?
          depth_image := 1 x H x W
          self.edge_index = 3 x E   (pix_index, h, w)
          self.center_index = 2 x E (h, w)
          self.local_index = 2 X E  
        '''
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        depth_image = pooled_depth_image
        weights = 1 / torch.abs(depth_image[self.center_index[0,:],self.center_index[1,:]] - depth_image[self.edge_index[1,:], self.edge_index[2,:]] + 1e-15) # TODO: (E, )

        # x should be [num_nodes, num_nodes_features]
        # edge_index should be [2, num_edges] torch.longs
        # weights should be [num_edges, num_features]
        return self.propagate(edge_index=self.vec_edge_index, x=x, weights=weights)

    def message(self, x_j: Tensor, weights: Tensor, index) -> Tensor:
        # TODO: check the local index on the convolution kernel
        h, w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        # reshape for scalar broadcast
        out = x_j * softmax(weights,index).reshape(-1,1) * self.kernel[h + self.local_index[0, :], w + self.local_index[1, :]].reshape(-1,1)  
        return out


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

        self.use_range_convolution = cfg.MODEL.RANGE.RANGE_CONV

        if self.use_range_convolution:
            # Compress the depth image(64x1024->64x64) by average pooling to fit the size of innermost layer of RangeNet
            self.image_pool = nn.Sequential(
                                nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1), count_include_pad=False),
                                nn.AvgPool2d(kernel_size=3, stride=(1,2), padding=1, count_include_pad=False),
                                nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1), count_include_pad=False),
                                nn.AvgPool2d(kernel_size=3, stride=(1,2), padding=1, count_include_pad=False)
            )       # TODO:  Vertical Pooling ?
            
            self.range_convolution = RangeConv(H=64,W=64)

    def batch_range_conv(self, batch_fea, batch_depth_image):
        '''
            batch_fea := [bs, C, H, W]
            batch_depth_image := [bs, H, W]
            return:
              conv_batch_fea := [bs, C, H, W]
        '''
        if self.use_range_convolution:
            assert len(batch_fea.shape) == 4, print(batch_fea.shape)
            batch_size = batch_fea.shape[0]
            fea_channel = batch_fea.shape[1]
            h, w = batch_fea.shape[2], batch_fea.shape[3]
            conv_fea_list = []
            for batch_i in range(batch_size):
                vec_img = batch_fea[batch_i].permute(1,2,0).reshape(h*w,fea_channel)    #[C, H, W] => [H*W, C]
                conv_fea = self.range_convolution(vec_img, batch_depth_image[batch_i])
                conv_fea = conv_fea.reshape(h,w,-1).permute(2,0,1)
                conv_fea_list.append(conv_fea[None,:])      # expand dim as [1, C, H, W]

            return torch.cat(conv_fea_list,dim=0)
        else:
            return None
    
    def forward(self, x):
        # Convolutional Encoder
        skip_1 = self.conv1b(x).detach()    # b x 64 x 64 x 1024
        skip_2 = self.conv1a(x)             # b x 64 x 64 x 512

        skip_3 = self.conv2(skip_2)
        skip_2 = skip_2.detach()

        skip_4 = self.conv3(skip_3)
        skip_3 = skip_3.detach()

        code = self.conv4(skip_4)
        skip_4 = skip_4.detach()

        # batch-wise range convolution
        if self.use_range_convolution:
            depth_image = x[:,0,:,:]
            pooled_depth_image = self.image_pool(depth_image)
            code = self.batch_range_conv(code, pooled_depth_image)

        # Convolutional Decoder
        out = self.upconv1(code) + skip_4
        out = self.upconv2(out) + skip_3
        out = self.upconv3(out) + skip_2
        out = self.upconv4(out) + skip_1

        out = self.FC(out)

        return out

