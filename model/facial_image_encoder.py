import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
from torchvision.transforms import v2, Lambda

""" MFN """

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class Mix_Depth_Wise(nn.Module):
    def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, kernel_size=[3,5,7], split_out_channels=[64,32,32]):
        super(Mix_Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = MDConv(channels=groups, kernel_size=kernel_size, split_out_channels=split_out_channels, stride=stride)
        self.CA = CoordAtt(groups, groups)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class Mix_Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[64,64]):
        super(Mix_Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Mix_Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups, kernel_size=kernel_size, split_out_channels=split_out_channels ))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)


class MixedFeatureNet(nn.Module):
    def __init__(self, embedding_size=256, out_h=7, out_w=7):
        super(MixedFeatureNet, self).__init__()
        #224x224
        self.conv0 = Conv_block(3, 3, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        #112x112
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        #56x56
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Mix_Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, kernel_size=[3,5,7], split_out_channels=[64,32,32] )

        #28x28
        self.conv_3 = Mix_Residual(64, num_block=9, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[96,32])
        self.conv_34 = Mix_Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, kernel_size=[3,5,7],split_out_channels=[128,64,64] )

        #14x14
        self.conv_4 = Mix_Residual(128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), kernel_size=[3,5], split_out_channels=[192,64])
        self.conv_5 = Mix_Depth_Wise(128, 512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=512*2, kernel_size=[3,5,7,9],split_out_channels=[128*2,128*2,128*2,128*2] )


    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_5(out)

        return l2_norm(out)
    
    
""" VGG 19 """

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.convblock(x)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class DepthWiseSeperableConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(DepthWiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channel,
                                   in_channel,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channel,
                                   bias=bias
                                   )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channel,
                                   out_channel,
                                   kernel_size=1,
                                   bias=bias
                                   )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class PatchExtraction(nn.Module):
    """ Patch extraction block:
            - Depthwise separable convolutional layer
            - Depthwise separable convolutional layer
            - Pointwise convolutional layer

        - MobileNet outputs feature maps from the MobileNetV1 that are padded to
        the dimension of 16x16

        - First depthwise separable convolutional layer splits into 4 patches
        ------------------------------------------
        Input Size:  (N, 512, 16, 16)
    """
    def __init__(self):
        super(PatchExtraction, self).__init__()
        self.conv1 = DepthWiseSeperableConv(in_channel=512,
                                            out_channel=256,
                                            kernel_size=4,
                                            stride=4,
                                            padding=2)
        self.conv2 = DepthWiseSeperableConv(in_channel=256,
                                            out_channel=256,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0)
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=49,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class MultiheadedSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 attn_dropout=0.5,
                 proj_dropout=0.5,
                 ):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads."
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x) # B, N, (3*C)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) # B, N, 3(qkv), H(eads), embed_dim
            .permute(2, 0, 3, 1, 4) # 3, B, H(eads), N, emb_dim
        )
        q, k, v = torch.chunk(qkv, 3) # B, H, N, dim
        # B,H,N,dim x B,H,dim,N -> B,H,N,N
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # <q,k> / sqrt(d)
        attn = attn.softmax(dim=-1) # Softmax over embedding dim
        attn = self.attn_dropout(attn)

        x = ( # B, H, N, N
            torch.matmul(attn, v) # B,H,N,N x B,H,N,dim -> B, H, N, dim
            .transpose(1, 2) # B, N, H, dim
            .reshape(B, N, C) # B, N, (H*dim)
        )
        x = self.projection(x)
        x = self.proj_dropout(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=192,
                 num_heads=8,
                 attn_dropout=0.5,
                 proj_dropout=0.5,
                 mlp_dropout=0.1,
                 feedforward_dim=768,
            ):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.MHA = MultiheadedSelfAttention(embed_dim,
                                        num_heads,
                                        attn_dropout,
                                        proj_dropout,
                   )
        self.ff = nn.Sequential(nn.Linear(embed_dim, feedforward_dim),
                                nn.GELU(),
                                nn.Dropout(mlp_dropout),
                                nn.Linear(feedforward_dim, embed_dim),
                                nn.Dropout(mlp_dropout),
                 )

    def forward(self, x):
        mha = self.norm_1(x)
        mha = self.MHA(mha)
        x = x + mha # Residual connection (Add)

        x = self.norm_2(x)
        x2 = self.ff(x)
        x = x + x2  # Residual connection (Add)

        return x

""" VCCT Original """


class VCCT(nn.Module):
    def __init__(self,
                 num_encoders=1,
                 num_classes=8,
                 embed_dim=192,
                 num_heads=8,
                 attn_dropout=0.5,
                 proj_dropout=0.5,
                 mlp_dropout=0.1,
                 feedforward_dim=768,
            ):
        super(VGGT, self).__init__()
        self.vgg = MixedFeatureNet()
        self.patcher = PatchExtraction()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1, embed_dim)
        self.transformer = self.create_encoders(embed_dim, num_heads,
                                                attn_dropout, proj_dropout,
                                                mlp_dropout, feedforward_dim,
                                                num_encoders)

        self.attention_pool = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        #for param in self.vgg.parameters():
        #    param.requires_grad = False



    def create_encoders(self, embed_dim=192,
                        num_heads=8,
                        attn_dropout=0.5,
                        proj_dropout=0.5,
                        mlp_dropout=0.1,
                        feedforward_dim=768,
                        num_layers=2,
                       ):
        return nn.Sequential(*[EncoderLayer(embed_dim, num_heads, attn_dropout, proj_dropout, mlp_dropout, feedforward_dim) for _ in range(num_layers)])


    def forward(self, x):
        x = self.vgg(x)
        x = self.patcher(x)
        x = self.gap(x)
        x = self.fc1(x).squeeze(dim=2)
        x = self.transformer(x)
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x)
        x = self.fc(x).squeeze(1)
        return x


""" VCCT encoder """

@torch.no_grad()
class VCCT_encoder(nn.Module):
    def __init__(self,
                 num_encoders=1,
                 num_classes=8,
                 embed_dim=192,
                 num_heads=8,
                 attn_dropout=0.5,
                 proj_dropout=0.5,
                 mlp_dropout=0.1,
                 feedforward_dim=768,
            ):
        super(VGGT, self).__init__()
        self.vgg = MixedFeatureNet()
        self.patcher = PatchExtraction()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1, embed_dim)
        self.transformer = self.create_encoders(embed_dim, num_heads,
                                                attn_dropout, proj_dropout,
                                                mlp_dropout, feedforward_dim,
                                                num_encoders)

        self.attention_pool = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        #for param in self.vgg.parameters():
        #    param.requires_grad = False



    def create_encoders(self, embed_dim=192,
                        num_heads=8,
                        attn_dropout=0.5,
                        proj_dropout=0.5,
                        mlp_dropout=0.1,
                        feedforward_dim=768,
                        num_layers=2,
                       ):
        return nn.Sequential(*[EncoderLayer(embed_dim, num_heads, attn_dropout, proj_dropout, mlp_dropout, feedforward_dim) for _ in range(num_layers)])


    def forward(self, x):
        x = self.vgg(x)
        x = self.patcher(x)
        x = self.gap(x)
        x = self.fc1(x).squeeze(dim=2)
        x = self.transformer(x)
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x)
        fer_answer = self.fc(x).squeeze(1)
        return x, fer_answer