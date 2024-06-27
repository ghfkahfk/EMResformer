import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math
from function.modules import Subtraction, Subtraction2, Aggregation
from torchvision.models import vgg19

class SEBlock(nn.Module):
    def __init__(self, input_dim=settings.channel, reduction=4):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)
class Residual_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Residual_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if self.in_channel != self.out_channel:
            self.convert = nn.Conv2d(self.in_channel, self.out_channel, 1, 1)
        self.res = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1),
        )

    def forward(self, x):
        if self.in_channel != self.out_channel:
            convert = self.convert(x)
        else:
            convert = x
        out = convert + self.res(x)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return input*self.sigmoid(x)

# class SpaceAware(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self):
#         super(SpaceAware, self).__init__()
#         self.channel = settings.channel
#         self.pool1 = nn.MaxPool2d(1, 1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.pool4 = nn.MaxPool2d(4, 4)
#
#
#         self.conv12 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv22 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.conv42 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#
#
#         self.spa1 = SpatialAttention(kernel_size=3)
#         self.spa2 = SpatialAttention(kernel_size=3)
#         self.spa4 = SpatialAttention(kernel_size=3)
#
#
#         self.fusion = nn.Conv2d(3*self.channel, self.channel,1,1)
#
#     def forward(self, x):
#         pool1 = self.pool1(x)
#         b1, c1, h1, w1 = pool1.size()
#         pool2 = self.pool2(x)
#         b2, c2, h2, w2 = pool2.size()
#         pool4 = self.pool4(x)
#
#         spa1 = self.spa1(pool1)
#         spa2 = self.spa2(pool2)
#         spa4 = self.spa4(pool4)
#
#         conv12 = self.conv12(spa1)
#         conv22 = self.conv22(spa2)
#         conv42 = self.conv42(spa4)
#
#         conv = self.fusion(torch.cat([conv12,F.upsample(conv22, [h1, w1]),F.upsample(conv42, [h1, w1])],dim=1))
#
#         return conv


class ChannelAware(nn.Module):
    def __init__(self, input_dim=settings.channel, reduction=4):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)
def adaptive_instance_normalization(center_feat, knn_feat, eps=1e-5):
    # center_feat = center_feat.contiguous()
    # knn_feat = knn_feat.contiguous()
    b,c,h,w = center_feat.size()
    _,_,_,_ = knn_feat.size()
    knn_feat_ori = knn_feat
    knn_feat = knn_feat.view(b, c, -1)
    center_feat = center_feat.view(b, c, -1)

    center_var = center_feat.var(dim=2) + eps
    center_std = center_var.sqrt().view(b, c, 1, 1)
    center_mean = center_feat.mean(dim=2).view(b, c, 1, 1)

    knn_var = knn_feat.var(dim=2) + eps
    knn_std = knn_var.sqrt().view(b, c, 1, 1)
    knn_mean = knn_feat.mean(dim=2).view(b, c, 1, 1)

    size = knn_feat_ori.size()
    normalized_feat = (knn_feat_ori - knn_mean.expand(size)) / knn_std.expand(size)

    return normalized_feat * center_std.expand(size) + center_mean.expand(size)


class ScaleAwareLightRepresentationModel(nn.Module):
    def __init__(self, input_dim=settings.channel):
        super().__init__()
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.conv = nn.Sequential(Residual_Block(input_dim,input_dim),Residual_Block(input_dim,input_dim))
        self.R1 = nn.Sequential(nn.Conv2d(input_dim,input_dim,3,1,1),nn.LeakyReLU(0.2))
        self.I1 = nn.Sequential(nn.Conv2d(input_dim,input_dim,3,1,1),nn.LeakyReLU(0.2))
        if settings.spaceaware is True:
            self.spaceaware1 = SpatialAttention()
        if settings.channelaware is True:
            self.channelaware1 = ChannelAware()

        self.R2 = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.I2 = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        if settings.spaceaware is True:
            self.spaceaware2 = SpatialAttention()
        if settings.channelaware is True:
            self.channelaware2 = ChannelAware()

        self.R4 = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.I4 = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, 1, 1), nn.LeakyReLU(0.2))
        if settings.spaceaware is True:
            self.spaceaware4 = SpatialAttention()
        if settings.channelaware is True:
            self.channelaware4 = ChannelAware()

        self.fusion = nn.Sequential(nn.Conv2d(3*input_dim, input_dim, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        if settings.scale is True:
            conv1 = self.conv(x)
            pool1 = self.pool1(conv1)
            b1, c1, h1, w1 = pool1.size()
            pool2 = self.pool2(conv1)
            b2, c2, h2, w2 = pool2.size()
            pool4 = self.pool4(conv1)
        else:
            conv1 = self.conv(x)
            pool1 = conv1
            b1, c1, h1, w1 = pool1.size()
            pool2 = conv1
            b2, c2, h2, w2 = pool2.size()
            pool4 = conv1

        R1 = self.R1(pool1)
        I1 = self.I1(pool1)
        if settings.spaceaware is True:
            space1 = self.spaceaware1(R1)
        else:
            space1 = R1
        if settings.channelaware is True:
            channel1 = self.channelaware1(I1)
        else:
            channel1 = I1
        if settings.norm is True:
            R1AdaIN = adaptive_instance_normalization(R1, space1)
            I1AdaIN = adaptive_instance_normalization(I1, channel1)
        else:
            R1AdaIN = space1
            I1AdaIN = channel1

        R2 = self.R2(pool2)
        I2 = self.I2(pool2)
        if settings.spaceaware is True:
            space2 = self.spaceaware2(R2)
        else:
            space2 = R2
        if settings.channelaware is True:
            channel2 = self.channelaware2(I2)
        else:
            channel2 = I2
        if settings.norm is True:
            R2AdaIN = adaptive_instance_normalization(R2, space2)
            I2AdaIN = adaptive_instance_normalization(I2, channel2)
        else:
            R2AdaIN = space2
            I2AdaIN = channel2

        R4 = self.R4(pool4)
        I4 = self.I4(pool4)
        if settings.spaceaware is True:
            space4 = self.spaceaware4(R4)
        else:
            space4 = R4
        if settings.channelaware is True:
            channel4 = self.channelaware4(I4)
        else:
            channel4 = I4
        if settings.norm is True:
            R4AdaIN = adaptive_instance_normalization(R4, space4)
            I4AdaIN = adaptive_instance_normalization(I4, channel4)
        else:
            R4AdaIN = space4
            I4AdaIN = channel4
        if settings.cat_formmer is True:
            R_cat = torch.cat([R1AdaIN, F.upsample(R2AdaIN, [h1, w1]), F.upsample(R4AdaIN, [h1, w1])], dim=1)
            I_cat = torch.cat([I1AdaIN, F.upsample(I2AdaIN, [h1, w1]), F.upsample(I4AdaIN, [h1, w1])], dim=1)
            multi = R_cat * I_cat
            fusion = self.fusion(multi)
            out = fusion + x
            return out
        else:
            fusion1 = R1AdaIN * I1AdaIN
            fusion2 = R2AdaIN * I2AdaIN
            fusion4 = R4AdaIN * I4AdaIN
            fusion = self.fusion(torch.cat([fusion1, F.upsample(fusion2, [h1, w1]), F.upsample(fusion4, [h1, w1])], dim=1))
            out = fusion + x
            return out

class ResidualGroup(nn.Module):
    def __init__(self,):
        super(ResidualGroup, self).__init__()
        self.channel = settings.channel
        self.conv = nn.Sequential(Residual_Block(self.channel,self.channel),
                                  Residual_Block(self.channel,self.channel),
                                  Residual_Block(self.channel,self.channel))
    def forward(self, x):
        conv = self.conv(x)
        return conv

class LightRepresentation(nn.Module):
    def __init__(self,):
        super(LightRepresentation, self).__init__()
        self.channel = settings.channel
        if settings.representation is True:
            self.conv = ScaleAwareLightRepresentationModel()
        else:
            self.conv = nn.Sequential(Residual_Block(self.channel, self.channel),
                                      Residual_Block(self.channel, self.channel),
                                      Residual_Block(self.channel, self.channel))
    def forward(self, x):
        conv = self.conv(x)
        return conv


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.channel = settings.channel
        self.enter = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exit = nn.Conv2d(self.channel, 3, 3, 1, 1)
        self.conv11 = LightRepresentation()
        self.conv12 = LightRepresentation()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv21 = LightRepresentation()
        self.conv22 = LightRepresentation()
        self.conv23 = LightRepresentation()
        self.deconv11 = LightRepresentation()
        self.deconv12 = LightRepresentation()
    def forward(self, x):
        enter = self.enter(x)
        conv11 = self.conv11(enter)
        conv12 = self.conv12(conv11)
        b1, c1, h1, w1 = conv12.size()
        pool1 = self.pool1(conv12)
        conv21 = self.conv21(pool1)

        conv22 = self.conv22(conv21)
        b2, c2, h2, w2 = conv22.size()
        pool2 = self.pool1(conv22)
        conv23 = self.conv23(F.upsample(pool2, [h2, w2])) + conv21
        deconv11 = self.deconv11(F.upsample(conv23, [h1, w1]))+conv12
        deconv12 = self.deconv12(deconv11)+conv11


        out = self.exit(deconv12)

        return out,deconv12


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.channel = settings.channel
        self.enter = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exit = nn.Conv2d(self.channel, 3, 3, 1, 1)
        self.conv11 = LightRepresentation()
        self.conv12 = LightRepresentation()
        self.conv13 = LightRepresentation()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        enter = self.enter(x)
        conv11 = self.conv11(enter)
        b1, c1, h1, w1 = conv11.size()
        pool1 = self.pool(conv11)
        conv12 = self.conv12(pool1)
        conv13 = self.conv13(F.upsample(conv12, [h1, w1])) + conv11
        out = self.exit(conv13)

        return out, conv13

class Net8(nn.Module):
    def __init__(self, num_unit):
        super(Net8, self).__init__()
        self.channel = settings.channel
        self.enter = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv2d(self.channel * 8, self.channel * 8, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv1 = Residual_Block(self.channel, self.channel)
        # self.deconv1 = nn.Sequential(nn.Conv2d(self.channel * 8, self.channel * 8, 3, 1, 1), nn.LeakyReLU(0.2))
        self.deconv1 = Residual_Block(self.channel, self.channel)
        self.exit = nn.Conv2d(self.channel, 3, 3, 1, 1)
        # self.group_phy1 = nn.ModuleList()
        # self.group_phy2 = nn.ModuleList()
        self.num_unit = num_unit
        # for i in range(int(self.num_unit/2)):
        self.group_phy1=DenseConnection(self.num_unit, self.channel)
        # for i in range(int(self.num_unit / 2)):
        # self.group_phy2=DenseConnection(int(self.num_unit / 2), self.channel * 8)

    def forward(self, x):
        enter = self.enter(x)
        conv1 = self.conv1(enter)
        # for i in range(int(self.num_unit/2)):
        conv1 = self.group_phy1(conv1)
        # temp = conv1
        # # for i in range(int(self.num_unit/2)):
        # conv1 = self.group_phy2(conv1)
        deconv1 = self.deconv1(conv1)
        out = self.exit(deconv1) + x

        return out, deconv1


class PyramidNet(nn.Module):
    def __init__(self):
        super(PyramidNet, self).__init__()
        self.channel = settings.channel
        self.enter = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exit = nn.Conv2d(self.channel, 3, 3, 1, 1)
        self.conv11 = LightRepresentation()
        self.conv12 = LightRepresentation()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv21 = LightRepresentation()
        self.conv22 = LightRepresentation()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv31 = LightRepresentation()
        self.conv32 = LightRepresentation()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv33 = nn.Sequential(LightRepresentation(),LightRepresentation())

        self.conv34 = LightRepresentation()
        # self.fusion34 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))
        self.conv35 = LightRepresentation()
        # self.fusion35 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))

        self.deconv11 = LightRepresentation()
        # self.fusiondeconv11 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))
        self.deconv12 = LightRepresentation()
        # self.fusiondeconv12 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))

        self.deconv21 = LightRepresentation()
        # self.fusiondeconv21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))
        self.deconv22 = LightRepresentation()
        # self.fusiondeconv22 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))

        self.net21 = Net2()
        self.net22 = Net2()
        self.fusionformer2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))
        self.fusionlatter2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))

        self.net41 = Net4()
        self.net42 = Net4()
        self.fusionformer4 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))
        self.fusionlatter4 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1),nn.LeakyReLU(0.2))

    def forward(self, x1, x2, x4):
        encoder = []
        decoder = []

        enter = self.enter(x1)
        b1, c1, h1, w1 = enter.size()

        conv11 = self.conv11(enter)
        conv12 = self.conv12(conv11)
        encoder.append(conv12)

        pool1 = self.pool1(conv12)
        b2, c2, h2, w2 = pool1.size()

        out21, former2 = self.net21(x2)
        fusionformer2 = former2+ pool1

        conv21 = self.conv21(fusionformer2)
        conv22 = self.conv22(conv21)
        encoder.append(conv22)

        pool2 = self.pool2(conv22)
        b4, c4, h4, w4 = pool2.size()

        out41, former4 = self.net41(x4)
        fusionformer4 = former4+ pool2

        conv31 = self.conv31(fusionformer4)
        conv32 = self.conv32(conv31)
        encoder.append(conv32)

        pool3 = self.pool3(conv32)

        conv33 = self.conv33(pool3)
        upsampling4 = F.upsample(conv33, [h4, w4])

        out42, later4 = self.net42(out41)

        fusionlatter4 = later4+ upsampling4

        conv34 = self.conv34(fusionlatter4) + conv32
        conv35 = self.conv35(conv34) + conv31
        decoder.append(conv35)

        upsampling2 = F.upsample(conv35, [h2, w2])

        out22, later2 = self.net22(out21)
        # fusionlatter2 = self.fusionlatter2(torch.cat([later2, upsampling2], dim=1))
        fusionlatter2 = later2+ upsampling2

        deconv11 = self.deconv11(fusionlatter2)+ conv22
        deconv12 = self.deconv12(deconv11)+ conv21
        decoder.append(deconv12)

        upsampling1 = F.upsample(deconv12, [h1, w1])

        deconv21 = self.deconv21(upsampling1)+ conv12
        deconv22 = self.deconv22(deconv21)+ conv11

        decoder.append(deconv22)

        out1 = self.exit(deconv22)
        out = []
        out.append(out1)
        out.append(out21)
        out.append(out22)
        out.append(out41)
        out.append(out42)

        return out, encoder, decoder

class NoPyramidNet(nn.Module):
    def __init__(self):
        super(NoPyramidNet, self).__init__()
        self.channel = settings.channel
        self.enter = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.exit = nn.Conv2d(self.channel, 3, 3, 1, 1)
        self.conv11 = LightRepresentation()
        self.conv12 = LightRepresentation()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv21 = LightRepresentation()
        self.conv22 = LightRepresentation()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv31 = LightRepresentation()
        self.conv32 = LightRepresentation()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv33 = nn.Sequential(LightRepresentation(),LightRepresentation())
        self.conv34 = LightRepresentation()
        self.conv35 = LightRepresentation()

        self.deconv11 = LightRepresentation()
        self.deconv12 = LightRepresentation()

        self.deconv21 = LightRepresentation()
        self.deconv22 = LightRepresentation()


    def forward(self, x1,x2,x3):
        encoder = []
        decoder = []
        map = []
        enter = self.enter(x1)
        b1, c1, h1, w1 = enter.size()

        conv11 = self.conv11(enter)
        conv12 = self.conv12(conv11)
        encoder.append(conv12)
        map.append(conv12)

        pool1 = self.pool1(conv12)
        b2, c2, h2, w2 = pool1.size()

        conv21 = self.conv21(pool1)

        conv22 = self.conv22(conv21)
        encoder.append(conv22)
        map.append(conv22)

        pool2 = self.pool2(conv22)
        b3, c3, h3, w3 = pool2.size()


        conv31 = self.conv31(pool2)
        conv32 = self.conv32(conv31)
        encoder.append(conv32)
        map.append(conv32)
        pool3 = self.pool3(conv32)

        conv33 = self.conv33(pool3)

        conv34 = self.conv34(F.upsample(conv33, [h3, w3])) + conv32
        conv35 = self.conv35(conv34) + conv31
        decoder.append(conv35)
        map.append(conv35)

        upsampling2 = F.upsample(conv35, [h2, w2])

        deconv11 = self.deconv11(upsampling2) + conv22
        deconv12 = self.deconv12(deconv11) + conv21
        decoder.append(deconv11)
        map.append(deconv11)

        upsampling1 = F.upsample(deconv12, [h1, w1])

        deconv21 = self.deconv21(upsampling1) + conv12
        deconv22 = self.deconv22(deconv21) + conv11
        decoder.append(deconv22)
        map.append(deconv22)

        out1 = self.exit(deconv22)
        out = []
        out.append(out1)

        return out, map


class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        if settings.pyramid is True:
            self.net = PyramidNet()
        else:
            self.net = NoPyramidNet()

    def forward(self, x1,x2,x3):
        out, map = self.net(x1, x2, x3)
        return out, map


import torch.nn.utils.spectral_norm as spectral_norm

class PyramidDiscriminator_Common(nn.Module):
    def __init__(self):
        super(PyramidDiscriminator_Common, self).__init__()
        self.channel = 24
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn):
        enter = self.enter(syn)
        # print('enter',enter.size())
        # print('features[0]', features[0].size())
        conv1 = self.conv1(enter)

        pool1 = self.pool(conv1)
        conv2 = self.conv2(pool1)

        pool2 = self.pool(conv2)
        conv3 = self.conv3(pool2)

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4

class PyramidDiscriminator_Fusion(nn.Module):
    def __init__(self):
        super(PyramidDiscriminator_Fusion, self).__init__()
        self.channel = 48
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + 2*settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + 2*settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + 2*settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn, encoder_features, decoder_features):
        enter = self.enter(syn)
        # print('enter',enter.size())
        # print('features[0]', features[0].size())
        conv1 = self.conv1(torch.cat([enter, encoder_features[0], decoder_features[2]],dim=1))

        pool1 = self.pool(conv1)
        conv2 = self.conv2(torch.cat([pool1, encoder_features[1], decoder_features[1]],dim=1))

        pool2 = self.pool(conv2)
        conv3 = self.conv3(torch.cat([pool2, encoder_features[2], decoder_features[0]],dim=1))

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4

class PyramidDiscriminator_Encoder(nn.Module):
    def __init__(self):
        super(PyramidDiscriminator_Encoder, self).__init__()
        self.channel = 24
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn, features):
        enter = self.enter(syn)
        # print('enter',enter.size())
        # print('features[0]', features[0].size())
        conv1 = self.conv1(torch.cat([enter, features[0]],dim=1))

        pool1 = self.pool(conv1)
        conv2 = self.conv2(torch.cat([pool1, features[1]],dim=1))

        pool2 = self.pool(conv2)
        conv3 = self.conv3(torch.cat([pool2, features[2]],dim=1))

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4


class PyramidDiscriminator_Decoder(nn.Module):
    def __init__(self):
        super(PyramidDiscriminator_Decoder, self).__init__()
        self.channel = 24
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn, features):
        enter = self.enter(syn)
        conv1 = self.conv1(torch.cat([enter, features[-1]],dim=1))

        pool1 = self.pool(conv1)
        conv2 = self.conv2(torch.cat([pool1, features[-2]],dim=1))

        pool2 = self.pool(conv2)
        conv3 = self.conv3(torch.cat([pool2, features[-3]],dim=1))

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4

class NoPyramidDiscriminator_Encoder(nn.Module):
    def __init__(self):
        super(NoPyramidDiscriminator_Encoder, self).__init__()
        self.channel = 24
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn, features):
        enter = self.enter(syn)
        # print('enter',enter.size())
        # print('features[0]', features[0].size())
        conv1 = self.conv1(torch.cat([enter, features[0]],dim=1))

        pool1 = self.pool(conv1)
        conv2 = self.conv2(torch.cat([pool1, features[1]],dim=1))

        pool2 = self.pool(conv2)
        conv3 = self.conv3(torch.cat([pool2, features[2]],dim=1))

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4


class NoPyramidDiscriminator_Decoder(nn.Module):
    def __init__(self):
        super(NoPyramidDiscriminator_Decoder, self).__init__()
        self.channel = 24
        self.enter = nn.Sequential(spectral_norm(nn.Conv2d(3, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel + settings.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(spectral_norm(nn.Conv2d(self.channel, self.channel, 3, 1, 1)), nn.LeakyReLU(0.2),
                                   nn.Conv2d(self.channel, 1, 3, 1, 1))

        # self.exit = nn.Sequential(spectral_norm(nn.Conv2d(self.channel*16, 1, 3, 1, 1)))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, syn, features):
        enter = self.enter(syn)
        conv1 = self.conv1(torch.cat([enter, features[-1]],dim=1))

        pool1 = self.pool(conv1)
        conv2 = self.conv2(torch.cat([pool1, features[-2]],dim=1))

        pool2 = self.pool(conv2)
        conv3 = self.conv3(torch.cat([pool2, features[-3]],dim=1))

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # exit = self.exit(conv4)

        return conv4



class Discriminator_Encoder(nn.Module):
    def __init__(self):
        super(Discriminator_Encoder, self).__init__()
        if settings.pyramid is True:
            self.gan = PyramidDiscriminator_Encoder()
        else:
            self.gan = NoPyramidDiscriminator_Encoder()

    def forward(self, syn, features):
        out = self.gan(syn, features)
        return out


class Discriminator_Decoder(nn.Module):
    def __init__(self):
        super(Discriminator_Decoder, self).__init__()
        if settings.pyramid is True:
            self.gan = PyramidDiscriminator_Decoder()
        else:
            self.gan = NoPyramidDiscriminator_Decoder()

    def forward(self, syn, features):
        out = self.gan(syn, features)
        return out

class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(1, 3, 5, 9, 13), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features