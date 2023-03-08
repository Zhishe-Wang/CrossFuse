import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import utils


# Convolution operation 填充卷积激活
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Prelu=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Prelu = use_Prelu
        self.PReLU=nn.PReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.use_Prelu is True:
            out = self.PReLU(out)
        return out

class ConvLayer_dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_Leakyrelu=True):
        super(ConvLayer_dis, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_Leakyrelu = use_Leakyrelu
        self.LeakyReLU=nn.LeakyReLU(0.2)
    def forward(self, x):

        out = self.conv2d(x)
        if self.use_Leakyrelu is True:
            out = self.LeakyReLU(out)
        return out

class CAIM(torch.nn.Module):
    def __init__(self,channels):
        super(CAIM, self).__init__()
        self.ca_avg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels,channels//2,kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels//2,channels,kernel_size=1),
                    )

        self.ca_max = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Conv2d(channels, channels // 2, kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels // 2, channels, kernel_size=1),

    )
        self.sigmod = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2*channels,channels,1,1)
    def forward(self,x):
        #CA
        w_avg = self.ca_avg(x)
        w_max = self.ca_max(x)
        w = torch.cat([w_avg,w_max], dim=1)
        w = self.conv1(w)
        #SA
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        output = self.sigmod(w * x1)
        return output

class UpsampleReshape(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, shape, x):
        x = self.up(x)

        shape = shape.size()
        shape_x = x.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape[3] != shape_x[3]:
            lef_right = shape[3] - shape_x[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape[2] != shape_x[2]:
            top_bot = shape[2] - shape_x[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        kernel_size_2 = 3
        self.save = utils.save_feat
        # encoder_convlayer
        self.MCB1 = nn.Sequential(ConvLayer(1,16,3,1,True),ConvLayer(16, 16, 3, 1, True))
        self.MCB2 = nn.Sequential(ConvLayer(16, 32, 3, 1, True),ConvLayer(32, 32, 3, 2, True))
        self.MCB3 = nn.Sequential(ConvLayer(32, 64, 3, 1, True), ConvLayer(64, 64, 3, 2, True))
        self.MCB4 = nn.Sequential(ConvLayer(64, 128, 3, 1, True),ConvLayer(128, 128, 3, 2, True))

        self.CAIM1 = CAIM(16)
        self.CAIM2 = CAIM(32)
        self.CAIM3 = CAIM(64)
        self.CAIM4 = CAIM(128)

        # decoder
        self.UP = UpsampleReshape()

        self.conv_1 = ConvLayer(256, 128, 3, stride=1, use_Prelu=True)
        self.conv_2 = ConvLayer(256, 128, 3, stride=1, use_Prelu=True)
        self.conv_3 = ConvLayer(128, 64, 3, stride=1, use_Prelu=True)
        self.conv_4 = ConvLayer(128, 64, 3, stride=1, use_Prelu=True)
        self.conv_5 = ConvLayer(64, 32, 3, stride=1, use_Prelu=True)
        self.conv_6 = ConvLayer(64, 32, 3, stride=1, use_Prelu=True)
        self.conv_7 = ConvLayer(32, 16, 3, stride=1, use_Prelu=True)
        self.conv_8 = ConvLayer(32, 1, 3, stride=1, use_Prelu=False)
        self.tanh = nn.Tanh()
    def forward(self,input_ir,input_vis):
    #Encoder
        # MCB1
        ir1 = self.MCB1(input_ir) # 16    256
        # MCB2
        ir2 = self.MCB2(ir1)# 32     128
        # MCB3
        ir3 = self.MCB3(ir2) #64     64
        # MCB4
        ir4 = self.MCB4(ir3) #128   32
        # MCB1
        vis1 = self.MCB1(input_vis)
        # MCB2
        vis2 = self.MCB2(vis1)
        # MCB3
        vis3 = self.MCB3(vis2)
        # MCB4
        vis4 = self.MCB4(vis3)
    #CROSS-SCALE-DECODER
        # CAIM4
        F_4_ = torch.cat([ir4,vis4],dim=1)#256
        F_4 = self.conv_1(F_4_)#128
        fusion4_w = self.CAIM4(F_4)
        fusion4_=torch.cat([fusion4_w*ir4 ,(1-fusion4_w)*vis4],dim=1)#256
        fusion4=self.conv_2(fusion4_)

        # CAIM3
        F_3_ = self.UP(ir3,fusion4)
        F_3 = self.conv_3(F_3_)
        fusion3_w = self.CAIM3(F_3)
        fusion3_ = torch.cat([fusion3_w * ir3, (1 - fusion3_w) * vis3], dim=1)#128
        fusion3 = self.conv_4(fusion3_)

        # CAIM2
        F_2_ = self.UP(ir2, fusion3)
        F_2 = self.conv_5(F_2_)
        fusion2_w = self.CAIM2(F_2)
        fusion2_ = torch.cat([fusion2_w * ir2, (1 - fusion2_w) * vis2], dim=1)  # 64
        fusion2 = self.conv_6(fusion2_)#32

        # CAIM1
        F_1_ = self.UP(ir1, fusion2)
        F_1 = self.conv_7(F_1_)
        fusion1_w = self.CAIM1(F_1)
        fusion1 = torch.cat([fusion1_w * ir1, (1 - fusion1_w) * vis1], dim=1)  #32
        out = self.conv_8(fusion1) #32-1
        output = self.tanh(out)

        return output




class  D_IR(nn.Module):
    def __init__(self):
        super( D_IR, self).__init__()
        fliter = [1,16,32,64,128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_Leakyrelu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_Leakyrelu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_Leakyrelu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_Leakyrelu=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out =self.tanh(out)

        return out

class  D_VI(nn.Module):
    def __init__(self):
        super( D_VI, self).__init__()
        fliter = [1,16,32,64,128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_Leakyrelu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_Leakyrelu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_Leakyrelu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_Leakyrelu=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.tanh(out)

        return out