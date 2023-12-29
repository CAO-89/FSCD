import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

from .help_funcs import *

class SUNet18(nn.Module):
    def __init__(self, in_ch, out_ch,
        nonlinearity =  partial(F.relu, inplace=True),
        base_model = models.resnet18(pretrained = True),
        share_encoder = False,
        base_model2 = models.resnet18(pretrained = True),
        last_layer = 'tanh',
        ):

        super().__init__()

        self.name = 'SUNet'
        self.share_encoder = share_encoder

        C = [32, 64, 128, 256, 512, 1024]

        resnet1 = base_model

        # Encoder first (and second) image
        self.firstconv1 = resnet1.conv1
        self.firstbn1 = resnet1.bn1
        self.firstrelu1 = resnet1.relu
        self.firstmaxpool1 = resnet1.maxpool

        self.encoder11 = resnet1.layer1
        self.encoder12 = resnet1.layer2
        self.encoder13 = resnet1.layer3
        self.encoder14 = resnet1.layer4

        if share_encoder:
          pass
        else:
          resnet2 = base_model2

          # Encoder second image
          self.firstconv2 = resnet2.conv1
          self.firstbn2 = resnet2.bn1
          self.firstrelu2 = resnet2.relu
          self.firstmaxpool2 = resnet2.maxpool

          self.encoder21 = resnet2.layer1
          self.encoder22 = resnet2.layer2
          self.encoder23 = resnet2.layer3
          self.encoder24 = resnet2.layer4
        
        # Decoder
        # self.conv5d = DecBlock(C[4], C[4], C[3])
        # self.conv4d = DecBlock(C[3]+C[3], C[3], C[2])
        # self.conv3d = DecBlock(C[2]+C[2], C[2], C[1])
        # self.conv2d = DecBlock(C[1]+C[1], C[1], C[1])
        # self.conv1d = DecBlock(C[1]+C[1], C[1], C[0]) #, out_ch, bn=False, act=False)
        self.conv1d_1 = Conv3x3(C[3], C[2], bn=True, act=True)
        self.conv1d_2 = DecBlock(C[2], C[2], C[1])
        self.conv1d_3 = DecBlock1(C[1], C[0])

        self.finaldeconv11 = nn.ConvTranspose2d(C[0], 32, 4, 2, 1)
        self.finalrelu11 = nonlinearity
        self.finalconv12 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu12 = nonlinearity
        self.finalconv13 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.finalnonlin1 = nn.LogSoftmax(dim=1)

        
    def forward(self, st1, st2, smask, qt1, qt2):
      
        # st1-->encoder
        sx11 = self.firstconv1(st1)
        sx11 = self.firstbn1(sx11)
        sx11 = self.firstrelu1(sx11)
        sxp11 = self.firstmaxpool1(sx11) 
        sxp12 = self.encoder11(sxp11)
        sxp13 = self.encoder12(sxp12)
        # sxp14 = self.encoder13(sxp13)
        # sxp15 = self.encoder14(sxp14)
        
        # st2-->encoder
        sx21 = self.firstconv1(st2)
        sx21 = self.firstbn1(sx21)
        sx21 = self.firstrelu1(sx21)
        sxp21 = self.firstmaxpool1(sx21) 
        sxp22 = self.encoder11(sxp21)
        sxp23 = self.encoder12(sxp22)
        # sxp24 = self.encoder13(sxp23)
        # sxp25 = self.encoder14(sxp24)
        
        # qt1-->encoder
        qx11 = self.firstconv1(qt1)
        qx11 = self.firstbn1(qx11)
        qx11 = self.firstrelu1(qx11)
        qxp11 = self.firstmaxpool1(qx11) 
        qxp12 = self.encoder11(qxp11)
        qxp13 = self.encoder12(qxp12)
        # qxp14 = self.encoder13(qxp13)
        # qxp15 = self.encoder14(qxp14)

        # qt2-->encoder
        qx21 = self.firstconv1(qt2)
        qx21 = self.firstbn1(qx21)
        qx21 = self.firstrelu1(qx21)
        qxp21 = self.firstmaxpool1(qx21) 
        qxp22 = self.encoder11(qxp21)
        qxp23 = self.encoder12(qxp22)
        # qxp24 = self.encoder13(qxp23)
        # qxp25 = self.encoder14(qxp24)

        # concate st1,st2
        sup = torch.cat((sxp13,sxp23), dim=1)

        # concate qt1,qt2
        que = torch.cat((qxp13,qxp23), dim=1)
        smask = smask.unsqueeze(1)
        smask = torch.as_tensor(smask, dtype=torch.float32)
        
        # mask * support
        sup_mask = F.interpolate(smask, sup.shape[-2:], mode='bilinear',align_corners=True)
        h,w=sup.shape[-2:][0],sup.shape[-2:][1]

        area = F.avg_pool2d(sup_mask, sup.shape[-2:]) * h * w + 0.0005
        z = sup_mask * sup

        z = F.avg_pool2d(input=z,
                         kernel_size=sup.shape[-2:]) * h * w / area
        z = z.expand(-1, -1, sup.shape[-2:][0], sup.shape[-2:][1])
        
        # # concate qeu,sup
        # xd = self.conv3d(torch.cat((skip13,skip23), dim=1), xd)

        # # Stage 2d
        # xd = self.conv2d(torch.cat((skip12,skip22), dim=1), xd)

        # # Stage 1d
        # xd = self.conv1d(torch.cat((x11,x21), dim=1), xd)        
        # Decode
        que = self.conv1d_1(que)
        z = self.conv1d_1(z)
        xd = self.conv1d_2(que, z)
        xd = self.conv1d_3(xd)

        # Final deconv for 2D map
        out2d = self.finaldeconv11(xd)
        out2d = self.finalrelu11(out2d)
        out2d = self.finalconv12(out2d)
        out2d = self.finalrelu12(out2d)
        out2d = self.finalconv13(out2d)
        out2d = self.finalnonlin1(out2d)
          
        return out2d