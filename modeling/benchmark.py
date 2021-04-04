import torch
from torch import nn

from modeling.base import BaseNetwork
from modeling.ifrnet import Flatten
from modules.blocks import DestyleResBlock, Destyler, ResBlock


class UNet(BaseNetwork):
    def __init__(self, base_n_channels):
        super(UNet, self).__init__()

        self.ds_res1 = ResBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=1, padding=2)
        self.ds_res2 = ResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2, padding=1)
        self.ds_res3 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.ds_res4 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, kernel_size=3, stride=2, padding=1)
        self.ds_res5 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.ds_res6 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.res1 = ResBlock(channels_in=base_n_channels * 8, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.res2 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.res3 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res4 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res5 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(base_n_channels, 3, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, x):
        out = self.ds_res1(x)
        out = self.ds_res2(out)
        out = self.ds_res3(out)
        out = self.ds_res4(out)
        out = self.ds_res5(out)
        aux = self.ds_res6(out)

        out = self.upsample(aux)
        out = self.res1(out)
        out = self.res2(out)
        out = self.upsample(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.upsample(out)
        out = self.res5(out)
        out = self.conv1(out)

        return out, aux


if __name__ == '__main__':
    import torchvision
    x = torch.rand((2, 3, 256, 256)).cuda()
    unet = UNet(32, 32).cuda()
    vgg16 = torchvision.models.vgg16(pretrained=True).features.eval().cuda()
    with torch.no_grad():
        vgg_feat = vgg16(x)
    out = unet(x, vgg_feat)

    print(out.size())
