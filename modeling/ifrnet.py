import torch
from torch import nn
from torch.nn.utils import spectral_norm

from modeling.base import BaseNetwork
from modules.blocks import DestyleResBlock, Destyler, ResBlock


class IFRNet(BaseNetwork):
    def __init__(self, base_n_channels, destyler_n_channels):
        super(IFRNet, self).__init__()
        self.destyler = Destyler(in_features=32768, num_features=destyler_n_channels)  # from vgg features

        self.ds_fc1 = nn.Linear(destyler_n_channels, base_n_channels * 2)
        self.ds_res1 = DestyleResBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=1, padding=2)
        self.ds_fc2 = nn.Linear(destyler_n_channels, base_n_channels * 4)
        self.ds_res2 = DestyleResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2, padding=1)
        self.ds_fc3 = nn.Linear(destyler_n_channels, base_n_channels * 4)
        self.ds_res3 = DestyleResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.ds_fc4 = nn.Linear(destyler_n_channels, base_n_channels * 8)
        self.ds_res4 = DestyleResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, kernel_size=3, stride=2, padding=1)
        self.ds_fc5 = nn.Linear(destyler_n_channels, base_n_channels * 8)
        self.ds_res5 = DestyleResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.ds_fc6 = nn.Linear(destyler_n_channels, base_n_channels * 16)
        self.ds_res6 = DestyleResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.res1 = ResBlock(channels_in=base_n_channels * 8, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.res2 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.res3 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res4 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res5 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(base_n_channels, 3, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, x, vgg_feat):
        b_size, ch, h, w = vgg_feat.size()
        vgg_feat = vgg_feat.view(b_size, ch * h * w)
        vgg_feat = self.destyler(vgg_feat)

        out = self.ds_res1(x, self.ds_fc1(vgg_feat))
        out = self.ds_res2(out, self.ds_fc2(vgg_feat))
        out = self.ds_res3(out, self.ds_fc3(vgg_feat))
        out = self.ds_res4(out, self.ds_fc4(vgg_feat))
        out = self.ds_res5(out, self.ds_fc5(vgg_feat))
        aux = self.ds_res6(out, self.ds_fc6(vgg_feat))

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


class MLP(nn.Module):
    def __init__(self, base_n_channels, num_class=14):
        super(MLP, self).__init__()
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(base_n_channels * 8, base_n_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(base_n_channels * 4, base_n_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            # nn.Conv2d(base_n_channels * 2, base_n_channels * 1, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(base_n_channels * 8 * 8 * 2, num_class),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.aux_classifier(x)


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


class Discriminator(BaseNetwork):
    def __init__(self, base_n_channels):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.image_to_features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(4 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            # nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        output_size = 8 * base_n_channels * 3 * 3
        self.features_to_prob = nn.Sequential(
            spectral_norm(nn.Conv2d(8 * base_n_channels, 2 * base_n_channels, 5, 2, 1)),
            Flatten(),
            nn.Linear(output_size, 1)
        )

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, input_data):
        x = self.image_to_features(input_data)
        return self.features_to_prob(x)


class PatchDiscriminator(Discriminator):
    def __init__(self, base_n_channels):
        super(PatchDiscriminator, self).__init__(base_n_channels)

        self.features_to_prob = nn.Sequential(
            spectral_norm(nn.Conv2d(8 * base_n_channels, 1, 1)),
            Flatten()
        )

    def forward(self, input_data):
        x = self.image_to_features(input_data)
        return self.features_to_prob(x)


if __name__ == '__main__':
    import torchvision
    ifrnet = IFRNet(32, 128).cuda()
    x = torch.rand((2, 3, 256, 256)).cuda()
    vgg16 = torchvision.models.vgg16(pretrained=True).features.eval().cuda()
    with torch.no_grad():
        vgg_feat = vgg16(x)
    output, aux_out = ifrnet(x, vgg_feat)
    print(output.size())
    print(aux_out.size())

    disc = Discriminator(32).cuda()
    d_out = disc(output)
    print(d_out.size())

    patch_disc = PatchDiscriminator(32).cuda()
    p_d_out = patch_disc(output)
    print(p_d_out.size())

