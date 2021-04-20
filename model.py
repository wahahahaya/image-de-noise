import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.up(x)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        embedding = x6.view(-1, 512 * 8 * 8)

        return embedding, x6

class Decoder_denoise(nn.Module):
    def __init__(self):
        super(Decoder_denoise, self).__init__()
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        logits = self.outc(x5)

        return logits


class Decoder_SP(nn.Module):
    def __init__(self):
        super(Decoder_SP, self).__init__()
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        logits = self.outc(x5)

        return logits


class Decoder_AWGN(nn.Module):
    def __init__(self):
        super(Decoder_AWGN, self).__init__()
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.up5 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        x5 = self.up5(x4)
        logits = self.outc(x5)

        return logits


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()

        self.output_shape = (1, 64, 64)

        self.dis = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(),
            #M-1
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-2
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-3
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-4
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-5
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-6
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-7
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(),
            #M-8
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.dis(x)


class VGG19_fea(nn.Module):
    def __init__(self):
        super(VGG19_fea, self).__init__()
        self.conv1to3 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        model = models.vgg19_bn(pretrained=True)
        self.vgg19_fea = model.features
        
    def forward(self, x):
        x = self.conv1to3(x)
        x = self.relu(x)
        x = self.vgg19_fea(x)

        return(x)


if __name__ == "__main__":
   device = "cuda" if torch.cuda.is_available() else "cpu"
   x = torch.randn(1 ,3, 256, 256).to(device)
   model = VGG19_fea().to(device)
   dec = Decoder_AWGN().to(device)
   print(model(x).shape)
