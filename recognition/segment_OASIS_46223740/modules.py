import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------- #
# Basic building block: Double Convolution #
# ---------------------------------------- #
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)
    
# ---------------------------------- #
#             Base U-Net             #
# ---------------------------------- #
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits
    

class ResidualDoubleConv(nn.Module):
    """(Conv-BN-ReLU-Conv-BN + Residual connection)"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        # if input and output channels differ, use 1x1 conv to match dimensions
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual

        return self.relu(out)
        

class ImprovedUNet(nn.Module):
    """Improved U-Net with BatchNorm, Dropout, and Residual connections"""
    def __init__(self, n_channels, n_classes, dropout=0.3):
        super().__init__()
        self.inc = ResidualDoubleConv(n_channels, 64, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(64, 128, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(128, 256, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(256, 512, dropout))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualDoubleConv(512, 1024, dropout))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = ResidualDoubleConv(1024, 512, dropout)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = ResidualDoubleConv(512, 256, dropout)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = ResidualDoubleConv(256, 128, dropout)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = ResidualDoubleConv(128, 64, dropout)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits
    

# if __name__ == "__main__":
#     model = ImprovedUNet(n_channels=1, n_classes=3)
#     x = torch.randn(1, 1, 256, 256)
#     y = model(x)
#     print(y.shape)