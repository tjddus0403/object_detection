#############################################################################################################
#############################################################################################################
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101

class DeepLabV3ResNet101(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(DeepLabV3ResNet101, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=weights, progress=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# 사용 예시
# num_classes = 1
# weights = 'COCO_WITH_VOC_LABELS_V1'  # Replace this with the path to your custom weights file if needed.
# model = DeepLabV3ResNet101(num_classes, weights=weights)

#############################################################################################################
#############################################################################################################

import torch
import torch.nn as nn
from torchvision.models import resnet34

class UNet_ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(UNet_ResNet34, self).__init__()
        self.encoder = resnet34(weights="DEFAULT")

        # Use the encoder's layers as the first few layers for our model
        self.conv = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.enc1 = nn.Sequential(self.encoder.layer1)
        self.enc2 = nn.Sequential(self.encoder.layer2)
        self.enc3 = nn.Sequential(self.encoder.layer3)
        self.enc4 = nn.Sequential(self.encoder.layer4)

        # Define the U-Net decoder layers
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        # Define the decoder layers with skip connections
        self.dec4 = self._decoder_block(1024, 256)
        self.dec3 = self._decoder_block(512, 128)
        self.dec2 = self._decoder_block(256, 64)
        self.dec1 = self._decoder_block(128, 64)
        # self.dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    # def _decoder_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True)
    #     )
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("x",x.shape)
        conv = self.conv(x)
        # print("conv",conv.shape)
        enc1 = self.enc1(conv)
        # print("1", enc1.shape)
        enc2 = self.enc2(enc1)
        # print("2", enc2.shape)
        enc3 = self.enc3(enc2)
        # print("3", enc3.shape)
        enc4 = self.enc4(enc3)
        # print("4",enc4.shape)
        # enc5 = self.enc5(enc4)
        # print(enc5.shape)
        center = self.center(enc4)
        # print("center", center.shape)
        # print(torch.cat([center, enc4], 1).shape)
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        # print("4",dec4.shape)
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        # print("3",dec3.shape)
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        # print("2",dec2.shape)
        # print(torch.cat([dec2, enc1], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        # print("1",dec1.shape)
        final = self.final(dec1)

        return final
    
#############################################################################################################
#############################################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # PyramidNet basic block
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# # PyramidNet encoder backbone
# class PyramidNetEncoder(nn.Module):
#     def __init__(self, block, num_blocks, alpha, num_classes=10):
#         super(PyramidNetEncoder, self).__init__()
#         self.in_channels = 16
#         self.alpha = alpha
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             self.in_channels = self.in_channels + self.alpha
#             layers += [block(self.in_channels, out_channels, stride)]
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
    
# class UpConv(nn.Module):
#     def __init__(self, in_channels, out_channels, output_padding=0):
#         super(UpConv, self).__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=output_padding)

#     def forward(self, x):
#         return self.upconv(x)

# class PyramidNetUNet(nn.Module):
#     def __init__(self, pyramidnet_encoder, n_classes):
#         super(PyramidNetUNet, self).__init__()
#         self.encoder = pyramidnet_encoder
#         self.middle = BasicBlock(64, 128, stride=2)
        
#         self.upconv1 = UpConv(128, 64)
#         self.decoder1 = BasicBlock(128, 64, stride=1)
#         self.upconv2 = UpConv(64, 32)
#         self.decoder2 = BasicBlock(64, 32, stride=1)
#         self.upconv3 = UpConv(32, 16)
#         self.decoder3 = BasicBlock(32, 16, stride=1)
        
#         self.upconv_final = nn.Conv2d(16, n_classes, kernel_size=1)
        
#     def forward(self, x):
#         out_encoder1 = self.encoder.layer1(self.encoder.bn1(self.encoder.conv1(x)))
#         out_encoder2 = self.encoder.layer2(out_encoder1)
#         out_encoder3 = self.encoder.layer3(out_encoder2)
        
#         out_middle = self.middle(out_encoder3)
        
#         out_upconv1 = self.upconv1(out_middle)
#         out_decoder1 = self.decoder1(torch.cat([out_encoder3, out_upconv1], dim=1))
        
#         out_upconv2 = self.upconv2(out_decoder1)
#         out_decoder2 = self.decoder2(torch.cat([out_encoder2, out_upconv2], dim=1))
        
#         out_upconv3 = self.upconv3(out_decoder2)
#         out_decoder3 = self.decoder3(torch.cat([out_encoder1, out_upconv3], dim=1))
        
#         out_final = self.upconv_final(out_decoder3)
        
#         return out_final

# def main():
#     net = PyramidNetUNet(PyramidNetEncoder(BasicBlock, [4, 9, 4], alpha=16), n_classes=10)
#     print(net)

# if __name__ == '__main__':
#     main()
#############################################################################################################
#############################################################################################################
import torch
import torch.nn as nn
from torchvision.models import resnet50

class UNet_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(UNet_ResNet50, self).__init__()
        self.encoder = resnet50(weights='DEFAULT')
        
        # Use the encoder's layers as the first few layers for our model
        self.conv = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        self.enc1 = self.encoder.layer1
        self.enc2 = self.encoder.layer2
        self.enc3 = self.encoder.layer3
        self.enc4 = self.encoder.layer4
        
        # Define the U-Net decoder layers
        self.center = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        )

        self.dec4 = self._decoder_block(1024 + 2048, 512)
        self.dec3 = self._decoder_block(512 + 1024, 256)
        self.dec2 = self._decoder_block(256 + 512, 128)
        self.dec1 = self._decoder_block(128 + 256, 64, stride=1)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _decoder_block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        conv = self.conv(x)
        enc1 = self.enc1(conv)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))

        final = self.final(dec1)
        print(final.shape)

        return final
