import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from efficientnet_pytorch import EfficientNet
    
# Efficient Local Attention module
class ELA(nn.Module):
    """Constructs an Efficient Local Attention module.
    Args:
        channel: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2,
                             groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(B, C, H, 1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(B, C, 1, W)
        return x * x_h * x_w

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, 
                                         stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        
        return x

# U-Net with EfficientNet backbone (no skip connections and no attention modules)
class UNetEfficientNet(nn.Module):
    def __init__(self, num_classes=1, encoder_name='efficientnet-b5', pretrained=True):
        super(UNetEfficientNet, self).__init__()

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(encoder_name)
        else:
            self.encoder = EfficientNet.from_name(encoder_name)

        self.encoder._fc = nn.Identity()
        self.encoder._avg_pooling = nn.Identity()
        self.encoder._dropout = nn.Identity()

        # Detect encoder output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            x = self.encoder.extract_features(dummy_input)
            encoder_out_channels = x.shape[1]

        print(f"Encoder output channels: {encoder_out_channels}")

        # Bottleneck
        self.center = nn.Sequential(
            nn.Conv2d(encoder_out_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()

        # Encoder
        x = self.encoder.extract_features(x)

        # Bottleneck
        x = self.center(x)

        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        # Final classifier
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        x = self.final(x)

        return x

#U-Net with EfficientNet backbone + skip connections but no attention modules
class UNetEfficientNet_Skip(nn.Module):
    def __init__(self, num_classes=1, encoder_name='efficientnet-b5', pretrained=True):
        super(UNetEfficientNet_Skip, self).__init__()

        # Load EfficientNet backbone
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(encoder_name)
        else:
            self.encoder = EfficientNet.from_name(encoder_name)

        self.encoder._fc = nn.Identity()
        self.encoder._avg_pooling = nn.Identity()
        self.encoder._dropout = nn.Identity()

        # Determine encoder channels dynamically
        with torch.no_grad():
            test_input = torch.randn(1, 3, 128, 128)
            feats = self._extract_features(test_input)
            encoder_channels = [f.shape[1] for f in feats]

        print(f"Detected encoder channels: {encoder_channels}")

        # Bottleneck
        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = DecoderBlock(512 + encoder_channels[-2], 256)
        self.decoder3 = DecoderBlock(256 + encoder_channels[-3], 128)
        self.decoder2 = DecoderBlock(128 + encoder_channels[-4], 64)
        self.decoder1 = DecoderBlock(64 + encoder_channels[-5], 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def _extract_features(self, x):
        """Extract intermediate features from EfficientNet backbone"""
        features = []
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        features.append(x)

        for idx, block in enumerate(self.encoder._blocks):
            x = block(x)
            # capture skip features at specific stages
            if idx in [1, 2, 5, 10, 15] or idx == len(self.encoder._blocks) - 1:
                features.append(x)

        if hasattr(self.encoder, '_conv_head'):
            x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
            features.append(x)  # append head

        return features[-5:]  # return last 5 stages

    def forward(self, x):
        input_size = x.size()
        features = self._extract_features(x)

        # Center (bottleneck)
        x = self.center(features[-1])

        # Decoder 4
        x = torch.cat([x, features[-2]], dim=1)
        x = self.decoder4(x)

        # Decoder 3
        x = torch.cat([x, features[-3]], dim=1)
        x = self.decoder3(x)

        # Decoder 2
        x = torch.cat([x, features[-4]], dim=1)
        x = self.decoder2(x)

        # Decoder 1
        x = torch.cat([x, features[-5]], dim=1)
        x = self.decoder1(x)

        # Final classifier
        x = nn.functional.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        x = self.final(x)
        return x

# # -------------------------------
# # U-Net with EfficientNet + Skips + ELA after each decoder
# # -------------------------------
class UNetEfficientNet_Skip_ELA(nn.Module):
    def __init__(self, num_classes=1, encoder_name='efficientnet-b5', pretrained=True):
        super(UNetEfficientNet_Skip_ELA, self).__init__()

        if pretrained:
            self.encoder = EfficientNet.from_pretrained(encoder_name)
        else:
            self.encoder = EfficientNet.from_name(encoder_name)

        self.encoder._fc = nn.Identity()
        self.encoder._avg_pooling = nn.Identity()
        self.encoder._dropout = nn.Identity()

        # Determine encoder channels dynamically
        with torch.no_grad():
            test_input = torch.randn(1, 3, 128, 128)
            feats = self._extract_features(test_input)
            encoder_channels = [f.shape[1] for f in feats]

        print(f"Detected encoder channels: {encoder_channels}")

        # Bottleneck
        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = DecoderBlock(512 + encoder_channels[-2], 256)
        self.decoder3 = DecoderBlock(256 + encoder_channels[-3], 128)
        self.decoder2 = DecoderBlock(128 + encoder_channels[-4], 64)
        self.decoder1 = DecoderBlock(64 + encoder_channels[-5], 32)

        # ELA after each decoder
        self.ela4 = ELA(256)
        self.ela3 = ELA(128)
        self.ela2 = ELA(64)
        self.ela1 = ELA(32)

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def _extract_features(self, x):
        """Extract intermediate features from EfficientNet backbone"""
        features = []
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        features.append(x)

        for idx, block in enumerate(self.encoder._blocks):
            x = block(x)
            if idx in [1, 2, 5, 10, 15] or idx == len(self.encoder._blocks) - 1:
                features.append(x)

        if hasattr(self.encoder, '_conv_head'):
            x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
            features.append(x)

        return features[-5:]

    def forward(self, x):
        input_size = x.size()
        features = self._extract_features(x)

        # Bottleneck
        x = self.center(features[-1])

        # Decoder 4 + ELA
        x = torch.cat([x, features[-2]], dim=1)
        x = self.decoder4(x)
        x = self.ela4(x)

        # Decoder 3 + ELA
        x = torch.cat([x, features[-3]], dim=1)
        x = self.decoder3(x)
        x = self.ela3(x)

        # Decoder 2 + ELA
        x = torch.cat([x, features[-4]], dim=1)
        x = self.decoder2(x)
        x = self.ela2(x)

        # Decoder 1 + ELA
        x = torch.cat([x, features[-5]], dim=1)
        x = self.decoder1(x)
        x = self.ela1(x)

        # Final classifier
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        x = self.final(x)
        return x