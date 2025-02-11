"""
Implementation of ESDNet for image demoireing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class my_model(nn.Module):
    
    def __init__(
        self,
        en_feature_num: int,
        en_inter_num  : int,
        de_feature_num: int,
        de_inter_num  : int,
        sam_number    : int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(
            feature_num = en_feature_num,
            inter_num   = en_inter_num,
            sam_number  = sam_number
        )
        self.decoder = Decoder(
            en_num      = en_feature_num,
            feature_num = de_feature_num,
            inter_num   = de_inter_num,
            sam_number  = sam_number
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_1, y_2, y_3       = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(y_1, y_2, y_3)
        return out_1, out_2, out_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    
    def __init__(
        self,
        en_num     : int,
        feature_num: int,
        inter_num  : int,
        sam_number : int
    ):
        super().__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)
        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)
        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1: torch.Tensor, y_2: torch.Tensor, y_3: torch.Tensor) -> torch.Tensor:
        x_3           = y_3
        x_3           = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)
        x_2           = torch.cat([y_2, feat_3], dim=1)
        x_2           = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)
        x_1           = torch.cat([y_1, feat_2], dim=1)
        x_1           = self.preconv_1(x_1)
        out_1         = self.decoder_1(x_1, feat=False)
        return out_1, out_2, out_3


class Encoder(nn.Module):
    
    def __init__(self, feature_num: int, inter_num: int, sam_number: int):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(    feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)
        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3                 = self.encoder_3(down_feature_2)
        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    
    def __init__(self, feature_num: int, inter_num: int, level: int, sam_number: int):
        super().__init__()
        self.rdb        = RDB(in_channels=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channels=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    
    def __init__(self, feature_num: int, inter_num: int, sam_number: int):
        super().__init__()
        self.rdb        = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channels=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channels=feature_num, out_channels=12, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, feat: bool = True) -> torch.Tensor:
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)
        if feat:
            feature = F.interpolate(x, scale_factor=2, mode="bilinear")
            return out, feature
        else:
            return out


class DB(nn.Module):
    
    def __init__(self, in_channels: int, d_list: list, inter_num: int):
        super().__init__()
        self.d_list      = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channels
        for i in range(len(d_list)):
            dense_conv = conv_relu(
                in_channels   = c,
                out_channels  = inter_num,
                kernel_size   = 3,
                dilation_rate = d_list[i],
                padding       = d_list[i]
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channels=c, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t  = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    
    def __init__(self, in_channels: int, d_list: list, inter_num: int):
        super().__init__()
        self.basic_block   = DB(in_channels=in_channels, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channels=in_channels, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channels=in_channels, d_list=d_list, inter_num=inter_num)
        self.fusion        = CSAF(3 * in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5,  mode="bilinear")
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear")
        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)
        y_2 = F.interpolate(y_2, scale_factor=2, mode="bilinear")
        y_4 = F.interpolate(y_4, scale_factor=4, mode="bilinear")
        y   = self.fusion(y_0, y_2, y_4)
        y   = x + y
        return y


class CSAF(nn.Module):
    
    def __init__(self, in_channels: int, ratio: int = 4):
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1  = nn.Conv2d(in_channels,          in_channels // ratio, 1, 1, 0)
        self.compress2  = nn.Conv2d(in_channels // ratio, in_channels // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels // ratio, in_channels,          1, 1, 0)

    def forward(self, x0: torch.Tensor, x2: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out  = torch.cat([out0, out2, out4], dim=1)
        out  = self.compress1(out)
        out  = F.relu(out)
        out  = self.compress2(out)
        out  = F.relu(out)
        out  = self.excitation(out)
        out  = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x    = x0 * w0 + x2 * w2 + x4 * w4
        return x


class RDB(nn.Module):
    
    def __init__(self, in_channels: int, d_list: list, inter_num: int):
        super().__init__()
        self.d_list      = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channels
        for i in range(len(d_list)):
            dense_conv = conv_relu(
                in_channels   = c,
                out_channels  = inter_num,
                kernel_size   = 3,
                dilation_rate = d_list[i],
                padding       = d_list[i]
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channels=c, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : _size_2_t,
        dilation_rate: _size_2_t       = 1,
        padding      : _size_2_t | str = 0,
        stride       : _size_2_t       = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            bias         = True,
            dilation     = dilation_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class conv_relu(nn.Module):
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : _size_2_t,
        dilation_rate: _size_2_t       = 1,
        padding      : _size_2_t | str = 0,
        stride       : _size_2_t       = 1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                bias         = True,
                dilation     = dilation_rate
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
