import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SqueezeExcite
"""----------------------------------------GMA所需模块-------------------------------------------------"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(ChannelAttention, self).__init__()
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // self.rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // self.rotio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out
        return self.sigmoid(out)
class MultiChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(MultiChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, (1, 1), bias=False))

        self.conv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=(1, 1),
                              stride=(1, 1), padding=(0, 0))
        self.batch_norm = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, shallow_channel_attention_map):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))

        x2 = shallow_channel_attention_map

        x2 = self.conv(x2)
        x2 = self.batch_norm(x2)
        x2 = self.relu(x2)
        x2 = self.sharedMLP(x2)
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out + self.coefficient3 * x2
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)
class MultiSpatialAttention(nn.Module):
    def __init__(self):
        super(MultiSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.batch_norm0 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x, shallow_spatial_attention_map):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x0 = self.conv0(shallow_spatial_attention_map)
        x0 = self.batch_norm0(x0)
        x0 = self.relu(x0)
        x0 = nn.AvgPool2d(2)(x0)
        x = torch.cat([avg_out, max_out, x0], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class CLMA_Net(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C):
        super(CLMA_Net, self).__init__()
        self.name = 'LliuMK_Net3'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C

        self.relu = nn.ReLU()

        self.CA1 = ChannelAttention(in_planes=self.band)

        self.MCA1 = MultiChannelAttention(in_planes=self.band)
        self.MCA2 = MultiChannelAttention(in_planes=self.band)

        self.SA1 = SpatialAttention()

        self.MSA1 = MultiSpatialAttention()
        self.MSA2 = MultiSpatialAttention()
        self.conv11 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm11 = nn.BatchNorm2d(self.band)

        self.conv12 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm12 = nn.BatchNorm2d(self.band)

        self.conv13 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm13 = nn.BatchNorm2d(self.band)

        self.conv21 = nn.Conv2d(in_channels=self.band, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm21 = nn.BatchNorm2d(64)

        self.conv22 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch_norm22 = nn.BatchNorm2d(128)

        self.conv23 = nn.Conv2d(in_channels=128, out_channels=self.band, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm23 = nn.BatchNorm2d(self.band)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.finally_fc_classification = nn.Linear(self.band * 2, self.classes)

    def forward(self, patchX, pixelX):
        patch_size = patchX.shape[-1] // 2
        input_spectral = patchX[:, :, (patch_size - 1):(patch_size + 2),
                         (patch_size - 1):(patch_size + 2)]

        x11 = self.conv11(input_spectral)
        x11 = self.batch_norm11(x11)
        x11 = self.relu(x11)
        ca1 = self.CA1(x11)
        x11 = x11 * ca1

        x12 = x11
        x12 = self.conv12(x12)
        x12 = self.batch_norm12(x12)
        x12 = self.relu(x12)
        mca1 = self.MCA1(x12, ca1)
        x12 = x12 * mca1

        x13 = x12
        x13 = self.conv13(x13)
        x13 = self.batch_norm13(x13)
        x13 = self.relu(x13)
        mca2 = self.MCA2(x13, mca1)
        x13 = x13 * mca2

        x13 = self.global_pooling(x13)
        x13 = x13.view(x13.size(0), -1)
        output_spectral = x13

        input_spatial = patchX
        x21 = self.conv21(input_spatial)
        x21 = self.batch_norm21(x21)
        x21 = self.relu(x21)
        sa1 = self.SA1(x21)
        x21 = x21 * sa1
        x21 = nn.MaxPool2d(2)(x21)

        x22 = self.conv22(x21)
        x22 = self.batch_norm22(x22)
        x22 = self.relu(x22)
        msa1 = self.MSA1(x22, sa1)
        x22 = x22 * msa1

        x22 = nn.MaxPool2d(2)(x22)

        x23 = self.conv23(x22)
        x23 = self.batch_norm23(x23)
        x23 = self.relu(x23)
        msa2 = self.MSA2(x23, msa1)
        x23 = x23 * msa2

        x23 = nn.MaxPool2d(2)(x23)

        x25 = self.global_pooling(x23)
        x25 = x25.view(x25.size(0), -1)
        output_spatial = x25

        output = torch.cat((output_spectral, output_spatial), dim=1)
        output = self.finally_fc_classification(output)
        output = F.softmax(output, dim=1)

        return output
"""--------------------------------------------------------------------------------------------------"""
# ================模块组件===================
# -------------------BaseNet-----------------------
class PSS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.contrast_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(out_channels)

        nn.init.dirac_(self.contrast_conv.weight)
        if self.contrast_conv.bias is not None:
            nn.init.zeros_(self.contrast_conv.bias)

    def forward(self, x):
        x = self.base_conv(x)
        avg_neighbor = self.contrast_conv(x)
        x = x - 0.5 * avg_neighbor
        x = self.bn(self.act(x))
        return x
# -------------SFE注意力机制--------------
class MultiFeatureAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Sigmoid()
        )

        self.attn_fusion = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        nn.init.constant_(self.attn_fusion.weight, 1 / 2)

    def forward(self, x, modal2=None):
        B, C, H, W = x.shape

        sa = self.spectral_attn(x)
        sa = sa.expand(-1, -1, H, W)
        spectral_feat = x * sa

        spa = self.spatial_attn(x)
        spatial_feat = x * spa

        sa_spatial = sa.mean(dim=1, keepdim=True)
        attn_concat = torch.cat([sa_spatial, spa], dim=1)
        fusion_weight = self.attn_fusion(attn_concat)
        fusion_weight = F.sigmoid(fusion_weight)

        fused_feat = spectral_feat + spatial_feat
        fused_feat = fused_feat * fusion_weight

        return fused_feat, fusion_weight
# ---------------------------------------
# 3. 增强空间局部特征的SFE模块（多尺度+局部注意力）
class SFE(nn.Module):
    def __init__(self, msi_channels):
        super().__init__()
        # 多尺度空洞卷积
        self.scale1 = nn.Sequential(
            nn.Conv2d(msi_channels, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(msi_channels, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(msi_channels, 16, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )

        self.local_attn = MultiFeatureAttention(48, 8)

    def forward(self, msi):
        # 多尺度特征融合
        s1 = self.scale1(msi)
        s2 = self.scale2(msi)
        s3 = self.scale3(msi)
        features = torch.cat([s1, s2, s3], dim=1)

        # 应用注意力
        fused_feat, attn_weight = self.local_attn(features)

        return fused_feat
# 4. 修复残差连接的ES2CM模块
class ES2CM(nn.Module):
    def __init__(self, input_channels=48, compression_ratio=32 / 48):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = max(16, int(input_channels * compression_ratio))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.LeakyReLU(0.2)
        )

        self.detail_branch = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.residual_proj = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_channels)
        )

    def forward(self, fused_features):
        residual = self.residual_proj(fused_features)
        out = self.conv(fused_features)
        detail_weights = self.detail_branch(fused_features)
        return self.res_scale * residual + out * detail_weights
# --------------------HFFG组件------------------------
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_a):
        gate_a = self.gate_a(feat_a)
        fused = feat_a * gate_a
        return fused, gate_a
# ----------------------------------------------------
# HFFG模块
class HFFG(nn.Module):
    def __init__(self, stm_dim, es2cm_dim, fusion_dim=32):
        super().__init__()

        self.stm_proj = nn.Conv2d(stm_dim, fusion_dim, 1)
        self.es2cm_proj = nn.Conv2d(es2cm_dim, fusion_dim, 1)

        self.fusion = GatedFusion(fusion_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, groups=fusion_dim),
            nn.Conv2d(fusion_dim, fusion_dim, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU()
        )

    def forward(self, stm_feat, es2cm_feat):
        stm_proc = self.stm_proj(stm_feat)
        es2cm_proc = self.es2cm_proj(es2cm_feat)

        fused, gate = self.fusion(stm_proc)
        refined = self.refine(fused)

        return refined, {"gate": gate}
# -------------------BaseNet-----------------------
class Spectral_Attention_block(nn.Module):
    def __init__(self, dim):
        super(Spectral_Attention_block, self).__init__()

        reduced_dim = max(1,dim//16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class MultiGroupSpatialAttention(nn.Module):
    def __init__(self, channel, patch_size):
        super(MultiGroupSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x)), self.sigmoid(self.conv(x))
# --- 修改后的 Multi_Attention 类 ---
class Multi_Attention(nn.Module):
    def __init__(self, channelNumber, patch_size):
        super(Multi_Attention, self).__init__()
        self.total_Channel_num = channelNumber
        self.patch_size = patch_size

        self.num_groups = 3
        base_channels = channelNumber // self.num_groups
        remainder = channelNumber % self.num_groups
        self.group_channels = []
        for i in range(self.num_groups):
            if i < remainder:
                self.group_channels.append(base_channels + 1)
            else:
                self.group_channels.append(base_channels)
        self.mgca_list = nn.ModuleList()
        self.mgsa_list = nn.ModuleList()

        for i in range(self.num_groups):
            self.mgca_list.append(Spectral_Attention_block(dim=self.group_channels[i]))
            self.mgsa_list.append(
                MultiGroupSpatialAttention(channel=self.group_channels[i], patch_size=self.patch_size))
        total_fusion_channels = sum(self.group_channels) * self.num_groups
        self.safusion_conv_layer = nn.Sequential(
            nn.Conv2d(total_fusion_channels, self.total_Channel_num, (1, 1)),
            nn.BatchNorm2d(self.total_Channel_num),
            nn.ReLU()
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.total_Channel_num * 2, out_channels=self.total_Channel_num, kernel_size=(1, 1),
                      padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(self.total_Channel_num),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.total_Channel_num, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        ca_outputs = []
        sa_maps = []
        current_channel = 0
        for i in range(self.num_groups):
            start = current_channel
            end = current_channel + self.group_channels[i]
            current_channel = end

            group_x = x[:, start:end, :, :]

            ca_out = self.mgca_list[i](group_x)
            ca_outputs.append(ca_out)

            sa_out, sa_map = self.mgsa_list[i](group_x)
            sa_maps.append(sa_map)

        ca_output = torch.cat(ca_outputs, dim=1)

        safusion_parts = []
        for i in range(self.num_groups):
            sap = torch.mean(sa_maps[i], dim=1).unsqueeze(1)

            current_fusion_parts = []
            current_channel = 0
            for j in range(self.num_groups):
                start = current_channel
                end = current_channel + self.group_channels[j]
                current_channel = end

                if i == j:
                    weighted_part = sa_maps[i] * x[:, start:end, :, :]
                else:
                    weighted_part = (1 - sap) * x[:, start:end, :, :]

                current_fusion_parts.append(weighted_part)

            safusion_part = torch.cat(current_fusion_parts, dim=1)
            safusion_parts.append(safusion_part)

        safusion_conv_input = torch.cat(safusion_parts, dim=1)
        safusion_conv = self.safusion_conv_layer(safusion_conv_input)

        safusion_conv = safusion_conv + x

        output = torch.cat([safusion_conv, ca_output], dim=1)

        output = self.fusion_layer(output)

        return output
class GMABlock(nn.Module):
    def __init__(self, HSI_Data_Shape_C, patch_size):
        super(GMABlock, self).__init__()
        self.name = 'GMA_Net'
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size

        self.gma = Multi_Attention(channelNumber=self.band, patch_size=self.patch_size)
        self.relu = nn.ReLU()

        """Pixel branch"""
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        """Patch branch"""
        self.conv31 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm31 = nn.BatchNorm2d(32)

        self.conv32 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm32 = nn.BatchNorm2d(64)

        self.conv33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm33 = nn.BatchNorm2d(128)

        self.conv34 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm34 = nn.BatchNorm2d(256)

        """特征融合卷积"""
        self.fusion_conv = nn.Conv2d(
            in_channels=256 + 256,
            out_channels=self.band,
            kernel_size=(1, 1)
        )
        self.fusion_bn = nn.BatchNorm2d(self.band)

    def forward(self, patchX, pixelX):
        input_1 = pixelX.unsqueeze(-1).unsqueeze(-1)
        x11 = self.conv11(input_1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        pixel_feat = F.interpolate(
            x14,
            size=(patchX.shape[2] // 8, patchX.shape[3] // 8),
            mode='bilinear',
            align_corners=True
        )

        """Patch branch"""
        gma_out = self.gma(patchX)

        x31 = self.relu(self.batch_norm31(self.conv31(gma_out)))
        x32 = nn.MaxPool2d(2)(x31)
        x32 = self.relu(self.batch_norm32(self.conv32(x32)))
        x33 = nn.MaxPool2d(2)(x32)
        x33 = self.relu(self.batch_norm33(self.conv33(x33)))
        x34 = nn.MaxPool2d(2)(x33)
        patch_feat = self.relu(self.batch_norm34(self.conv34(x34)))

        """特征融合"""
        fused = torch.cat([pixel_feat, patch_feat], dim=1)
        output = self.relu(self.fusion_bn(self.fusion_conv(fused)))

        output = F.interpolate(output, size=(patchX.shape[2], patchX.shape[3]), mode='bilinear', align_corners=True)

        return output
class BaseNet(nn.Module):
    def __init__(self, msi_channels=24,
                 hidden_dim=32,
                 es2cm_compression=32/48,
                 fusion_dim=32,
                 ):
        super().__init__()

        self.gma = GMABlock(HSI_Data_Shape_C=msi_channels, patch_size=64)
        self.conv_1x1 = nn.Conv2d(msi_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.sfe = SFE(msi_channels=msi_channels)
        self.es2cm = ES2CM(
            input_channels=48,
            compression_ratio=es2cm_compression
        )

        self.fusion = HFFG(
            stm_dim=hidden_dim,
            es2cm_dim=self.es2cm.output_channels,
            fusion_dim=fusion_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, msi_data):
        device = next(self.parameters()).device
        msi_data = msi_data.to(device)
        B,C,H,W = msi_data.shape

        patchX = msi_data
        center_h,center_w = H//2,W//2
        pixelX = patchX[:, :, center_h,center_w]
        spectral_features = self.gma(patchX, pixelX)
        spectral_features = self.conv_1x1(spectral_features)
        spatial_enhanced = self.sfe(msi_data)
        es2cm_features = self.es2cm(spatial_enhanced)

        B, _, H_spec, W_spec = spectral_features.shape
        B, _, H_spatial, W_spatial = es2cm_features.shape
        assert H_spec == H_spatial and W_spec == W_spatial, \
            f"光谱流空间尺寸({H_spec}x{W_spec})与空间流({H_spatial}x{W_spatial})不匹配"

        fused_features, attn_weights = self.fusion(spectral_features, es2cm_features)

        aux_features = {
            'GMA': spectral_features,
            'SFE': spatial_enhanced,
            'ES2CM': es2cm_features,
        }

        return fused_features, aux_features
# -----------------------------------------------------------------------------
# LSNet核心模块
# -----------------------------------------------------------------------------
class LSConv(nn.Module):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.lkp = nn.Conv2d(dim, dim, 7, 1, 3, groups=groups)
        self.ska = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x) * self.lkp(x)) + x
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False
        ))
        self.add_module('bn', nn.BatchNorm2d(out_ch))
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        conv, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        fused_conv = nn.Conv2d(
            w.size(1) * conv.groups, w.size(0), w.shape[2:],
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            groups=conv.groups, device=conv.weight.device
        )
        fused_conv.weight.data.copy_(w)
        fused_conv.bias.data.copy_(b)
        return fused_conv
class Residual(nn.Module):
    def __init__(self, module, dropout=0.):
        super().__init__()
        self.module = module
        self.dropout = dropout

    def forward(self, x):
        if self.training and self.dropout > 0:
            residual = self.module(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.dropout).div(1 - self.dropout).detach()
        else:
            residual = self.module(x)
        return x + residual
class FFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.pw1 = Conv2d_BN(dim, hidden_dim, 1)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(hidden_dim, dim, 1)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))
class RepVGGDW(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv3x3 = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.conv1x1 = Conv2d_BN(dim, dim, 1, 1, 0, groups=dim)
        self.dim = dim

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv3x3 = self.conv3x3.fuse()
        conv1x1 = self.conv1x1.fuse()

        conv1x1_weight = F.pad(conv1x1.weight, [1, 1, 1, 1])
        identity_weight = F.pad(torch.eye(self.dim, device=conv1x1.weight.device).view(self.dim, self.dim, 1, 1), [1, 1, 1, 1])

        fused_weight = conv3x3.weight + conv1x1_weight + identity_weight
        fused_bias = conv3x3.bias + conv1x1.bias

        conv3x3.weight.data.copy_(fused_weight)
        conv3x3.bias.data.copy_(fused_bias)
        return conv3x3
class LSBlock(nn.Module):
    def __init__(self, dim, depth, stage):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(dim)
            self.se = SqueezeExcite(dim, 0.25)
        else:
            self.mixer = LSConv(dim)
            self.se = nn.Identity()

        self.ffn = Residual(FFN(dim))

    def forward(self, x):
        x = self.se(self.mixer(x))
        x = self.ffn(x)
        return x

class OverviewNet(nn.Module):
    def __init__(self, in_channels=32, ctx_dim=64, downscale=4):
        super().__init__()
        self.downscale = downscale
        self.downsample = nn.Sequential(
            PSS(in_channels, ctx_dim),
            nn.MaxPool2d(2, stride=2),
            PSS(ctx_dim, ctx_dim),
            nn.MaxPool2d(2, stride=2)
        )
        self.ctx_enhance = Multi_Attention(channelNumber=ctx_dim, patch_size=64 // downscale)
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(in_channels, ctx_dim, 1),
            nn.BatchNorm2d(ctx_dim),
            nn.ReLU()
        )

    def forward(self, base_feat):
        down_feat = self.downsample(base_feat)
        ctx_enhanced = self.ctx_enhance(down_feat)
        ctx = self.ctx_proj(ctx_enhanced)
        return ctx

class FocusNet(nn.Module):
    def __init__(self, base_dim=32, ctx_dim=64, num_classes=4, depth=4):
        super().__init__()
        self.base_proj = nn.Sequential(
            Conv2d_BN(base_dim, ctx_dim, 1),
            nn.ReLU()
        )
        self.ctx_attn = nn.Sequential(
            Conv2d_BN(ctx_dim * 2, ctx_dim // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2d_BN(ctx_dim // 2, ctx_dim, 1),
            nn.Sigmoid()
        )
        self.deep_blocks = nn.Sequential(*[
            LSBlock(dim=ctx_dim, depth=i,stage=i) for i in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ctx_dim, ctx_dim // 2),
            nn.BatchNorm1d(ctx_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(ctx_dim // 2, num_classes)
        )

    def forward(self, base_feat, ctx):
        base_aligned = self.base_proj(base_feat)

        ctx_up = F.interpolate(
            ctx, size=base_feat.shape[2:], mode='bilinear', align_corners=False
        )

        attn_input = torch.cat([base_aligned, ctx_up], dim=1)
        attn_weight = self.ctx_attn(attn_input)
        fused_feat = base_aligned * attn_weight + ctx_up * (1 - attn_weight)

        fused_feat = self.deep_blocks(fused_feat)

        logits = self.classifier(fused_feat)
        return logits,fused_feat

# -----------------------------------------------------------------------------
# 总网络：GMA_MSI
# -----------------------------------------------------------------------------
class GMA_MSI(nn.Module):
    def __init__(self, msi_channels=24, num_classes=4,
                 hidden_dim=32, es2cm_compression=0.8, fusion_dim=32, ctx_dim=64):
        super().__init__()
        self.base_net = BaseNet(msi_channels=msi_channels, hidden_dim=hidden_dim, es2cm_compression=es2cm_compression, fusion_dim=fusion_dim)
        self.overview_net = OverviewNet(in_channels=fusion_dim, ctx_dim=ctx_dim)
        self.focus_net = FocusNet(base_dim=fusion_dim, ctx_dim=ctx_dim, num_classes=num_classes, depth=4)

    def forward(self, msi_data):
        base_feat, base_aux = self.base_net(msi_data)
        ctx = self.overview_net(base_feat)
        logits, fused_feat = self.focus_net(base_feat, ctx)
        aux_features = {**base_aux, 'base_feat': base_feat, 'ctx': ctx}
        return logits, aux_features
# -----------------------------------------------------------------------------
# 测试代码
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 模拟输入
    msi = torch.randn(2, 24, 32, 32)

    # 初始化网络
    model = GMA_MSI(num_classes=4)

    # 前向传播
    logits, aux_features = model(msi)

    # 验证输出维度
    print(f"分类输出维度: {logits.shape} (期望: [2, 4])")
    print(f"Base_feat维度: {aux_features['base_feat'].shape}")
    print(f"Ctx维度: {aux_features['ctx'].shape}")

    # 验证LSBlock是否正常工作
    ls_block = model.focus_net.deep_blocks[0]
    test_feat = torch.randn(2, 64, 32, 32)
    ls_output = ls_block(test_feat)
    print(f"LSBlock输入输出维度是否一致: {test_feat.shape == ls_output.shape} (期望: True)")
