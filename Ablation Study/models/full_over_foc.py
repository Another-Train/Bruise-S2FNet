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
        self.cross_modal_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attn_fusion = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)
        nn.init.constant_(self.attn_fusion.weight, 1 / 3)

    def forward(self, x, modal2=None):
        B, C, H, W = x.shape
        sa = self.spectral_attn(x).expand(-1, -1, H, W)
        spectral_feat = x * sa
        spa = self.spatial_attn(x)
        spatial_feat = x * spa

        if modal2 is not None:
            modal_diff = torch.abs(x - modal2)
            cross = self.cross_modal_attn(torch.cat([x, modal2, modal_diff], dim=1))
            cross_feat = x * cross
        else:
            cross = torch.ones(B, 1, H, W, device=x.device)
            cross_feat = x

        sa_spatial = sa.mean(dim=1, keepdim=True)
        attn_concat = torch.cat([sa_spatial, spa, cross], dim=1)
        fusion_weight = torch.sigmoid(self.attn_fusion(attn_concat))
        fused_feat = spectral_feat + spatial_feat + cross_feat
        fused_feat = fused_feat * fusion_weight
        return fused_feat, fusion_weight


class SFE(nn.Module):
    def __init__(self, msi_channels, rgb_channels):
        super().__init__()
        self.rgb_adjust = nn.Conv2d(rgb_channels, msi_channels, kernel_size=1, bias=False)
        self.rgb_adj_proj = nn.Conv2d(msi_channels, 48, kernel_size=1)
        self.scale1 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.local_attn = MultiFeatureAttention(48, 8)

    def forward(self, msi, rgb):
        rgb_adj = self.rgb_adjust(rgb)
        rgb_adj_proj = self.rgb_adj_proj(rgb_adj)
        combined = torch.cat([msi, rgb_adj], dim=1)
        s1 = self.scale1(combined)
        s2 = self.scale2(combined)
        s3 = self.scale3(combined)
        features = torch.cat([s1, s2, s3], dim=1)
        fused_feat, attn_weight = self.local_attn(features, rgb_adj_proj)
        return fused_feat


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
        self.gate_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_a, feat_b):
        gate_a = self.gate_a(feat_a)
        gate_b = self.gate_b(feat_b)
        norm = gate_a + gate_b + 1e-6
        fused = (feat_a * gate_a + feat_b * gate_b) / norm
        return fused, (gate_a, gate_b)


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
        fused, gates = self.fusion(stm_proc, es2cm_proc)
        refined = self.refine(fused)
        return refined, {"gates": gates}


def generate_patches_and_points(data, patchsize=30):
    patch_size = patchsize + 1
    batch_size, channels, height, width = data.shape
    images = []
    points = []
    for i in range(batch_size):
        for h in range(0, height - patch_size + 1, patch_size - 1):
            for w in range(0, width - patch_size + 1, patch_size - 1):
                patch = data[i, :, h:h + patch_size, w:w + patch_size]
                images.append(patch)
                center_point = [h + patch_size // 2, w + patch_size // 2]
                points.append(center_point)
    images = torch.stack(images)
    points = torch.tensor(points, dtype=torch.long)
    return images, points


class Spectral_Attention_block(nn.Module):
    def __init__(self, dim):
        super(Spectral_Attention_block, self).__init__()
        reduced_dim = max(1, dim // 16)
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
        self.conv31 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm31 = nn.BatchNorm2d(32)
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm32 = nn.BatchNorm2d(64)
        self.conv33 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm33 = nn.BatchNorm2d(128)
        self.conv34 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm34 = nn.BatchNorm2d(256)
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
        pixel_feat = F.interpolate(x14, size=(patchX.shape[2] // 8, patchX.shape[3] // 8), mode='bilinear',
                                   align_corners=True)
        gma_out = self.gma(patchX)
        x31 = self.relu(self.batch_norm31(self.conv31(gma_out)))
        x32 = nn.MaxPool2d(2)(x31)
        x32 = self.relu(self.batch_norm32(self.conv32(x32)))
        x33 = nn.MaxPool2d(2)(x32)
        x33 = self.relu(self.batch_norm33(self.conv33(x33)))
        x34 = nn.MaxPool2d(2)(x33)
        patch_feat = self.relu(self.batch_norm34(self.conv34(x34)))
        fused = torch.cat([pixel_feat, patch_feat], dim=1)
        output = self.relu(self.fusion_bn(self.fusion_conv(fused)))
        output = F.interpolate(output, size=(patchX.shape[2], patchX.shape[3]), mode='bilinear', align_corners=True)
        return output


class BaseNet(nn.Module):
    def __init__(self, msi_channels=24, rgb_channels=3,
                 hidden_dim=32,
                 es2cm_compression=32 / 48,
                 fusion_dim=32,
                 num_classes=4):
        super().__init__()

        self.gma = GMABlock(HSI_Data_Shape_C=msi_channels, patch_size=64)
        self.conv_1x1 = nn.Conv2d(24, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.sfe = SFE(msi_channels=msi_channels, rgb_channels=rgb_channels)
        self.es2cm = ES2CM(
            input_channels=48,
            compression_ratio=es2cm_compression
        )
        self.fusion = HFFG(
            stm_dim=hidden_dim,
            es2cm_dim=self.es2cm.output_channels,
            fusion_dim=fusion_dim
        )

        # 直接接分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
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

    def forward(self, msi_data, rgb_img):
        device = next(self.parameters()).device
        msi_data = msi_data.to(device)
        rgb_img = rgb_img.to(device)
        B, C, H, W = msi_data.shape
        patchX = msi_data
        center_h, center_w = H // 2, W // 2
        pixelX = patchX[:, :, center_h, center_w]
        spectral_features = self.gma(patchX, pixelX)
        spectral_features = self.conv_1x1(spectral_features)
        spatial_enhanced = self.sfe(msi_data, rgb_img)
        es2cm_features = self.es2cm(spatial_enhanced)
        fused_features, attn_weights = self.fusion(spectral_features, es2cm_features)

        # 直接输出分类结果
        logits = self.classifier(fused_features)

        aux_features = {
            'GMA': spectral_features,
            'SFE': spatial_enhanced,

            'fused_features': fused_features
        }

        return logits, aux_features


class BruiseSTFNet(nn.Module):
    def __init__(self, msi_channels=24, rgb_channels=3, num_classes=4,
                 hidden_dim=32, es2cm_compression=0.8, fusion_dim=32):
        super().__init__()
        self.base_net = BaseNet(
            msi_channels=msi_channels,
            rgb_channels=rgb_channels,
            hidden_dim=hidden_dim,
            es2cm_compression=es2cm_compression,
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )

    def forward(self, msi_data, rgb_img):
        logits, aux_features = self.base_net(msi_data, rgb_img)
        return logits, aux_features


if __name__ == "__main__":
    # 模拟输入
    msi = torch.randn(2, 24, 32, 32)
    rgb = torch.randn(2, 3, 32, 32)

    # 初始化网络
    model = BruiseSTFNet(num_classes=4)

    # 前向传播
    logits, aux_features = model(msi, rgb)

    # 验证输出维度
    print(f"分类输出维度: {logits.shape} (期望: [2, 4])")
    print(f"融合特征维度: {aux_features['fused_features'].shape}")