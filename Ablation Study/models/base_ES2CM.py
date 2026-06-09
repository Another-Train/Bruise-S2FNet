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
        # self.up_sample_fc = nn.Sequential(nn.Linear(in_features=in_planes, out_features=in_planes),
        #                                   nn.BatchNorm1d(num_features=in_planes),
        #                                   nn.ReLU()
        #                                   )
        self.sigmoid = nn.Sigmoid()

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, shallow_channel_attention_map):  # 64,103,3,3   64,103,1,1
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))

        # x2 = shallow_channel_attention_map.squeeze(-1).squeeze(-1)
        # x2 = self.up_sample_fc(x2)
        # x2 = x2.unsqueeze(-1).unsqueeze(-1)
        # x2 = shallow_channel_attention_map.permute(0, 2, 1, 3)  # 64,103,1,1 -> 64,1,103,1
        x2 = shallow_channel_attention_map  # 64,103,1,1 -> 64,1,103,1

        x2 = self.conv(x2)
        x2 = self.batch_norm(x2)
        x2 = self.relu(x2)
        # x2 = x2.permute(0, 2, 1, 3)
        x2 = self.sharedMLP(x2)
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out + self.coefficient3 * x2
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [16,32,9,9]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        x_out = torch.cat([avg_out, max_out], dim=1)  # [16,2,9,9]                                    # 按维数1（列）拼接
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
        # shallow_spatial_attention_map = nn.MaxPool2d(2)(shallow_spatial_attention_map)

        x0 = self.conv0(shallow_spatial_attention_map)
        x0 = self.batch_norm0(x0)
        x0 = self.relu(x0)
        x0 = nn.AvgPool2d(2)(x0)  # 最大池化下采样效果貌似稍微好一点，还需要进一步实验验证
        x = torch.cat([avg_out, max_out, x0], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CLMA_Net(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C):  # band:103  classes=9
        super(CLMA_Net, self).__init__()
        self.name = 'LliuMK_Net3'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C

        # self.mish = mish()  # 也可以引用一下，等待后续改进
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

    def forward(self, patchX, pixelX):  # x:(16,103,9,9)
        """------------------------光谱分支------------------------"""
        patch_size = patchX.shape[-1] // 2
        input_spectral = patchX[:, :, (patch_size - 1):(patch_size + 2),
                         (patch_size - 1):(patch_size + 2)]  # [64,103,3,3]

        # input_spectral = input_spectral.permute(0, 2, 1, 3)  # [64,3,103,3]
        x11 = self.conv11(input_spectral)
        x11 = self.batch_norm11(x11)  # [64,103,3,3]
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

        """------------------------空间分支------------------------"""
        input_spatial = patchX
        x21 = self.conv21(input_spatial)  # (16,32,9,9)<—(16,103,9,9)
        x21 = self.batch_norm21(x21)  # (16,32,9,9)
        x21 = self.relu(x21)  # (16,32,9,9)
        sa1 = self.SA1(x21)
        x21 = x21 * sa1
        x21 = nn.MaxPool2d(2)(x21)

        x22 = self.conv22(x21)  # (16,24,1,9,9)
        x22 = self.batch_norm22(x22)  # (16,24,1,9,9)
        x22 = self.relu(x22)
        msa1 = self.MSA1(x22, sa1)
        x22 = x22 * msa1

        x22 = nn.MaxPool2d(2)(x22)

        x23 = self.conv23(x22)  # (16,24,1,9,9)
        x23 = self.batch_norm23(x23)  # (16,24,1,9,9)
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
        # 1. 光谱基础特征提取
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 2. 局部光谱对比度增强（突出瘀伤与周围的光谱差异）
        self.contrast_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.act = nn.LeakyReLU(0.2)  # 增大负斜率，减少梯度消失风险
        self.bn = nn.BatchNorm2d(out_channels)

        # 初始化保障：对比卷积权重初始化为单位矩阵（保留原始特征）
        nn.init.dirac_(self.contrast_conv.weight)
        if self.contrast_conv.bias is not None:
            nn.init.zeros_(self.contrast_conv.bias)

    def forward(self, x):
        # 基础特征
        x = self.base_conv(x)
        # 局部对比度增强：当前位置特征 - 邻域平均特征（突出差异）
        avg_neighbor = self.contrast_conv(x)
        x = x - 0.5 * avg_neighbor  # 控制差异强度，避免数值波动
        x = self.bn(self.act(x))
        return x


# -------------SFE注意力机制--------------
class MultiFeatureAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels

        # 1. 光谱显著性分支：捕捉通道维度的异常特征（如瘀伤的颜色差异）
        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化获取通道统计量 [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # 输出通道权重：异常通道（如瘀伤相关波段）权重高
        )

        # 2. 空间连续性分支：捕捉局部空间一致的区域（抑制孤立噪声）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),  # 压缩为单通道特征图
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # 用3x3均值滤波衡量局部一致性：值越高，局部越连续
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Sigmoid()  # 输出空间权重：连续区域权重高，孤立像素权重低
        )

        # 3. 跨模态一致性分支（若有RGB+多光谱双模态）：强化双模态一致的异常区域
        self.cross_modal_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),  # 融合双模态特征
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # 输出一致性权重：双模态均异常的区域权重高
        )

        # 注意力融合：自适应学习3个分支的权重
        self.attn_fusion = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)
        nn.init.constant_(self.attn_fusion.weight, 1 / 3)  # 初始权重平均

    def forward(self, x, modal2=None):
        """
        x: 主模态特征（如多光谱特征）[B, C, H, W]
        modal2: 辅模态特征（如RGB特征）[B, C, H, W]（可选，无则跳过跨模态分支）
        """
        B, C, H, W = x.shape

        # 1. 光谱显著性注意力：[B, C, 1, 1] → 扩展为[B, C, H, W]
        sa = self.spectral_attn(x)  # 通道权重
        sa = sa.expand(-1, -1, H, W)  # [B, C, H, W]（扩展后与x的空间维度匹配）
        spectral_feat = x * sa  # 增强异常通道特征

        # 2. 空间连续性注意力：[B, 1, H, W]
        spa = self.spatial_attn(x)  # 空间权重
        spatial_feat = x * spa  # 增强连续区域特征

        # 3. 跨模态一致性注意力（若有双模态）
        if modal2 is not None:
            # 计算双模态特征的差异（差异小的区域更可能是真实瘀伤）
            modal_diff = torch.abs(x - modal2)  # [B, C, H, W]
            cross = self.cross_modal_attn(torch.cat([x, modal2, modal_diff], dim=1))  # [B, 1, H, W]
            cross_feat = x * cross  # 增强双模态一致的区域
        else:
            cross = torch.ones(B, 1, H, W, device=x.device)  # 无辅模态时权重为1
            cross_feat = x

        # 融合3个分支的注意力权重（空间维度）
        # 将通道注意力转换为空间权重（取通道平均），与空间、跨模态权重拼接
        sa_spatial = sa.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        attn_concat = torch.cat([sa_spatial, spa, cross], dim=1)  # [B, 3, H, W]
        fusion_weight = self.attn_fusion(attn_concat)  # [B, 1, H, W]
        fusion_weight = F.sigmoid(fusion_weight)  # 最终注意力权重（0~1）

        # 用融合权重增强特征
        fused_feat = spectral_feat + spatial_feat + cross_feat
        fused_feat = fused_feat * fusion_weight  # 进一步突出综合高关注区域

        return fused_feat, fusion_weight


# ---------------------------------------
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
        # 新增：固定输出通道数
        self.output_channels = 48

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

# --------------------HFFG组件------------------------
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 轻量门控机制
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

        # 门控融合
        norm = gate_a + gate_b + 1e-6
        fused = (feat_a * gate_a + feat_b * gate_b) / norm

        return fused, (gate_a, gate_b)


# ----------------------------------------------------
# HFFG模块
class HFFG(nn.Module):
    def __init__(self, stm_dim, es2cm_dim, fusion_dim=32):
        super().__init__()

        self.stm_proj = nn.Conv2d(stm_dim, fusion_dim, 1)
        self.es2cm_proj = nn.Conv2d(es2cm_dim, fusion_dim, 1)

        # 使用门控融合替代交叉注意力
        self.fusion = GatedFusion(fusion_dim)

        # 轻量精炼
        self.refine = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, groups=fusion_dim),
            nn.Conv2d(fusion_dim, fusion_dim, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU()
        )

    def forward(self, stm_feat, es2cm_feat):
        stm_proc = self.stm_proj(stm_feat)
        es2cm_proc = self.es2cm_proj(es2cm_feat)

        # 高效融合
        fused, gates = self.fusion(stm_proc, es2cm_proc)
        refined = self.refine(fused)

        return refined, {"gates": gates}


# -------------------BaseNet-----------------------
def generate_patches_and_points(data, patchsize=30):
    """
    生成数据块和中心像素坐标。

    Args:
        data: 4D 张量，形状为 [batch_size, channels, height, width]
        patchsize: 数据块的边长（不包括中心像素），默认为 30

    Returns:
        images: 4D 张量，形状为 [new_batch_size, channels, patch_size+1, patch_size+1]
        points: 2D 张量，形状为 [new_batch_size, 2]，每行表示一个数据块的中心像素坐标
    """
    patch_size = patchsize + 1
    batch_size, channels, height, width = data.shape
    images = []
    points = []

    for i in range(batch_size):
        for h in range(0, height - patch_size + 1, patch_size - 1):
            for w in range(0, width - patch_size + 1, patch_size - 1):
                # 提取数据块
                patch = data[i, :, h:h + patch_size, w:w + patch_size]
                images.append(patch)
                # 生成中心像素坐标
                center_point = [h + patch_size // 2, w + patch_size // 2]
                points.append(center_point)

    # 将列表转换为 tensor
    images = torch.stack(images)
    points = torch.tensor(points, dtype=torch.long)

    return images, points


class Spectral_Attention_block(nn.Module):
    def __init__(self, dim):
        super(Spectral_Attention_block, self).__init__()

        reduced_dim = max(1, dim // 16)  # 防止除数为0
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

        # --- 动态分组 ---
        # 示例：将24通道分为3组，每组8通道
        self.num_groups = 3
        base_channels = channelNumber // self.num_groups
        remainder = channelNumber % self.num_groups
        # 创建分组通道数列表
        self.group_channels = []
        for i in range(self.num_groups):
            if i < remainder:
                self.group_channels.append(base_channels + 1)
            else:
                self.group_channels.append(base_channels)
        # 动态创建光谱和空间注意力模块
        self.mgca_list = nn.ModuleList()
        self.mgsa_list = nn.ModuleList()

        for i in range(self.num_groups):
            # 使用正确的通道数初始化每个组
            self.mgca_list.append(Spectral_Attention_block(dim=self.group_channels[i]))
            self.mgsa_list.append(
                MultiGroupSpatialAttention(channel=self.group_channels[i], patch_size=self.patch_size))
        # 注意：safusion_conv_layer 的输入通道数是 group数 * total_Channel_num
        total_fusion_channels = sum(self.group_channels) * self.num_groups
        self.safusion_conv_layer = nn.Sequential(
            nn.Conv2d(total_fusion_channels, self.total_Channel_num, (1, 1)),
            nn.BatchNorm2d(self.total_Channel_num),
            nn.ReLU()
        )

        # fusion_layer 输入通道数是 total_Channel_num (来自safusion) + total_Channel_num (来自ca)
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
        # x shape: [B, C, H, W] = [4, 24, 64, 64]

        ca_outputs = []
        sa_maps = []
        current_channel = 0
        # 对每个组应用注意力
        for i in range(self.num_groups):
            start = current_channel
            end = current_channel + self.group_channels[i]
            current_channel = end

            group_x = x[:, start:end, :, :]

            # 光谱注意力
            ca_out = self.mgca_list[i](group_x)
            ca_outputs.append(ca_out)

            # 空间注意力
            sa_out, sa_map = self.mgsa_list[i](group_x)
            sa_maps.append(sa_map)  # 保存空间注意力图用于后续融合

        # 拼接所有组的光谱注意力输出
        ca_output = torch.cat(ca_outputs, dim=1)  # Shape: [B, 24, H, W]

        # --- 空间注意力融合 (SA Fusion) ---
        safusion_parts = []
        for i in range(self.num_groups):
            sap = torch.mean(sa_maps[i], dim=1).unsqueeze(1)  # Shape: [B, 1, H, W]

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

        # 将所有组的SA融合结果拼接起来，并通过卷积层
        safusion_conv_input = torch.cat(safusion_parts, dim=1)  # Shape: [B, 3*24, H, W] = [B, 72, H, W]
        safusion_conv = self.safusion_conv_layer(safusion_conv_input)  # Shape: [B, 24, H, W]

        # 残差连接
        safusion_conv = safusion_conv + x

        # --- 最终融合 ---
        # 将SA融合的结果和CA融合的结果拼接
        output = torch.cat([safusion_conv, ca_output], dim=1)  # Shape: [B, 48, H, W]

        # 通过融合层得到最终特征
        output = self.fusion_layer(output)  # Shape: [B, 32, H, W]

        return output


class GMABlock(nn.Module):
    def __init__(self, HSI_Data_Shape_C, patch_size):
        super(GMABlock, self).__init__()
        self.name = 'GMA_Net'
        self.band = HSI_Data_Shape_C  # 输入通道数
        self.patch_size = patch_size

        # 使用修改后的 Multi_Attention 类
        self.gma = Multi_Attention(channelNumber=self.band, patch_size=self.patch_size)
        self.relu = nn.ReLU()

        """Pixel branch（保持空间维度）"""
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

        """Patch branch（保持空间维度）"""
        # Patch branch 的输入是经过 GMA 处理后的特征
        # GMA 的 fusion_layer 输出 32 通道
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
            in_channels=256 + 256,  # Pixel branch (256) + Patch branch (256)
            out_channels=self.band,  # 输出通道数与输入保持一致
            kernel_size=(1, 1)
        )
        self.fusion_bn = nn.BatchNorm2d(self.band)

    def forward(self, patchX, pixelX):
        # patchX shape: [4, 24, 64, 64]
        # pixelX shape: [4, 24]

        """Pixel branch"""
        input_1 = pixelX.unsqueeze(-1).unsqueeze(-1)  # [B, 24, 1, 1]
        x11 = self.conv11(input_1)  # [B, 32, 1, 1]
        x12 = self.conv12(x11)  # [B, 64, 1, 1]
        x13 = self.conv13(x12)  # [B, 128, 1, 1]
        x14 = self.conv14(x13)  # [B, 256, 1, 1]

        # 上采样到与 Patch branch 输出相同的空间维度 (64/2^3 = 8)
        pixel_feat = F.interpolate(
            x14,
            size=(patchX.shape[2] // 8, patchX.shape[3] // 8),
            mode='bilinear',
            align_corners=True
        )  # [B, 256, 8, 8]

        """Patch branch"""
        # 首先经过 GMA 模块
        gma_out = self.gma(patchX)  # [B, 32, 64, 64]

        x31 = self.relu(self.batch_norm31(self.conv31(gma_out)))  # [B, 32, 64, 64]
        x32 = nn.MaxPool2d(2)(x31)  # [B, 32, 32, 32]
        x32 = self.relu(self.batch_norm32(self.conv32(x32)))  # [B, 64, 32, 32]
        x33 = nn.MaxPool2d(2)(x32)  # [B, 64, 16, 16]
        x33 = self.relu(self.batch_norm33(self.conv33(x33)))  # [B, 128, 16, 16]
        x34 = nn.MaxPool2d(2)(x33)  # [B, 128, 8, 8]
        patch_feat = self.relu(self.batch_norm34(self.conv34(x34)))  # [B, 256, 8, 8]

        """特征融合"""
        fused = torch.cat([pixel_feat, patch_feat], dim=1)  # [B, 512, 8, 8]
        output = self.relu(self.fusion_bn(self.fusion_conv(fused)))  # [B, 24, 8, 8]

        output = F.interpolate(output, size=(patchX.shape[2], patchX.shape[3]), mode='bilinear', align_corners=True)

        return output

        """
        前向传播：输入MSI+RGB数据，输出四分类结果
        Args:
            msi_data: MSI多光谱数据，格式 [B, msi_channels, H, W]
            rgb_img: RGB彩色数据，格式 [B, rgb_channels, H, W]
        Returns:
            logits: 分类结果，格式 [B, 4]
            aux_features: 辅助特征字典（可选）
        """
        # 设备对齐
        device = next(self.parameters()).device
        msi_data = msi_data.to(device)
        rgb_img = rgb_img.to(device)
        B, C, H, W = msi_data.shape
        # images, points = generate_patches_and_points(msi_data)  # [new_batch_size, channels, patch_size+1, patch_size+1]

        patchX = msi_data  # [4, 24, 64, 64]
        center_h, center_w = H // 2, W // 2
        pixelX = patchX[:, :, center_h, center_w]
        spectral_features = self.gma(patchX, pixelX)  # [B, 32, H, W]
        spectral_features = self.conv_1x1(spectral_features)  # [B, hidden_dim, H, W]
        # 2. 空间特征增强流
        spatial_enhanced = self.sfe(msi_data, rgb_img)  # [B, 48, H, W]
        # es2cm_features = self.es2cm(spatial_enhanced)  # [B, es2cm_output_channels, H, W]

        # 3. 特征融合前校验
        B, _, H_spec, W_spec = spectral_features.shape
        # B, _, H_spatial, W_spatial = es2cm_features.shape
        B, _, H_spatial, W_spatial = spatial_enhanced.shape
        assert H_spec == H_spatial and W_spec == W_spatial, \
            f"光谱流空间尺寸({H_spec}x{W_spec})与空间流({H_spatial}x{W_spatial})不匹配"

        # 4. 核心特征融合
        # fused_features, attn_weights = self.fusion(spectral_features, es2cm_features)  # [B, fusion_dim, H, W]
        fused_features, attn_weights = self.fusion(spectral_features, spatial_enhanced)  # [B, fusion_dim, H, W]
        # 辅助特征（可选）
        aux_features = {
            'GMA': spectral_features,
            'SFE': spatial_enhanced,
            # 'ES2CM': es2cm_features,
        }

        # 输出分类结果和辅助特征
        return fused_features, aux_features


class BaseNet(nn.Module):
    def __init__(self, msi_channels=24, rgb_channels=3,
                 hidden_dim=32,
                 es2cm_compression=32/48,
                 fusion_dim=32,
                 num_classes=4):
        super().__init__()
        self.gma = GMABlock(HSI_Data_Shape_C=msi_channels, patch_size=64)
        self.conv_1x1 = nn.Conv2d(24, hidden_dim, kernel_size=1, stride=1, padding=0)
        # 使用位置参数调用 SFE
        self.sfe = SFE(msi_channels, rgb_channels)

        # ES2CM 已注释（消融实验）
        # self.es2cm = ES2CM(input_channels=48, compression_ratio=es2cm_compression)

        # 特征融合模块（修正：第二个参数传入 SFE 的输出通道）
        self.fusion = HFFG(
            stm_dim=hidden_dim,
            es2cm_dim=self.sfe.output_channels,   # 关键修正
            fusion_dim=fusion_dim
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fusion_dim, num_classes)
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

        fused_features, attn_weights = self.fusion(spectral_features, spatial_enhanced)

        logits = self.classifier(fused_features)

        aux_features = {
            'GMA': spectral_features,
            'SFE': spatial_enhanced,
            'fused': fused_features
        }

        return logits, aux_features


if __name__ == '__main__':
    # 模拟输入
    msi = torch.randn(2, 24, 32, 32)
    rgb = torch.randn(2, 3, 32, 32)

    model = BaseNet(num_classes=4)  # 默认4分类
    logits, aux = model(msi, rgb)

    print(logits.shape)  # torch.Size([2, 4])
