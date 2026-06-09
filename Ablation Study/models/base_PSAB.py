import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SqueezeExcite

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
# 3. 增强空间局部特征的SFE模块（多尺度+局部注意力）
class SFE(nn.Module):
    def __init__(self, msi_channels, rgb_channels):
        super().__init__()
        self.output_channels = 48
        self.rgb_adjust = nn.Conv2d(rgb_channels, msi_channels, kernel_size=1, bias=False)
        # 新增：将调整后的RGB特征通道数投影到48（与多尺度融合特征匹配）
        self.rgb_adj_proj = nn.Conv2d(msi_channels, 48, kernel_size=1)  # 关键：通道对齐
        # 多尺度空洞卷积（适应不同大小瘀伤）
        self.scale1 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(msi_channels * 2, 16, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )

        # 通用注意力：基于光谱异常、空间连续性、跨模态一致性（无形状约束）
        self.local_attn = MultiFeatureAttention(48, 8)

    def forward(self, msi, rgb):
        rgb_adj = self.rgb_adjust(rgb)
        # 新增：将RGB特征投影到48通道（与多尺度融合特征匹配）
        rgb_adj_proj = self.rgb_adj_proj(rgb_adj)  # [B, 48, H, W]
        combined = torch.cat([msi, rgb_adj], dim=1)  # [B, 2*C, H, W]

        # 多尺度特征融合
        s1 = self.scale1(combined)
        s2 = self.scale2(combined)
        s3 = self.scale3(combined)
        features = torch.cat([s1, s2, s3], dim=1)  # [B, 48, H, W]（多尺度融合）

        # 应用通用注意力（输入多光谱特征和RGB调整特征，强化跨模态一致性）
        fused_feat, attn_weight = self.local_attn(features,rgb_adj_proj)

        return fused_feat
# 4. 修复残差连接的ES2CM模块
class ES2CM(nn.Module):
    def __init__(self, input_channels=48, compression_ratio=32 / 48):  # 降低压缩率，保留更多特征
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = max(16, int(input_channels * compression_ratio))
        # NEW:残差缩放因子(可学习，初始值为0.5)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        # 主分支
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.LeakyReLU(0.2)  # 避免ReLU截断负梯度
        )

        # 细节分支（调整激活函数）
        self.detail_branch = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保权重为正，避免特征值符号突变
        )

        # 残差连接通道匹配
        self.residual_proj = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_channels)  # 新增归一化
        )

    def forward(self, fused_features):
        residual = self.residual_proj(fused_features)  # 引入输入特征的残差
        out = self.conv(fused_features)
        detail_weights = self.detail_branch(fused_features)
        return self.res_scale * residual + out * detail_weights  # 残差分支乘以缩放因子，平衡主分支和残差贡献
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
        self.es2cm_proj = nn.Conv2d(es2cm_dim, fusion_dim, 1)   # 改为 es2cm_proj
        self.fusion = GatedFusion(fusion_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, groups=fusion_dim),
            nn.Conv2d(fusion_dim, fusion_dim, 1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU()
        )

    def forward(self, stm_feat, es2cm_feat):
        stm_proc = self.stm_proj(stm_feat)
        es2cm_proc = self.es2cm_proj(es2cm_feat)   # 使用 es2cm_proj
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

        reduced_dim = max(1,dim//16)#防止除数为0
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

# ================ 修改后的 BaseNet（直接输出分类结果） ================
class BaseNet(nn.Module):
    """
    消融实验版本：删除 GMABlock，使用 RGB 简单卷积 + ES2CM 特征融合
    直接输出分类结果（包含分类头）
    """
    def __init__(self, msi_channels=24, rgb_channels=3,
                 hidden_dim=32,
                 es2cm_compression=32/48,
                 fusion_dim=32,
                 num_classes=4):          # 新增分类数参数
        super().__init__()

        # RGB 简单卷积
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # 空间特征增强流（SFE + ES2CM）
        self.sfe = SFE(msi_channels=msi_channels, rgb_channels=rgb_channels)
        self.es2cm = ES2CM(input_channels=48, compression_ratio=es2cm_compression)

        # 特征融合模块
        self.fusion = HFFG(
            stm_dim=hidden_dim,
            es2cm_dim=self.es2cm.output_channels,
            fusion_dim=fusion_dim
        )

        # 分类头（直接输出分类结果）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
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

        # RGB 简单卷积提取特征
        rgb_features = self.rgb_conv(rgb_img)          # [B, hidden_dim, H, W]

        # 空间特征增强流
        spatial_enhanced = self.sfe(msi_data, rgb_img) # [B, 48, H, W]
        es2cm_features = self.es2cm(spatial_enhanced)  # [B, es2cm_output_channels, H, W]

        # 特征融合
        fused_features, attn_weights = self.fusion(rgb_features, es2cm_features)

        # 分类
        logits = self.classifier(fused_features)       # [B, num_classes]

        aux_features = {
            'rgb_features': rgb_features,
            'SFE': spatial_enhanced,
            'ES2CM': es2cm_features,
            'fused_features': fused_features
        }
        return logits, aux_features


# -----------------------------------------------------------------------------
# 测试代码（验证修改后网络的正确性）
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    msi = torch.randn(2, 24, 32, 32)
    rgb = torch.randn(2, 3, 32, 32)
    model = BaseNet(num_classes=4)
    logits, aux_features = model(msi, rgb)
    print(f"分类输出维度: {logits.shape} (期望: [2, 4])")
    print(f"融合特征维度: {aux_features['fused_features'].shape}")