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
# 3. 增强空间局部特征的SFE模块（多尺度+局部注意力）
class SFE(nn.Module):
    def __init__(self, rgb_channels):
        super().__init__()
        # 多尺度空洞卷积（适应不同大小瘀伤）
        self.scale1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(rgb_channels, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(rgb_channels, 16, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(16),  # 单独归一化
            nn.LeakyReLU(0.2)
        )

        # 通用注意力：基于光谱异常、空间连续性
        self.local_attn = MultiFeatureAttention(48, 8)

    def forward(self, rgb):
        # 多尺度特征融合
        s1 = self.scale1(rgb)
        s2 = self.scale2(rgb)
        s3 = self.scale3(rgb)
        features = torch.cat([s1, s2, s3], dim=1)  # [B, 48, H, W]（多尺度融合）

        # 应用通用注意力（只输入RGB特征）
        fused_feat, attn_weight = self.local_attn(features)

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
# --------------------RGB_STM模块------------------------
class RGB_STM(nn.Module):
    """专为RGB数据设计的空间特征处理模块"""
    def __init__(self, spec_dim, spatial_dim=32, depth=2, heads=2, dropout=0.1):
        super().__init__()
        self.spec_dim = spec_dim
        self.spatial_dim = spatial_dim
        self.spec_proj = nn.Sequential(
            nn.Conv1d(spec_dim, spatial_dim, kernel_size=1),
            nn.BatchNorm1d(spatial_dim),
            nn.LeakyReLU(0.2)
        )
        self.spec_gate = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(spatial_dim, spatial_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spatial_dim // 2, spatial_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.layers = nn.ModuleList()
        self.res_scales = nn.ParameterList()
        for _ in range(depth):
            layer_modules = nn.ModuleList([
                nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1, groups=heads),
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1, groups=heads),
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1),
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(spatial_dim),
                nn.Dropout(dropout),
            ])
            self.layers.append(layer_modules)
            self.res_scales.append(nn.Parameter(torch.tensor(0.5)))
    def forward(self, rgb_feat):
        B, C, H, W = rgb_feat.shape
        assert C == self.spec_dim, f"输入通道维度 {C} 与模块预期 {self.spec_dim} 不匹配"
        spec_proj = self.spec_proj(rgb_feat.flatten(2)).view(B, self.spatial_dim, H, W)
        spec_attn = self.spec_gate(spec_proj)
        spec_gated = spec_proj * spec_attn
        x = spec_gated
        for i, (spec_conv, spatial_conv, fuse_gate, residual_conv, bn, dropout) in enumerate(self.layers):
            res_scale = self.res_scales[i]
            x_spec = spec_conv(x.flatten(2)).view(B, self.spatial_dim, H, W)
            x_spatial = spatial_conv(x)
            fuse_weight = torch.sigmoid(fuse_gate(x_spec + x_spatial))
            x_fused = fuse_weight * x_spec + (1 - fuse_weight) * x_spatial
            residual = torch.sigmoid(res_scale) * residual_conv(x_fused)
            x = bn(x + residual)
            x = dropout(x)
        return x
# --------------------HFFG组件------------------------
class SpatialPositionalEncoding(nn.Module):
    """空间位置编码：为特征添加2D位置信息，增强注意力对局部区域的定位能力"""
    def __init__(self, dim, H=32, W=32):
        super().__init__()
        # 生成x/y轴位置编码（正弦余弦编码，适配不同空间尺寸）
        self.pos_weight = nn.Parameter(torch.tensor(0.1))# 可学习权重
        pos_x = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)  # [H, W]
        pos_y = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)  # [H, W]
        # 位置编码投影到特征维度
        self.proj_x = nn.Conv2d(1, dim, kernel_size=1)
        self.proj_y = nn.Conv2d(1, dim, kernel_size=1)

    def forward(self, x):
        """x: [B, dim, H, W]"""
        B, dim, H, W = x.shape
        # 动态生成位置编码（不依赖初始化时的固定尺寸）
        pos_x = torch.linspace(0, 1, W, device=x.device).unsqueeze(0).repeat(H, 1)
        pos_y = torch.linspace(0, 1, H, device=x.device).unsqueeze(1).repeat(1, W)

        # 生成x/y轴位置特征
        pos_x_feat = self.proj_x(pos_x.expand(B, 1, H, W))  # [B, dim, H, W]
        pos_x_feat = F.normalize(pos_x_feat, dim=1)  # 通道维度归一化
        pos_y_feat = self.proj_y(pos_y.expand(B, 1, H, W))  # [B, dim, H, W]
        pos_y_feat = F.normalize(pos_y_feat, dim=1)  # 通道维度归一化
        # 位置特征与输入特征相加（保留原特征，补充位置信息）
        return x + self.pos_weight * (pos_x_feat + pos_y_feat)  # 小权重避免位置信息主导

class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力：让两个分支特征互相引导注意力，突出互补信息"""
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads  # 每头特征维度（需确保dim能被heads整除）
        assert self.head_dim * heads == dim, "融合维度必须能被注意力头数整除"

        # 线性投影：Query（Q）、Key（K）、Value（V）
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)  # 使用BatchNorm替代LayerNorm
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        nn.init.xavier_normal_(self.q_proj[0].weight)
        nn.init.xavier_normal_(self.k_proj[0].weight)
        nn.init.xavier_normal_(self.v_proj[0].weight)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)  # 注意力输出投影

        self.scale = self.head_dim ** -0.5  # 注意力缩放因子，避免梯度爆炸
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        """
        输入：
            x_a: [B, dim, H, W] → 作为Query的特征（如STM特征）
            x_b: [B, dim, H, W] → 作为Key/Value的特征（如ES2CM特征）
        输出：
            attn_out: [B, dim, H, W] → 注意力加权后的x_a特征
        """
        B, dim, H, W = x_a.shape
        N = H * W  # 空间维度展平为序列长度（N = H*W）

        # 1. 线性投影 + 多头拆分（[B, dim, H, W] → [B, heads, N, head_dim]）
        # Query: 从x_a生成
        q = self.q_proj(x_a).view(B, self.heads, self.head_dim, N).transpose(2, 3)  # [B, h, N, d]
        # Key/Value: 从x_b生成
        k = self.k_proj(x_b).view(B, self.heads, self.head_dim, N).transpose(2, 3)  # [B, h, N, d]
        v = self.v_proj(x_b).view(B, self.heads, self.head_dim, N).transpose(2, 3)  # [B, h, N, d]

        # 2. 计算交叉注意力分数（Q@K^T / sqrt(d)）
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn_weight = F.softmax(attn_score, dim=-1)  # [B, h, N, N]，空间位置间的注意力权重
        attn_weight = self.dropout(attn_weight)

        # 优化1：过滤低置信度权重（如权重<0.1视为噪声，设为0）
        conf_threshold = torch.mean(attn_weight, dim=-1, keepdim=True) * 0.5  # 动态阈值（均值的50%）
        attn_weight = torch.where(attn_weight < conf_threshold, torch.zeros_like(attn_weight), attn_weight)
        attn_weight = attn_weight / (attn_weight.sum(dim=-1, keepdim=True) + 1e-6)  # 重新归一化

        # 3. 注意力加权Value（突出x_b中与x_a相关的区域）
        attn_val = torch.matmul(attn_weight, v)  # [B, h, N, d]

        # 4. 多头合并 + 输出投影（恢复图像格式）
        attn_val = attn_val.transpose(2, 3).contiguous().view(B, dim, H, W)  # [B, dim, H, W]
        attn_out = self.out_proj(attn_val)

        return attn_out,attn_weight
# ----------------------------------------------------
# HFFG模块
class HFFG(nn.Module):
    """
        基于多头交叉注意力的STM-ES2CM融合模块：
        1. 双向交叉注意力：STM引导ES2CM聚焦光谱异常区域，ES2CM引导STM聚焦空间异常区域
        2. 位置编码增强局部定位：适配瘀伤的局部特征特性
        3. 残差精炼：确保特征融合后不丢失原始有效信息
        """

    def __init__(self, stm_dim, es2cm_dim, fusion_dim=32, heads=4, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.heads = heads

        # --------------------------
        # 1. 特征维度对齐（将STM和ES2CM映射到同一融合维度）
        # --------------------------
        self.stm_proj = nn.Sequential(
            nn.Conv2d(stm_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.LeakyReLU(0.2)  # 增强负特征梯度传递
        )
        self.es2cm_proj = nn.Sequential(
            nn.Conv2d(es2cm_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.LeakyReLU(0.2)
        )

        # --------------------------
        # 2. 空间位置编码（增强注意力对瘀伤局部的定位）
        # --------------------------
        # 注：H/W在forward中动态获取，此处先定义投影层，位置编码在forward中生成
        self.pos_enc = SpatialPositionalEncoding(dim=fusion_dim, H=1, W=1)  # 占位，H/W后续更新

        # --------------------------
        # 3. 双向多头交叉注意力（核心融合机制）
        # --------------------------
        # ① STM→ES2CM交叉注意力：用STM的光谱特征引导ES2CM聚焦空间异常
        self.cross_attn_stm2es2cm = MultiHeadCrossAttention(dim=fusion_dim, heads=heads, dropout=dropout)
        # ② ES2CM→STM交叉注意力：用ES2CM的空间特征引导STM聚焦光谱异常
        self.cross_attn_es2cm2stm = MultiHeadCrossAttention(dim=fusion_dim, heads=heads, dropout=dropout)

        # --------------------------
        # 4. 特征融合与精炼（强化瘀伤特征，稳定训练）
        # --------------------------
        # 双向注意力特征融合
        self.fuse_conv = nn.Conv2d(2 * fusion_dim, fusion_dim, kernel_size=3, padding=1, groups=fusion_dim)
        # 残差精炼块（深度卷积+点卷积，轻量化且保留局部特征）
        self.refine_block = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1, groups=fusion_dim),  # 深度卷积：保留空间局部
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1),  # 点卷积：通道交互
            nn.BatchNorm2d(fusion_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        # 跨分支残差（保留原始特征差异，增强对比）
        self.cross_residual = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1)

    def forward(self, stm_feat, es2cm_feat):
        """
        输入：
            stm_feat: [B, stm_dim, H, W] → RGB_STM输出的空间特征
            es2cm_feat: [B, es2cm_dim, H, W] → ES2CM输出的空间增强特征
        输出：
            fused_feat: [B, fusion_dim, H, W] → 融合后的瘀伤特征（空间-空间协同增强）
        """
        B, _, H, W = stm_feat.shape
        # 动态更新位置编码的H/W（适配输入特征的空间尺寸）
        self.pos_enc = SpatialPositionalEncoding(dim=self.fusion_dim, H=H, W=W).to(stm_feat.device)

        # --------------------------
        # 步骤1：特征维度对齐 + 位置编码
        # --------------------------
        stm_proc = self.stm_proj(stm_feat)  # [B, fusion_dim, H, W]
        es2cm_proc = self.es2cm_proj(es2cm_feat)  # [B, fusion_dim, H, W]
        # 加入空间位置编码（帮助注意力定位瘀伤局部）
        stm_proc = self.pos_enc(stm_proc)
        es2cm_proc = self.pos_enc(es2cm_proc)

        # 新增：分支特征相似度校验（计算每个空间位置的特征相似度）
        # 归一化特征后计算余弦相似度（值越近1，特征越一致；越近0，分歧越大）
        stm_norm = F.normalize(stm_proc, dim=1)  # [B, C, H, W]
        es2cm_norm = F.normalize(es2cm_proc, dim=1)
        sim_map = torch.sum(stm_norm * es2cm_norm, dim=1, keepdim=True)  # [B, 1, H, W]，相似度图
        sim_mask = torch.sigmoid(5 * (sim_map - 0.5))  # 动态掩码：相似度<0.5的区域权重降低

        # --------------------------
        # 步骤2：双向交叉注意力（核心融合）
        # --------------------------
        # ① ES2CM特征被STM引导：聚焦光谱异常对应的空间区域（如红光异常→瘀伤位置）
        es2cm_attended,attn_stm2es2cm = self.cross_attn_stm2es2cm(x_a=es2cm_proc*sim_mask, x_b=stm_proc*sim_mask)  # [B, fusion_dim, H, W]
        # ② STM特征被ES2CM引导：聚焦空间异常对应的光谱通道（如边缘区域→近红外通道）
        stm_attended,attn_es2cm2stm = self.cross_attn_es2cm2stm(x_a=stm_proc*sim_mask, x_b=es2cm_proc*sim_mask)  # [B, fusion_dim, H, W]

        # --------------------------
        # 步骤3：特征融合 + 精炼
        # --------------------------
        # 拼接双向注意力特征（保留双方引导后的有效信息）
        joint_feat = torch.cat([stm_attended, es2cm_attended], dim=1)  # [B, 2*fusion_dim, H, W]
        # 通道融合（深度卷积避免跨通道干扰）
        fused = self.fuse_conv(joint_feat)  # [B, fusion_dim, H, W]
        # 特征精炼（强化局部细节，如瘀伤边缘）
        refined = self.refine_block(fused)
        # 跨分支残差（增强STM与ES2CM的特征对比，突出瘀伤差异）
        residual = self.cross_residual(stm_proc - es2cm_proc)
        fused_feat = refined + residual

        return fused_feat,{
            "attn_stm2es2cm": attn_stm2es2cm, # STM引导ES2CM的注意力权重
            "attn_es2cm2stm": attn_es2cm2stm, # ES2CM引导STM的注意力权重
        }
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
class BaseNet(nn.Module):
    """
        改造后的BaseNet子网络：移除分类头，专注于RGB特征的提取与融合，
        输出融合后的图像格式特征，供后续OverviewNet下采样或其他网络处理。
        """

    def __init__(self, rgb_channels=3,
                 hidden_dim=32,
                 es2cm_compression=0.8,
                 fusion_dim=32,  # 显式指定融合后的特征通道数（供后续网络适配）
                 attn_heads=2):
        super().__init__()
        # 1. 空间特征提取流（输出图像格式特征，无分类相关操作）
        self.pss = PSS(in_channels=rgb_channels,out_channels=hidden_dim)  # 输出[B, hidden_dim, H, W]
        self.stm = RGB_STM(
            spec_dim=hidden_dim,  # 与PSS输出通道数匹配（图像格式的通道维度）
            spatial_dim=hidden_dim,  # 输出通道保持一致，确保后续融合维度兼容
            depth=1,  # 浅层设计避免梯度消失，适配子网络轻量化需求
            heads=attn_heads,
            dropout=0.1
        )

        # 2. 空间特征增强流（图像格式特征提取，专注RGB融合增强）
        self.sfe = SFE(rgb_channels=rgb_channels)  # 融合RGB空间特征
        self.es2cm = ES2CM(
            input_channels=48,  # 严格匹配SFE输出的48通道
            compression_ratio=es2cm_compression  # 压缩空间特征，降低后续计算量
        )

        # 3. 特征融合模块（融合空间流与空间流特征，输出统一维度特征）
        self.fusion = HFFG(
            stm_dim=hidden_dim,  # 空间流输出通道数
            es2cm_dim=self.es2cm.output_channels,  # 空间流压缩后通道数（动态适配压缩比）
            fusion_dim=fusion_dim,  # 融合后固定通道数（供后续网络明确输入维度）
            heads=4,
            dropout=0.1
        )

        # 初始化权重（保持原逻辑，确保子网络参数稳定）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化：适配卷积、线性层与归一化层，确保训练稳定性"""
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Kaiming初始化：适配LeakyReLU激活函数，避免梯度消失
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0，减少初始偏差
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            # 归一化层权重初始化为1，偏置为0，确保初始分布正常
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, rgb_img):
        """
        前向传播：输入RGB数据，输出融合后的特征（供后续网络使用）
        Args:
            rgb_img: RGB彩色数据，格式 [B, rgb_channels, H, W]
        Returns:
            fused_features: 空间-空间融合特征，格式 [B, fusion_dim, H, W]（适配后续下采样等操作）
            aux_features: 辅助特征字典（可选，供调试或中间特征复用）
        """
        # 设备对齐：确保输入数据与模型参数在同一设备（CPU/GPU）
        device = next(self.parameters()).device
        rgb_img = rgb_img.to(device)

        # 1. 空间特征提取流：聚焦RGB的空间信息，输出图像格式特征
        spatial_features = self.pss(rgb_img)  # [B, hidden_dim, H, W]
        stm_features = self.stm(spatial_features)  # [B, hidden_dim, H, W]（增强空间-空间关联）

        # 2. 空间特征增强流：融合RGB的空间信息，压缩后输出
        spatial_enhanced = self.sfe(rgb_img)  # [B, 48, H, W]（RGB空间融合）
        es2cm_features = self.es2cm(spatial_enhanced)  # [B, es2cm_output_channels, H, W]（压缩+细节增强）

        # 3. 特征融合前校验：确保空间流与空间流的空间尺寸完全匹配（避免后续融合维度错误）
        B, _, H_stm, W_stm = stm_features.shape
        B, _, H_es2cm, W_es2cm = es2cm_features.shape
        assert H_stm == H_es2cm and W_stm == W_es2cm, \
            f"空间流空间尺寸({H_stm}x{W_stm})与空间流({H_es2cm}x{W_es2cm})不匹配，请检查输入尺寸或模块设计"

        # 4. 核心特征融合：输出固定通道数的融合特征（供后续网络直接使用）
        fused_features,attn_weights = self.fusion(stm_features, es2cm_features)  # [B, fusion_dim, H, W]

        # 可选：返回辅助特征（如中间层特征，供后续网络灵活复用）
        aux_features = {
            'PSS': spatial_features,  # 空间流中间特征
            'SFE': spatial_enhanced,  # 空间流中间特征（未压缩）
            'STM': stm_features,    # 空间增强中间特征
            'ES2CM': es2cm_features,  # 空间流压缩后特征
            'cross_attention':attn_weights
        }

        # 主输出：融合特征（供OverviewNet下采样等后续操作）；辅助输出：中间特征（可选）
        return fused_features, aux_features
# -----------------------------------------------------------------------------
# LSNet核心模块（复用并保持一致性）
# -----------------------------------------------------------------------------
class LSConv(nn.Module):
    """LSNet核心卷积模块（带轻量注意力）"""
    def __init__(self, dim, groups=8):
        super().__init__()
        self.lkp = nn.Conv2d(dim, dim, 7, 1, 3, groups=groups)  # 大核感受野
        self.ska = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)    # 深度卷积（空间注意力）
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x) * self.lkp(x)) + x  # 注意力增强+残差
class Conv2d_BN(nn.Sequential):
    """LSNet基础卷积-BN模块"""
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
        """融合conv和bn（推理加速）"""
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
    """残差连接模块"""
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
    """LSNetFeed-Forward Network"""
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.pw1 = Conv2d_BN(dim, hidden_dim, 1)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(hidden_dim, dim, 1)  # 残差分支初始化权重为0

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))
class RepVGGDW(nn.Module):
    """LSNet轻量深度卷积模块（带重参数化）"""
    def __init__(self, dim):
        super().__init__()
        self.conv3x3 = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)  # 深度卷积
        self.conv1x1 = Conv2d_BN(dim, dim, 1, 1, 0, groups=dim)  # 点卷积（补充1x1感受野）
        self.dim = dim

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) + x  # 残差连接（原始特征+3x3+1x1）

    @torch.no_grad()
    def fuse(self):
        """融合为单个卷积（推理加速）"""
        conv3x3 = self.conv3x3.fuse()
        conv1x1 = self.conv1x1.fuse()

        # 1x1卷积pad为3x3
        conv1x1_weight = F.pad(conv1x1.weight, [1, 1, 1, 1])
        # 恒等映射（单位矩阵pad为3x3）
        identity_weight = F.pad(torch.eye(self.dim, device=conv1x1.weight.device).view(self.dim, self.dim, 1, 1), [1, 1, 1, 1])

        # 权重融合
        fused_weight = conv3x3.weight + conv1x1_weight + identity_weight
        fused_bias = conv3x3.bias + conv1x1.bias

        conv3x3.weight.data.copy_(fused_weight)
        conv3x3.bias.data.copy_(fused_bias)
        return conv3x3
class LSBlock(nn.Module):
    """LSNet基础块（交替使用RepVGGDW和LSConv）"""
    def __init__(self, dim, depth, stage):
        super().__init__()
        # 交替使用不同混合器（保持LSNet设计理念）
        if depth % 2 == 0:
            self.mixer = RepVGGDW(dim)
            self.se = SqueezeExcite(dim, 0.25)  # 通道注意力
        else:
            self.mixer = LSConv(dim)
            self.se = nn.Identity()  # 部分块不使用SE，避免过拟合

        self.ffn = Residual(FFN(dim))  # FFN残差连接

    def forward(self, x):
        x = self.se(self.mixer(x))
        x = self.ffn(x)
        return x

# ==============下采样必需模块（通道适配修改）================
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # b c h w -> b h w c
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # b h w c -> b c h w
        return x

class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    kernel_size = (kernel_size, kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = (padding, padding)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)

def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        # 保持原卷积核配置，仅适配通道数
        if kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block建议使用kernel_size=5或7以平衡性能和内存')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x

# 增强初期特征缩放能力，防止特征值累积异常
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-2,device=None):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))  # 增强初始调节强度
    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)  # 按通道缩放

class GRN(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class RepConvBlock(nn.Module):
    # 适配调整后的通道数，保持结构轻量化
    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False):
        super().__init__()

        mlp_dim = int(dim * mlp_ratio)  # 按比例计算MLP维度，避免固定高维度

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
                                attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),  # 1x1卷积降参
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),  # 1x1卷积回原维度
            nn.Dropout(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.norm_mid = nn.BatchNorm2d(dim)# 中间层归一化
    def forward_features(self, x):
        x = self.dwconv(x)
        drop_path = self.proj[-1]
        x = x + drop_path(self.ls(self.proj[:-1](x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

# ---- Overview-Net（通道调整，与BaseNet输出对齐） ---------
class OverviewNet(nn.Module):
    """
        适配 BruiseSTFBaseNet 的 OverViewNet：
        1. 输入通道数与 BaseNet 的 fusion_dim 严格对齐（默认32，可随 BaseNet 配置灵活调整）
        2. 优化空洞卷积参数（适配 BaseNet 输出的原始分辨率，避免过度扩张）
        3. 保持通道维度稳定（输入=输出，供后续网络直接复用特征维度）
        4. 兼容 BaseNet 的设备对齐逻辑，确保跨设备训练稳定
        """

    def __init__(self,
                 in_chans=32,  # 核心适配：与 BaseNet 的 fusion_dim 一致（默认32，需和 BaseNet 保持同步）
                 embed_dim=32,  # 输出通道数与输入一致，避免后续网络维度适配成本
                 depth=2,  # 保持轻量结构，与 BaseNet 深度（depth=1）匹配，避免梯度消失
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=1,
                 drop_path_rate=0,
                 norm_layer=LayerNorm2d):
        super().__init__()

        # 输入适配模块：将 BaseNet 输出的特征映射到 OverViewNet 内部维度（此处输入=输出，简化映射）
        # 优化空洞卷积：BaseNet 输出为原始分辨率（如 H×W），dilation=2 平衡感受野与参数效率（原dilation=6易导致参数冗余）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            # padding = (kernel_size-1)*dilation // 2 → (3-1)*2//2=2，确保输出分辨率与输入完全一致
            norm_layer(embed_dim)  # 用传入的 norm_layer（默认LayerNorm2d），保持与 BaseNet 归一化逻辑兼容
        )

        # 构建轻量 RepConvBlock 堆叠：深度=2，与 BaseNet 轻量定位匹配
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 线性分配drop_path概率，避免过拟合
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                RepConvBlock(
                    dim=embed_dim,  # 通道数=32，与输入/输出维度一致
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    drop_path=dpr[i],  # 逐层递增drop_path，增强泛化性
                    norm_layer=norm_layer,
                )
            )

        # 权重初始化：保持与 BaseNet 一致的初始化逻辑，确保训练稳定性
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        与 BruiseSTFBaseNet 对齐的权重初始化逻辑：
        - 卷积/线性层：Truncated Normal 初始化（避免初始权重过大导致梯度爆炸）
        - 归一化层：权重=1、偏置=0（确保初始分布正常，无额外偏移）
        """
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if m.groups > 1:  # 分组卷积单独处理
                # 按组拆分权重并分别初始化
                group_size = m.weight.size(0) // m.groups
                for i in range(m.groups):
                    start = i * group_size
                    end = (i + 1) * group_size
                    nn.init.trunc_normal_(m.weight[start:end], std=.01)  # 降低分组初始化标准差
            else:
                nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)  # 截断正态分布，减少极端值
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0，避免初始偏差
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)  # 归一化层权重=1，保持特征尺度
            nn.init.constant_(m.bias, 0)  # 归一化层偏置=0，避免额外偏移

    def forward(self, x):
        """
        前向传播逻辑：适配 BaseNet 输出的特征格式
        输入 x：BaseNet 输出的融合特征，格式 [B, fusion_dim, H, W] → 此处为 [B, 32, H, W]
        输出 x：OverViewNet 处理后的特征，格式 [B, embed_dim, H, W] → 此处为 [B, 32, H, W]（分辨率不变）
        """
        # 设备对齐：与 BaseNet 保持一致，避免跨设备数据类型错误
        device = next(self.parameters()).device
        x = x.to(device)

        # 1. 输入特征映射：保持分辨率不变，增强局部感受野
        x = self.patch_embed(x)  # [B, 32, H, W] → 分辨率与输入完全一致

        # 2. 特征增强：通过 RepConvBlock 堆叠提取深层特征
        for blk in self.blocks:
            x = blk(x)  # 每块处理后仍保持 [B, 32, H, W]

        # 输出：与后续网络兼容的特征格式（通道数32，原始分辨率）
        return x

# 适配LSNet的基础模块
class LSConv(nn.Module):
    """轻量级光谱-空间卷积模块，专注RGB特征提取"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FocusNet(nn.Module):
    """
    适配 BruiseSTFBaseNet 的 FocusNet：
    1. 输入：BaseNet 融合特征（32通道）+ OverviewNet 上下文特征（32通道）
    2. 核心：轻量 LSConv 堆叠，避免过拟合，适配RGB单模态数据
    3. 输出：分类头（2类：瘀伤/非瘀伤），分辨率与输入一致
    """

    def __init__(self, base_dim=32, ctx_dim=32, num_classes=2, depth=4):
        super().__init__()

        # 1. 特征融合：BaseNet + OverviewNet（通道数均为32，直接相加）
        # 注：OverviewNet 输出分辨率与 BaseNet 一致，无需上采样，避免计算冗余
        self.fuse = nn.Conv2d(base_dim + ctx_dim, base_dim, kernel_size=1)  # 1x1卷积降维到32通道
        self.fuse_bn = nn.BatchNorm2d(base_dim)
        self.fuse_relu = nn.LeakyReLU(0.2)

        # 2. 轻量特征提取：LSConv 堆叠（深度=4，与原LSNet深度一致）
        self.layers = nn.ModuleList()
        for i in range(depth):
            # 保持通道数稳定在32，避免维度膨胀
            self.layers.append(LSConv(in_channels=base_dim, out_channels=base_dim))

        # 3. 分类头：从32通道映射到2类输出
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_dim, base_dim // 2),
            nn.BatchNorm1d(base_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(base_dim // 2, num_classes)
        )

        # 权重初始化：保持与 BaseNet/OverviewNet 一致的逻辑
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """与 BruiseSTFBaseNet 对齐的权重初始化逻辑"""
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, base_feat, ctx):
        """
        前向传播逻辑：
        输入 base_feat：BaseNet 输出的融合特征 [B, 32, H, W]
        输入 ctx：OverviewNet 输出的上下文特征 [B, 32, H, W]（分辨率与 base_feat 一致）
        输出 logits：分类结果 [B, 2]
        输出 fused_feat：融合后的特征 [B, 32, H, W]
        """
        # 设备对齐：避免跨设备数据类型错误
        device = next(self.parameters()).device
        base_feat = base_feat.to(device)
        ctx = ctx.to(device)

        # 1. 特征融合：BaseNet 局部特征 + OverviewNet 上下文特征
        fused = torch.cat([base_feat, ctx], dim=1)  # [B, 64, H, W]
        fused_feat = self.fuse_relu(self.fuse_bn(self.fuse(fused)))  # [B, 32, H, W]

        # 2. 深层特征提取：通过 LSConv 堆叠增强特征表达
        for layer in self.layers:
            fused_feat = layer(fused_feat)  # 每层处理后仍保持 [B, 32, H, W]

        # 3. 分类：映射到2类输出
        logits = self.classifier(fused_feat)  # [B, 2]

        return logits, fused_feat

# -----------------------------------------------------------------------------
# 总网络：BruiseSTFNet（整合BaseNet + OverviewNet + FocusNet）
# -----------------------------------------------------------------------------
class BruiseSTFNet(nn.Module):
    """
    适配 BruiseSTFBaseNet 的总网络：
    1. 输入：RGB 单模态数据 [B, 3, H, W]
    2. 输出：二分类结果 [B, 2]（瘀伤/非瘀伤）
    3. 核心组件：
       - BaseNet：RGB 特征提取与融合（输出32通道特征）
       - OverviewNet：上下文特征增强（输出32通道特征，分辨率与 BaseNet 一致）
       - FocusNet：深层特征提取与分类（输出2类预测）
    """

    def __init__(self, rgb_channels=3, num_classes=2):
        super().__init__()

        # 1. BaseNet：RGB 特征提取与融合（固定 fusion_dim=32）
        self.base_net = BaseNet(
            rgb_channels=rgb_channels,
            hidden_dim=32,  # 空间流输出通道数
            fusion_dim=32   # 融合后固定通道数，与 OverviewNet 输入对齐
        )

        # 2. OverviewNet：上下文特征增强（输入/输出通道均为32，分辨率与 BaseNet 一致）
        self.overview_net = OverviewNet(
            in_chans=32,  # 与 BaseNet 的 fusion_dim 严格对齐
            embed_dim=32  # 输出通道与输入一致，与 FocusNet 输入对齐
        )

        # 3. FocusNet：深层特征提取与分类（输入通道均为32，输出2类）
        self.focus_net = FocusNet(
            base_dim=32,  # 与 BaseNet 的 fusion_dim 对齐
            ctx_dim=32,   # 与 OverviewNet 的 embed_dim 对齐
            num_classes=num_classes  # 二分类：瘀伤/非瘀伤
        )

        # 权重初始化：统一初始化策略，确保训练稳定性
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """与子网络一致的权重初始化逻辑"""
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, rgb_img):
        """
        前向传播逻辑：
        输入 rgb_img：RGB 单模态数据 [B, 3, H, W]
        输出 logits：二分类结果 [B, 2]
        输出 aux_features：辅助特征字典（可选）
        """
        # 设备对齐：确保输入数据与模型参数在同一设备
        device = next(self.parameters()).device
        rgb_img = rgb_img.to(device)

        # 1. BaseNet：提取 RGB 融合特征
        base_feat, base_aux = self.base_net(rgb_img)  # [B, 32, H, W]

        # 2. OverviewNet：增强上下文特征
        ctx = self.overview_net(base_feat)  # [B, 32, H, W]（分辨率与 BaseNet 一致）

        # 3. FocusNet：深层特征提取与分类
        logits, fused_feat = self.focus_net(base_feat, ctx)  # [B, 2]

        # 辅助特征（可选）
        aux_features = {**base_aux, 'base_feat': base_feat, 'ctx': ctx, 'fused_feat': fused_feat}

        return logits, aux_features
# -----------------------------------------------------------------------------
# 测试代码（验证修改后网络的正确性）
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 模拟输入
    rgb = torch.randn(4, 3, 64, 64)

    # 初始化网络
    model = BruiseSTFNet(num_classes=2)  # 二分类：瘀伤/非瘀伤

    # 前向传播
    logits, aux_features = model(rgb)

    # 验证输出维度
    print(f"分类输出维度: {logits.shape} (期望: [4, 2])")
    print(f"Base_feat维度: {aux_features['base_feat'].shape} (期望: [4, 32, 64, 64])")
    print(f"Ctx维度: {aux_features['ctx'].shape} (期望: [4, 32, 64, 64])")
    print(f"Fused_feat维度: {aux_features['fused_feat'].shape} (期望: [4, 32, 64, 64])")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params / 1e6:.2f} M")

    # 验证FocusNet的LSConv是否正常工作
    ls_conv = model.focus_net.layers[0]
    test_feat = torch.randn(4, 32, 64, 64)
    ls_output = ls_conv(test_feat)
    print(f"LSConv输入输出维度是否一致: {test_feat.shape == ls_output.shape} (期望: True)")