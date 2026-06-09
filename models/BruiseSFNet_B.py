'''
适用二分类任务的模型
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import to_2tuple, DropPath
from torch.cuda import device
from torch.utils.checkpoint import checkpoint

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

# 修改完的STM模块
class MSI_STM(nn.Module):
    """专为MSI数据设计的光谱-空间特征处理模块"""

    def __init__(self,
                 spec_dim,  # 光谱通道数（MSI的输入光谱维度）
                 spatial_dim=32,  # 空间特征维度
                 depth=2,  # 模块深度（建议1-2层，避免梯度消失）
                 heads=2,  # 注意力头数（适配光谱分组）
                 dropout=0.1):
        super().__init__()
        self.spec_dim = spec_dim
        self.spatial_dim = spatial_dim
        # 1. 光谱-空间特征对齐（将MSI的光谱维度映射到空间特征维度）
        self.spec_proj = nn.Sequential(
            nn.Conv1d(spec_dim, spatial_dim, kernel_size=1),  # 光谱维度压缩/扩张
            nn.BatchNorm1d(spatial_dim),
            nn.LeakyReLU(0.2)# 负斜率从0.1增至0.2，增强负特征梯度传递
        )

        # 2.  替换光谱门控的全局池化为局部池化（3x3窗口捕捉局部光谱异常）
        self.spec_gate = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # 空间局部池化 [B, C, H, W]
            nn.Conv2d(spatial_dim, spatial_dim // 2, kernel_size=1),  # 适配2D输入
            nn.GELU(),
            nn.Conv2d(spatial_dim // 2, spatial_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 堆叠轻量级光谱-空间注意力层
        self.layers = nn.ModuleList()
        self.res_scales = nn.ParameterList()  # 新增：用ParameterList存储可学习缩放因子
        for _ in range(depth):
            layer_modules = (nn.ModuleList([
                # 光谱注意力分支（处理光谱间依赖）
                nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1, groups=heads),
                # 空间注意力分支（处理空间位置依赖）
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1, groups=heads),
                # 跨分支融合门控
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1),
                # 残差连接（简化梯度路径）
                nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1, bias=False),
                # 归一化与激活
                nn.BatchNorm2d(spatial_dim),
                nn.Dropout(dropout),
            ]))
            self.layers.append(layer_modules)
            # 单独存储缩放因子参数
            self.res_scales.append(nn.Parameter(torch.tensor(0.5)))

    def forward(self, msi_feat):
        """
        输入: msi_feat - MSI特征，形状为 [B, spec_dim, H, W]（原始MSI数据或其特征）
        输出: 融合光谱-空间特征，形状为 [B, spatial_dim, H, W]（可直接用于后续图像级融合）
        """
        B, C, H, W = msi_feat.shape
        assert C == self.spec_dim, f"输入光谱维度 {C} 与模块预期 {self.spec_dim} 不匹配"

        # 步骤1: 光谱维度预处理（压缩+门控）
        # 将空间维度展平，聚焦光谱处理: [B, C, H, W] → [B, C, H*W]
        spec_proj = self.spec_proj(msi_feat.flatten(2))  # [B, spec_dim, N]，N=H*W
        spec_proj = spec_proj.view(B,self.spatial_dim,H,W)  # 恢复空间维度 [B, C, H, W]
        # 局部光谱门控：基于3x3窗口的光谱注意力
        spec_attn = self.spec_gate(spec_proj) # [B, C, H, W]（空间维度对齐）
        spec_gated = spec_proj * spec_attn  # 局部光谱增强
        x = spec_gated

        # 步骤2: 堆叠光谱-空间注意力层(强化残差梯度)
        for i, (spec_conv, spatial_conv, fuse_gate, residual_conv, bn, dropout) in enumerate(self.layers):
            # 光谱分支（基于1D卷积，保持空间结构）
            # [B, C, H, W] → 展平空间 → [B, C, N] → 光谱卷积 → [B, C, N] → 恢复图像
            res_scale = self.res_scales[i] # 获取当前层的缩放因子
            x_spec = x.flatten(2)  # [B, C, N]
            x_spec = spec_conv(x_spec).view(B, self.spatial_dim, H, W)  # [B, C, H, W]

            # 空间分支（直接2D卷积）
            x_spatial = spatial_conv(x)  # [B, C, H, W]

            # 跨分支融合（门控机制控制融合比例）
            fuse_weight = torch.sigmoid(fuse_gate(x_spec + x_spatial))  # [B, C, H, W]
            x_fused = fuse_weight * x_spec + (1 - fuse_weight) * x_spatial  # 加权融合

            # 残差连接添加缩放因子，平衡原始特征与融合特征
            residual = torch.sigmoid(res_scale) * residual_conv(x_fused) # 缩放残差强度
            x = bn(x + residual)  # 原始特征+缩放后的融合残差，梯度路径更稳定
            x = dropout(x)

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

        #  # 通用注意力：基于光谱异常、空间连续性、跨模态一致性（无形状约束）
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
        fused_feat, attn_weight = self.local_attn(features, modal2=rgb_adj_proj)

        return fused_feat

# 4. 修复残差连接的ES2CM模块
class ES2CM(nn.Module):
    def __init__(self, input_channels=48, compression_ratio=32/48):  # 降低压缩率，保留更多特征
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
        # # 初始化位置编码（固定参数，不参与训练）
        # self.register_buffer('pos_x', pos_x.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
        # self.register_buffer('pos_y', pos_y.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]

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
            LayerNorm2d(dim)  # 注意LayerNorm的维度适配（需动态调整或用1D对通道归一化）
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            LayerNorm2d(dim)  # 注意LayerNorm的维度适配（需动态调整或用1D对通道归一化）
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            LayerNorm2d(dim)  # 注意LayerNorm的维度适配（需动态调整或用1D对通道归一化）
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
            stm_feat: [B, stm_dim, H, W] → MSI_STM输出的光谱-空间特征
            es2cm_feat: [B, es2cm_dim, H, W] → ES2CM输出的空间增强特征
        输出：
            fused_feat: [B, fusion_dim, H, W] → 融合后的瘀伤特征（光谱-空间协同增强）
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


# ==============下采样必需模块（通道适配修改）================
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()
class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
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
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
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
# ==============核心网络（通道调整关键模块）================

# ----  Base-Net（通道压缩：[96,192,384]→[64,128,256]）---
class BaseNet(nn.Module):
    """
        改造后的BaseNet子网络：移除分类头，专注于MSI+RGB特征的提取与融合，
        输出融合后的图像格式特征，供后续OverviewNet下采样或其他网络处理。
        """

    def __init__(self, msi_channels=24, rgb_channels=3,
                 hidden_dim=32,
                 es2cm_compression=0.8,
                 fusion_dim=32,  # 显式指定融合后的特征通道数（供后续网络适配）
                 attn_heads=2):
        super().__init__()
        # 1. 光谱特征提取流（输出图像格式特征，无分类相关操作）
        self.pss = PSS(in_channels=msi_channels,out_channels=hidden_dim)  # 输出[B, hidden_dim, H, W]
        self.stm = MSI_STM(
            spec_dim=hidden_dim,  # 与PSS输出通道数匹配（图像格式的通道维度）
            spatial_dim=hidden_dim,  # 输出通道保持一致，确保后续融合维度兼容
            depth=1,  # 浅层设计避免梯度消失，适配子网络轻量化需求
            heads=attn_heads,
            dropout=0.1
        )

        # 2. 空间特征增强流（图像格式特征提取，专注MSI+RGB融合增强）
        self.sfe = SFE(msi_channels=msi_channels, rgb_channels=rgb_channels)  # 融合MSI与RGB空间特征
        self.es2cm = ES2CM(
            input_channels=48,  # 严格匹配SFE输出的32通道
            compression_ratio=es2cm_compression  # 压缩空间特征，降低后续计算量
        )

        # 3. 特征融合模块（融合光谱流与空间流特征，输出统一维度特征）
        self.fusion = HFFG(
            stm_dim=hidden_dim,  # 光谱流输出通道数下
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

    def forward(self, msi_data, rgb_img):
        """
        前向传播：输入MSI+RGB数据，输出融合后的特征（供后续网络使用）
        Args:
            msi_data: MSI多光谱数据，格式 [B, msi_channels, H, W]
            rgb_img: RGB彩色数据，格式 [B, rgb_channels, H, W]
        Returns:
            fused_features: 光谱-空间融合特征，格式 [B, fusion_dim, H, W]（适配后续下采样等操作）
            aux_features: 辅助特征字典（可选，供调试或中间特征复用）
        """
        # 设备对齐：确保输入数据与模型参数在同一设备（CPU/GPU）
        device = next(self.parameters()).device
        msi_data = msi_data.to(device)
        rgb_img = rgb_img.to(device)

        # 1. 光谱特征提取流：聚焦MSI的光谱信息，输出图像格式特征
        spectral_features = self.pss(msi_data)  # [B, hidden_dim, H, W]
        stm_features = self.stm(spectral_features)  # [B, hidden_dim, H, W]（增强光谱-空间关联）

        # 2. 空间特征增强流：融合MSI与RGB的空间信息，压缩后输出
        spatial_enhanced = self.sfe(msi_data, rgb_img)  # [B, 32, H, W]（MSI+RGB空间融合）
        es2cm_features = self.es2cm(spatial_enhanced)  # [B, es2cm_output_channels, H, W]（压缩+细节增强）

        # 3. 特征融合前校验：确保光谱流与空间流的空间尺寸完全匹配（避免后续融合维度错误）
        B, _, H_stm, W_stm = stm_features.shape
        B, _, H_es2cm, W_es2cm = es2cm_features.shape
        assert H_stm == H_es2cm and W_stm == W_es2cm, \
            f"光谱流空间尺寸({H_stm}x{W_stm})与空间流({H_es2cm}x{W_es2cm})不匹配，请检查输入尺寸或模块设计"

        # 4. 核心特征融合：输出固定通道数的融合特征（供后续网络直接使用）
        fused_features,attn_weights = self.fusion(stm_features, es2cm_features)  # [B, fusion_dim, H, W]

        # 可选：返回辅助特征（如中间层特征，供后续网络灵活复用）
        aux_features = {
            'PSS': spectral_features,  # 光谱流中间特征
            'SFE': spatial_enhanced,  # 空间流中间特征（未压缩）
            'STM': stm_features,    # 光谱增强中间特征
            'ES2CM': es2cm_features,  # 空间流压缩后特征
            'cross_attention':attn_weights
        }

        # 主输出：融合特征（供OverviewNet下采样等后续操作）；辅助输出：中间特征（可选）
        return fused_features, aux_features

# ---- Overview-Net（通道调整，与BaseNet输出对齐） ---------
class OverViewNet(nn.Module):
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

# ---------------多尺度特征融合：适应瘀伤尺度差异------------
# 新增特征金字塔（FPN）结构：融合不同分辨率特征
class FPN(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        # 高分辨率分支（1x，捕捉小瘀伤）
        self.high_res = nn.Conv2d(in_channels, 32, kernel_size=1)
        # 中分辨率分支（1/2x，捕捉中等瘀伤）
        self.mid_res = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
        # 低分辨率分支（1/4x，捕捉大瘀伤）
        self.low_res = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
        # 融合卷积
        self.fuse = nn.Conv2d(32*3, 32, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.high_res(x)
        m = F.interpolate(self.mid_res(x), size=x.shape[2:], mode='bilinear')  # 上采样到原尺寸
        l = F.interpolate(self.low_res(x), size=x.shape[2:], mode='bilinear')
        fused = torch.cat([h, m, l], dim=1)
        return self.fuse(fused)  # 融合多尺度特征

# ---- 7. PSCA：适配256维通道 ----
class PSCA(nn.Module):
    def __init__(self, d_model=256, d_ff=512):  # d_ff=2*d_model，避免过度膨胀
        super().__init__()
        self.w_1 = nn.Conv3d(d_model, d_ff, 1, bias=False)  # 1x1x1卷积降参
        self.w_2 = nn.Conv3d(d_ff, d_model, 1, bias=False)

    def forward(self, x, attention):
        x = attention * x
        x = self.w_1(x)
        x = F.silu(x) # GELU → Swish（silu是PyTorch对Swish的实现）
        x = self.w_2(x)
        return x

# 新增动态注意力生成器
class DynamicAttentionGenerator(nn.Module):
    def __init__(self, d_model=256,device=None):
        super().__init__()
        self.w_3 = nn.Conv3d(d_model, d_model, 1, bias=False,device=device)# 显式指定设备
        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        return 1 + self.w_3(x)

# ---- 8. MHRSA：适配256维通道 ----
class MHRSA(nn.Module):
    def __init__(self, channels=256, multi_head=True, ffn=True):
        super().__init__()
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if self.ffn:
            self.ffn1 = MLP(channels)
            self.ffn2 = MLP(channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        Z, F = self._conv_step(inputs)

        # 多头部处理（适配256维，拆分后每头128维）
        if inputs.size(2) == 1 and self.multi_head:
            multi_head = False
        else:
            multi_head = self.multi_head

        if multi_head:
            Z1, Z2 = Z.split(self.channels // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(reversed(Z.split(1, 2)), reversed(F.split(1, 2))))):
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)
        return y

class MLP(nn.Module):
    def __init__(self, channel=256, bias=True):
        super().__init__()
        self.w_1 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)

    def forward(self, x):
        return self.w_2(F.tanh(self.w_1(x)))

# ---- 9. TGRM Block（适配256维通道） ----
class TGRM(nn.Module):
    def __init__(self, in_channels=256, channels=256, attention=None, multi_head=True, ffn=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, channels, 3, 1, 1, bias=True)  # 3x3x3卷积，保持通道
        # 3D卷积专用初始化：Kaiming初始化适配3D核（调整增益）
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_in',nonlinearity='leaky_relu',a=0.2)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.01)  # 偏置初始化为小正数，避免初期输出为负
        self.attention = attention
        self.intra_sa = PSCA(channels, channels * 2)  # d_ff=2*channels
        self.inter_sa = MHRSA(channels, multi_head=multi_head, ffn=ffn)

    def forward(self, x, attention, reverse=False):
        x = self.conv(x)
        x = self.intra_sa(x, attention)
        x = self.inter_sa(x, reverse=reverse)
        return x

# ---- FocusNet（通道调整：512→256，与BaseNet/OverViewNet对齐） ----
'''
class FocusNet(nn.Module):
    def __init__(self, base_channels=256,  # BaseNet输出256维
                 cp_channels=256,  # OverViewNet输出256维
                 tgrm_channels=256,  # 核心调整：512→256
                 time_steps=7, spatial_reduction=4):
        super().__init__()
        self.time_steps = time_steps
        self.spatial_reduction = spatial_reduction

        # BaseNet输出调整：256→256（1x1卷积，无通道膨胀）
        self.base_adjust = nn.Sequential(
            nn.Conv2d(base_channels, tgrm_channels, kernel_size=1, padding=0),  # 替换3x3 dilation，降参
            nn.BatchNorm2d(tgrm_channels),
            nn.ReLU(inplace=True)
        )

        # 上下文先验调整：256→256（1x1卷积，无通道膨胀）
        self.cp_adjust = nn.Sequential(
            nn.Conv2d(cp_channels, tgrm_channels, kernel_size=1, padding=0),  # 替换3x3 dilation，降参
            nn.BatchNorm2d(tgrm_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力生成器（256维输入）
        self.attention_generator = nn.Sequential(
            nn.Conv2d(tgrm_channels, tgrm_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # TGRM模块（256维输入输出）
        self.tgrm1 = TGRM(
            in_channels=tgrm_channels,
            channels=tgrm_channels,
            attention=None,
            multi_head=True,
            ffn=True
        )

        # 下采样层（256→256，空间降维）
        self.downsample = nn.Sequential(
            nn.Conv2d(tgrm_channels, tgrm_channels, kernel_size=3,
                      stride=spatial_reduction, padding=1),
            nn.BatchNorm2d(tgrm_channels),
            nn.ReLU(inplace=True)
        )

        # 第二个TGRM模块（256维输入输出）
        self.tgrm2 = TGRM(
            in_channels=tgrm_channels,
            channels=tgrm_channels,
            attention=None,
            multi_head=True,
            ffn=True
        )

        # 时间序列投影：256→time_steps×256（与DEE输入对齐）
        self.time_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(tgrm_channels, time_steps * tgrm_channels),
            nn.Unflatten(1, (time_steps, tgrm_channels))  # [B, time_steps, 256]
        )

        # 输出投影：2×256→256（通道合并后回原维度）
        self.output_proj = nn.Conv2d(tgrm_channels * 2, tgrm_channels, kernel_size=1)

    def forward(self, base_out, context_prior):
        # BaseNet输出调整：256→256
        base_processed = self.base_adjust(base_out)  # [B, 256, H/2, W/2]

        # 上下文先验调整：256→256
        cp_processed = self.cp_adjust(context_prior)  # [B, 256, H/2, W/2]

        # 生成注意力图
        attention_map = self.attention_generator(cp_processed)  # [B, 256, H/2, W/2]
        attention_5d = attention_map.unsqueeze(2)  # [B, 256, 1, H/2, W/2]

        # Base特征转5D：[B, 256, 1, H/2, W/2]
        base_5d = base_processed.unsqueeze(2)

        # 第一个TGRM处理
        tgrm1_out = self.tgrm1(base_5d, attention_5d)  # [B, 256, 1, H/2, W/2]
        tgrm1_out_4d = tgrm1_out.squeeze(2)  # [B, 256, H/2, W/2]

        # 下采样：空间降维，通道保持256
        downsampled = self.downsample(tgrm1_out_4d)  # [B, 256, H/(2*s), W/(2*s)]
        cp_processed_2 = self.downsample(cp_processed)  # [B, 256, H/(2*s), W/(2*s)]

        # 生成第二个TGRM的注意力图
        attention_map2 = self.attention_generator(cp_processed_2)  # [B, 256, H/(2*s), W/(2*s)]
        attention2_5d = attention_map2.unsqueeze(2)  # [B, 256, 1, H/(2*s), W/(2*s)]

        # 合并特征：256+256→512
        combined = torch.cat([downsampled, cp_processed_2], dim=1)  # [B, 512, H/(2*s), W/(2*s)]
        # 投影回256维
        combined_adjusted = self.output_proj(combined)  # [B, 256, H/(2*s), W/(2*s)]

        # 转5D输入TGRM
        combined_5d = combined_adjusted.unsqueeze(2)  # [B, 256, 1, H/(2*s), W/(2*s)]
        tgrm2_out = self.tgrm2(combined_5d, attention2_5d)  # [B, 256, 1, H/(2*s), W/(2*s)]

        # 转4D后投影到时间序列
        output_4d = tgrm2_out.squeeze(2)  # [B, 256, H/(2*s), W/(2*s)]
        output_3d = self.time_projection(output_4d)  # [B, time_steps, 256]

        return output_3d
'''
class FocusNet(nn.Module):
    """
    瘀伤任务适配的FocusNet：仅保留特征融合功能，
    移除时间序列模块与分类头，输出融合后的空间特征
    """
    def __init__(self,
                 base_channels=32,  # 与BaseNet输出通道(fusion_dim)一致
                 cp_channels=32,    # 与OverViewNet输出通道(embed_dim)一致
                 hidden_channels=32):  # 内部融合特征的通道数（保持与输入一致）
        super().__init__()

        # 1. 特征维度对齐：统一BaseNet与OverViewNet的特征通道
        self.base_adjust = nn.Sequential(
            nn.Conv2d(base_channels, hidden_channels, kernel_size=1),  # 1x1卷积轻量对齐
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.cp_adjust = nn.Sequential(
            nn.Conv2d(cp_channels, hidden_channels, kernel_size=1),  # 上下文特征通道对齐
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 注意力生成器：基于上下文特征强化瘀伤关键区域（如边缘、光谱异常区）
        self.attention_generator = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid()  # 输出0-1权重图，突出有效特征
        )

        # 3. 第一级TGRM：融合Base特征与注意力，强化光谱-空间关联
        self.tgrm1 = TGRM(
            in_channels=hidden_channels,
            channels=hidden_channels,
            attention=None,
            multi_head=True,
            ffn=True
        )

        # 4. 空间下采样：聚焦核心特征，降低后续计算量（保留瘀伤细节）
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      stride=2, padding=1),  # 空间尺寸减半（H→H/2, W→W/2），通道不变
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 5. 第二级TGRM：进一步融合下采样后的Base特征与上下文特征
        self.tgrm2 = TGRM(
            in_channels=hidden_channels,
            channels=hidden_channels,
            attention=None,
            multi_head=True,
            ffn=True
        )

        # 6. 特征输出调整：确保输出特征格式统一（无通道跳跃）
        self.output_adjust = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels)  # 标准化特征，便于后续模块使用
        )

    def forward(self, base_out, context_prior):
        """
        输入：
            base_out: BaseNet输出的融合特征 → shape [B, 32, H, W]
            context_prior: OverViewNet输出的上下文特征 → shape [B, 32, H, W]
        输出：
            fused_feat: 最终融合特征 → shape [B, 32, H/2, W/2]（下采样后，通道稳定）
        """
        # 设备对齐：避免跨设备数据类型错误
        device = next(self.parameters()).device
        base_out = base_out.to(device)
        context_prior = context_prior.to(device)

        # 步骤1：统一BaseNet与OverViewNet的特征维度
        base_processed = self.base_adjust(base_out)  # [B, 32, H, W]
        cp_processed = self.cp_adjust(context_prior)  # [B, 32, H, W]

        # 步骤2：生成注意力图（基于上下文特征，突出瘀伤区域）
        attention_map = self.attention_generator(cp_processed)  # [B, 32, H, W]
        attention_5d = attention_map.unsqueeze(2)  # 适配TGRM的5D输入 → [B, 32, 1, H, W]

        # 步骤3：第一级TGRM融合（Base特征+注意力）
        base_5d = base_processed.unsqueeze(2)  # Base特征转为5D → [B, 32, 1, H, W]
        tgrm1_out = self.tgrm1(base_5d, attention_5d)  # TGRM处理 → [B, 32, 1, H, W]
        tgrm1_out_4d = tgrm1_out.squeeze(2)  # 转回4D特征 → [B, 32, H, W]

        # 步骤5：第二级TGRM融合（下采样后的双分支特征）
        # 双分支特征拼接后压缩通道（适配TGRM输入维度）
        combined = torch.cat([tgrm1_out_4d, cp_processed], dim=1)  # [B, 64, H/2, W/2]
        combined_adjusted = nn.Conv2d(64, 32, kernel_size=1).to(device)(combined)  # [B, 32, H/2, W/2]
        # 转为5D输入TGRM
        combined_5d = combined_adjusted.unsqueeze(2)  # [B, 32, 1, H/2, W/2]
        attention2_5d = self.attention_generator(cp_processed).unsqueeze(2)  # 新注意力图 → [B, 32, 1, H/2, W/2]
        tgrm2_out = self.tgrm2(combined_5d, attention2_5d)  # TGRM融合 → [B, 32, 1, H/2, W/2]

        # 步骤6：调整输出特征（标准化，确保格式统一）
        fused_feat = self.output_adjust(tgrm2_out.squeeze(2))  # [B, 32, H/2, W/2]

        return fused_feat

# -----------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 生成空间注意力权重（突出瘀伤区域）
        attn_weights = self.attn(x)  # [B, C, H, W]
        # 加权求和（替代全局平均）
        weighted_sum = (x * attn_weights).sum(dim=(2, 3)) / (attn_weights.sum(dim=(2, 3)) + 1e-6)
        return weighted_sum

class Classifier(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.attn_pool = AttentionPooling(in_channels)  # 注意力加权池化
        self.fc = nn.Linear(in_channels, 2)  # 二分类

    def forward(self, x):
        feat = self.attn_pool(x)  # [B, C]（聚焦瘀伤区域特征）
        return self.fc(feat)
# -------------------------------------------------------------
# ---- 12. 主网络 Bruise-STFNet（通道统一为256核心） ----
class BruiseSTFNet(nn.Module):
    """
    瘀伤二分类主网络：整合BaseNet（特征提取）、OverViewNet（上下文增强）、
    FocusNet（特征融合）与分类头，实现从MSI+RGB输入到二分类结果的端到端处理
    """
    def __init__(self,
                 msi_channels=24,        # MSI多光谱通道数
                 rgb_channels=3,         # RGB图像通道数
                 num_classes=2,          # 二分类（有瘀伤/无瘀伤）
                 hidden_dim=32,          # BaseNet内部特征维度
                 fusion_dim=32,          # 特征融合维度（贯穿全网络）
                 attn_heads=2,           # 注意力头数
                 spatial_reduction=2):   # FocusNet下采样率
        super().__init__()

        # 1. 基础特征提取网络（BaseNet）：MSI+RGB特征初步融合
        self.base_net = BaseNet(
            msi_channels=msi_channels,
            rgb_channels=rgb_channels,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            attn_heads=attn_heads
        )

        # 2. 上下文增强网络（OverViewNet）：强化全局上下文特征
        self.overview_net = OverViewNet(
            in_chans=fusion_dim,      # 输入与BaseNet输出维度一致
            embed_dim=fusion_dim      # 输出保持fusion_dim，便于后续融合
        )
        self.fpn = FPN(in_channels=32)# 在OverViewNet之后
        # 3. 特征融合网络（FocusNet）：融合Base与上下文特征，输出高级特征
        self.focus_net = FocusNet(
            base_channels=fusion_dim,
            cp_channels=fusion_dim,
            hidden_channels=fusion_dim
        )

        # 4. 分类头：从融合特征输出二分类结果
        self.classifier = Classifier(in_channels=32)

        # 初始化权重（确保全网络参数分布一致）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """全网络统一权重初始化：适配卷积、线性层与归一化层"""
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            # Kaiming初始化：适配ReLU/GELU等激活函数，避免梯度消失
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0，减少初始偏差
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 归一化层权重=1，偏置=0，确保初始特征分布稳定
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, msi_data, rgb_img):
        """
        前向传播：输入MSI+RGB图像，输出二分类结果
        Args:
            msi_data: MSI多光谱数据，shape [B, msi_channels, H, W]
            rgb_img: RGB图像数据，shape [B, rgb_channels, H, W]
        Returns:
            logits: 二分类预测结果，shape [B, num_classes]（0=无瘀伤，1=有瘀伤）
            features: 中间特征字典（可选，用于调试或特征可视化）
        """
        # 1. BaseNet提取基础融合特征
        base_feat, aux_feat = self.base_net(msi_data, rgb_img)  # [B, fusion_dim, H, W]

        # 2. OverViewNet增强上下文特征
        overview_feat = self.overview_net(base_feat)  # [B, fusion_dim, H, W]
        multi_scale_feat = self.fpn(overview_feat)  # 多尺度容二虎
        # 3. FocusNet融合Base与上下文特征
        focus_feat = self.focus_net(base_feat, multi_scale_feat)  # [B, fusion_dim, H/2, W/2]

        # 4. 分类头输出二分类结果
        logits = self.classifier(focus_feat)  # [B, num_classes]

        # 可选：返回中间特征（便于分析网络各阶段特征）
        features = {
            'base': base_feat,
            'overview': overview_feat,
            'focus': focus_feat,
            'aux_feat': aux_feat
        }

        return logits, features