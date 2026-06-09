"""
加载 BruiseSTFNet_B.pth 模型，用 SHAP 分析特征重要性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ==================== 配置 ====================
CSV_PATH = r'/teams/TXLJ_1745741277/xzk/visualize/results/features.csv'
MODEL_PATH = r'/teams/TXLJ_1745741277/xzk/visualize/BruiseSTFNet_B.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =============================================

# ==================== 模型定义（与你训练时完全一致）====================
# 这里需要你之前提供的完整模型定义
# 由于模型代码很长，我假设你已经把它放在一个单独的文件里
# 你可以用 from model import BruiseSTFNet 导入

# 如果还没有导入，请把之前的模型定义代码粘贴在这里
# 注意：这里需要完整的 BruiseSTFNet 类定义
# 为了简洁，我假设你已经有了模型定义文件
try:
    from BruiseSTFNet_B import BruiseSTFNet
except ImportError:
    print("请将 BruiseSTFNet 模型定义代码放在这里或导入")
    # 如果你还没有模型定义文件，需要把之前完整的模型定义代码粘贴在这里
    raise ImportError("需要 BruiseSTFNet 模型定义")

# ==================== 1. 读取数据 ====================
print("=" * 60)
print("Step 1: 读取特征数据")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"数据形状: {df.shape}")

# ==================== 2. 构建标签 ====================
print("\n" + "=" * 60)
print("Step 2: 根据文件名构建标签")
print("=" * 60)


def get_label(filename):
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]
    parts = basename.split('-')
    if len(parts) < 4:
        return -1
    fruit_type = parts[0]
    try:
        time_str = parts[1]
        if time_str == '37':
            return -1
        drop_time = int(time_str)
    except (IndexError, ValueError):
        return -1
    if fruit_type == "jjk":
        return 0
    elif fruit_type != "ys" and drop_time == 0:
        return 0
    else:
        return 1


df['label'] = df['filename'].apply(get_label)
df = df[df['label'] != -1]

print(f"标签分布:")
print(f"  损伤(1): {(df['label'] == 1).sum()} 个")
print(f"  健康(0): {(df['label'] == 0).sum()} 个")

# ==================== 3. 准备特征 ====================
print("\n" + "=" * 60)
print("Step 3: 准备特征和目标变量")
print("=" * 60)

feature_cols = ['R', 'G', 'B'] + [f'band_{i}' for i in range(24)]
X = df[feature_cols].values
y = df['label'].values

print(f"特征矩阵形状: {X.shape}")

# ==================== 4. 划分数据集 ====================
print("\n" + "=" * 60)
print("Step 4: 划分训练集和测试集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

# ==================== 5. 加载 PyTorch 模型 ====================
print("\n" + "=" * 60)
print("Step 5: 加载 BruiseSTFNet 模型")
print("=" * 60)

# 创建模型实例
model = BruiseSTFNet(msi_channels=24, rgb_channels=3)

# 加载权重
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 处理不同的保存格式
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint, strict=False)
else:
    model = checkpoint

model.to(DEVICE)
model.eval()
print("模型加载成功")

# ==================== 6. 模型评估 ====================
print("\n" + "=" * 60)
print("Step 6: 模型评估")
print("=" * 60)


def predict_with_model(features):
    """输入特征矩阵，返回正类概率 —— 修复 1x1 报错"""
    with torch.no_grad():
        # 1. 转 tensor + 升维 [B, 27] → [B, 27, 1, 1]
        x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, 27, 1, 1]

        # 2. 关键修复：上采样到 8x8，让模型能正常运行
        x = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)

        # 3. 拆分多光谱 + RGB
        msi_data = x[:, :24, :, :]  # [B, 24, 8, 8]
        rgb_img = x[:, 24:27, :, :]  # [B, 3, 8, 8]

        # 4. 模型推理
        logits, _ = model(msi_data, rgb_img)
        probs = torch.softmax(logits, dim=1)

    return probs[:, 1].cpu().numpy()


def predict_with_model_detailed(features):
    """详细预测函数，返回类别和概率"""
    with torch.no_grad():
        # 1. 转 tensor + 升维 [B, 27] → [B, 27, 1, 1]
        x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, 27, 1, 1]

        # 2. 关键修复：上采样到 8x8，让模型能正常运行
        x = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)

        # 3. 拆分多光谱 + RGB
        msi_data = x[:, :24, :, :]  # [B, 24, 8, 8]
        rgb_img = x[:, 24:27, :, :]  # [B, 3, 8, 8]

        # 4. 模型推理
        logits, _ = model(msi_data, rgb_img)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()


# 分析数据分布
print("\n" + "=" * 60)
print("分析数据分布")
print("=" * 60)
print(f"训练集健康样本比例: {(y_train == 0).sum() / len(y_train):.4f}")
print(f"测试集健康样本比例: {(y_test == 0).sum() / len(y_test):.4f}")
print(f"总体健康样本比例: {(y == 0).sum() / len(y):.4f}")

# 检查特征值范围
print("\n" + "=" * 60)
print("检查特征值范围")
print("=" * 60)
print(f"特征值最小值: {X.min():.4f}")
print(f"特征值最大值: {X.max():.4f}")
print(f"特征值平均值: {X.mean():.4f}")
print(f"特征值标准差: {X.std():.4f}")

# 评估测试集
y_pred_proba = predict_with_model(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['健康(0)', '损伤(1)']))

# 分析健康样本的预测情况
print("\n" + "=" * 60)
print("分析健康样本预测情况")
print("=" * 60)
healthy_indices = np.where(y_test == 0)[0]
unhealthy_indices = np.where(y_test == 1)[0]

print(f"健康样本数量: {len(healthy_indices)}")
print(f"损伤样本数量: {len(unhealthy_indices)}")

if len(healthy_indices) > 0:
    healthy_probs = y_pred_proba[healthy_indices]
    print(f"健康样本的预测概率范围: {healthy_probs.min():.4f} - {healthy_probs.max():.4f}")
    print(f"健康样本的平均预测概率: {healthy_probs.mean():.4f}")
    print(f"健康样本被预测为健康的数量: {(healthy_probs < 0.5).sum()}")
    print(f"健康样本被预测为损伤的数量: {(healthy_probs >= 0.5).sum()}")
    
    # 显示前10个健康样本的预测情况
    print("\n前10个健康样本的预测情况:")
    for i, idx in enumerate(healthy_indices[:10]):
        prob = y_pred_proba[idx]
        print(f"健康样本 {i+1}: 预测损伤概率 = {prob:.4f}, 预测类别 = {1 if prob >= 0.5 else 0}")

if len(unhealthy_indices) > 0:
    unhealthy_probs = y_pred_proba[unhealthy_indices]
    print(f"\n损伤样本的预测概率范围: {unhealthy_probs.min():.4f} - {unhealthy_probs.max():.4f}")
    print(f"损伤样本的平均预测概率: {unhealthy_probs.mean():.4f}")

# 检查模型输出的详细概率
detailed_probs = predict_with_model_detailed(X_test[:10])
print("\n前10个样本的详细预测概率:")
for i, prob in enumerate(detailed_probs):
    print(f"样本 {i+1}: 健康概率 = {prob[0]:.4f}, 损伤概率 = {prob[1]:.4f}, 真实标签 = {y_test[i]}, 预测标签 = {1 if prob[1] >= 0.5 else 0}")

# 尝试不同的阈值
print("\n" + "=" * 60)
print("尝试不同的预测阈值")
print("=" * 60)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    print(f"\n阈值 = {threshold}:")
    print(f"健康样本被正确预测的数量: {(y_pred_threshold[healthy_indices] == 0).sum()}")
    print(f"损伤样本被正确预测的数量: {(y_pred_threshold[unhealthy_indices] == 1).sum()}")
    print(f"总体准确率: {accuracy_score(y_test, y_pred_threshold):.4f}")

# 分析模型可能的问题
print("\n" + "=" * 60)
print("分析模型可能的问题")
print("=" * 60)
print("1. 类别不平衡问题:")
print(f"   - 训练集健康样本: {(y_train == 0).sum()}, 损伤样本: {(y_train == 1).sum()}")
print(f"   - 测试集健康样本: {(y_test == 0).sum()}, 损伤样本: {(y_test == 1).sum()}")
print("2. 模型预测偏向:")
print(f"   - 健康样本平均预测损伤概率: {healthy_probs.mean():.4f} (如果接近1，说明模型严重偏向损伤类别)")
print(f"   - 损伤样本平均预测损伤概率: {unhealthy_probs.mean():.4f}")
print("3. 可能的原因:")
print("   - 训练数据中损伤样本远多于健康样本，导致模型过度拟合到损伤类别")
print("   - 健康样本的特征与损伤样本差异不明显，模型无法有效区分")
print("   - 模型权重可能存在问题，导致对健康样本的预测始终偏向损伤类别")
print("   - 特征预处理不当，健康样本的特征值范围可能不适合模型")

# ==================== 7. 计算 SHAP 值 ====================
print("\n" + "=" * 60)
print("Step 7: 计算 SHAP 值（这可能需要几分钟）")
print("=" * 60)

# 注意：PyTorch 模型需要用 KernelExplainer，比 TreeExplainer 慢
# 如果样本太多，可以只用一部分计算 SHAP
n_samples_for_shap = min(len(X), 200)  # 用 200 个样本计算 SHAP
X_sample = X[:n_samples_for_shap]

# 选择背景样本（用训练集的一部分）
background = X_train[:100]

print(f"使用 {n_samples_for_shap} 个样本计算 SHAP 值")
print(f"背景样本数: {len(background)}")
print("正在计算 SHAP 值，可能需要几分钟...")

# 创建 KernelExplainer
explainer = shap.KernelExplainer(predict_with_model, background)

# 计算 SHAP 值
shap_values = explainer.shap_values(X_sample, nsamples=100)  # nsamples 控制计算量

print(f"SHAP 值形状: {shap_values.shape}")

# ==================== 8. 特征重要性排序 ====================
print("\n" + "=" * 60)
print("Step 8: 特征重要性排序")
print("=" * 60)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print("\n所有特征重要性排序（前15）:")
print(feature_importance.head(15).to_string(index=False))

# ==================== 9. 选出重要波段 ====================
print("\n" + "=" * 60)
print("Step 9: 选出重要的多光谱波段")
print("=" * 60)

msi_bands = [f'band_{i}' for i in range(24)]
msi_importance = feature_importance[feature_importance['feature'].isin(msi_bands)]
top_msi = msi_importance.head(8)

print("\n前8个最重要的多光谱波段:")
print(top_msi.to_string(index=False))

# ==================== 10. 绘制双轴 SHAP 可视化 ====================
print("\n" + "=" * 60)
print("Step 10: 绘制双轴 SHAP 可视化图")
print("=" * 60)

# 选择要显示的特征：RGB + 前8个重要波段
selected_features = ['R', 'G', 'B'] + top_msi['feature'].tolist()
print(f"\n选定的特征: {selected_features}")

# 获取选定特征的索引和数据
selected_indices = [feature_cols.index(f) for f in selected_features]
X_selected = X_sample[:, selected_indices]
shap_selected = shap_values[:, selected_indices]

# 计算平均绝对 SHAP 值，用于排序
mean_abs_selected = np.abs(shap_selected).mean(axis=0)

# 按平均 SHAP 值从大到小排序
sort_idx = np.argsort(mean_abs_selected)[::-1]
sorted_features = [selected_features[i] for i in sort_idx]
sorted_shap = shap_selected[:, sort_idx]
sorted_X = X_selected[:, sort_idx]
sorted_mean_abs = mean_abs_selected[sort_idx]

print(f"\n排序后的特征顺序（从大到小）: {sorted_features}")

# 动态计算图形高度
n_features = len(sorted_features)
fig_height = max(6, n_features * 0.45)
fig_width = 12

fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# 左侧 y 轴位置 - 反转顺序，使得最重要的特征显示在最上方
y_pos = np.arange(n_features)[::-1]

# ========== 左侧：蜂巢图 ==========
print("绘制蜂巢图...")

cmap = plt.cm.RdBu_r
all_feat_vals = sorted_X.flatten()
vmin = np.percentile(all_feat_vals, 2)
vmax = np.percentile(all_feat_vals, 98)
if vmin == vmax:
    vmin = all_feat_vals.min()
    vmax = all_feat_vals.max()
norm = plt.Normalize(vmin, vmax)

for i, feat_name in enumerate(sorted_features):
    shap_vals = sorted_shap[:, i]
    feat_vals = sorted_X[:, i]

    jitter = np.random.normal(0, 0.08, len(shap_vals))
    # 反转y轴位置，使得最重要的特征显示在最上方
    y_jitter = (n_features - 1 - i) + jitter

    scatter = ax1.scatter(shap_vals, y_jitter, c=feat_vals, cmap=cmap,
                          norm=norm, alpha=0.7, s=8, edgecolors='none', rasterized=True)

    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# 设置左侧 y 轴
ax1.set_yticks(np.arange(n_features))
ax1.set_yticklabels(sorted_features, fontsize=10)
ax1.set_ylabel('Features', fontsize=11)
ax1.set_xlabel('SHAP value (impact on damage prediction)', fontsize=11)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax1, fraction=0.05, pad=0.02)
cbar.set_label('Feature value', fontsize=9)
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels(['Low', 'High'])

# ========== 右侧：条形图 ==========
ax2 = ax1.twiny()
# 反转条形图的y轴位置
bars = ax2.barh(np.arange(n_features)[::-1], sorted_mean_abs, height=0.6, alpha=0.25, color='steelblue')
ax2.set_xlabel('Mean |SHAP| (feature importance)', fontsize=11)
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()

max_val = sorted_mean_abs.max()
for i, (bar, val) in enumerate(zip(bars, sorted_mean_abs)):
    # 反转文本的y轴位置
    ax2.text(val + 0.02 * max_val, (n_features - 1 - i), f'{val:.4f}',
             va='center', fontsize=8, alpha=0.6)

ax1.set_ylim(-0.8, n_features - 0.2)

# 调整subplots_adjust参数，为标题和标签留出足够空间
plt.subplots_adjust(left=0.2, right=0.85, top=0.85, bottom=0.08, wspace=0.3)

# 设置标题，确保在最顶部
plt.suptitle('Feature Importance Analysis (BruiseSTFNet)', fontsize=12, y=0.98)

# 调整ax2的xlabel位置，确保在标题下方
ax2.set_xlabel('Mean |SHAP| (feature importance)', fontsize=11, labelpad=15)

# 保存图片
plt.savefig(r'/teams/TXLJ_1745741277/xzk/visualize/results/dual_shap_pytorch.png', dpi=300, bbox_inches='tight')
plt.savefig(r'/teams/TXLJ_1745741277/xzk/visualize/results/dual_shap_pytorch.pdf', bbox_inches='tight')
plt.show()

print("\n双轴 SHAP 图已保存为 /teams/TXLJ_1745741277/xzk/visualize/results/dual_shap_pytorch.png 和 .pdf")

# ==================== 11. 保存结果 ====================
print("\n" + "=" * 60)
print("Step 11: 保存分析结果")
print("=" * 60)

feature_importance.to_csv(r'/teams/TXLJ_1745741277/xzk/visualize/results/feature_importance_pytorch.csv', index=False)
print("特征重要性已保存到: /teams/TXLJ_1745741277/xzk/visualize/results/feature_importance_pytorch.csv")

top_msi.to_csv(r'/teams/TXLJ_1745741277/xzk/visualize/results/top_bands_pytorch.csv', index=False)
print("重要波段列表已保存到: /teams/TXLJ_1745741277/xzk/visualize/results/top_bands_pytorch.csv")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)