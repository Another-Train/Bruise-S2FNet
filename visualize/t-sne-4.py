import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 导入我们的模型
from GMA_changed import BruiseSTFNet

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== 配置参数（集中管理，便于调整） ==========
class Config:
    DATA_ROOT = "D://OverLoCK//data"  # 根目录：包含train/test/val
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01', '6', '12', '18', '24', '30', '36']  # 8个时间点（四分类_BaseNet）
    TIME_TO_LABEL = {
        '0': 0,  # 0h → 标签0（健康）
        '01': 0,  # 0.1h → 标签0
        '6': 1,  # 6h → 标签1（瘀伤早期）
        '12': 1,  # 12h → 标签1（瘀伤早期）
        '18': 2,  # 18h → 标签2（瘀伤中期）
        '24': 2,  # 24h → 标签2（瘀伤中期）
        '30': 3,  # 30h → 标签3（瘀伤后期）
        '36': 3  # 36h → 标签3（瘀伤后期）
    }
    IMG_SIZE = (64,64)
    BATCH_SIZE = 2
    NUM_CLASSES = 4  # 四分类_BaseNet
    MSI_CHANNELS = 24  # 根据实际高光谱数据通道数设置（关键！）
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 随机种子
    SEED = 42

# 固定随机种子（确保可复现）
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# ========== 数据集类（优化数据加载逻辑） ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, subset="train", transform=None):
        self.subset = subset
        self.transform = transform
        self.samples = self._collect_samples()  # (msi_path, rgb_path, label)
        if not self.samples:
            raise ValueError(f"{subset}集未找到有效样本，请检查数据路径或格式")

    def _collect_samples(self):
        samples = []
        subset_dir = os.path.join(Config.DATA_ROOT, self.subset)
        print(f"当前搜索的子集目录：{subset_dir}")  # 打印当前处理的子集目录（如data/train）
        for time in Config.TIME_POINTS:
            label = Config.TIME_TO_LABEL[time]
            msi_time_dir = os.path.join(subset_dir, "MSI", str(time))
            rgb_time_dir = os.path.join(subset_dir, "RGB", str(time))
            print(f"\n检查时间点{time}：")
            print(f"MSI目录：{msi_time_dir}（是否存在：{os.path.isdir(msi_time_dir)}）")
            print(f"RGB目录：{rgb_time_dir}（是否存在：{os.path.isdir(rgb_time_dir)}）")
            # 跳过不存在的目录
            if not os.path.isdir(msi_time_dir) or not os.path.isdir(rgb_time_dir):
                continue
            # 获取MSI的.hdr文件前缀
            msi_hdr_files = glob.glob(os.path.join(msi_time_dir, "*.hdr"))
            print(f"MSI目录下找到{len(msi_hdr_files)}个.hdr文件：{[os.path.basename(f) for f in msi_hdr_files]}")
            msi_prefixes = {os.path.splitext(os.path.basename(f))[0] for f in msi_hdr_files}
            # 获取RGB的.png文件前缀
            rgb_png_files = glob.glob(os.path.join(rgb_time_dir, "*.png"))
            print(f"RGB目录下找到{len(rgb_png_files)}个.png文件：{[os.path.basename(f) for f in rgb_png_files]}")
            rgb_prefixes = {os.path.splitext(os.path.basename(f))[0] for f in rgb_png_files}
            # 匹配共同前缀
            common_prefixes = msi_prefixes & rgb_prefixes
            print(f"匹配到{len(common_prefixes)}个共同前缀样本")
            for prefix in common_prefixes:
                # 将jjk数据标签改为健康标签0
                if "jjk" in prefix.lower():
                    # jjk类别强制标记为健康标签0
                    final_label = 0
                    print(f"检测到jjk类别样本: {prefix}，强制标记为健康标签0")
                else:
                    # 非jjk类别使用原始时间标签
                    final_label = label
                msi_path = os.path.join(msi_time_dir, f"{prefix}.hdr")  # 加上MSI文件所在目录
                rgb_path = os.path.join(rgb_time_dir, f"{prefix}.png")  # 加上RGB文件所在目录
                samples.append((msi_path, rgb_path, final_label))
        print(f"\n最终收集到{len(samples)}个样本")
        return samples

    def _read_msi(self, msi_hdr_path):
        """使用spectral库读取高光谱数据（适配.float格式数据文件，修复参数错误）"""
        try:
            # 1. 生成带.float扩展名的数据文件路径
            msi_data_path = f"{os.path.splitext(msi_hdr_path)[0]}.float"
            # 2. 直接调用envi.open，并通过image参数指定数据文件（关键修复）
            from spectral.io import envi  # 显式导入envi模块
            img = envi.open(msi_hdr_path, image=msi_data_path)  # 这里用envi.open而非open_image
            msi_data = img.load()  # 读取数据， shape: (H, W, C)
            msi_data = np.transpose(msi_data, (2, 0, 1))  # (H,W,C) → (C,H,W)
            # 归一化（避免除以0）
            msi_min, msi_max = msi_data.min(), msi_data.max()
            if msi_max - msi_min < 1e-8:
                msi_data = np.zeros_like(msi_data)
            else:
                msi_data = (msi_data - msi_min) / (msi_max - msi_min)
            # 调整尺寸并转换为张量
            msi_tensor = torch.tensor(msi_data, dtype=torch.float32)
            msi_resized = F.interpolate(
                msi_tensor.unsqueeze(0),  # 加batch维度
                size=Config.IMG_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)  # 去掉batch维度
            return msi_resized
        except Exception as e:
            raise RuntimeError(
                f"使用spectral读取MSI文件失败：{msi_hdr_path}\n"
                f"请确保.hdr文件和对应.float数据文件在同一目录，且文件名前缀一致（如xxx.hdr和xxx.float）\n"
                f"错误详情：{str(e)}"
            )

    def _read_rgb(self, rgb_path):
        """读取RGB图像（优化：整合transform到读取流程）"""
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            if self.transform:
                rgb_img = self.transform(rgb_img)
            # 调整尺寸（transform已经包含了ToTensor，所以rgb_img已经是tensor）
            rgb_resized = F.interpolate(
                rgb_img.unsqueeze(0),
                size=Config.IMG_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            return rgb_resized
        except Exception as e:
            raise RuntimeError(f"读取RGB文件失败：{rgb_path}，错误：{str(e)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msi_path, rgb_path, label = self.samples[idx]
        msi = self._read_msi(msi_path)
        rgb = self._read_rgb(rgb_path)
        return msi, rgb, torch.tensor(label, dtype=torch.long)

# ==========================================
# 1. 特征提取函数
# ==========================================
def get_features_and_labels(model, loader, device, mode='fused'):
    model.eval()
    features_list = []
    labels_list = []

    print(f"正在提取阶段特征: {mode}...")
    with torch.no_grad():
        for msi_data, rgb_img, labels in loader:
            msi_data = msi_data.to(device)
            rgb_img = rgb_img.to(device)

            # 前向传播获取特征
            logits, aux_features = model(msi_data, rgb_img)

            if mode == 'rgb':
                # 使用SFE模块的空间增强特征
                if 'SFE' in aux_features:
                    sfe_feat = aux_features['SFE']
                    feat = F.adaptive_avg_pool2d(sfe_feat, 1).flatten(1)
                else:
                    # 如果没有SFE特征，使用fused特征
                    fused_feat = aux_features['base_feat']
                    feat = F.adaptive_avg_pool2d(fused_feat, 1).flatten(1)

            elif mode == 'hsi':
                # 使用GMA模块的光谱特征
                if 'GMA' in aux_features:
                    gma_feat = aux_features['GMA']
                    feat = F.adaptive_avg_pool2d(gma_feat, 1).flatten(1)
                else:
                    # 如果没有GMA特征，使用fused特征
                    fused_feat = aux_features['base_feat']
                    feat = F.adaptive_avg_pool2d(fused_feat, 1).flatten(1)

            elif mode == 'fused':
                # 使用融合后的特征
                fused_feat = aux_features['base_feat']
                feat = F.adaptive_avg_pool2d(fused_feat, 1).flatten(1)

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


# ==========================================
# 2. 运行可视化主函数
# ==========================================
def run_visualize():
    device = Config.DEVICE
    # 模型路径
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model_4.pth')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'tsne_results')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = BruiseSTFNet(
        msi_channels=Config.MSI_CHANNELS,  # MSI通道数为24
        rgb_channels=Config.RGB_CHANNELS,    # RGB通道数为3
        num_classes=Config.NUM_CLASSES      # 四分类
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"模型加载成功: {MODEL_PATH}")

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和加载器
    dataset = TimeSeriesDataset(subset="test", transform=transform)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 特征提取模式
    modes = ['rgb', 'hsi', 'fused']
    titles = ['(a) RGB Features', '(b) HSI Features', '(c) Fused Features']
    all_tsne_results = []

    # 提取特征并进行t-SNE降维
    for m in modes:
        feats, lbls = get_features_and_labels(model, loader, device, mode=m)
        print(f"正在计算 {m} 模式的 t-SNE 降维...")

        # 增加 max_iter 让计算更充分
        tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30, max_iter=1500)
        low_dim_embs = tsne.fit_transform(feats)
        all_tsne_results.append((low_dim_embs, lbls))

    # 设置绘图风格和中文字体
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制t-SNE结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 四分类的颜色
    colors = ['#8E7FB8', '#A2C9AE', '#F6B352', '#E88471']
    # 四分类标签映射
    labels_map = {0: '健康', 1: '瘀伤早期', 2: '瘀伤中期', 3: '瘀伤后期'}

    for i in range(3):
        embs, lbls = all_tsne_results[i]
        df = pd.DataFrame({
            'Dimension 1': embs[:, 0],
            'Dimension 2': embs[:, 1],
            'Class': [labels_map[l] for l in lbls]
        })

        # 绘制散点图
        sns.scatterplot(data=df, x='Dimension 1', y='Dimension 2', hue='Class',
                        palette={labels_map[0]: colors[0], labels_map[1]: colors[1],
                                 labels_map[2]: colors[2], labels_map[3]: colors[3]},
                        ax=axes[i], s=60, alpha=0.85, edgecolors='white', linewidth=0.5)

        axes[i].set_title(titles[i], fontsize=16, fontweight='bold', pad=20)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        if i < 2:
            if axes[i].get_legend():
                axes[i].get_legend().remove()
        else:
            axes[i].legend(title='类别', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'tsne_gma_net.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可视化结果已保存至: {save_path}")


if __name__ == "__main__":
    run_visualize()
