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
from BruiseSTFNet_B import BruiseSTFNet

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== 配置参数（集中管理，便于调整） ==========
class Config:
    DATA_DIR = "D://DATA//gtdata"  # 二分类数据根目录
    IMG_SIZE = (32, 32)  # 二分类使用的图像尺寸
    BATCH_SIZE = 4
    NUM_CLASSES = 2  # 二分类：健康(0)和瘀伤(1)
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

# ========== 数据解析与处理 ==========
def parse_filename(filename):
    import os
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]
    parts = basename.split('-')
    if len(parts) < 4:
        return None
    fruit_type = parts[0]
    try:
        time_str = parts[1]
        if time_str == '37':
            return None
        drop_time = int(time_str)
    except (IndexError, ValueError):
        return None
    if fruit_type == "jjk":
        bruise_label = 0
    elif fruit_type != "ys" and drop_time == 0:
        bruise_label = 0
    else:
        bruise_label = 1
    return {"bruise_label": bruise_label}


def find_msi_data_file(hdr_path):
    base_path = hdr_path.replace('.hdr', '')
    extensions = ['.float', '.dat', '.img', '.bin', '.raw', '']
    for ext in extensions:
        test_path = base_path + ext
        if os.path.exists(test_path):
            return test_path
    hdr_dir = os.path.dirname(hdr_path)
    hdr_basename = os.path.basename(hdr_path).replace('.hdr', '')
    for file in os.listdir(hdr_dir):
        file_base = file.split('.')[0]
        if file_base == hdr_basename and not file.endswith('.hdr'):
            return os.path.join(hdr_dir, file)
    return None


def load_rgb_image(path, img_size=Config.IMG_SIZE):
    try:
        from PIL import Image
        rgb_img = Image.open(path).convert("RGB")
        # 转换为张量并调整尺寸
        rgb_tensor = transforms.ToTensor()(rgb_img)
        rgb_resized = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=img_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        # 归一化
        rgb_resized = (rgb_resized - rgb_resized.min()) / (rgb_resized.max() - rgb_resized.min() + 1e-8)
        return rgb_resized
    except Exception as e:
        print(f"读取RGB文件失败：{path}，错误：{str(e)}")
        return torch.zeros(3, *img_size)


def load_msi_image(path, img_size=Config.IMG_SIZE, target_channels=None):
    try:
        data_file = find_msi_data_file(path)
        if data_file is None:
            return torch.zeros(target_channels, *img_size) if target_channels else torch.zeros(10, *img_size)
        from spectral.io import envi
        msi_data = envi.open(path, data_file)
        img = msi_data.load()
        img = torch.tensor(np.array(img), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = F.interpolate(img.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False)
        img = img.squeeze(0)
        img = img / (img.max() + 1e-8)
        if target_channels is not None:
            c = img.shape[0]
            if c < target_channels:
                pad = torch.zeros(target_channels - c, *img_size)
                img = torch.cat([img, pad], dim=0)
            elif c > target_channels:
                img = img[:target_channels]
        return img
    except Exception as e:
        print(f"读取MSI文件失败：{path}，错误：{str(e)}")
        return torch.zeros(target_channels, *img_size) if target_channels else torch.zeros(10, *img_size)


def collect_data_paths():
    rgb_paths = []
    msi_paths = []
    labels = []

    rgb_dir = os.path.join(Config.DATA_DIR, "rgb_cropped")
    msi_dir = os.path.join(Config.DATA_DIR, "hyperspectral_cropped")

    if not os.path.exists(rgb_dir):
        print(f"警告: RGB目录 {rgb_dir} 不存在")
        return rgb_paths, msi_paths, labels

    if not os.path.exists(msi_dir):
        print(f"警告: 高光谱目录 {msi_dir} 不存在")
        return rgb_paths, msi_paths, labels

    rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
    print(f"找到 {len(rgb_files)} 个RGB文件")

    msi_files = glob.glob(os.path.join(msi_dir, "*.hdr"))
    msi_basenames = {os.path.basename(f).replace('.hdr', '') for f in msi_files}
    print(f"找到 {len(msi_files)} 个高光谱HDR文件")

    for rgb_file in rgb_files:
        rgb_basename = os.path.basename(rgb_file).replace('.png', '')
        label_info = parse_filename(rgb_file)
        if label_info is None:
            print(f"跳过无效文件（无法解析标签）: {rgb_basename}")
            continue
        if rgb_basename in msi_basenames:
            msi_file = os.path.join(msi_dir, f"{rgb_basename}.hdr")
            if find_msi_data_file(msi_file):
                rgb_paths.append(rgb_file)
                msi_paths.append(msi_file)
                labels.append(label_info["bruise_label"])
            else:
                print(f"警告: 找到HDR文件但找不到对应的数据文件: {rgb_basename}")
        else:
            print(f"警告: 找不到与 {rgb_basename} 对应的高光谱文件")

    print(f"总共匹配了 {len(rgb_paths)} 对RGB-高光谱图像")
    healthy_count = sum(1 for label in labels if label == 0)
    bruised_count = sum(1 for label in labels if label == 1)
    print(f"健康样本: {healthy_count}, 瘀伤样本: {bruised_count}")

    return rgb_paths, msi_paths, labels

# ========== 数据集定义 ==========
class MultiModalFruitDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, labels, img_size=Config.IMG_SIZE, msi_channels=None, is_train=False):
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.labels = labels
        self.img_size = img_size
        self.msi_channels = msi_channels
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15)
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = load_rgb_image(self.rgb_paths[idx], self.img_size)
        msi = load_msi_image(self.msi_paths[idx], self.img_size, target_channels=self.msi_channels)

        rgb = self._preprocess_rgb(rgb)
        msi = self._preprocess_msi(msi)

        if self.is_train and self.transform:
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            torch.manual_seed(seed)
            # 注意：transform需要调整以适应tensor输入
            # 这里暂时不使用transform，因为我们只是进行可视化

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return msi, rgb, label

    def _preprocess_rgb(self, rgb):
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float()
        else:
            rgb = rgb.float()
        rgb = torch.nan_to_num(rgb, nan=torch.nanmean(rgb), posinf=torch.nanmean(rgb), neginf=torch.nanmean(rgb))
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        else:
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        return rgb

    def _preprocess_msi(self, msi_data):
        msi_data = torch.nan_to_num(msi_data, nan=0.0, posinf=3.0, neginf=-3.0)
        msi_data = torch.clamp(msi_data, min=-3.0, max=3.0)
        return msi_data

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
            logits, features = model(msi_data, rgb_img)

            # 打印features的键，以便了解模型返回的特征结构
            if len(features_list) == 0:
                print(f"模型返回的features键: {list(features.keys())}")
                if 'aux_feat' in features:
                    print(f"aux_feat键: {list(features['aux_feat'].keys())}")

            if mode == 'rgb':
                # 尝试获取RGB相关特征
                if 'aux_feat' in features and 'SFE' in features['aux_feat']:
                    sfe_feat = features['aux_feat']['SFE']
                    feat = F.adaptive_avg_pool2d(sfe_feat, 1).flatten(1)
                elif 'aux_feat' in features and 'spatial_enhanced' in features['aux_feat']:
                    spatial_feat = features['aux_feat']['spatial_enhanced']
                    feat = F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1)
                else:
                    # 如果没有RGB相关特征，使用overview特征
                    if 'overview' in features:
                        overview_feat = features['overview']
                        feat = F.adaptive_avg_pool2d(overview_feat, 1).flatten(1)
                    else:
                        # 最后使用logits
                        feat = logits

            elif mode == 'hsi':
                # 尝试获取HSI相关特征
                if 'aux_feat' in features and 'STM' in features['aux_feat']:
                    stm_feat = features['aux_feat']['STM']
                    feat = F.adaptive_avg_pool2d(stm_feat, 1).flatten(1)
                elif 'aux_feat' in features and 'PSS' in features['aux_feat']:
                    pss_feat = features['aux_feat']['PSS']
                    feat = F.adaptive_avg_pool2d(pss_feat, 1).flatten(1)
                else:
                    # 如果没有HSI相关特征，使用base特征
                    if 'base' in features:
                        base_feat = features['base']
                        feat = F.adaptive_avg_pool2d(base_feat, 1).flatten(1)
                    else:
                        # 最后使用logits
                        feat = logits

            elif mode == 'fused':
                # 尝试获取融合特征
                if 'focus' in features:
                    focus_feat = features['focus']
                    feat = F.adaptive_avg_pool2d(focus_feat, 1).flatten(1)
                elif 'base' in features:
                    base_feat = features['base']
                    feat = F.adaptive_avg_pool2d(base_feat, 1).flatten(1)
                else:
                    # 最后使用logits
                    feat = logits

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


# ==========================================
# 2. 运行可视化主函数
# ==========================================
def run_visualize():
    device = Config.DEVICE
    # 模型路径
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'BruiseSTFNet_B.pth')  # 二分类模型
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'tsne_results')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = BruiseSTFNet(
        msi_channels=Config.MSI_CHANNELS,  # MSI通道数为24
        rgb_channels=Config.RGB_CHANNELS,    # RGB通道数为3
        num_classes=Config.NUM_CLASSES      # 二分类
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"模型加载成功: {MODEL_PATH}")

    # 收集数据路径
    rgb_paths, msi_paths, labels = collect_data_paths()
    if not rgb_paths:
        print("没有找到有效数据，请检查数据路径")
        return

    # 创建数据集和加载器
    dataset = MultiModalFruitDataset(
        rgb_paths=rgb_paths,
        msi_paths=msi_paths,
        labels=labels,
        img_size=Config.IMG_SIZE,
        msi_channels=Config.MSI_CHANNELS,
        is_train=False
    )
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
    # 二分类的颜色
    colors = ['#8E7FB8', '#E88471']  # 健康和瘀伤的颜色
    # 二分类标签映射
    labels_map = {0: '健康', 1: '瘀伤'}

    for i in range(3):
        embs, lbls = all_tsne_results[i]
        df = pd.DataFrame({
            'Dimension 1': embs[:, 0],
            'Dimension 2': embs[:, 1],
            'Class': [labels_map[l] for l in lbls]
        })

        # 绘制散点图
        sns.scatterplot(data=df, x='Dimension 1', y='Dimension 2', hue='Class',
                        palette={labels_map[0]: colors[0], labels_map[1]: colors[1]},
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
    save_path = os.path.join(SAVE_DIR, 'tsne_BruiseSTFNet_B_binary_总数据.png')  # 二分类结果
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n可视化结果已保存至: {save_path}")


if __name__ == "__main__":
    run_visualize()
