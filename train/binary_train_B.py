import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import spectral
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL.features import features
from rasterio.errors import NotGeoreferencedWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

# 忽略警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 导入模型（确保路径正确）
from BruiseSTFNet_B import BruiseSTFNet

# ========== 配置 ==========
DATA_DIR = "D://DATA//gtdata"
IMG_SIZE = (32, 32)
BATCH_SIZE = 4
EPOCHS = 33
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 健康(0)和瘀伤(1)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 早停机制参数
PATIENCE = 30
MIN_DELTA = 0.001


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


def load_rgb_image(path, img_size=IMG_SIZE):
    try:
        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.nn.functional.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=img_size, mode='bilinear',
                                              align_corners=False)
        img = img.squeeze(0) / 255.0
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    except Exception:
        return torch.zeros(3, *img_size)


def load_msi_image(path, img_size=IMG_SIZE, target_channels=None):
    try:
        data_file = find_msi_data_file(path)
        if data_file is None:
            return torch.zeros(target_channels, *img_size) if target_channels else torch.zeros(10, *img_size)
        msi_data = spectral.envi.open(path, data_file)
        img = msi_data.load()
        img = torch.tensor(np.array(img), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False)
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
    except Exception:
        return torch.zeros(target_channels, *img_size) if target_channels else torch.zeros(10, *img_size)


def collect_data_paths():
    rgb_paths = []
    msi_paths = []
    labels = []

    rgb_dir = os.path.join(DATA_DIR, "rgb_cropped")
    msi_dir = os.path.join(DATA_DIR, "hyperspectral_cropped")

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


def bruise_fusion_loss(logits, labels, attn_weight, lambda_attn=0.1):
    """
    适配[B]形状的分类标签（0=正常，1=瘀伤）
    logits: 分类预测结果 [B, 2]
    labels: 真实标签 [B]（0/1）
    attn_weight: 交叉注意力权重 [B, heads, H*W, H*W]
    lambda_attn: 正则化系数
    """
    ce_loss = F.cross_entropy(logits, labels)
    B = labels.shape[0]
    N = attn_weight.shape[2]
    label_flat = labels.view(B, 1, 1).repeat(1, 1, N)
    attn_avg = attn_weight.mean(dim=1)
    bruise_attn = (attn_avg * label_flat.transpose(1, 2)).mean(dim=-1).mean(dim=-1)
    normal_attn = (attn_avg * (1 - label_flat.transpose(1, 2))).mean(dim=-1).mean(dim=-1)
    mask = labels.float()
    attn_reg = (F.relu(normal_attn - bruise_attn) * mask).mean()
    total_loss = ce_loss + lambda_attn * attn_reg
    return total_loss


# ========== 数据集定义 ==========
class MultiModalFruitDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, labels, img_size=IMG_SIZE, msi_channels=None, is_train=True):
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
            rgb = self.transform(rgb)

            random.seed(seed)
            torch.manual_seed(seed)
            msi = self.transform(msi)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb, msi, label

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


# ========== 评估函数（通用，支持验证/测试） ==========
def evaluate_model(model, loader, device, dataset_name="Validation"):
    """
    通用评估函数，返回完整指标
    :param model: 模型
    :param loader: 数据加载器
    :param device: 设备
    :param dataset_name: 数据集名称（用于打印）
    :return: 包含所有指标的字典
    """
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for rgb, msi, labels in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            rgb, msi, labels = rgb.to(device), msi.to(device), labels.to(device)
            out, _ = model(msi, rgb)

            # 计算损失
            loss = F.cross_entropy(out, labels)

            # 获取预测概率和预测标签
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()  # 瘀伤类概率
            preds = out.argmax(1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 收集结果
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)

            # 累计损失和准确率
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels_np).sum()
            total += labels.size(0)

    # 计算核心指标
    accuracy = total_correct / total
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    # 计算AUC-ROC（处理单类别情况）
    try:
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            auc = 0.5
            print(f"警告: {dataset_name}集只有一类样本，AUC设为0.5")
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = 0.5
        print(f"计算{dataset_name}集AUC时出错: {e}，AUC设为0.5")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 打印详细指标
    print(f"\n{dataset_name} 集详细指标:")
    print(f"损失: {total_loss / total:.4f} | 准确率: {accuracy:.4f} | 精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f} | F1分数: {f1:.4f} | AUC-ROC: {auc:.4f}")
    print(f"混淆矩阵:\n{cm}")

    return {
        'loss': total_loss / total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
        'confusion_matrix': cm
    }


# ========== 训练函数 ==========
def train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for rgb, msi, labels in tqdm(loader, desc="Training"):
        rgb, msi, labels = rgb.to(device), msi.to(device), labels.to(device)
        optimizer.zero_grad()
        out, features = model(msi, rgb)
        attn_weight = features['aux_feat']['cross_attention']['attn_stm2es2cm']

        loss = bruise_fusion_loss(
            logits=out,
            labels=labels,
            attn_weight=attn_weight,
            lambda_attn=0.1
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = total_correct / total
    print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
    return train_loss, train_acc


# ========== 主流程（训练+验证+测试） ==========
def main(RESULT_DIR="results/二分类/"):
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 50)
    print("开始数据收集...")
    print("=" * 50)

    # 1. 收集数据
    rgb_paths, msi_paths, labels = collect_data_paths()
    if len(rgb_paths) == 0:
        print("错误: 没有找到任何数据文件")
        return

    # 2. 数据划分（训练集70% + 验证集10% + 测试集20%）
    # 第一步：划分训练集和临时集（80%:20%）
    train_rgb, temp_rgb, train_msi, temp_msi, train_labels, temp_labels = train_test_split(
        rgb_paths, msi_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # 第二步：划分验证集和测试集（临时集拆分为50%:50%）
    val_rgb, test_rgb, val_msi, test_msi, val_labels, test_labels = train_test_split(
        temp_rgb, temp_msi, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # 打印数据划分结果
    print(f"\n数据划分结果:")
    print(f"训练集: {len(train_rgb)} 样本")
    print(f"验证集: {len(val_rgb)} 样本")
    print(f"测试集: {len(test_rgb)} 样本")

    # 3. 获取MSI通道数
    sample_msi = load_msi_image(msi_paths[0], IMG_SIZE)
    msi_channels = sample_msi.shape[0]
    print(f"\nMSI图像通道数: {msi_channels}")

    # 4. 创建数据集和加载器
    train_dataset = MultiModalFruitDataset(train_rgb, train_msi, train_labels, IMG_SIZE, msi_channels, is_train=True)
    val_dataset = MultiModalFruitDataset(val_rgb, val_msi, val_labels, IMG_SIZE, msi_channels, is_train=False)
    test_dataset = MultiModalFruitDataset(test_rgb, test_msi, test_labels, IMG_SIZE, msi_channels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

    # 5. 初始化模型和优化器
    model = BruiseSTFNet(msi_channels=msi_channels, rgb_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 6. 早停机制和指标记录
    best_f1 = -1
    counter = 0
    best_model_state = None
    best_val_metrics = None

    # 修复：统一键名（使用accuracy而非acc）
    train_metrics = {'loss': [], 'accuracy': []}
    val_metrics_log = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    test_metrics_final = None  # 测试集最终指标

    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    # 7. 训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_acc)  # 修复：键名改为accuracy

        # 验证
        val_metrics = evaluate_model(model, val_loader, DEVICE, dataset_name="Validation")
        # 修复：遍历val_metrics_log的键，确保和val_metrics的键完全匹配
        for key in val_metrics_log.keys():
            val_metrics_log[key].append(val_metrics[key])

        # 早停逻辑（基于验证集F1）
        current_f1 = val_metrics['f1']
        if current_f1 > best_f1 + MIN_DELTA:
            best_f1 = current_f1
            best_model_state = model.state_dict()
            best_val_metrics = val_metrics
            counter = 0
            print(f"✅ 验证集F1提升至 {best_f1:.4f}, 保存最佳模型")
        else:
            counter += 1
            print(f"❌ 验证集F1无显著提升，计数: {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print(f"早停触发！连续{counter}个epoch未提升")
                break

    # 8. 加载最佳模型评估测试集
    print("\n" + "=" * 50)
    print("评估测试集（使用最佳模型）")
    print("=" * 50)
    model.load_state_dict(best_model_state)
    test_metrics_final = evaluate_model(model, test_loader, DEVICE, dataset_name="Test")

    # 9. 可视化所有指标
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建综合可视化图（5个子图）
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # 9.1 损失曲线
    axes[0].plot(train_metrics['loss'], label="训练损失", marker='.')
    axes[0].plot(val_metrics_log['loss'], label="验证损失", marker='.')
    axes[0].axhline(y=test_metrics_final['loss'], color='red', linestyle='--',
                    label=f'测试损失: {test_metrics_final["loss"]:.4f}')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 9.2 准确率曲线（修复：键名改为accuracy）
    axes[1].plot(train_metrics['accuracy'], label="训练准确率", marker='.')
    axes[1].plot(val_metrics_log['accuracy'], label="验证准确率", marker='.')
    axes[1].axhline(y=test_metrics_final['accuracy'], color='red', linestyle='--',
                    label=f'测试准确率: {test_metrics_final["accuracy"]:.4f}')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 9.3 精确率/召回率/F1曲线
    axes[2].plot(val_metrics_log['precision'], label="验证精确率", marker='.')
    axes[2].plot(val_metrics_log['recall'], label="验证召回率", marker='.')
    axes[2].plot(val_metrics_log['f1'], label="验证F1", marker='.')
    axes[2].axhline(y=test_metrics_final['precision'], color='red', linestyle='--',
                    label=f'测试精确率: {test_metrics_final["precision"]:.4f}')
    axes[2].axhline(y=test_metrics_final['recall'], color='orange', linestyle='--',
                    label=f'测试召回率: {test_metrics_final["recall"]:.4f}')
    axes[2].axhline(y=test_metrics_final['f1'], color='purple', linestyle='--',
                    label=f'测试F1: {test_metrics_final["f1"]:.4f}')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("分数")
    axes[2].set_title("精确率/召回率/F1曲线")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # 9.4 AUC曲线
    axes[3].plot(val_metrics_log['auc'], label="验证AUC", marker='.')
    axes[3].axhline(y=test_metrics_final['auc'], color='red', linestyle='--',
                    label=f'测试AUC: {test_metrics_final["auc"]:.4f}')
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("AUC-ROC")
    axes[3].set_title("AUC-ROC曲线")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    # 9.5 测试集ROC曲线
    fpr, tpr, _ = roc_curve(test_metrics_final['labels'], test_metrics_final['probs'])
    axes[4].plot(fpr, tpr, label=f'测试集AUC = {test_metrics_final["auc"]:.4f}', color='red', linewidth=2)
    axes[4].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[4].set_xlabel('假阳性率 (FPR)')
    axes[4].set_ylabel('真阳性率 (TPR)')
    axes[4].set_title('测试集ROC曲线')
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "training_test_metrics.png"), dpi=300)
    plt.show()

    # 10. 保存模型和指标
    # 10.1 保存最佳模型
    torch.save(best_model_state, os.path.join(RESULT_DIR, "best_model_BruiseSTFNet.pth"))
    print(f"\n最佳模型保存至: {os.path.join(RESULT_DIR, 'best_model_BruiseSTFNet.pth')}")

    # 10.2 保存所有指标到文本文件
    metrics_log_path = os.path.join(RESULT_DIR, "all_metrics_log.txt")
    with open(metrics_log_path, 'w', encoding='utf-8') as f:
        # 写入训练/验证指标（修复：train_acc改为train_accuracy）
        f.write("Epoch,Train_Loss,Train_Accuracy,Val_Loss,Val_Accuracy,Val_Precision,Val_Recall,Val_F1,Val_AUC\n")
        for i in range(len(train_metrics['loss'])):
            f.write(f"{i + 1},"
                    f"{train_metrics['loss'][i]:.4f},"
                    f"{train_metrics['accuracy'][i]:.4f},"
                    f"{val_metrics_log['loss'][i]:.4f},"
                    f"{val_metrics_log['accuracy'][i]:.4f},"
                    f"{val_metrics_log['precision'][i]:.4f},"
                    f"{val_metrics_log['recall'][i]:.4f},"
                    f"{val_metrics_log['f1'][i]:.4f},"
                    f"{val_metrics_log['auc'][i]:.4f}\n")

        # 写入最佳验证指标
        best_epoch = val_metrics_log['f1'].index(max(val_metrics_log['f1'])) + 1
        f.write(f"\n最佳验证指标 (Epoch {best_epoch}):\n")
        f.write(f"损失: {val_metrics_log['loss'][best_epoch - 1]:.4f}\n")
        f.write(f"准确率: {val_metrics_log['accuracy'][best_epoch - 1]:.4f}\n")
        f.write(f"精确率: {val_metrics_log['precision'][best_epoch - 1]:.4f}\n")
        f.write(f"召回率: {val_metrics_log['recall'][best_epoch - 1]:.4f}\n")
        f.write(f"F1分数: {val_metrics_log['f1'][best_epoch - 1]:.4f}\n")
        f.write(f"AUC-ROC: {val_metrics_log['auc'][best_epoch - 1]:.4f}\n")

        # 写入测试集最终指标
        f.write("\n测试集最终指标:\n")
        f.write(f"损失: {test_metrics_final['loss']:.4f}\n")
        f.write(f"准确率: {test_metrics_final['accuracy']:.4f}\n")
        f.write(f"精确率: {test_metrics_final['precision']:.4f}\n")
        f.write(f"召回率: {test_metrics_final['recall']:.4f}\n")
        f.write(f"F1分数: {test_metrics_final['f1']:.4f}\n")
        f.write(f"AUC-ROC: {test_metrics_final['auc']:.4f}\n")
        f.write(f"混淆矩阵:\n{test_metrics_final['confusion_matrix']}\n")

    print(f"\n所有指标已保存至: {metrics_log_path}")

    # 11. 打印最终总结
    print("\n" + "=" * 50)
    print("训练流程总结")
    print("=" * 50)
    print(f"最佳验证集F1: {best_f1:.4f} (Epoch {best_epoch})")
    print("\n测试集最终结果:")
    print(f"准确率: {test_metrics_final['accuracy']:.4f}")
    print(f"精确率: {test_metrics_final['precision']:.4f}")
    print(f"召回率: {test_metrics_final['recall']:.4f}")
    print(f"F1分数: {test_metrics_final['f1']:.4f}")
    print(f"AUC-ROC: {test_metrics_final['auc']:.4f}")
    print(f"混淆矩阵:\n{test_metrics_final['confusion_matrix']}")


if __name__ == "__main__":
    main()