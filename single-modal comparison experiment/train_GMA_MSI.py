import glob
import os
import warnings
import logging
import time
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import spectral
import torch
import torch.nn as nn
import torch.optim as optim
from rasterio.errors import NotGeoreferencedWarning
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from thop import profile

# 忽略 NotGeoreferencedWarning 警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 导入模型
from models.GMA_changed import BruiseSTFNet

# ========== 配置 ==========
DATA_DIR = "/teams/TXLJ_1745741277/gtdata/gtdata"
FRUIT_TYPES = ["jjk", "ys", "jk", "bd"]
IMG_SIZE = (64, 64)
BATCH_SIZE = 2
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 二分类：健康(0)和瘀伤(1)

# 创建结果文件夹
RESULT_DIR = "results/GMA_MSI双模态/"
os.makedirs(RESULT_DIR, exist_ok=True)

# 配置日志
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(RESULT_DIR, f"training_log_{timestamp}.txt")

# 配置日志同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ========== 数据解析与处理 ==========
def parse_filename(filename):
    import os

    # 移除路径和扩展名，保留基础文件名
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]

    # 处理可能的预处理前缀（根据需求保留或移除）
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]

    # 按连字符分割文件名
    parts = basename.split('-')
    if len(parts) < 4:  # 新格式至少有4部分
        return {
            "fruit_type": "unknown",
            "fruit_label": 0,
            "drop_time": 0,
            "label_range": "0-0",
            "quality_label": 0
        }

    # 解析各部分
    fruit_type = parts[0]  # 瘀伤类型（如bd）
    drop_time = int(parts[1])  # 掉落时间
    quality_label = int(parts[2])  # 质量标签
    side_indicator = parts[3]  # 正反面标识（f/z）

    # 构造标签范围（单个标签格式化为"X-X"）
    label_range = f"{quality_label}-{quality_label}"

    # 二分类标签：jjk所有样本为健康(0):334，其他水果0小时为健康(0):74+58，其他时间为瘀伤(1)
    if fruit_type == "jjk":
        bruise_label = 0  # jjk所有样本视为健康
    elif fruit_type != "ys" and drop_time == 0:
        bruise_label = 0
    else:
        bruise_label = 1  # 除ys外其他水果0小时为健康，其他时间为瘀伤

    return {
        "fruit_type": fruit_type,
        "bruise_label": bruise_label,  # 瘀伤标签
        "drop_time": drop_time,
        "label_range": label_range,  # 格式化的标签范围
        "quality_label": quality_label  # 质量标签
    }


def find_msi_data_file(hdr_path):
    # 首先尝试使用相同的基名但不同扩展名
    base_path = hdr_path.replace('.hdr', '')

    # 检查可能的扩展名，包括新的.float扩展名
    extensions = ['.float', '.dat', '.img', '.bin', '.raw', '']
    for ext in extensions:
        test_path = base_path + ext
        if os.path.exists(test_path):
            return test_path

    # 如果上述方法失败，尝试在目录中查找匹配的文件
    hdr_dir = os.path.dirname(hdr_path)
    hdr_basename = os.path.basename(hdr_path).replace('.hdr', '')

    for file in os.listdir(hdr_dir):
        file_base = file.split('.')[0]  # 移除所有扩展名
        if file_base == hdr_basename and not file.endswith('.hdr'):
            return os.path.join(hdr_dir, file)

    return None


def load_msi_image(path, img_size=IMG_SIZE, target_channels=None):
    try:
        data_file = find_msi_data_file(path)
        if data_file is None:
            logger.error(f"找不到MSI数据文件: {path}")
            if target_channels:
                return torch.zeros(target_channels, *img_size)
            else:
                return torch.zeros(10, *img_size)
        msi_data = spectral.envi.open(path, data_file)
        img = msi_data.load()
        img = torch.tensor(np.array(img), dtype=torch.float32)  # (H, W, C)
        img = img.permute(2, 0, 1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False)
        img = img.squeeze(0)
        
        # 数据预处理：确保数据不会产生NaN值
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 归一化
        max_val = img.max()
        min_val = img.min()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = torch.zeros_like(img)
        
        if target_channels is not None:
            c = img.shape[0]
            if c < target_channels:
                pad = torch.zeros(target_channels - c, *img_size)
                img = torch.cat([img, pad], dim=0)
            elif c > target_channels:
                img = img[:target_channels]
        return img
    except Exception as e:
        logger.error(f"加载MSI图像时出错: {path}, 错误: {e}")
        if target_channels:
            return torch.zeros(target_channels, *img_size)
        else:
            return torch.zeros(10, *img_size)


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
        # 归一化
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    except Exception as e:
        logger.error(f"加载RGB图像时出错: {path}, 错误: {e}")
        return torch.zeros(3, *img_size)


def collect_data_paths():
    rgb_paths = []
    msi_paths = []
    labels = []

    # 遍历RGB和高光谱目录
    rgb_dir = os.path.join(DATA_DIR, "rgb_cropped")
    msi_dir = os.path.join(DATA_DIR, "hyperspectral_cropped")

    if not os.path.exists(rgb_dir):
        logger.warning(f"RGB目录 {rgb_dir} 不存在")
        return rgb_paths, msi_paths, labels

    if not os.path.exists(msi_dir):
        logger.warning(f"高光谱目录 {msi_dir} 不存在")
        return rgb_paths, msi_paths, labels

    # 遍历所有RGB图像文件
    rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
    logger.info(f"找到 {len(rgb_files)} 个RGB文件")

    # 获取所有高光谱HDR文件
    msi_files = glob.glob(os.path.join(msi_dir, "*.hdr"))
    logger.info(f"找到 {len(msi_files)} 个高光谱HDR文件")

    # 创建MSI文件字典，以便快速查找
    msi_file_dict = {}
    for msi_file in msi_files:
        msi_basename = os.path.basename(msi_file).replace('.hdr', '')
        if find_msi_data_file(msi_file):
            msi_file_dict[msi_basename] = msi_file

    # 遍历RGB文件，寻找对应的MSI文件
    for rgb_file in rgb_files:
        rgb_basename = os.path.basename(rgb_file).replace('.png', '')
        
        # 处理可能的前缀
        if rgb_basename.startswith('registered_'):
            rgb_basename = rgb_basename.replace('registered_', '')

        # 查找对应的MSI文件
        if rgb_basename in msi_file_dict:
            msi_file = msi_file_dict[rgb_basename]
            
            # 解析文件名获取标签信息
            label_info = parse_filename(rgb_file)
            
            rgb_paths.append(rgb_file)
            msi_paths.append(msi_file)
            labels.append(label_info["bruise_label"])  # 使用瘀伤标签
        else:
            logger.warning(f"找不到对应的MSI文件: {rgb_basename}")

    logger.info(f"总共匹配了 {len(rgb_paths)} 个RGB-MSI对")

    # 统计各类别样本数量
    healthy_count = sum(1 for label in labels if label == 0)
    bruised_count = sum(1 for label in labels if label == 1)
    logger.info(f"健康样本: {healthy_count}, 瘀伤样本: {bruised_count}")

    return rgb_paths, msi_paths, labels


# ========== PyTorch Dataset ==========
import random
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, labels, img_size=IMG_SIZE, msi_channels=None, is_train=True):
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.labels = labels
        self.img_size = img_size
        self.msi_channels = msi_channels
        self.is_train = is_train

        # 定义训练时的数据增强
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = load_rgb_image(self.rgb_paths[idx], self.img_size)
        msi = load_msi_image(self.msi_paths[idx], self.img_size, target_channels=self.msi_channels)
        label = self.labels[idx]

        # 应用数据增强
        if self.is_train and self.transform:
            rgb = self.transform(rgb)
            msi = self.transform(msi)

        return rgb, msi, label


# ========== 训练和验证函数 ==========
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 用于AUC计算的概率

    for rgb, msi, labels in loader:
        rgb, msi, labels = rgb.to(device), msi.to(device), labels.to(device)
        
        # 检查输入数据
        if torch.isnan(rgb).any() or torch.isnan(msi).any():
            logger.warning("输入数据包含NaN值，跳过此批次")
            continue
        
        optimizer.zero_grad()

        out, features = model(msi_data=msi, rgb_img=rgb)
        
        # 检查模型输出
        if torch.isnan(out).any():
            logger.warning("模型输出包含NaN值，跳过此批次")
            continue
        
        loss = criterion(out, labels)
        
        # 检查损失值
        if torch.isnan(loss):
            logger.warning("损失值为NaN，跳过此批次")
            continue
        
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 计算概率
        probs = torch.softmax(out, dim=1)[:, 1]  # 取正类的概率

        # 收集预测结果和标签
        preds = out.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    # 计算指标
    if total > 0:
        acc = total_correct / total
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # 处理NaN值
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(all_probs)
        if np.sum(valid_mask) > 0:
            valid_probs = all_probs[valid_mask]
            valid_labels = all_labels[valid_mask]
            
            # 检查是否有足够的样本计算AUC
            if len(np.unique(valid_labels)) >= 2:
                auc = roc_auc_score(valid_labels, valid_probs)
            else:
                auc = 0.0
        else:
            auc = 0.0
    else:
        acc = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0

    return (total_loss / (total + 1e-8), acc, precision, recall, f1, auc)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 用于AUC计算的概率

    with torch.no_grad():
        for rgb, msi, labels in loader:
            rgb, msi, labels = rgb.to(device), msi.to(device), labels.to(device)
            
            # 检查输入数据
            if torch.isnan(rgb).any() or torch.isnan(msi).any():
                logger.warning("输入数据包含NaN值，跳过此批次")
                continue

            out, features = model(msi_data=msi, rgb_img=rgb)
            
            # 检查模型输出
            if torch.isnan(out).any():
                logger.warning("模型输出包含NaN值，跳过此批次")
                continue
            
            loss = criterion(out, labels)
            
            # 检查损失值
            if torch.isnan(loss):
                logger.warning("损失值为NaN，跳过此批次")
                continue

            # 计算概率
            probs = torch.softmax(out, dim=1)[:, 1]  # 取正类的概率

            # 收集预测结果和标签
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    # 计算指标
    if total > 0:
        acc = total_correct / total
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # 处理NaN值
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(all_probs)
        if np.sum(valid_mask) > 0:
            valid_probs = all_probs[valid_mask]
            valid_labels = all_labels[valid_mask]
            
            # 检查是否有足够的样本计算AUC
            if len(np.unique(valid_labels)) >= 2:
                auc = roc_auc_score(valid_labels, valid_probs)
            else:
                auc = 0.0
        else:
            auc = 0.0
    else:
        acc = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0

    return (total_loss / (total + 1e-8), acc, precision, recall, f1, auc, all_labels, all_probs)

# 计算模型参数量、推理速度和吞吐量
def calculate_model_metrics(model, device, msi_input_size, rgb_input_size):
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_million = total_params / 1e6
    
    # 确保模型在 eval 模式
    model.eval()
    
    # 计算FLOPS - 使用 batch size 2 避免 BatchNorm 问题
    dummy_msi = torch.randn(2, *msi_input_size).to(device)
    dummy_rgb = torch.randn(2, *rgb_input_size).to(device)
    
    # 创建包装模型处理多输出问题
    class FLOPSWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, msi, rgb):
            with torch.no_grad():
                output, _ = self.model(msi_data=msi, rgb_img=rgb)
            return output
    
    try:
        wrapped_model = FLOPSWrapper(model)
        flops, params = profile(wrapped_model, inputs=(dummy_msi, dummy_rgb))
        flops_g = flops / 1e9  # 转换为十亿单位
    except Exception as e:
        logger.error(f"计算FLOPS时出错: {e}")
        flops_g = 0
    
    # 预热 - 使用 batch size 1
    dummy_msi_single = torch.randn(1, *msi_input_size).to(device)
    dummy_rgb_single = torch.randn(1, *rgb_input_size).to(device)
    model.eval()  # 再次确保模型在 eval 模式
    for _ in range(10):
        with torch.no_grad():
            model(msi_data=dummy_msi_single, rgb_img=dummy_rgb_single)
    
    # 测量推理时间 - 使用 batch size 1
    start_time = time.time()
    num_inferences = 100
    model.eval()  # 再次确保模型在 eval 模式
    for _ in range(num_inferences):
        with torch.no_grad():
            model(msi_data=dummy_msi_single, rgb_img=dummy_rgb_single)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_inferences
    throughput = 1 / avg_inference_time
    
    return params_million, flops_g, avg_inference_time, throughput


# ========== 主流程 ==========
def main():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 确认使用设备
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"使用GPU加速训练: {device_name}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
    else:
        logger.info("使用CPU训练（建议使用GPU以加速训练）")

    logger.info("收集数据路径...")
    rgb_paths, msi_paths, labels = collect_data_paths()
    if len(rgb_paths) == 0:
        logger.error("错误: 没有找到任何数据文件")
        return
    logger.info(f"找到 {len(rgb_paths)} 个样本")

    # 划分训练集和测试集
    train_rgb, test_rgb, train_msi, test_msi, train_labels, test_labels = train_test_split(
        rgb_paths, msi_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 估算msi通道数
    sample_msi = load_msi_image(msi_paths[0], IMG_SIZE)
    msi_channels = sample_msi.shape[0]
    logger.info(f"MSI图像通道数: {msi_channels}")

    logger.info("构建数据集和加载器...")
    train_dataset = MultiModalDataset(train_rgb, train_msi, train_labels, IMG_SIZE, msi_channels, is_train=True)
    test_dataset = MultiModalDataset(test_rgb, test_msi, test_labels, IMG_SIZE, msi_channels, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

    logger.info("构建模型...")
    model = BruiseSTFNet(msi_channels=msi_channels, num_classes=NUM_CLASSES).to(DEVICE)

    # 计算模型指标
    msi_input_size = (msi_channels, *IMG_SIZE)
    rgb_input_size = (3, *IMG_SIZE)
    params_million, flops_g, avg_inference_time, throughput = calculate_model_metrics(model, DEVICE, msi_input_size, rgb_input_size)
    logger.info(f"模型总参数量: {params_million:.2f}M")
    logger.info(f"FLOPS: {flops_g:.2f}G")
    logger.info(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
    logger.info(f"吞吐量: {throughput:.2f} imgs/s")

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    logger.info("开始训练...")
    best_acc = 0
    best_auc = 0
    # 存储所有指标的历史记录
    metrics = {
        'train': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
        'val': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    }
    best_model_state = None
    best_labels = None
    best_probs = None

    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_labels, val_probs = eval_one_epoch(
            model, test_loader, criterion, DEVICE)

        # 更新学习率
        scheduler.step(val_acc)

        # 保存指标
        metrics['train']['loss'].append(train_loss)
        metrics['train']['acc'].append(train_acc)
        metrics['train']['precision'].append(train_precision)
        metrics['train']['recall'].append(train_recall)
        metrics['train']['f1'].append(train_f1)
        metrics['train']['auc'].append(train_auc)

        metrics['val']['loss'].append(val_loss)
        metrics['val']['acc'].append(val_acc)
        metrics['val']['precision'].append(val_precision)
        metrics['val']['recall'].append(val_recall)
        metrics['val']['f1'].append(val_f1)
        metrics['val']['auc'].append(val_auc)

        # 打印 epoch 结果
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS}\n"
            f"Train - loss: {train_loss:.4f}, acc: {train_acc:.4f}, precision: {train_precision:.4f}, "
            f"recall: {train_recall:.4f}, f1: {train_f1:.4f}, auc: {train_auc:.4f}\n"
            f"Val   - loss: {val_loss:.4f}, acc: {val_acc:.4f}, precision: {val_precision:.4f}, "
            f"recall: {val_recall:.4f}, f1: {val_f1:.4f}, auc: {val_auc:.4f}"
        )

        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_labels = val_labels
            best_probs = val_probs
            model_path = os.path.join(RESULT_DIR, "best_model_BruiseSTFNet.pth")
            torch.save(best_model_state, model_path)
            logger.info(f"新的最佳模型已保存，准确率: {best_acc:.4f}, AUC: {best_auc:.4f}")

        # 提前停止条件：验证集准确率大于83%时停止训练
        if val_acc >= 0.83:
            logger.info(f"验证集准确率达到{val_acc:.4f}，已达到目标准确率83%，提前停止训练")
            break

    # 绘制训练历史 - 损失和准确率
    plt.figure(figsize=(16, 12))

    # 损失
    plt.subplot(2, 3, 1)
    plt.plot(metrics['train']['loss'], label="训练损失")
    plt.plot(metrics['val']['loss'], label="验证损失")
    plt.legend()
    plt.title("损失曲线")

    # 准确率
    plt.subplot(2, 3, 2)
    plt.plot(metrics['train']['acc'], label="训练准确率")
    plt.plot(metrics['val']['acc'], label="验证准确率")
    plt.legend()
    plt.title("准确率曲线")

    # 精确率
    plt.subplot(2, 3, 3)
    plt.plot(metrics['train']['precision'], label="训练精确率")
    plt.plot(metrics['val']['precision'], label="验证精确率")
    plt.legend()
    plt.title("精确率曲线")

    # 召回率
    plt.subplot(2, 3, 4)
    plt.plot(metrics['train']['recall'], label="训练召回率")
    plt.plot(metrics['val']['recall'], label="验证召回率")
    plt.legend()
    plt.title("召回率曲线")

    # F1分数
    plt.subplot(2, 3, 5)
    plt.plot(metrics['train']['f1'], label="训练F1分数")
    plt.plot(metrics['val']['f1'], label="验证F1分数")
    plt.legend()
    plt.title("F1分数曲线")

    # AUC
    plt.subplot(2, 3, 6)
    plt.plot(metrics['train']['auc'], label="训练AUC")
    plt.plot(metrics['val']['auc'], label="验证AUC")
    plt.legend()
    plt.title("AUC曲线")

    plt.tight_layout()
    metrics_plot_path = os.path.join(RESULT_DIR, f'training_metrics_{timestamp}.png')
    plt.savefig(metrics_plot_path)
    plt.close()

    # 绘制最佳模型的ROC曲线
    if best_labels is not None and best_probs is not None:
        fpr, tpr, _ = roc_curve(best_labels, best_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {best_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 随机猜测的基准线
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('最佳模型的ROC曲线')
        plt.legend()
        roc_plot_path = os.path.join(RESULT_DIR, f'roc_curve_{timestamp}.png')
        plt.savefig(roc_plot_path)
        plt.close()

    # 加载最佳模型进行最终测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_metrics = eval_one_epoch(model, test_loader, criterion, DEVICE)
        final_loss, final_acc, final_precision, final_recall, final_f1, final_auc, _, _ = final_metrics

        logger.info("\n===== 最终测试结果 =====")
        logger.info(f"损失: {final_loss:.4f}")
        logger.info(f"准确率: {final_acc:.4f}")
        logger.info(f"精确率: {final_precision:.4f}")
        logger.info(f"召回率: {final_recall:.4f}")
        logger.info(f"F1分数: {final_f1:.4f}")
        logger.info(f"AUC: {final_auc:.4f}")
        logger.info(f"最佳验证集准确率: {best_acc:.4f}")
        logger.info(f"最佳验证集AUC: {best_auc:.4f}")
        logger.info(f"模型参数量: {params_million:.2f}M")
        logger.info(f"FLOPS: {flops_g:.2f}G")
        logger.info(f"平均推理时间: {avg_inference_time*1000:.2f}ms")
        logger.info(f"吞吐量: {throughput:.2f} imgs/s")

    logger.info("训练完成！所有结果已保存到results/GMA_MSI双模态文件夹")


if __name__ == "__main__":
    main()
