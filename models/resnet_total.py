import os
import glob
import warnings
import random
import numpy as np
import spectral
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import sys
import datetime
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

import torchvision.models as models
import torch.nn as nn

# 忽略无关警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from torch.hub import load_state_dict_from_url

__all__ = ['HSIRGBResNet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """ResNet18使用的基础块（不同于ResNet50的Bottleneck）"""
    expansion = 1  # 特征图通道数扩展倍数为1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FeatureExtractor(nn.Module):
    """特征提取器，基于ResNet18结构"""
    def __init__(self, block, layers, in_channels):
        super(FeatureExtractor, self).__init__()
        self.inplanes = 64
        
        # 针对不同模态调整输入卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet18的层配置（不同于ResNet50）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class HSIRGBResNet18(nn.Module):
    """高光谱+RGB双模态融合ResNet18模型"""
    def __init__(self, hsi_channels, num_classes=2, pretrained_rgb=True):
        super(HSIRGBResNet18, self).__init__()
        # RGB分支 (3通道)，使用ResNet18的层配置[2,2,2,2]
        self.rgb_extractor = FeatureExtractor(BasicBlock, [2, 2, 2, 2], in_channels=3)
        # 高光谱分支，同样使用ResNet18结构
        self.hsi_extractor = FeatureExtractor(BasicBlock, [2, 2, 2, 2], in_channels=hsi_channels)
       
        # 加载RGB分支预训练权重
        if pretrained_rgb:
            rgb_pretrained = load_state_dict_from_url(model_urls['resnet18'])
            # 过滤掉输入卷积层（通道数不匹配）
            filtered = {k: v for k, v in rgb_pretrained.items() if not k.startswith('conv1.')}
            self.rgb_extractor.load_state_dict(filtered, strict=False)
        
        # 融合后的全连接层（ResNet18每个分支输出512维，拼接后为1024维）
        self.fc = nn.Linear(512 * 2, num_classes)  # 512 * 1（BasicBlock.expansion）*2（两个分支）
        # 融合层正则化
        self.bn_fuse = nn.BatchNorm1d(512 * 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hsi, rgb):
        # 提取RGB特征
        rgb_feat = self.rgb_extractor(rgb)  # 输出形状: [batch_size, 512]
        # 提取高光谱特征
        hsi_feat = self.hsi_extractor(hsi)  # 输出形状: [batch_size, 512]
        
        # 特征融合（拼接）
        fused = torch.cat([rgb_feat, hsi_feat], dim=1)  # 输出形状: [batch_size, 1024]
        
        # 分类头
        fused = self.bn_fuse(fused)
        fused = self.dropout(fused)
        out = self.fc(fused)
        
        return out

# ========== 配置参数 ==========
class Config:
    DATA_ROOT = "data2"
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01', '6', '12', '18', '24', '30', '36']  # 原始时间点
    LABEL_TO_CLASS = {
        0: "健康",
        1: "瘀伤", 
    }
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    MSI_CHANNELS = 24
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 20
    MIN_DELTA = 0.001
    RESULT_DIR = r"二分类results/resnet/resnet_total"
    SEED = 42
    # 新增：类别权重参数
    CLASS_WEIGHTS = [1.0, 0.3]  # 健康样本权重1.0，瘀伤样本权重0.3
    # 新增：过采样参数
    OVERSAMPLE_RATIO = 2.0  # 健康样本过采样比例

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(Config.SEED)

# ========== 日志设置 ==========
def setup_logging(result_dir):
    """设置日志记录"""
    log_file = os.path.join(result_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ========== 数据集类 ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, subset="train", transform=None, oversample=False):
        self.subset = subset
        self.transform = transform
        self.oversample = oversample
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"{subset}集未找到有效样本")
        
        # 如果需要过采样，对健康样本进行过采样
        if self.oversample and subset == "train":
            self._oversample_healthy_samples()
        
        # 统计按照文件名分类的样本分布
        self._log_label_distribution()

    def _get_label_by_filename(self, prefix):
        """
        根据文件名确定标签
        文件名格式类似"jjk-0-2-z"，使用"-"分割
        如果第一部分是"jjk"，类别为0
        如果第一部分不是"jjk"，第二部分是"0"，类别为0
        如果第一部分不是"jjk"，第二部分不是"0"，类别为1
        """
        # 使用"-"分割文件名
        parts = prefix.split('-')
        
        # 检查文件名格式是否正确
        if len(parts) < 2:
            logging.warning(f"文件名格式不符合预期: {prefix}")
            # 如果格式不符合预期，使用默认规则：如果包含"jjk"则为健康
            if "jjk" in prefix.lower():
                return 0
            else:
                return 1
        
        # 第一部分判断
        if parts[0].lower() == "jjk":
            return 0  # 健康
        else:
            # 第二部分判断
            if parts[1] == "0":
                return 0  # 健康
            else:
                return 1  # 瘀伤

    def _oversample_healthy_samples(self):
        """对健康样本进行过采样以平衡类别分布"""
        # 分离健康样本和瘀伤样本
        healthy_samples = [s for s in self.samples if s[2] == 0]  # 健康样本
        bruise_samples = [s for s in self.samples if s[2] == 1]   # 瘀伤样本
        
        logging.info(f"过采样前: 健康样本 {len(healthy_samples)}个, 瘀伤样本 {len(bruise_samples)}个")
        
        # 计算需要过采样的数量
        target_count = int(len(bruise_samples) * Config.OVERSAMPLE_RATIO)
        oversample_count = max(0, target_count - len(healthy_samples))
        
        if oversample_count > 0:
            # 随机选择健康样本进行复制
            oversampled = random.choices(healthy_samples, k=oversample_count)
            self.samples.extend(oversampled)
            
            logging.info(f"过采样后: 健康样本 {len(healthy_samples) + oversample_count}个, 瘀伤样本 {len(bruise_samples)}个")
            logging.info(f"过采样比例: {Config.OVERSAMPLE_RATIO:.1f}, 新增样本: {oversample_count}个")
        else:
            logging.info("健康样本数量已足够，无需过采样")

    def _log_label_distribution(self):
        """记录按照文件名分类的样本分布"""
        jjk_first_part_count = 0
        zero_second_part_count = 0
        other_count = 0
        label_0_count = 0
        label_1_count = 0
        
        for sample in self.samples:
            msi_path, rgb_path, label, sample_name, time_point = sample
            prefix = sample_name.split('_time')[0]  # 获取文件名前缀
            
            # 重新计算标签以统计分布
            calculated_label = self._get_label_by_filename(prefix)
            
            # 统计分类依据
            parts = prefix.split('-')
            if len(parts) >= 2:
                if parts[0].lower() == "jjk":
                    jjk_first_part_count += 1
                elif parts[1] == "0":
                    zero_second_part_count += 1
                else:
                    other_count += 1
            else:
                other_count += 1
                
            if calculated_label == 0:
                label_0_count += 1
            else:
                label_1_count += 1
        
        # 使用self.subset而不是subset
        logging.info(f"\n{self.subset}集标签分布统计:")
        logging.info(f"  按第一部分是'jjk'分类: {jjk_first_part_count}个样本")
        logging.info(f"  按第二部分是'0'分类: {zero_second_part_count}个样本")
        logging.info(f"  其他情况: {other_count}个样本")
        logging.info(f"  健康样本 (标签0): {label_0_count}个")
        logging.info(f"  瘀伤样本 (标签1): {label_1_count}个")
        logging.info(f"  健康样本比例: {label_0_count/len(self.samples):.2%}")
        logging.info(f"  瘀伤样本比例: {label_1_count/len(self.samples):.2%}")

    def _collect_samples(self):
        samples = []
        subset_dir = os.path.join(Config.DATA_ROOT, self.subset)
        logging.info(f"当前搜索的子集目录：{subset_dir}")

        for time in Config.TIME_POINTS:
            msi_time_dir = os.path.join(subset_dir, "MSI", str(time))
            rgb_time_dir = os.path.join(subset_dir, "RGB", str(time))

            if not os.path.isdir(msi_time_dir) or not os.path.isdir(rgb_time_dir):
                continue

            msi_hdr_files = glob.glob(os.path.join(msi_time_dir, "*.hdr"))
            msi_prefixes = {os.path.splitext(os.path.basename(f))[0] for f in msi_hdr_files}

            rgb_png_files = glob.glob(os.path.join(rgb_time_dir, "*.png"))
            rgb_prefixes = {os.path.splitext(os.path.basename(f))[0] for f in rgb_png_files}

            common_prefixes = msi_prefixes & rgb_prefixes
            for prefix in common_prefixes:
                msi_path = os.path.join(msi_time_dir, f"{prefix}.hdr")
                rgb_path = os.path.join(rgb_time_dir, f"{prefix}.png")
                sample_name = f"{prefix}_time{time}"
                
                # 根据文件名确定标签
                label = self._get_label_by_filename(prefix)
                samples.append((msi_path, rgb_path, label, sample_name, time))

        logging.info(f"最终收集到{len(samples)}个样本")
        return samples

    def _read_msi(self, msi_hdr_path):
        try:
            msi_data_path = f"{os.path.splitext(msi_hdr_path)[0]}.float"
            from spectral.io import envi
            img = envi.open(msi_hdr_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))

            msi_min, msi_max = msi_data.min(), msi_data.max()
            if msi_max - msi_min < 1e-8:
                msi_data = np.zeros_like(msi_data)
            else:
                msi_data = (msi_data - msi_min) / (msi_max - msi_min)

            msi_tensor = torch.tensor(msi_data, dtype=torch.float32)
            msi_resized = F.interpolate(
                msi_tensor.unsqueeze(0),
                size=Config.IMG_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            return msi_resized

        except Exception as e:
            raise RuntimeError(f"读取MSI文件失败：{msi_hdr_path}\n错误详情：{str(e)}")

    def _read_rgb(self, rgb_path):
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            if self.transform:
                rgb_img = self.transform(rgb_img)
            rgb_tensor = transforms.ToTensor()(rgb_img)
            rgb_resized = F.interpolate(
                rgb_tensor.unsqueeze(0),
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
        msi_path, rgb_path, label, sample_name, time_point = self.samples[idx]
        msi = self._read_msi(msi_path)
        rgb = self._read_rgb(rgb_path)
        return msi, rgb, torch.tensor(label, dtype=torch.long), sample_name, time_point

# ========== 损失函数 ==========
def weighted_cross_entropy(logits, labels, class_weights=None):
    """
    带类别权重的交叉熵损失
    class_weights: 每个类别的权重，例如[1.0, 0.3]表示健康样本权重1.0，瘀伤样本权重0.3
    """
    if class_weights is None:
        return F.cross_entropy(logits, labels)
    
    # 将类别权重转换为张量
    weights = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)
    
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(logits, labels, weight=weights)
    return ce_loss

def multi_class_fusion_loss(logits, labels, lambda_attn=0.01, class_weights=None):
    """融合损失函数，支持类别权重"""
    ce_loss = weighted_cross_entropy(logits, labels, class_weights)
    return ce_loss

# ========== 数据增强函数 ==========
def get_strong_augmentation():
    """为健康样本提供更强的数据增强"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),  # 更大的旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
    ])

def get_standard_augmentation():
    """为瘀伤样本提供标准的数据增强"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
    ])

# ========== 平衡数据加载器 ==========
class BalancedDataLoader:
    """平衡数据加载器，确保每个batch中各类别样本比例均衡"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 获取样本标签
        labels = [sample[2] for sample in dataset.samples]
        
        # 计算每个类别的样本索引
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        
        # 计算每个类别的样本数
        self.class_counts = {label: len(indices) for label, indices in class_indices.items()}
        
        # 找到最大类别的样本数
        self.max_class_count = max(self.class_counts.values())
        
        # 计算每个batch中每个类别的样本数
        self.samples_per_class = self.batch_size // self.num_classes
        if self.samples_per_class == 0:
            self.samples_per_class = 1
        
        logging.info(f"平衡数据加载器: 每批每个类别样本数={self.samples_per_class}, 总类别数={self.num_classes}")
        
    def __iter__(self):
        # 为每个类别创建索引列表
        class_indices_copy = {label: indices.copy() for label, indices in self.class_indices.items()}
        
        # 对每个类别的索引进行洗牌
        if self.shuffle:
            for indices in class_indices_copy.values():
                random.shuffle(indices)
        
        # 为每个类别创建迭代器
        class_iterators = {}
        for label, indices in class_indices_copy.items():
            # 如果样本数不足，循环使用
            if len(indices) < self.samples_per_class:
                # 重复样本直到达到所需数量
                repeated_indices = []
                while len(repeated_indices) < self.samples_per_class:
                    remaining = self.samples_per_class - len(repeated_indices)
                    repeated_indices.extend(indices[:min(remaining, len(indices))])
                class_iterators[label] = iter(repeated_indices)
            else:
                class_iterators[label] = iter(indices)
        
        # 生成批次
        batch_indices = []
        while True:
            try:
                # 从每个类别中取样本
                for label in class_iterators:
                    for _ in range(self.samples_per_class):
                        try:
                            batch_indices.append(next(class_iterators[label]))
                        except StopIteration:
                            # 如果某个类别的样本用完，重新开始
                            if self.shuffle:
                                indices = class_indices_copy[label].copy()
                                random.shuffle(indices)
                            else:
                                indices = class_indices_copy[label].copy()
                            class_iterators[label] = iter(indices)
                            batch_indices.append(next(class_iterators[label]))
                
                # 如果批次大小不是类别数的整数倍，添加额外样本
                remaining = self.batch_size - len(batch_indices)
                if remaining > 0:
                    # 从所有类别中随机选择样本
                    all_indices = []
                    for indices in class_indices_copy.values():
                        all_indices.extend(indices)
                    
                    if self.shuffle:
                        random.shuffle(all_indices)
                    
                    batch_indices.extend(all_indices[:remaining])
                
                yield batch_indices
                batch_indices = []
                
            except StopIteration:
                break
    
    def __len__(self):
        return (self.max_class_count * self.num_classes) // self.batch_size + 1

# ========== 训练和验证函数 ==========
def train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0, class_weights=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="训练"):
        if isinstance(batch, list):
            # 如果是平衡数据加载器，需要先通过索引获取数据
            indices = batch
            batch_data = []
            for idx in indices:
                batch_data.append(loader.dataset[idx])
            
            # 重新组织批次数据
            msi_batch = torch.stack([item[0] for item in batch_data])
            rgb_batch = torch.stack([item[1] for item in batch_data])
            labels_batch = torch.stack([item[2] for item in batch_data])
            
            msi, rgb, labels = msi_batch.to(device), rgb_batch.to(device), labels_batch.to(device)
        else:
            # 普通数据加载器
            msi, rgb, labels, _, _ = batch
            msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)
        
        optimizer.zero_grad()

        out = model(msi, rgb)
        loss = multi_class_fusion_loss(logits=out, labels=labels, lambda_attn=0.1, class_weights=class_weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc

def eval_one_epoch(model, loader, device, compute_auc=False, record_errors=False, 
                   error_file_path=None, group_by_time=False, time_metrics_file=None):
    """评估函数"""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels, all_times = [], [], []
    all_probs = []
    error_files = []
    time_group_data = {}  # 按时间点分组的数据

    with torch.no_grad():
        for batch in loader:
            if record_errors or group_by_time:
                msi, rgb, labels, sample_names, time_points = batch
            else:
                msi, rgb, labels, _, _ = batch
            
            msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)
            out = model(msi, rgb)

            loss = F.cross_entropy(out, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.extend(probs)

            preds = out.argmax(1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            total_correct += (preds == labels_np).sum()
            total += labels.size(0)

            # 按时间点分组数据
            if group_by_time:
                time_points_np = np.array(time_points)
                for time_point in np.unique(time_points_np):
                    mask = time_points_np == time_point
                    if time_point not in time_group_data:
                        time_group_data[time_point] = {
                            'preds': [], 'labels': [], 'probs': [], 'sample_names': []
                        }
                    
                    time_group_data[time_point]['preds'].extend(preds[mask])
                    time_group_data[time_point]['labels'].extend(labels_np[mask])
                    time_group_data[time_point]['probs'].extend(probs[mask])
                    if record_errors:
                        time_group_data[time_point]['sample_names'].extend(np.array(sample_names)[mask])

            # 记录错误预测
            if record_errors:
                for i in range(len(preds)):
                    if preds[i] != labels_np[i]:
                        error_files.append({
                            'file_name': sample_names[i],
                            'true_label': labels_np[i],
                            'pred_label': preds[i],
                            'confidence': torch.softmax(out[i], dim=0).max().item(),
                            'time_point': time_points[i]
                        })

    # 总体指标
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 只在需要时计算AUC
    auc_macro, auc_micro = 0.0, 0.0
    if compute_auc:
        try:
            # 检查实际出现的类别
            unique_labels = np.unique(all_labels)
            num_actual_classes = len(unique_labels)
            logging.info(f"实际出现的类别: {unique_labels}, 类别数量: {num_actual_classes}")
            logging.info(f"各类别样本数量: {np.bincount(all_labels)}")
            
            all_probs_array = np.array(all_probs)
            
            # 二分类情况特殊处理
            if Config.NUM_CLASSES == 2:
                # 对于二分类，直接使用roc_auc_score，不需要label_binarize
                if num_actual_classes == 2:
                    # 两个类别都存在的情况
                    # 使用正类（索引1）的概率计算AUC
                    auc_macro = roc_auc_score(all_labels, all_probs_array[:, 1])
                    auc_micro = auc_macro  # 二分类中macro和micro AUC相同
                elif num_actual_classes == 1:
                    # 只有一个类别存在的情况
                    logging.warning("数据集中只存在一个类别，无法计算AUC")
                else:
                    logging.warning(f"预期二分类但实际有{num_actual_classes}个类别")
            else:
                # 多分类情况
                if num_actual_classes == Config.NUM_CLASSES:
                    # 类别数量匹配
                    y_true_bin = label_binarize(all_labels, classes=range(Config.NUM_CLASSES))
                    auc_macro = roc_auc_score(y_true_bin, all_probs_array, average='macro', multi_class='ovr')
                    auc_micro = roc_auc_score(y_true_bin, all_probs_array, average='micro', multi_class='ovr')
                else:
                    # 类别数量不匹配，只计算存在的类别
                    logging.warning(f"实际类别数({num_actual_classes})与模型输出类别数({Config.NUM_CLASSES})不匹配")
                    if num_actual_classes >= 2:
                        # 只对实际存在的类别计算AUC
                        y_true_bin = label_binarize(all_labels, classes=unique_labels)
                        # 调整概率矩阵，只保留存在的类别
                        adjusted_probs = all_probs_array[:, unique_labels]
                        auc_macro = roc_auc_score(y_true_bin, adjusted_probs, average='macro', multi_class='ovr')
                        auc_micro = roc_auc_score(y_true_bin, adjusted_probs, average='micro', multi_class='ovr')
        except Exception as e:
            logging.error(f"计算AUC时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
    
    result = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

    # 按时间点计算指标
    if group_by_time:
        time_metrics = {}
        for time_point, data in time_group_data.items():
            if len(data['labels']) == 0:
                continue
                
            # 计算该时间点的指标
            time_acc = accuracy_score(data['labels'], data['preds'])
            time_precision = precision_score(data['labels'], data['preds'], average='macro', zero_division=0)
            time_recall = recall_score(data['labels'], data['preds'], average='macro', zero_division=0)
            time_f1 = f1_score(data['labels'], data['preds'], average='macro', zero_division=0)
            
            # 只在需要时计算时间点的AUC
            time_auc_macro, time_auc_micro = 0.0, 0.0
            if compute_auc:
                try:
                    time_labels = np.array(data['labels'])
                    time_probs = np.array(data['probs'])
                    time_unique_labels = np.unique(time_labels)
                    time_num_classes = len(time_unique_labels)
                    
                    if time_num_classes >= 2:
                        if Config.NUM_CLASSES == 2:
                            # 二分类情况
                            if time_num_classes == 2:
                                # 使用正类（索引1）的概率计算AUC
                                time_auc_macro = roc_auc_score(time_labels, time_probs[:, 1])
                                time_auc_micro = time_auc_macro
                        else:
                            # 多分类情况
                            if time_num_classes == Config.NUM_CLASSES:
                                y_true_bin_time = label_binarize(time_labels, classes=range(Config.NUM_CLASSES))
                                time_auc_macro = roc_auc_score(y_true_bin_time, time_probs, average='macro', multi_class='ovr')
                                time_auc_micro = roc_auc_score(y_true_bin_time, time_probs, average='micro', multi_class='ovr')
                            elif time_num_classes >= 2:
                                # 只对存在的类别计算AUC
                                y_true_bin_time = label_binarize(time_labels, classes=time_unique_labels)
                                adjusted_time_probs = time_probs[:, time_unique_labels]
                                time_auc_macro = roc_auc_score(y_true_bin_time, adjusted_time_probs, average='macro', multi_class='ovr')
                                time_auc_micro = roc_auc_score(y_true_bin_time, adjusted_time_probs, average='micro', multi_class='ovr')
                except Exception as e:
                    logging.error(f"时间点 {time_point} AUC计算错误: {e}")
            
            time_metrics[time_point] = {
                '样本数量': len(data['labels']),
                '准确率': time_acc,
                '精确率': time_precision,
                '召回率': time_recall,
                'F1分数': time_f1,
                'AUC_macro': time_auc_macro,
                'AUC_micro': time_auc_micro
            }
        
        result['time_metrics'] = time_metrics
        
        # 保存时间点指标到文件
        if time_metrics_file:
            df_time = pd.DataFrame.from_dict(time_metrics, orient='index')
            df_time.index.name = '时间点(小时)'
            df_time.to_csv(time_metrics_file, encoding='utf-8-sig')
            logging.info(f"各时间点指标已保存至: {time_metrics_file}")

    # 记录错误文件
    if record_errors and error_file_path:
        logging.info(f"预测错误的文件（共{len(error_files)}个）：")
        for error in error_files:
            logging.info(f"时间点{error['time_point']}h: {error['file_name']}, "
                        f"真实标签: {Config.LABEL_TO_CLASS[error['true_label']]}, "
                        f"预测标签: {Config.LABEL_TO_CLASS[error['pred_label']]}, "
                        f"置信度: {error['confidence']:.4f}")
        
        with open(error_file_path, 'w', encoding='utf-8') as f:
            f.write("预测错误的文件列表：\n")
            f.write(f"总错误数: {len(error_files)}\n")
            f.write("时间点,文件名称,真实标签,预测标签,置信度\n")
            for error in error_files:
                f.write(f"{error['time_point']},{error['file_name']},"
                       f"{Config.LABEL_TO_CLASS[error['true_label']]},"
                       f"{Config.LABEL_TO_CLASS[error['pred_label']]},"
                       f"{error['confidence']:.4f}\n")
        
        result['error_files'] = error_files

    return result

# ========== 主训练流程 ==========
def main():
    # 创建结果目录
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    # 设置日志
    logger, log_file = setup_logging(Config.RESULT_DIR)
    logging.info(f"日志文件: {log_file}")
    logging.info(f"结果保存目录: {Config.RESULT_DIR}")

    # 数据增强
    # 为健康样本使用更强的数据增强
    strong_transform = get_strong_augmentation()
    standard_transform = get_standard_augmentation()

    # 构建数据集和加载器
    # 训练集使用过采样
    train_dataset = TimeSeriesDataset(subset="train", transform=standard_transform, oversample=True)
    val_dataset = TimeSeriesDataset(subset="val", transform=None)
    test_dataset = TimeSeriesDataset(subset="test", transform=None)

    # 分析数据集分布
    def analyze_dataset_distribution(train_dataset, val_dataset, test_dataset):
        """分析数据集的正反比例分布"""
        def get_label_distribution(dataset, dataset_name):
            labels = [sample[2] for sample in dataset.samples]
            total_count = len(labels)
            label_0_count = labels.count(0)
            label_1_count = labels.count(1)
            
            label_0_ratio = label_0_count / total_count if total_count > 0 else 0
            label_1_ratio = label_1_count / total_count if total_count > 0 else 0
            
            return {
                '数据集': dataset_name,
                '总样本数': total_count,
                '健康样本数(0)': label_0_count,
                '瘀伤样本数(1)': label_1_count,
                '健康比例': label_0_ratio,
                '瘀伤比例': label_1_ratio
            }
        
        train_dist = get_label_distribution(train_dataset, '训练集')
        val_dist = get_label_distribution(val_dataset, '验证集')
        test_dist = get_label_distribution(test_dataset, '测试集')
        
        # 输出分布信息
        logging.info("\n" + "="*50)
        logging.info("数据集正反比例分析")
        logging.info("="*50)
        
        for dist in [train_dist, val_dist, test_dist]:
            logging.info(f"\n{dist['数据集']}:")
            logging.info(f"  总样本数: {dist['总样本数']}")
            logging.info(f"  健康样本(0): {dist['健康样本数(0)']} ({dist['健康比例']:.2%})")
            logging.info(f"  瘀伤样本(1): {dist['瘀伤样本数(1)']} ({dist['瘀伤比例']:.2%})")
        
        return train_dist, val_dist, test_dist

    train_dist, val_dist, test_dist = analyze_dataset_distribution(train_dataset, val_dataset, test_dataset)

    # 使用平衡数据加载器
    train_loader = BalancedDataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = HSIRGBResNet18(hsi_channels=24).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 早停机制
    best_f1 = -1.0
    counter = 0
    best_model_state = None

    # 记录训练指标
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 
        'test_f1': [], 'test_auc_macro': [], 'test_auc_micro': []
    }

    logging.info(f"开始训练（设备：{Config.DEVICE}，类别数：{Config.NUM_CLASSES}）")
    logging.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")
    logging.info(f"使用类别权重: {Config.CLASS_WEIGHTS}")
    logging.info(f"使用过采样比例: {Config.OVERSAMPLE_RATIO}")
    
    for epoch in range(Config.EPOCHS):
        # 训练（使用类别权重）
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.DEVICE, class_weights=Config.CLASS_WEIGHTS)
        
        # 验证（不计算AUC，不按时间点分组）
        val_metrics = eval_one_epoch(model, val_loader, Config.DEVICE, 
                                   compute_auc=False, record_errors=False, group_by_time=False)
        
        # 测试集评估（计算AUC，不记录错误，不按时间点分组）
        test_metrics = eval_one_epoch(model, test_loader, Config.DEVICE, 
                                    compute_auc=True, record_errors=False, group_by_time=False)
        
        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_f1'].append(val_metrics['f1'])
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['test_acc'].append(test_metrics['accuracy'])
        metrics['test_precision'].append(test_metrics['precision'])
        metrics['test_recall'].append(test_metrics['recall'])
        metrics['test_f1'].append(test_metrics['f1'])
        metrics['test_auc_macro'].append(test_metrics['auc_macro'])
        metrics['test_auc_micro'].append(test_metrics['auc_micro'])

        # 打印epoch信息
        logging.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        logging.info(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.4f}")
        logging.info(f"验证: 损失={val_metrics['loss']:.4f}, 准确率={val_metrics['accuracy']:.4f}, "
                    f"F1={val_metrics['f1']:.4f}")
        logging.info(f"测试: 损失={test_metrics['loss']:.4f}, 准确率={test_metrics['accuracy']:.4f}, "
                    f"精确率={test_metrics['precision']:.4f}, 召回率={test_metrics['recall']:.4f}, "
                    f"F1={test_metrics['f1']:.4f}, AUC_macro={test_metrics['auc_macro']:.4f}, "
                    f"AUC_micro={test_metrics['auc_micro']:.4f}")

        # 学习率调度和早停
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_f1 + Config.MIN_DELTA:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            counter = 0
            logging.info(f"最佳F1更新至 {best_f1:.4f}")
        else:
            counter += 1
            logging.info(f"早停计数: {counter}/{Config.PATIENCE}")
            if counter >= Config.PATIENCE:
                logging.info("早停触发，终止训练")
                break

    # 保存最佳模型
    if best_model_state is not None:
        model_path = os.path.join(Config.RESULT_DIR, "best_model.pth")
        torch.save(best_model_state, model_path)
        logging.info(f"最佳模型已保存至: {model_path}")

    # 测试集最终评估（使用最佳模型，计算AUC，按时间点分组）
    model.load_state_dict(best_model_state)
    error_file_path = os.path.join(Config.RESULT_DIR, "error_predictions.txt")
    time_metrics_file = os.path.join(Config.RESULT_DIR, "time_metrics.csv")
    
    test_metrics = eval_one_epoch(
        model, test_loader, Config.DEVICE, 
        compute_auc=True,  # 只在最终测试时计算AUC
        record_errors=True, 
        error_file_path=error_file_path,
        group_by_time=True,
        time_metrics_file=time_metrics_file
    )
    
    # 打印各时间点指标
    if 'time_metrics' in test_metrics:
        logging.info("\n各时间点详细指标：")
        for time_point, metrics in test_metrics['time_metrics'].items():
            logging.info(f"\n时间点 {time_point}小时:")
            logging.info(f"  样本数量: {metrics['样本数量']}")
            logging.info(f"  准确率: {metrics['准确率']:.4f}")
            logging.info(f"  精确率: {metrics['精确率']:.4f}")
            logging.info(f"  召回率: {metrics['召回率']:.4f}")
            logging.info(f"  F1分数: {metrics['F1分数']:.4f}")
            if metrics['AUC_macro'] > 0:  # 只显示有效的AUC
                logging.info(f"  AUC_macro: {metrics['AUC_macro']:.4f}")
                logging.info(f"  AUC_micro: {metrics['AUC_micro']:.4f}")
    
    logging.info("\n测试集总体指标：")
    logging.info(f"准确率: {test_metrics['accuracy']:.4f}, "
                f"精确率: {test_metrics['precision']:.4f}, "
                f"召回率: {test_metrics['recall']:.4f}, "
                f"F1: {test_metrics['f1']:.4f}")
    
    # 只显示有效的AUC指标
    if test_metrics['auc_macro'] > 0:
        logging.info(f"AUC_macro: {test_metrics['auc_macro']:.4f}, "
                    f"AUC_micro: {test_metrics['auc_micro']:.4f}")
    
    logging.info("训练完成！")
    
if __name__ == '__main__':
    main()