import glob
import os
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.BruiseSFNet import BruiseSTFNet
# 忽略无关警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更友好的CUDA错误提示
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ========== 配置参数（集中管理，便于调整） ==========
class Config:
    DATA_ROOT = "data"  # 根目录：包含train/test/val
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01', '6', '12', '18', '24', '30', '36']  # 8个时间点（四分类_BaseNet）
    TIME_TO_LABEL = {
        '0': 0,  # 0h → 标签0（健康）
        '01': 0,  # 0.1h → 标签0
        '6': 1,  # 6h → 标签1（瘀伤早期）
        '12': 1,  # 12h → 标签1（瘀伤早期）
        '18': 1,  # 18h → 标签2（瘀伤中期）
        '24': 1,  # 24h → 标签2（瘀伤中期）
        '30': 1,  # 30h → 标签3（瘀伤后期）
        '36': 1  # 36h → 标签3（瘀伤后期）
    }
    IMG_SIZE = (64,64)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2  # 四分类_BaseNet
    MSI_CHANNELS = 24  # 根据实际高光谱数据通道数设置（关键！）
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 早停参数
    PATIENCE = 20
    MIN_DELTA = 0.001
    # 结果保存
    RESULT_DIR = "results/四分类_B/"
    # 错误样本保存路径
    ERROR_SAMPLES_DIR = os.path.join(RESULT_DIR, "error_samples")  # 新增：错误样本记录目录
    # 随机种子
    SEED = 71
    # 新增OverviewNet相关配置
    OVERVIEW_HIDDEN_DIMS = [64, 128]
    CONTEXT_PRIOR_DIM = 64
    USE_AFFINITY_LOSS = True
    AFFINITY_TEMPERATURE = 0.1
    # 损失权重
    ALPHA = 0.7  # 增强分类损失权重
    GAMMA = 0.1  # 亲和度损失权重
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
            # 转换为张量并调整尺寸
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
        msi_path, rgb_path, label = self.samples[idx]
        msi = self._read_msi(msi_path)
        rgb = self._read_rgb(rgb_path)
        return msi, rgb, torch.tensor(label, dtype=torch.long),idx
# ========== 健壮性改进的训练和验证函数 ==========
def train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    successful_batches = 0
    nan_batches = 0
    for msi, rgb, labels, _ in tqdm(loader, desc="训练"):
        try:
            msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)

            # 检查输入数据是否有NaN
            if torch.isnan(msi).any() or torch.isnan(rgb).any():
                print("Warning: NaN detected in input data, skipping batch")
                nan_batches += 1
                continue

            optimizer.zero_grad()

            # 【核心修改1】适配原始 BruiseSTFNet 前向传播输出
            # 原始模型输出：(分类logits, 特征字典)
            logits, _ = model(msi, rgb)

            # 检查模型输出是否有NaN
            if torch.isnan(logits).any():
                print("Warning: NaN in model outputs, using uniform distribution")
                batch_size, num_classes = logits.shape
                logits = torch.ones(batch_size, num_classes, device=device) / num_classes

            # 【核心修改2】简化损失计算：仅使用交叉熵损失（无亲和度损失）
            loss = F.cross_entropy(logits, labels)

            # 检查损失是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Invalid loss detected, skipping batch")
                nan_batches += 1


            # 反向传播
            loss.backward()

            # 检查梯度是否为NaN
            grad_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"Warning: Invalid gradients in {name}, skipping update")
                    grad_nan = True
                    break

            if grad_nan:
                nan_batches += 1
                optimizer.zero_grad()
                continue

            # 梯度裁剪防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # 统计
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            successful_batches += 1

        except Exception as e:
            print(f"Exception in training batch: {e}, continuing...")
            nan_batches += 1
            continue
    if nan_batches > 0:
        print(f"Training: {nan_batches} batches had issues, {successful_batches} successful")
    avg_loss = total_loss / total if total > 0 else float('inf')
    avg_acc = total_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc
def eval_one_epoch(model, loader, dataset,device,save_errors=True):
    """
       评估模型并记录错误样本
       :param model: 模型
       :param loader: 数据加载器
       :param dataset: 对应的数据集（用于获取样本路径）
       :param device: 设备
       :param save_errors: 是否保存错误样本信息
       :return: 评估指标 + 错误样本列表（若save_errors=True）
       """
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    error_samples = []# 存储错误样本信息：(msi_path, rgb_path, true_label, pred_label)
    nan_count = 0
    total_batches = 0

    with torch.no_grad():
        for msi, rgb, labels,indices in loader:
            total_batches += 1
            try:
                msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)

                # 检查输入数据
                if torch.isnan(msi).any() or torch.isnan(rgb).any():
                    print("Warning: NaN in input data during evaluation, skipping batch")
                    nan_count += 1
                    continue

                # 新网络的前向传播（测试时不传入labels）
                enhanced_logits,aux_features = model(msi, rgb)

                # 检查输出并处理NaN
                if torch.isnan(enhanced_logits).any():
                    nan_count += 1
                    print(f"Warning: NaN detected in evaluation batch {total_batches}, using uniform distribution")
                    batch_size, num_classes = enhanced_logits.shape
                    enhanced_logits = torch.ones(batch_size, num_classes, device=device) / num_classes

                # 使用增强分类器的输出计算损失
                loss = F.cross_entropy(enhanced_logits, labels)

                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"Warning: Invalid loss in evaluation batch {total_batches}, skipping")
                    continue

                total_loss += loss.item() * labels.size(0)

                # 计算概率并再次检查
                probs = F.softmax(enhanced_logits, dim=1)
                if torch.isnan(probs).any():
                    nan_count += 1
                    print(f"Warning: NaN after softmax in batch {total_batches}, using uniform distribution")
                    batch_size, num_classes = enhanced_logits.shape
                    probs = torch.ones(batch_size, num_classes, device=device) / num_classes

                # 存储预测结果
                preds = enhanced_logits.argmax(1).cpu().numpy()
                probs_np = probs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # 记录错误样本（如果需要保存）
                if save_errors:
                    for idx, true_label, pred_label in zip(indices, labels_np, preds):
                        if true_label != pred_label:
                            # 从数据集中获取样本路径
                            msi_path, rgb_path, _ = dataset.samples[idx]
                            error_samples.append({
                                "msi_path": msi_path,
                                "rgb_path": rgb_path,
                                "true_label": true_label,
                                "pred_label": pred_label
                            })

                all_preds.extend(preds)
                all_labels.extend(labels_np)
                all_probs.extend(probs_np)
                total += labels.size(0)

            except Exception as e:
                nan_count += 1
                print(f"Error in evaluation batch {total_batches}: {e}, skipping batch")
                continue

    # 报告NaN情况但不中断
    if nan_count > 0:
        print(f"Evaluation: {nan_count}/{total_batches} batches had NaN issues, but continuing...")

    # 转换为numpy数组并最终清理
    all_probs_array = np.array(all_probs)
    all_labels_array = np.array(all_labels)

    # 安全处理NaN值，用0替换NAN值
    all_probs_array = np.nan_to_num(all_probs_array, nan=0.0, posinf=1.0, neginf=0.0)

    # 确保概率和为1
    row_sums = all_probs_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # 避免除零
    all_probs_array = all_probs_array / row_sums

    # 1. 计算混淆矩阵（用于TP/FP/FN统计）
    cm = confusion_matrix(all_labels_array, all_preds, labels=range(Config.NUM_CLASSES))
    TP = cm.diagonal()  # 各类别TP
    FP = cm.sum(axis=0) - TP  # 各类别FP
    FN = cm.sum(axis=1) - TP  # 各类别FN

    # 2. 图片定义的ACC：sum(TPk) / sum(TPk+FPk+FNk)
    sum_TP = TP.sum()
    sum_denominator = (TP + FP + FN).sum()
    acc = sum_TP / sum_denominator if sum_denominator > 0 else 0.0

    # 3. Micro平均指标
    precision_micro = precision_score(all_labels_array, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels_array, all_preds, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels_array, all_preds, average='micro', zero_division=0)

    # 4. 安全地计算AUC-ROC
    auc = 0.0
    try:
        if len(np.unique(all_labels_array)) > 1 and not np.isnan(all_probs_array).any():
            auc = roc_auc_score(
                all_labels_array, all_probs_array,
                multi_class='ovr',  # 每个类别视为正类，其他为负类
                average='macro',  # 各类别AUC取平均
            )
        else:
            print("Warning: Could not calculate AUC due to single class or NaN values")
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}, setting to 0")

    avg_loss = total_loss / total if total > 0 else float('inf')
    result = {
        'loss': avg_loss,
        'accuracy': acc,  # 替换为图片定义的ACC
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'auc_roc': auc,
        'nan_batches': nan_count,
        'total_batches': total_batches
    }
    if save_errors:
        return result, error_samples
    return result
# 保存错误样本信息到文件
def save_error_samples(error_samples, subset_name):
    """保存错误样本信息到文本文件"""
    os.makedirs(Config.ERROR_SAMPLES_DIR, exist_ok=True)
    save_path = os.path.join(Config.ERROR_SAMPLES_DIR, f"{subset_name}_errors.txt")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"错误样本总数: {len(error_samples)}\n")
        f.write("=" * 50 + "\n")
        for i, sample in enumerate(error_samples, 1):
            f.write(f"样本 {i}:\n")
            f.write(f"  MSI路径: {sample['msi_path']}\n")
            f.write(f"  RGB路径: {sample['rgb_path']}\n")
            f.write(f"  真实标签: {sample['true_label']}\n")
            f.write(f"  预测标签: {sample['pred_label']}\n")
            f.write("-" * 50 + "\n")

    print(f"错误样本信息已保存至: {save_path}")
# 检查数据分布
def analyze_data_distribution():
    """data:
        训练集样本数: 755
        验证集样本数: 168
        测试集样本数: 170
        训练集类别分布: [ 73 205 241 236]
        验证集类别分布: [18 50 53 47]
        data0:
        训练集样本数: 1008
        验证集样本数: 216
        测试集样本数: 222
        训练集类别分布: [ 92 381 274 261]
        验证集类别分布: [20 82 58 56]
        差距353个样本
    """
    train_dataset = TimeSeriesDataset(subset="train")
    val_dataset = TimeSeriesDataset(subset="val")
    test_dataset = TimeSeriesDataset(subset="test")

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 检查类别分布
    train_labels = [sample[2] for sample in train_dataset.samples]
    val_labels = [sample[2] for sample in val_dataset.samples]

    print(f"训练集类别分布: {np.bincount(train_labels)}")
    print(f"验证集类别分布: {np.bincount(val_labels)}")
# ========== 主训练流程（优化逻辑和可视化） ==========
def main():
    analyze_data_distribution()
    # 创建结果目录
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    # 数据增强（仅训练集）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
    ])

    # 构建数据集和加载器（移除重复初始化）
    train_dataset = TimeSeriesDataset(subset="train", transform=train_transform)
    val_dataset = TimeSeriesDataset(subset="val", transform=None)
    test_dataset = TimeSeriesDataset(subset="test", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型、优化器和调度器
    model = BruiseSTFNet(
        msi_channels=Config.MSI_CHANNELS,
        rgb_channels=Config.RGB_CHANNELS,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 使用更稳定的优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-5,
        eps=1e-8  # 增加数值稳定性
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控F1提升
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 早停机制变量
    best_f1 = -1.0
    counter = 0
    nan_epochs = 0  # 记录连续出现NaN的epoch数
    best_model_state = None
    best_epoch = -1  # 新增：记录最佳epoch
    best_test_metrics = None  # 新增：记录最佳模型在测试集上的指标

    # 记录训练指标
    metrics = {
        'train_loss': [], 'train_acc': [],  # 训练集ACC保留原逻辑，或按需替换
        'val_loss': [], 'val_acc': [], 'val_precision_micro': [],
        'val_recall_micro': [], 'val_f1_micro': [], 'val_auc_roc': [],
        'test_loss': [], 'test_acc': [], 'test_precision_micro': [],
        'test_recall_micro': [], 'test_f1_micro': [], 'test_auc_roc': [],
        'nan_batches_train': [], 'nan_batches_val': [], 'nan_batches_test': []
    }

    print(f"开始训练（设备：{Config.DEVICE}，类别数：{Config.NUM_CLASSES}）")
    print(f"使用网络：BruiseSTFNet_B")
    print(f"亲和度损失：{'启用' if Config.USE_AFFINITY_LOSS else '禁用'}")

    for epoch in range(Config.EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        # 测试并保存错误样本
        test_metrics, test_errors = eval_one_epoch(
            model, test_loader, test_dataset, Config.DEVICE, save_errors=True
        )
        # 保存测试集错误样本
        save_error_samples(test_errors, "test")
        # 验证集错误样本（如需分析）
        val_metrics, val_errors = eval_one_epoch(
            model, val_loader, val_dataset, Config.DEVICE, save_errors=True
        )
        save_error_samples(val_errors, "val")
        # 记录NaN情况
        metrics['nan_batches_val'].append(val_metrics['nan_batches'])
        metrics['nan_batches_test'].append(test_metrics['nan_batches'])

        # 在每个epoch的验证之后添加测试集评估
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_precision_micro'].append(val_metrics['precision_micro'])
        metrics['val_recall_micro'].append(val_metrics['recall_micro'])
        metrics['val_f1_micro'].append(val_metrics['f1_micro'])
        metrics['val_auc_roc'].append(val_metrics['auc_roc'])
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['test_acc'].append(test_metrics['accuracy'])
        metrics['test_precision_micro'].append(test_metrics['precision_micro'])
        metrics['test_recall_micro'].append(test_metrics['recall_micro'])
        metrics['test_f1_micro'].append(test_metrics['f1_micro'])
        metrics['test_auc_roc'].append(test_metrics['auc_roc'])

        # 打印epoch信息（扩展指标显示）
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print(f"训练：损失={train_loss:.4f}，准确率={train_acc:.4f}")
        print(f"验证：损失={val_metrics['loss']:.4f}，ACC={val_metrics['accuracy']:.4f}，"
              f"Micro-P={val_metrics['precision_micro']:.4f}，Micro-R={val_metrics['recall_micro']:.4f}，"
              f"Micro-F1={val_metrics['f1_micro']:.4f}，AUC-ROC={val_metrics['auc_roc']:.4f}，"
              f"NaN批次={val_metrics['nan_batches']}/{val_metrics['total_batches']}")
        print(f"测试：损失={test_metrics['loss']:.4f}，ACC={test_metrics['accuracy']:.4f}，"
              f"Micro-P={test_metrics['precision_micro']:.4f}，Micro-R={test_metrics['recall_micro']:.4f}，"
              f"Micro-F1={test_metrics['f1_micro']:.4f}，AUC-ROC={test_metrics['auc_roc']:.4f}，"
              f"NaN批次={test_metrics['nan_batches']}/{test_metrics['total_batches']}")

        # 学习率调度
        scheduler.step(val_metrics['f1_micro'])

        # 如果这个epoch有NaN问题，记录但不停止
        if val_metrics['nan_batches'] > 0 or test_metrics['nan_batches'] > 0:
            nan_epochs += 1
            print(f"Warning: {nan_epochs} consecutive epochs with NaN issues")

            # 如果连续多个epoch都有NaN，考虑调整学习率
            if nan_epochs >= 3:
                print("Multiple NaN epochs, adjusting learning rate...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                nan_epochs = 0
        else:
            nan_epochs = 0  # 重置计数器

        # 早停逻辑（监控验证集F1）
        if val_metrics['f1_micro'] > best_f1 + Config.MIN_DELTA:
            best_f1 = val_metrics['f1_micro']
            best_model_state = model.state_dict().copy()  # 使用copy()确保状态不被后续训练影响
            best_epoch = epoch + 1  # 记录最佳epoch
            best_test_metrics = test_metrics.copy()  # 记录最佳模型对应的测试集指标
            counter = 0
            print(f"最佳F1更新至 {best_f1:.4f}，保存模型")
        else:
            counter += 1
            print(f"早停计数：{counter}/{Config.PATIENCE}")
            if counter >= Config.PATIENCE:
                print("早停触发，终止训练")
                break

    # 保存最佳模型
    if best_model_state is not None:
        model_path = os.path.join(Config.RESULT_DIR, "best_model.pth")
        torch.save(best_model_state, model_path)
        print(f"最佳模型已保存至：{model_path}")

    # 输出最佳模型在测试集上的指标
    if best_test_metrics is not None:
        best_test_log = (
            f"\n🎯 最佳模型测试集指标（第{best_epoch}轮）:\n"
            f"ACC：{best_test_metrics['accuracy']:.4f}，"
            f"Micro-P：{best_test_metrics['precision_micro']:.4f}，"
            f"Micro-R：{best_test_metrics['recall_micro']:.4f}，"
            f"Micro-F1：{best_test_metrics['f1_micro']:.4f}，"
            f"AUC-ROC：{best_test_metrics['auc_roc']:.4f}"
        )
        print(best_test_log)
    else:
        print("警告：未找到最佳模型，使用最终模型进行评估")
        model.load_state_dict(best_model_state)
        best_test_metrics = eval_one_epoch(model, test_loader, test_dataset,Config.DEVICE,save_errors=False)
        best_test_log = (
            f"\n🎯 最终模型测试集指标:\n"
            f"ACC：{best_test_metrics['accuracy']:.4f}，"
            f"Micro-P：{best_test_metrics['precision_micro']:.4f}，"
            f"Micro-R：{best_test_metrics['recall_micro']:.4f}，"
            f"Micro-F1：{best_test_metrics['f1_micro']:.4f}，"
            f"AUC-ROC：{best_test_metrics['auc_roc']:.4f}"
        )
        print(best_test_log)

    # 可视化训练曲线（优化中文显示，新增测试集曲线）
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 新增AUC-ROC曲线子图
    plt.figure(figsize=(18, 6))

    # 原有损失曲线（保持不变）
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label="训练损失", linewidth=2)
    plt.plot(metrics['val_loss'], label="验证损失", linewidth=2)
    plt.plot(metrics['test_loss'], label="测试损失", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("损失值", fontsize=10)
    plt.title("损失曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    # 原有准确率曲线（保持不变）
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_acc'], label="训练准确率", linewidth=2)
    plt.plot(metrics['val_acc'], label="验证准确率", linewidth=2)
    plt.plot(metrics['test_acc'], label="测试准确率", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("准确率", fontsize=10)
    plt.title("准确率曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    # 新增AUC-ROC曲线
    plt.subplot(1, 3, 3)
    plt.plot(metrics['val_auc_roc'], label="验证AUC-ROC", linewidth=2)
    plt.plot(metrics['test_auc_roc'], label="测试AUC-ROC", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("AUC-ROC", fontsize=10)
    plt.title("AUC-ROC曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_DIR, "training_test_curves_full.png"), dpi=300)
    plt.show()
if __name__ == "__main__":
    main()