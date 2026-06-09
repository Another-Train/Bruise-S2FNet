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
from models.GMA_changed_23 import BruiseSTFNet  # 确保导入原始模型类名正确
# 忽略无关警告d d
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ========== 配置参数（移除 CPNet/亲和度损失相关配置） ==========
class Config:
    DATA_ROOT = "data"
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0',
                   '01',
                   '6',
                   '12',
                   '18',
                   '24',
                   '30',
                   '36'
                   ]
    TIME_TO_LABEL = {
        '0': 0,
        '01': 0,
        '6': 1,
        '12': 1,
        '18': 2,
        '24': 2,
        '30': 3,
        '36': 3
    }
    IMG_SIZE = (64,64)
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    MSI_CHANNELS = 23 # 在_read_msi()函数中删除了异常波段898.7nm,所以是24-1个波段
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 10
    MIN_DELTA = 0.001
    RESULT_DIR = "results/四分类_GMA/"
    ERROR_SAMPLES_DIR = os.path.join(RESULT_DIR, "error_samples")
    SEED = 42

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


# ========== 增强的数据增强类 ==========
class MultiModalTransform:
    """多模态数据增强：确保RGB和MSI应用相同的几何变换"""

    def __init__(self, rgb_transform=None, msi_transform=None, apply_geometric=True):
        self.rgb_transform = rgb_transform
        self.msi_transform = msi_transform
        self.apply_geometric = apply_geometric

        # 几何变换参数（RGB和MSI共享）
        self.geometric_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),#随机垂直反转
            transforms.RandomRotation(degrees=15),#随机旋转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),#随机仿射变换
        ]) if apply_geometric else None

        # 弹性变换参数
        self.elastic_transform = None
        if apply_geometric:
            try:
                from torchvision.transforms import ElasticTransform
                self.elastic_transform = ElasticTransform(alpha=50.0, sigma=5.0)
            except ImportError:
                print("ElasticTransform not available, skipping")

    def __call__(self, rgb_img, msi_tensor):
        # 确保输入是PIL图像和tensor
        if not isinstance(rgb_img, Image.Image):
            rgb_img = transforms.ToPILImage()(rgb_img)

        # 保存随机状态以确保几何变换一致性
        if self.apply_geometric and self.geometric_transforms:
            # 生成随机参数
            state = torch.get_rng_state()

            # 应用相同的几何变换到RGB和MSI
            rgb_img = self.geometric_transforms(rgb_img)

            # 对MSI应用相同的几何变换（需要转换为PIL再转换回来）
            msi_pil = []
            for i in range(msi_tensor.shape[0]):
                channel = msi_tensor[i].numpy()
                channel_pil = Image.fromarray((channel * 255).astype(np.uint8))
                channel_pil = self.geometric_transforms(channel_pil)
                msi_pil.append(np.array(channel_pil) / 255.0)

            msi_tensor = torch.tensor(np.stack(msi_pil), dtype=torch.float32)

            # 恢复随机状态
            torch.set_rng_state(state)

        # 应用弹性变换（如果可用）
        if self.elastic_transform and random.random() < 0.3:
            # 对RGB应用弹性变换
            rgb_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)
            rgb_tensor = self.elastic_transform(rgb_tensor).squeeze(0)
            rgb_img = transforms.ToPILImage()(rgb_tensor)

            # 对MSI应用弹性变换
            msi_tensor = self.elastic_transform(msi_tensor.unsqueeze(0)).squeeze(0)

        # 应用模态特定的变换
        if self.rgb_transform:
            rgb_img = self.rgb_transform(rgb_img)

        if self.msi_transform:
            msi_tensor = self.msi_transform(msi_tensor)

        return rgb_img, msi_tensor

# ========== 数据集类（增强数据增强） ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, subset="train", transform=None):
        self.subset = subset
        self.transform = transform  # 现在是一个MultiModalTransform实例
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"{subset}集未找到有效样本，请检查数据路径或格式")

    def _collect_samples(self):
        samples = []
        subset_dir = os.path.join(Config.DATA_ROOT, self.subset)
        print(f"当前搜索的子集目录：{subset_dir}")

        for time in Config.TIME_POINTS:
            label = Config.TIME_TO_LABEL[time]
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
                if "jjk" in prefix.lower():
                    final_label = 0
                else:
                    final_label = label

                msi_path = os.path.join(msi_time_dir, f"{prefix}.hdr")
                rgb_path = os.path.join(rgb_time_dir, f"{prefix}.png")
                samples.append((msi_path, rgb_path, final_label))

        print(f"\n{self.subset}集最终收集到{len(samples)}个样本")
        return samples

    def _read_msi(self, msi_hdr_path):
        try:
            msi_data_path = f"{os.path.splitext(msi_hdr_path)[0]}.float"
            from spectral.io import envi
            img = envi.open(msi_hdr_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))

            # 修改:删除索引为18的异常波段（898.7）
            if msi_data.shape[0] > 18:  # 确保索引有效
                msi_data = np.delete(msi_data, 18, axis=0)

            # 增强的MSI预处理
            msi_min, msi_max = msi_data.min(), msi_data.max()
            if msi_max - msi_min < 1e-8:
                msi_data = np.zeros_like(msi_data)
            else:
                # 使用更稳健的归一化
                msi_data = (msi_data - msi_min) / (msi_max - msi_min)
                # 添加对比度增强
                msi_data = np.power(msi_data, 0.8)  # Gamma校正

            msi_tensor = torch.tensor(msi_data, dtype=torch.float32)
            msi_resized = F.interpolate(
                msi_tensor.unsqueeze(0),
                size=Config.IMG_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            return msi_resized
        except Exception as e:
            raise RuntimeError(f"读取MSI文件失败：{msi_hdr_path}，错误：{str(e)}")

    def _read_rgb(self, rgb_path):
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")

            # 增强的RGB预处理
            if self.transform:
                # 如果是训练集且有变换，先应用模态特定的变换
                if isinstance(self.transform, MultiModalTransform):
                    # 创建一个临时的MSI tensor用于几何变换一致性
                    dummy_msi = torch.zeros((24, *Config.IMG_SIZE))
                    rgb_img, _ = self.transform(rgb_img, dummy_msi)
                else:
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
        msi_path, rgb_path, label = self.samples[idx]

        # 读取MSI数据
        msi = self._read_msi(msi_path)

        # 读取RGB数据
        rgb = self._read_rgb(rgb_path)

        # 应用多模态数据增强（确保几何变换一致性）
        if self.transform and isinstance(self.transform, MultiModalTransform) and self.subset == "train":
            # 将RGB转换回PIL进行几何变换
            rgb_pil = transforms.ToPILImage()(rgb)
            rgb_pil, msi = self.transform(rgb_pil, msi)
            rgb = transforms.ToTensor()(rgb_pil)

        return msi, rgb, torch.tensor(label, dtype=torch.long), idx

# ========== 训练和验证函数（核心修改：损失函数适配） ==========
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
            logits, _ = model(msi, rgb)

            # 检查模型输出是否有NaN
            if torch.isnan(logits).any():
                print("Warning: NaN in model outputs, using uniform distribution")
                batch_size, num_classes = logits.shape
                logits = torch.ones(batch_size, num_classes, device=device) / num_classes

            loss = F.cross_entropy(logits, labels)

            # 检查损失是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Invalid loss detected, skipping batch")
                nan_batches += 1
                continue

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
                enhanced_logits, aux_features = model(msi, rgb)

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

# ========== 其他辅助函数（保持不变） ==========
def save_error_samples(error_samples, subset_name):
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

def analyze_data_distribution():
    train_dataset = TimeSeriesDataset(subset="train")
    val_dataset = TimeSeriesDataset(subset="val")
    test_dataset = TimeSeriesDataset(subset="test")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    train_labels = [sample[2] for sample in train_dataset.samples]
    val_labels = [sample[2] for sample in val_dataset.samples]
    print(f"训练集类别分布: {np.bincount(train_labels)}")
    print(f"验证集类别分布: {np.bincount(val_labels)}")
# MSI特定的变换（光谱增强）
def msi_specific_transform(msi_tensor):
    # 随机通道丢弃（模拟光谱噪声）
    if random.random() < 0.2:
        channels_to_drop = random.randint(1, 3)
        channels = random.sample(range(msi_tensor.shape[0]), channels_to_drop)
        for channel in channels:
            msi_tensor[channel] = 0

    # 随机光谱偏移
    if random.random() < 0.3:
        shift_amount = random.uniform(-0.1, 0.1)
        msi_tensor = msi_tensor + shift_amount
        msi_tensor = torch.clamp(msi_tensor, 0, 1)
    return msi_tensor
# ========== 主训练流程（修改模型初始化） ==========
def main():
    analyze_data_distribution()
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    # # 增强的数据增强策略
    # rgb_specific_transforms = transforms.Compose([
    #     transforms.ColorJitter(
    #         brightness=0.3,  # 增加亮度扰动
    #         contrast=0.3,  # 增加对比度扰动
    #         saturation=0.3,  # 增加饱和度扰动
    #         hue=0.1  # 轻微色调变化
    #     ),
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    #     transforms.RandomAutocontrast(p=0.3),
    #     transforms.RandomEqualize(p=0.3),
    # ])



    # 创建多模态变换
    multimodal_transform = MultiModalTransform(
        # rgb_transform=rgb_specific_transforms,
        # msi_transform=msi_specific_transform,
        apply_geometric=True  # 启用几何变换
    )

    # 构建数据集和加载器（保持不变）
    train_dataset = TimeSeriesDataset(subset="train", transform=multimodal_transform)
    val_dataset = TimeSeriesDataset(subset="val", transform=None)
    test_dataset = TimeSeriesDataset(subset="test", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True # 新增：丢弃最后一个不足批次的样本
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


    model = BruiseSTFNet(
        msi_channels=Config.MSI_CHANNELS,
        rgb_channels=Config.RGB_CHANNELS,
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)

    # 打印模型参数数量（保持不变）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 优化器和调度器（保持不变）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-5,
        eps=1e-8
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,# 增加耐心值
        verbose=True
    )
    # 早停机制变量（基于验证集准确率）
    best_val_acc = -1.0  # 跟踪最佳验证集准确率
    counter = 0
    best_model_state = None
    best_epoch = -1
    best_val_metrics = None

    # 记录训练指标（保持不变）
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision_micro': [],
        'val_recall_micro': [], 'val_f1_micro': [], 'val_auc_roc': [],
        'test_loss': [], 'test_acc': [], 'test_precision_micro': [],
        'test_recall_micro': [], 'test_f1_micro': [], 'test_auc_roc': [],
        'nan_batches_train': [], 'nan_batches_val': [], 'nan_batches_test': []
    }

    print(f"开始训练（设备：{Config.DEVICE}，类别数：{Config.NUM_CLASSES}）")
    print(f"使用网络：BruiseSTFNet")

    for epoch in range(Config.EPOCHS):
        # 训练（使用修改后的train_one_epoch）
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        # 测试
        test_metrics, test_errors = eval_one_epoch(
            model, test_loader, test_dataset, Config.DEVICE, save_errors=True
        )
        save_error_samples(test_errors, "test")
        # 验证
        val_metrics, val_errors = eval_one_epoch(
            model, val_loader, val_dataset, Config.DEVICE, save_errors=True
        )
        save_error_samples(val_errors, "val")

        # 记录指标（保持不变）
        metrics['nan_batches_val'].append(val_metrics['nan_batches'])
        metrics['nan_batches_test'].append(test_metrics['nan_batches'])
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

        # 打印epoch信息（保持不变）
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

        # 学习率调度（保持不变）
        scheduler.step(val_metrics['f1_micro'])

        # NaN处理（保持不变）
        if val_metrics['nan_batches'] > 0 or test_metrics['nan_batches'] > 0:
            nan_epochs += 1
            print(f"Warning: {nan_epochs} consecutive epochs with NaN issues")
            if nan_epochs >= 3:
                print("Multiple NaN epochs, adjusting learning rate...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                nan_epochs = 0
        else:
            nan_epochs = 0

        # 早停逻辑（核心：基于验证集准确率）
        if val_metrics['accuracy'] > best_val_acc + Config.MIN_DELTA:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            best_val_metrics = val_metrics.copy()
            counter = 0
            print(f"最佳验证集准确率更新至 {best_val_acc:.4f}，保存模型")
        else:
            counter += 1
            print(f"早停计数：{counter}/{Config.PATIENCE}")
            if counter >= Config.PATIENCE:
                print(f"早停触发，终止训练（最佳epoch为 {best_epoch}）")
                break

    # 保存最佳模型（保持不变）
    if best_model_state is not None:
        model_path = os.path.join(Config.RESULT_DIR, "best_model.pth")
        torch.save(best_model_state, model_path)
        print(f"最佳模型已保存至：{model_path}")

    # 输出最佳指标（保持不变）
    if best_val_metrics is not None:
        best_test_log = (
            f"\n🎯 最佳模型测试集指标（第{best_epoch}轮）:\n"
            f"ACC：{best_val_metrics['accuracy']:.4f}，"
            f"Micro-P：{best_val_metrics['precision_micro']:.4f}，"
            f"Micro-R：{best_val_metrics['recall_micro']:.4f}，"
            f"Micro-F1：{best_val_metrics['f1_micro']:.4f}，"
            f"AUC-ROC：{best_val_metrics['auc_roc']:.4f}"
        )
        print(best_test_log)
    else:
        print("警告：未找到最佳模型，使用最终模型进行评估")
        best_val_metrics = eval_one_epoch(model, test_loader, test_dataset, Config.DEVICE, save_errors=False)
        best_test_log = (
            f"\n🎯 最终模型测试集指标:\n"
            f"ACC：{best_val_metrics['accuracy']:.4f}，"
            f"Micro-P：{best_val_metrics['precision_micro']:.4f}，"
            f"Micro-R：{best_val_metrics['recall_micro']:.4f}，"
            f"Micro-F1：{best_val_metrics['f1_micro']:.4f}，"
            f"AUC-ROC：{best_val_metrics['auc_roc']:.4f}"
        )
        print(best_test_log)

    # 可视化训练曲线（保持不变）
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label="训练损失", linewidth=2)
    plt.plot(metrics['val_loss'], label="验证损失", linewidth=2)
    plt.plot(metrics['test_loss'], label="测试损失", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("损失值", fontsize=10)
    plt.title("损失曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_acc'], label="训练准确率", linewidth=2)
    plt.plot(metrics['val_acc'], label="验证准确率", linewidth=2)
    plt.plot(metrics['test_acc'], label="测试准确率", linewidth=2, linestyle='--')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("准确率", fontsize=10)
    plt.title("准确率曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
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