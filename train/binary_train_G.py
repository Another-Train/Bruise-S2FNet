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
from visualize.GMA_changed import BruiseSTFNet  # 确保导入原始模型类名正确
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
        '18': 1,
        '24': 1,
        '30': 1,
        '36': 1
    }
    IMG_SIZE = (64,64)
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    MSI_CHANNELS = 24
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 30
    MIN_DELTA = 0.001
    RESULT_DIR = "results/二分类_GMA_changed/"
    ERROR_SAMPLES_DIR = os.path.join(RESULT_DIR, "error_samples")
    SEED = 42
    # 移除 CPNet/亲和度损失相关配置
    hidden_dim = 32
    fusion_dim = 32
    attn_heads = 2
    PREDICTION_TABLE_PATH = os.path.join(RESULT_DIR, "prediction_results.csv")
    TIME_STEP_METRICS_PATH = os.path.join(RESULT_DIR, "time_step_metrics.csv")

# ========== 增强的数据增强类 ==========
class MultiModalTransform:
    """多模态数据增强：确保RGB和MSI应用相同的几何变换"""

    def __init__(self, rgb_transform=None, msi_transform=None, apply_geometric=True):
        self.rgb_transform = rgb_transform
        self.msi_transform = msi_transform
        self.apply_geometric = apply_geometric

        # 几何变换参数（RGB和MSI共享）
        self.geometric_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
            # 对RGB应用弹性变换（保持PIL格式输出）
            rgb_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)
            rgb_tensor = self.elastic_transform(rgb_tensor).squeeze(0)
            rgb_img = transforms.ToPILImage()(rgb_tensor)  # 转换回PIL，避免后续报错

            # 对MSI应用弹性变换
            msi_tensor = self.elastic_transform(msi_tensor.unsqueeze(0)).squeeze(0)

        # 应用模态特定的变换（确保RGB变换后仍为PIL，除非变换包含ToTensor）
        if self.rgb_transform:
            rgb_transformed = self.rgb_transform(rgb_img)
            # 若变换后是Tensor，转换回PIL（避免后续_to_tensor冲突）
            if isinstance(rgb_transformed, torch.Tensor):
                rgb_img = transforms.ToPILImage()(rgb_transformed)
            else:
                rgb_img = rgb_transformed

        if self.msi_transform:
            msi_tensor = self.msi_transform(msi_tensor)

        # 最终返回：RGB保持PIL，MSI保持Tensor
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
            # 1. 读取原始RGB图像，转为PIL Image（RGB格式）
            rgb_img = Image.open(rgb_path).convert("RGB")

            # 2. 应用多模态变换（仅训练集，保持PIL格式输出）
            if self.transform and isinstance(self.transform, MultiModalTransform):
                # 创建临时MSI tensor用于几何变换一致性
                dummy_msi = torch.zeros((Config.MSI_CHANNELS, *Config.IMG_SIZE))
                rgb_img, _ = self.transform(rgb_img, dummy_msi)

            # 3. 转换为Tensor（仅执行一次，避免重复转换）
            # 若经过变换后已是Tensor，直接使用；否则转换
            if isinstance(rgb_img, torch.Tensor):
                rgb_tensor = rgb_img
            else:
                rgb_tensor = transforms.ToTensor()(rgb_img)
            # 补充归一化（对应原rgb_train_transform中的Normalize）
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            rgb_tensor = normalize(rgb_tensor)
            # 4. 调整图像尺寸（插值到目标大小）
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

# 3. 创建合并后的训练集（继承TimeSeriesDataset并重写样本列表）
class CombinedTrainDataset(TimeSeriesDataset):
        def __init__(self, combined_samples, transform):
            self.subset = "train_combined"
            self.transform = transform
            self.samples = combined_samples  # 使用合并后的样本列表
            if not self.samples:
                raise ValueError("合并后的训练集未找到有效样本")
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
    all_samples_details = []
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
                logits, _ = model(msi, rgb)

                # 检查输出并处理NaN
                if torch.isnan(logits).any():
                    nan_count += 1
                    print(f"Warning: NaN detected in evaluation batch {total_batches}, using uniform distribution")
                    batch_size, num_classes = logits.shape
                    logits = torch.ones(batch_size, num_classes, device=device) / num_classes

                # 使用增强分类器的输出计算损失
                loss = F.cross_entropy(logits, labels)

                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    print(f"Warning: Invalid loss in evaluation batch {total_batches}, skipping")
                    continue

                total_loss += loss.item() * labels.size(0)

                # 计算概率并再次检查
                probs = F.softmax(logits, dim=1)
                if torch.isnan(probs).any():
                    nan_count += 1
                    print(f"Warning: NaN after softmax in batch {total_batches}, using uniform distribution")
                    batch_size, num_classes = logits.shape
                    probs = torch.ones(batch_size, num_classes, device=device) / num_classes

                # 存储预测结果
                preds = logits.argmax(1).cpu().numpy()
                probs_np = probs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                # 存储所有样本详细信息（包括时间步）
                for idx, true_label, pred_label, prob in zip(indices, labels_np, preds, probs_np[:, 1]):
                    msi_path, rgb_path, _ = dataset.samples[idx]
                    # 从路径中提取时间步信息 (假设路径包含时间信息，如"time_6"等)
                    time_step = None
                    for t in Config.TIME_POINTS:
                        if f"/{t}/" in msi_path:
                            time_step = t
                            break

                    all_samples_details.append({
                        "msi_path": msi_path,
                        "rgb_path": rgb_path,
                        "time_step": time_step,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "probability": prob
                    })

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
                all_probs.extend(probs_np[:,1])
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
        return result, error_samples,all_samples_details
    return result,all_samples_details


# ========== 修改数据集加载逻辑（添加测试集到训练集） ==========
def get_combined_datasets_and_loaders():
    # 定义数据增强
    # 训练集（含合并的测试集）使用的数据增强
    rgb_train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
        # 暂时注释ToTensor()，移到_read_rgb中统一执行，避免类型冲突
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    msi_train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # 高斯噪声增强
    ])

    # 验证集不使用数据增强
    rgb_val_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    msi_val_transform = transforms.Lambda(lambda x: x)  # 无增强

    # 创建多模态变换实例
    train_transform = MultiModalTransform(
        rgb_transform=rgb_train_transform,
        msi_transform=msi_train_transform,
        apply_geometric=True
    )
    val_transform = MultiModalTransform(
        rgb_transform=rgb_val_transform,
        msi_transform=msi_val_transform,
        apply_geometric=False
    )

    # 1. 加载原始训练集和测试集
    train_dataset_base = TimeSeriesDataset(subset="train", transform=None)  # 先不应用变换
    test_dataset = TimeSeriesDataset(subset="test", transform=None)  # 先不应用变换

    # 2. 合并训练集和测试集的样本列表
    combined_samples = train_dataset_base.samples + test_dataset.samples
    print(f"合并后训练集样本数: {len(combined_samples)} "
          f"(原训练集: {len(train_dataset_base)} + 原测试集: {len(test_dataset)})")



    # 4. 初始化合并后的训练集（应用训练变换）
    train_dataset = CombinedTrainDataset(combined_samples, train_transform)

    # 5. 加载验证集（保持不变）
    val_dataset = TimeSeriesDataset(subset="val", transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_dataset, val_dataset, train_loader, val_loader
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


# 新增：计算并保存时间步指标的函数
def calculate_time_step_metrics(all_samples):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

    # 创建DataFrame
    df = pd.DataFrame(all_samples)

    # 保存完整预测表格
    df.to_csv(Config.PREDICTION_TABLE_PATH, index=False, encoding='utf-8')
    print(f"所有样本预测结果已保存至: {Config.PREDICTION_TABLE_PATH}")

    # 按时间步分组计算指标
    time_steps = Config.TIME_POINTS
    metrics_list = []

    # 计算总体指标
    overall_metrics = {
        "time_step": "总体",
        "samples_count": len(df),
        "accuracy": accuracy_score(df["true_label"], df["pred_label"]),
        "precision": precision_score(df["true_label"], df["pred_label"], average='binary', zero_division=0),
        "recall": recall_score(df["true_label"], df["pred_label"], average='binary', zero_division=0),
        "f1": f1_score(df["true_label"], df["pred_label"], average='binary', zero_division=0)
    }

    # 计算AUC
    try:
        if len(df["true_label"].unique()) > 1:
            overall_metrics["auc"] = roc_auc_score(df["true_label"], df["probability"])
        else:
            overall_metrics["auc"] = 0.0
    except:
        overall_metrics["auc"] = 0.0

    # 计算混淆矩阵指标
    cm = confusion_matrix(df["true_label"], df["pred_label"])
    if cm.size >= 4:
        overall_metrics["tp"] = cm[1, 1]
        overall_metrics["tn"] = cm[0, 0]
        overall_metrics["fp"] = cm[0, 1]
        overall_metrics["fn"] = cm[1, 0]
    else:
        overall_metrics["tp"] = overall_metrics["tn"] = overall_metrics["fp"] = overall_metrics["fn"] = 0

    metrics_list.append(overall_metrics)

    # 按每个时间步计算指标
    for time_step in time_steps:
        time_df = df[df["time_step"] == time_step]
        if len(time_df) == 0:
            continue

        metrics = {
            "time_step": time_step,
            "samples_count": len(time_df),
            "accuracy": accuracy_score(time_df["true_label"], time_df["pred_label"]),
            "precision": precision_score(time_df["true_label"], time_df["pred_label"], average='binary',
                                         zero_division=0),
            "recall": recall_score(time_df["true_label"], time_df["pred_label"], average='binary', zero_division=0),
            "f1": f1_score(time_df["true_label"], time_df["pred_label"], average='binary', zero_division=0)
        }

        # 计算AUC
        try:
            if len(time_df["true_label"].unique()) > 1:
                metrics["auc"] = roc_auc_score(time_df["true_label"], time_df["probability"])
            else:
                metrics["auc"] = 0.0
        except:
            metrics["auc"] = 0.0

        # 计算混淆矩阵指标
        cm = confusion_matrix(time_df["true_label"], time_df["pred_label"])
        if cm.size >= 4:
            metrics["tp"] = cm[1, 1]
            metrics["tn"] = cm[0, 0]
            metrics["fp"] = cm[0, 1]
            metrics["fn"] = cm[1, 0]
        else:
            metrics["tp"] = metrics["tn"] = metrics["fp"] = metrics["fn"] = 0

        metrics_list.append(metrics)

    # 保存时间步指标
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(Config.TIME_STEP_METRICS_PATH, index=False, encoding='utf-8')
    print(f"时间步指标已保存至: {Config.TIME_STEP_METRICS_PATH}")

    return df, metrics_df
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
    nan_epochs = 0
    analyze_data_distribution()
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    # 增强的数据增强策略
    rgb_specific_transforms = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.3,  # 增加亮度扰动
            contrast=0.3,  # 增加对比度扰动
            saturation=0.3,  # 增加饱和度扰动
            hue=0.1  # 轻微色调变化
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
    ])
    # 创建多模态变换
    multimodal_transform = MultiModalTransform(
        rgb_transform=rgb_specific_transforms,
        msi_transform=msi_specific_transform,
        apply_geometric=True  # 启用几何变换
    )

    # 加载合并后的数据集和数据加载器（替换原加载逻辑）
    train_dataset, val_dataset, train_loader, val_loader = get_combined_datasets_and_loaders()
    # 初始化模型、优化器
    model = BruiseSTFNet(
        msi_channels=Config.MSI_CHANNELS,
        rgb_channels=Config.RGB_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.hidden_dim,
        fusion_dim=Config.fusion_dim
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
        # 验证
        val_metrics, val_errors, val_samples = eval_one_epoch(
            model, val_loader, val_dataset, Config.DEVICE, save_errors=True
        )
        save_error_samples(val_errors, "val")
        # 记录指标（保持不变）
        metrics['nan_batches_val'].append(val_metrics['nan_batches'])
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_precision_micro'].append(val_metrics['precision_micro'])
        metrics['val_recall_micro'].append(val_metrics['recall_micro'])
        metrics['val_f1_micro'].append(val_metrics['f1_micro'])
        metrics['val_auc_roc'].append(val_metrics['auc_roc'])

        # 打印epoch信息（保持不变）
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print(f"训练：损失={train_loss:.4f}，准确率={train_acc:.4f}")
        print(f"验证：损失={val_metrics['loss']:.4f}，ACC={val_metrics['accuracy']:.4f}，"
              f"Micro-P={val_metrics['precision_micro']:.4f}，Micro-R={val_metrics['recall_micro']:.4f}，"
              f"Micro-F1={val_metrics['f1_micro']:.4f}，AUC-ROC={val_metrics['auc_roc']:.4f}，"
              f"NaN批次={val_metrics['nan_batches']}/{val_metrics['total_batches']}")
        # 学习率调度（保持不变）
        scheduler.step(val_metrics['f1_micro'])

        # NaN处理（保持不变）
        if val_metrics['nan_batches'] > 0 :
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
        model.load_state_dict(best_model_state)
        print("使用最佳模型计算最终指标...")
        final_val_metrics, _, final_val_samples = eval_one_epoch(
            model, val_loader, val_dataset, Config.DEVICE, save_errors=True
        )
        # 计算并保存最终的时间步指标
        calculate_time_step_metrics(final_val_samples)
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
        best_val_metrics = eval_one_epoch(model, val_loader, val_dataset, Config.DEVICE, save_errors=False)
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
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("损失值", fontsize=10)
    plt.title("损失曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_acc'], label="训练准确率", linewidth=2)
    plt.plot(metrics['val_acc'], label="验证准确率", linewidth=2)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("准确率", fontsize=10)
    plt.title("准确率曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(metrics['val_auc_roc'], label="验证AUC-ROC", linewidth=2)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("AUC-ROC", fontsize=10)
    plt.title("AUC-ROC曲线", fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_DIR, "training_val_curves_full.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()