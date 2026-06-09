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
import sys
import datetime

import torchvision.models as models
import torch.nn as nn

# 忽略无关警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ========== 配置参数 ==========
class Config:
    DATA_ROOT = "data2"
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01', '6', '12', '18', '24', '30', '36']  # 7个时间点（七分类）
    TIME_TO_LABEL = {
        '0': 0,  # 0h → 标签0（健康）
        '01': 0,  # 1h → 标签0（健康）
        '6': 1,  # 6h → 标签0（瘀伤早期）
        '12': 1,  # 12h → 标签1（瘀伤早期）
        '18': 1,  # 18h → 标签1（瘀伤中期）
        '24': 1,  # 24h → 标签2（瘀伤中期）
        '30': 1,  # 30h → 标签2（瘀伤后期）
        '36': 1  # 36h → 标签3（瘀伤后期）
    }
    LABEL_TO_CLASS = {
        0: "健康",
        1: "瘀伤早期", 
    }
    IMG_SIZE = (32, 32)  # 图像尺寸（高度, 宽度）
    BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    MSI_CHANNELS = 24
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 20
    MIN_DELTA = 0.001
    RESULT_DIR = r"二分类results/RNN/RNN_total"
    SEED = 42


class BruiseDetectionLSTM(nn.Module):
    def __init__(self, rgb_channels=3, msi_bands=Config.MSI_CHANNELS, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 存储输入维度信息
        self.rgb_channels = rgb_channels
        self.msi_bands = msi_bands
        self.seq_length = 32  # 序列长度（图像高度）
        
        # 计算输入维度（与原始处理逻辑一致）
        self.rgb_feature_dim = rgb_channels * 32  # RGB特征维度: 3 * 32 = 192
        self.msi_feature_dim = msi_bands * 32     # MSI特征维度: bands * 32
        self.input_size = self.rgb_feature_dim + self.msi_feature_dim  # 总输入维度
        
        # LSTM网络
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 分类器 - 修改为4分类输出
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, Config.NUM_CLASSES)  # 输出4个类别
        )
        
        print(f"模型初始化: RGB通道={rgb_channels}, MSI波段={msi_bands}")
        print(f"特征维度: RGB={self.rgb_feature_dim}, MSI={self.msi_feature_dim}, 总输入={self.input_size}")

    def _process_modalities(self, rgb_data, msi_data):
        """处理RGB和MSI数据，将其转换为序列格式"""
        batch_size = rgb_data.size(0)
        
        # RGB序列化: [batch, 3, 64, 64] -> [batch, 64, 192]
        rgb_seq = rgb_data.permute(0, 2, 1, 3)  # [batch, 64, 3, 64]
        rgb_seq = rgb_seq.reshape(batch_size, self.seq_length, -1)  # [batch, 64, 3 * 64]
        
        # MSI序列化: [batch, bands, 64, 64] -> [batch, 64, bands*64]
        msi_seq = msi_data.permute(0, 2, 1, 3)  # [batch, 64, bands, 64]
        msi_seq = msi_seq.reshape(batch_size, self.seq_length, -1)
        
        # 拼接RGB和MSI序列
        combined_seq = torch.cat([rgb_seq, msi_seq], dim=-1)  # [batch, 64, 192 + bands*64]
        
        return combined_seq

    def forward(self, rgb_data, msi_data):
        """
        前向传播
        rgb_data: [batch, 3, 64, 64]
        msi_data: [batch, bands, 64, 64]
        返回: 分类logits和特征
        """
        # 处理多模态数据
        combined_seq = self._process_modalities(rgb_data, msi_data)  # [batch, seq_len, input_size]
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(combined_seq)  # lstm_out: [batch, seq_len, hidden_size]
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        # 分类
        logits = self.classifier(last_output)  # [batch, num_classes]
        # 返回分类结果和特征（为了与训练代码兼容）
        # features = last_output  # 使用LSTM最后一个时间步的输出作为特征
        return logits


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

# ========== 数据集类 ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, subset="train", transform=None):
        self.subset = subset
        self.transform = transform
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"{subset}集未找到有效样本")

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
                sample_name = f"{prefix}_time{time}"
                samples.append((msi_path, rgb_path, label, sample_name, time))  # 添加时间点信息

        print(f"\n最终收集到{len(samples)}个样本")
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
def multi_class_fusion_loss(logits, labels, lambda_attn=0.01):
    ce_loss = F.cross_entropy(logits, labels)
    return ce_loss

# ========== 训练和验证函数 ==========
def train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for msi, rgb, labels, _, _ in tqdm(loader, desc="训练"):
        msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)
        optimizer.zero_grad()

        out = model(msi, rgb)
        loss = multi_class_fusion_loss(logits=out, labels=labels, lambda_attn=0.1)
        
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
    """
    评估函数
    compute_auc: 是否计算AUC指标（只在最终测试时计算）
    """
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
            print(f"实际出现的类别: {unique_labels}, 类别数量: {num_actual_classes}")
            print(f"各类别样本数量: {np.bincount(all_labels)}")
            
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
                    print("警告: 数据集中只存在一个类别，无法计算AUC")
                else:
                    print(f"警告: 预期二分类但实际有{num_actual_classes}个类别")
            else:
                # 多分类情况
                if num_actual_classes == Config.NUM_CLASSES:
                    # 类别数量匹配
                    y_true_bin = label_binarize(all_labels, classes=range(Config.NUM_CLASSES))
                    auc_macro = roc_auc_score(y_true_bin, all_probs_array, average='macro', multi_class='ovr')
                    auc_micro = roc_auc_score(y_true_bin, all_probs_array, average='micro', multi_class='ovr')
                else:
                    # 类别数量不匹配，只计算存在的类别
                    print(f"警告: 实际类别数({num_actual_classes})与模型输出类别数({Config.NUM_CLASSES})不匹配")
                    if num_actual_classes >= 2:
                        # 只对实际存在的类别计算AUC
                        y_true_bin = label_binarize(all_labels, classes=unique_labels)
                        # 调整概率矩阵，只保留存在的类别
                        adjusted_probs = all_probs_array[:, unique_labels]
                        auc_macro = roc_auc_score(y_true_bin, adjusted_probs, average='macro', multi_class='ovr')
                        auc_micro = roc_auc_score(y_true_bin, adjusted_probs, average='micro', multi_class='ovr')
        except Exception as e:
            print(f"计算AUC时出错: {e}")
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
                    print(f"时间点 {time_point} AUC计算错误: {e}")
            
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
            print(f"\n各时间点指标已保存至: {time_metrics_file}")

    # 记录错误文件
    if record_errors and error_file_path:
        print(f"\n预测错误的文件（共{len(error_files)}个）：")
        for error in error_files:
            print(f"时间点{error['time_point']}h: {error['file_name']}, "
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

# ========== 可视化函数 ==========
def plot_time_metrics(time_metrics, result_dir):
    """绘制各时间点指标对比图"""
    if not time_metrics:
        return
    
    # 准备数据
    times = list(time_metrics.keys())
    metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC_macro', 'AUC_micro']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [time_metrics[time][metric] for time in times]
        axes[i].bar(times, values, color='skyblue', alpha=0.7)
        axes[i].set_title(f'{metric}随时间的變化')
        axes[i].set_xlabel('时间点(小时)')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "time_metrics_comparison.png"), dpi=300)
    plt.show()

# ========== 日志记录类 ==========
class Logger(object):
    """同时将输出记录到文件和终端"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# ========== 主训练流程 ==========
def main():
    # 创建结果目录
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    # 设置日志记录
    log_file = os.path.join(Config.RESULT_DIR, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_file)
    
    print(f"训练开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print(f"结果目录: {Config.RESULT_DIR}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
    ])

    # 构建数据集和加载器
    train_dataset = TimeSeriesDataset(subset="train", transform=train_transform)
    val_dataset = TimeSeriesDataset(subset="val", transform=None)
    test_dataset = TimeSeriesDataset(subset="test", transform=None)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = BruiseDetectionLSTM().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 早停机制
    best_f1 = -1.0
    counter = 0
    best_model_state = None

    # 记录训练指标 - 增加测试集完整指标记录
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
        'test_auc_macro': [], 'test_auc_micro': []
    }

    print(f"开始训练（设备：{Config.DEVICE}，类别数：{Config.NUM_CLASSES}）")
    for epoch in range(Config.EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        
        # 验证（不计算AUC，不按时间点分组）
        val_metrics = eval_one_epoch(model, val_loader, Config.DEVICE, 
                                   compute_auc=False, record_errors=False, group_by_time=False)
        
        # 测试集评估（计算完整指标，包括AUC）
        test_metrics = eval_one_epoch(model, test_loader, Config.DEVICE, 
                                    compute_auc=True, record_errors=False, group_by_time=False)
        
        # 记录所有指标
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_precision'].append(val_metrics['precision'])
        metrics['val_recall'].append(val_metrics['recall'])
        metrics['val_f1'].append(val_metrics['f1'])
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['test_acc'].append(test_metrics['accuracy'])
        metrics['test_precision'].append(test_metrics['precision'])
        metrics['test_recall'].append(test_metrics['recall'])
        metrics['test_f1'].append(test_metrics['f1'])
        metrics['test_auc_macro'].append(test_metrics['auc_macro'])
        metrics['test_auc_micro'].append(test_metrics['auc_micro'])

        # 打印epoch信息
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print(f"训练：损失={train_loss:.4f}，准确率={train_acc:.4f}")
        print(f"验证：损失={val_metrics['loss']:.4f}，准确率={val_metrics['accuracy']:.4f}，"
              f"精确率={val_metrics['precision']:.4f}，召回率={val_metrics['recall']:.4f}，F1={val_metrics['f1']:.4f}")
        print(f"测试：损失={test_metrics['loss']:.4f}，准确率={test_metrics['accuracy']:.4f}，"
              f"精确率={test_metrics['precision']:.4f}，召回率={test_metrics['recall']:.4f}，F1={test_metrics['f1']:.4f}")
        
        # 显示AUC指标（如果有效）
        if test_metrics['auc_macro'] > 0:
            print(f"测试AUC：macro={test_metrics['auc_macro']:.4f}，micro={test_metrics['auc_micro']:.4f}")

        # 学习率调度和早停
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_f1 + Config.MIN_DELTA:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            counter = 0
            print(f"最佳F1更新至 {best_f1:.4f}")
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

    # 保存训练过程中的所有指标
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(Config.RESULT_DIR, "training_metrics.csv"), index_label='epoch', encoding='utf-8-sig')
    print(f"训练指标已保存至：{os.path.join(Config.RESULT_DIR, 'training_metrics.csv')}")

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
        print("\n各时间点详细指标：")
        for time_point, metrics in test_metrics['time_metrics'].items():
            print(f"\n时间点 {time_point}小时:")
            print(f"  样本数量: {metrics['样本数量']}")
            print(f"  准确率: {metrics['准确率']:.4f}")
            print(f"  精确率: {metrics['精确率']:.4f}")
            print(f"  召回率: {metrics['召回率']:.4f}")
            print(f"  F1分数: {metrics['F1分数']:.4f}")
            if metrics['AUC_macro'] > 0:  # 只显示有效的AUC
                print(f"  AUC_macro: {metrics['AUC_macro']:.4f}")
                print(f"  AUC_micro: {metrics['AUC_micro']:.4f}")
    
    print("\n测试集总体指标：")
    print(f"准确率：{test_metrics['accuracy']:.4f}，"
          f"精确率：{test_metrics['precision']:.4f}，"
          f"召回率：{test_metrics['recall']:.4f}，"
          f"F1：{test_metrics['f1']:.4f}")
    
    # 只显示有效的AUC指标
    if test_metrics['auc_macro'] > 0:
        print(f"AUC_macro：{test_metrics['auc_macro']:.4f}，"
              f"AUC_micro：{test_metrics['auc_micro']:.4f}")
    
    # 保存测试指标
    with open(os.path.join(Config.RESULT_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("测试集最终指标：\n")
        f.write(f"准确率：{test_metrics['accuracy']:.4f}\n")
        f.write(f"精确率：{test_metrics['precision']:.4f}\n")
        f.write(f"召回率：{test_metrics['recall']:.4f}\n")
        f.write(f"F1：{test_metrics['f1']:.4f}\n")
        if test_metrics['auc_macro'] > 0:
            f.write(f"AUC_macro：{test_metrics['auc_macro']:.4f}\n")
            f.write(f"AUC_micro：{test_metrics['auc_micro']:.4f}\n")
        if 'error_files' in test_metrics:
            f.write(f"\n预测错误文件数量：{len(test_metrics['error_files'])}\n")
    
    # 恢复标准输出
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    
    print(f"\n训练完成！所有输出已记录到: {log_file}")
    print(f"结果目录: {Config.RESULT_DIR}")
            
if __name__ == '__main__':
    main()