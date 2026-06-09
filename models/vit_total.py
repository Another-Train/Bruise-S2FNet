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
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import torchvision.models as models
import torch.nn as nn
import datetime

# 忽略无关警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ========== 配置参数 ==========
class Config:
    DATA_ROOT = "data2"
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01','6', '12', '18', '24', '30', '36']  # 7个时间点（七分类）
    TIME_TO_LABEL = {
        '0': 0,  # 0h → 标签0（健康）
        '01': 1,  # 1h → 标签0（健康）
        '6': 1,  # 6h → 标签0（瘀伤早期）
        '12': 1,  # 12h → 标签1（瘀伤早期）
        '18': 1,  # 18h → 标签1（瘀伤中期）
        '24': 1,  # 24h → 标签2（瘀伤中期）
        '30': 1,  # 30h → 标签2（瘀伤后期）
        '36': 1  # 36h → 标签3（瘀伤后期）
    }
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
    RESULT_DIR = r"二分类results/vit/vit_total"
    SEED = 42
    HS_PCA_DIM = 20  # 高光谱PCA降维维度

# ========== 日志记录器 ==========
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.csv")
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("训练日志\n")
            f.write("=" * 50 + "\n")
            f.write(f"训练开始时间: {timestamp}\n")
            f.write(f"数据根目录: {Config.DATA_ROOT}\n")
            f.write(f"批次大小: {Config.BATCH_SIZE}\n")
            f.write(f"最大训练轮数: {Config.EPOCHS}\n")
            f.write(f"学习率: {Config.LEARNING_RATE}\n")
            f.write(f"设备: {Config.DEVICE}\n")
            f.write(f"随机种子: {Config.SEED}\n")
            f.write("=" * 50 + "\n\n")
        
        # 初始化CSV文件头
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc_macro,val_auc_micro,"
                   "test_loss,test_acc,test_precision,test_recall,test_f1,test_auc_macro,test_auc_micro,lr\n")
    
    def log_epoch(self, epoch, train_metrics, val_metrics, test_metrics, lr):
        """记录一个epoch的指标到txt和csv文件"""
        
        # 写入txt日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  训练 - 损失: {train_metrics['loss']:.6f}, 准确率: {train_metrics['accuracy']:.6f}\n")
            f.write(f"  验证 - 损失: {val_metrics['loss']:.6f}, 准确率: {val_metrics['accuracy']:.6f}, "
                   f"精确率: {val_metrics['precision']:.6f}, 召回率: {val_metrics['recall']:.6f}, "
                   f"F1: {val_metrics['f1']:.6f}, AUC宏: {val_metrics['auc_macro']:.6f}\n")
            f.write(f"  测试 - 损失: {test_metrics['loss']:.6f}, 准确率: {test_metrics['accuracy']:.6f}, "
                   f"精确率: {test_metrics['precision']:.6f}, 召回率: {test_metrics['recall']:.6f}, "
                   f"F1: {test_metrics['f1']:.6f}, AUC宏: {test_metrics['auc_macro']:.6f}\n")
            f.write(f"  学习率: {lr:.8f}\n")
            f.write("-" * 80 + "\n")
        
        # 保存到CSV文件
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_metrics['loss']:.6f},{train_metrics['accuracy']:.6f},"
                   f"{val_metrics['loss']:.6f},{val_metrics['accuracy']:.6f},{val_metrics['precision']:.6f},"
                   f"{val_metrics['recall']:.6f},{val_metrics['f1']:.6f},{val_metrics['auc_macro']:.6f},"
                   f"{val_metrics['auc_micro']:.6f},{test_metrics['loss']:.6f},{test_metrics['accuracy']:.6f},"
                   f"{test_metrics['precision']:.6f},{test_metrics['recall']:.6f},{test_metrics['f1']:.6f},"
                   f"{test_metrics['auc_macro']:.6f},{test_metrics['auc_micro']:.6f},{lr:.8f}\n")
    
    def log_best_model(self, best_epoch, best_f1):
        """记录最佳模型信息"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n最佳模型: Epoch {best_epoch}, 最佳F1: {best_f1:.6f}\n")
    
    def plot_training_curves(self):
        """从CSV文件读取数据并绘制训练曲线图"""
        try:
            # 读取CSV文件
            df = pd.read_csv(self.metrics_file)
            if len(df) == 0:
                print("警告：没有训练数据可绘制")
                return
            
            epochs = df['epoch'].values
            
            # 提取指标
            train_loss = df['train_loss'].values
            val_loss = df['val_loss'].values
            test_loss = df['test_loss'].values
            
            train_acc = df['train_acc'].values
            val_acc = df['val_acc'].values
            test_acc = df['test_acc'].values
            
            val_precision = df['val_precision'].values
            val_recall = df['val_recall'].values
            val_f1 = df['val_f1'].values
            
            test_precision = df['test_precision'].values
            test_recall = df['test_recall'].values
            test_f1 = df['test_f1'].values
            
            val_auc_macro = df['val_auc_macro'].values
            val_auc_micro = df['val_auc_micro'].values
            test_auc_macro = df['test_auc_macro'].values
            test_auc_micro = df['test_auc_micro'].values
            
            learning_rates = df['lr'].values
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 损失曲线
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
            plt.plot(epochs, test_loss, 'g-', label='测试损失', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('训练、验证和测试损失曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 准确率曲线
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_acc, 'b-', label='训练准确率', linewidth=2)
            plt.plot(epochs, val_acc, 'r-', label='验证准确率', linewidth=2)
            plt.plot(epochs, test_acc, 'g-', label='测试准确率', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('训练、验证和测试准确率曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 验证集指标曲线
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, val_precision, 'b-', label='精确率', linewidth=2)
            plt.plot(epochs, val_recall, 'r-', label='召回率', linewidth=2)
            plt.plot(epochs, val_f1, 'g-', label='F1分数', linewidth=2)
            plt.plot(epochs, val_auc_macro, 'm-', label='AUC宏平均', linewidth=2)
            plt.plot(epochs, val_auc_micro, 'c-', label='AUC微平均', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('验证集指标曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 测试集指标曲线
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, test_precision, 'b-', label='精确率', linewidth=2)
            plt.plot(epochs, test_recall, 'r-', label='召回率', linewidth=2)
            plt.plot(epochs, test_f1, 'g-', label='F1分数', linewidth=2)
            plt.plot(epochs, test_auc_macro, 'm-', label='AUC宏平均', linewidth=2)
            plt.plot(epochs, test_auc_micro, 'c-', label='AUC微平均', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('测试集指标曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'test_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. AUC对比曲线
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, val_auc_macro, 'b-', label='验证集AUC宏平均', linewidth=2)
            plt.plot(epochs, val_auc_micro, 'b--', label='验证集AUC微平均', linewidth=2)
            plt.plot(epochs, test_auc_macro, 'r-', label='测试集AUC宏平均', linewidth=2)
            plt.plot(epochs, test_auc_micro, 'r--', label='测试集AUC微平均', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.title('AUC指标对比曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'auc_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. 学习率曲线
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, learning_rates, 'purple', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('学习率变化曲线')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 7. 综合指标对比图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 损失对比
            ax1.plot(epochs, train_loss, 'b-', label='训练损失')
            ax1.plot(epochs, val_loss, 'r-', label='验证损失')
            ax1.plot(epochs, test_loss, 'g-', label='测试损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('损失对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 准确率对比
            ax2.plot(epochs, train_acc, 'b-', label='训练准确率')
            ax2.plot(epochs, val_acc, 'r-', label='验证准确率')
            ax2.plot(epochs, test_acc, 'g-', label='测试准确率')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('准确率对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # F1分数对比
            ax3.plot(epochs, val_f1, 'r-', label='验证F1')
            ax3.plot(epochs, test_f1, 'g-', label='测试F1')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('F1分数对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # AUC对比
            ax4.plot(epochs, val_auc_macro, 'r-', label='验证AUC宏')
            ax4.plot(epochs, test_auc_macro, 'g-', label='测试AUC宏')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('AUC')
            ax4.set_title('AUC宏平均对比')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"所有训练曲线图已保存至: {self.log_dir}")
            
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")

# ========== 高光谱PCA处理器 ==========
class HsProcessor:
    def __init__(self, pca_dim=Config.HS_PCA_DIM):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim)
        self.fitted = False

    def fit_pca(self, train_hs_list):
        """使用训练集拟合PCA模型"""
        hs_flatten = [np.mean(hs, axis=(1, 2)) for hs in train_hs_list]
        hs_flatten = np.array(hs_flatten)
        hs_scaled = self.scaler.fit_transform(hs_flatten)
        self.pca.fit(hs_scaled)
        self.fitted = True
        print(f"PCA拟合完成：输入波段数={hs_flatten.shape[1]} | 输出维度={Config.HS_PCA_DIM}")

    def transform_hs(self, hs_data):
        """对高光谱数据进行PCA降维"""
        if not self.fitted:
            raise RuntimeError("需先调用fit_pca()拟合训练集！")
        hs_mean = np.mean(hs_data, axis=(1, 2)).reshape(1, -1)
        return self.pca.transform(self.scaler.transform(hs_mean)).squeeze(0)


import torch
import torchvision

def load_vit_simple():
    """使用torchvision内置的下载功能，更可靠"""
    try:
        # 新版本的torchvision使用weights参数
        # 首先尝试使用新API
        try:
            # 对于torchvision 0.13+
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            model = torchvision.models.vit_b_16(weights=weights)
            print("使用新API加载ViT模型成功！")
        except AttributeError:
            # 回退到旧方法
            print("使用旧方法加载ViT模型...")
            model = torchvision.models.vit_b_16(pretrained=True)
        
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


class ViTWithProb(nn.Module):
    def __init__(self, vit_model, hs_dim=Config.HS_PCA_DIM, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.vit = vit_model
        self.softmax = nn.Softmax(dim=1)  # 用于输出概率计算AUC
        # 高光谱特征处理，转换为 512 维
        self.hs_processor = nn.Sequential(
            nn.Linear(hs_dim, 512),
            nn.ReLU(True)
        )
        self.final_classifier = nn.Linear(1000 + 512, num_classes)
    def forward(self, hs, rgb, return_prob=False):
        rgb = torch.nn.functional.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)
        rgb_features = self.vit(rgb)
        if return_prob:
            return self.softmax(rgb_features)
        
        # 高光谱特征处理
        hs_features = self.hs_processor(hs)  # 输出512维
        fused = torch.cat([rgb_features, hs_features], dim=1)
        output = self.final_classifier(fused)
        return output

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

# ========== 修改后的数据集类 ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, subset="train", transform=None, hs_processor=None):
        self.subset = subset
        self.transform = transform
        self.hs_processor = hs_processor
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
                samples.append((msi_path, rgb_path, final_label, sample_name, time))

        print(f"{self.subset}集最终收集到{len(samples)}个样本")
        return samples

    def _read_msi(self, msi_hdr_path):
        """读取高光谱数据并应用PCA降维"""
        try:
            msi_data_path = f"{os.path.splitext(msi_hdr_path)[0]}.float"
            from spectral.io import envi
            img = envi.open(msi_hdr_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))  # [bands, height, width]

            # 应用PCA降维
            if self.hs_processor and self.hs_processor.fitted:
                hs_features = self.hs_processor.transform_hs(msi_data)
                hs_tensor = torch.tensor(hs_features, dtype=torch.float32)
                return hs_tensor
            else:
                # 如果未提供PCA处理器，返回全零特征
                return torch.zeros(Config.HS_PCA_DIM, dtype=torch.float32)

        except Exception as e:
            print(f"读取MSI文件失败：{msi_hdr_path}\n错误详情：{str(e)}")
            return torch.zeros(Config.HS_PCA_DIM, dtype=torch.float32)

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
        msi = self._read_msi(msi_path)  # 现在返回的是PCA降维后的特征向量
        rgb = self._read_rgb(rgb_path)
        return msi, rgb, torch.tensor(label, dtype=torch.long), sample_name, time_point

# ========== 准备PCA处理器 ==========
def prepare_hs_pca():
    """准备高光谱PCA处理器"""
    # 收集训练集的高光谱数据用于PCA拟合
    train_samples = []
    train_dir = os.path.join(Config.DATA_ROOT, "train")
    
    for time in Config.TIME_POINTS:
        msi_time_dir = os.path.join(train_dir, "MSI", str(time))
        if not os.path.isdir(msi_time_dir):
            continue
            
        msi_hdr_files = glob.glob(os.path.join(msi_time_dir, "*.hdr"))
        for hdr_path in msi_hdr_files:
            train_samples.append(hdr_path)
    
    print(f"收集到{len(train_samples)}个训练样本用于PCA拟合")
    
    # 读取高光谱数据
    hs_processor = HsProcessor()
    train_hs_data = []
    
    for hdr_path in tqdm(train_samples, desc="读取高光谱数据"):
        try:
            msi_data_path = f"{os.path.splitext(hdr_path)[0]}.float"
            from spectral.io import envi
            img = envi.open(hdr_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))
            train_hs_data.append(msi_data)
        except Exception as e:
            print(f"跳过有问题的文件 {hdr_path}: {e}")
            continue
    
    # 拟合PCA
    if train_hs_data:
        hs_processor.fit_pca(train_hs_data)
    else:
        print("警告：未找到有效的高光谱数据，PCA将使用默认参数")
    
    return hs_processor

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

def eval_one_epoch(model, loader, device, record_errors=False, error_file_path=None, 
                   group_by_class=False, class_metrics_file=None):
    """评估函数 - 修改为按类别计算指标"""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels, all_times = [], [], []
    all_probs = []
    error_files = []
    class_group_data = {}  # 按类别分组的数据

    with torch.no_grad():
        for batch in loader:
            if record_errors or group_by_class:
                msi, rgb, labels, sample_names, time_points = batch
            else:
                msi, rgb, labels, _, _ = batch
            
            msi, rgb, labels = msi.to(device), rgb.to(device), labels.to(device)
            out = model(msi, rgb)

            loss = F.cross_entropy(out, labels)
            total_loss += loss.item() * labels.size(0)

            # 概率计算
            try:
                # 检查输出是否有效
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print("警告：模型输出包含NaN或Inf值，使用随机值替代")
                    probs = torch.softmax(torch.randn_like(out), dim=1).cpu().numpy()
                else:
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                
                # 确保概率有效
                if np.isnan(probs).any() or np.isinf(probs).any():
                    print("警告：概率包含NaN或Inf值，使用均匀分布")
                    probs = np.ones_like(probs) / Config.NUM_CLASSES
                else:
                    # 确保概率和为1（处理数值精度问题）
                    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
                    
            except Exception as e:
                print(f"概率计算错误: {e}，使用均匀分布")
                batch_size = out.shape[0]
                probs = np.ones((batch_size, Config.NUM_CLASSES)) / Config.NUM_CLASSES
            
            all_probs.extend(probs)

            preds = out.argmax(1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            total_correct += (preds == labels_np).sum()
            total += labels.size(0)

            # 按类别分组数据
            if group_by_class:
                for class_label in range(Config.NUM_CLASSES):
                    mask = labels_np == class_label
                    if np.sum(mask) > 0:
                        if class_label not in class_group_data:
                            class_group_data[class_label] = {
                                'preds': [], 'labels': [], 'probs': [], 'sample_names': []
                            }
                        
                        class_group_data[class_label]['preds'].extend(preds[mask])
                        class_group_data[class_label]['labels'].extend(labels_np[mask])
                        class_group_data[class_label]['probs'].extend(probs[mask])
                        if record_errors:
                            class_group_data[class_label]['sample_names'].extend(np.array(sample_names)[mask])

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
    
    # 确保有预测结果
    if len(all_labels) == 0:
        print("警告：验证/测试集没有样本")
        precision = recall = f1 = auc_macro = auc_micro = 0.0
    else:
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 改进的AUC计算
        auc_macro = 0.0
        auc_micro = 0.0
        
        try:
            # 检查是否有足够的样本和类别
            if len(set(all_labels)) < 2:
                print("警告：类别数少于2，无法计算AUC")
            else:
                y_true_bin = label_binarize(all_labels, classes=range(Config.NUM_CLASSES))
                all_probs_array = np.array(all_probs)
                
                # 检查概率矩阵形状
                if all_probs_array.shape[1] != y_true_bin.shape[1]:
                    print(f"概率矩阵形状不匹配: probs={all_probs_array.shape}, y_true={y_true_bin.shape}")
                    # 调整概率矩阵形状
                    if all_probs_array.shape[1] < y_true_bin.shape[1]:
                        # 填充缺失的类别概率为0
                        padding = np.zeros((all_probs_array.shape[0], y_true_bin.shape[1] - all_probs_array.shape[1]))
                        all_probs_array = np.hstack([all_probs_array, padding])
                    else:
                        # 截断多余的类别概率
                        all_probs_array = all_probs_array[:, :y_true_bin.shape[1]]
                
                # 计算AUC
                if len(np.unique(all_labels)) > 1:
                    auc_macro = roc_auc_score(y_true_bin, all_probs_array, average='macro', multi_class='ovr')
                    auc_micro = roc_auc_score(y_true_bin, all_probs_array, average='micro', multi_class='ovr')
                    
        except Exception as e:
            print(f"AUC计算错误: {e}")
            # 调试信息
            print(f"真实标签分布: {np.bincount(all_labels)}")
            print(f"预测标签分布: {np.bincount(all_preds)}")
            if len(all_probs) > 0:
                print(f"概率矩阵形状: {np.array(all_probs).shape}")

    result = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro
    }

    # 按类别计算指标
    if group_by_class:
        class_metrics = {}
        for class_label, data in class_group_data.items():
            if len(data['labels']) == 0:
                continue
                
            # 计算该类别的指标
            class_acc = accuracy_score(data['labels'], data['preds'])
            class_precision = precision_score(data['labels'], data['preds'], average='macro', zero_division=0)
            class_recall = recall_score(data['labels'], data['preds'], average='macro', zero_division=0)
            class_f1 = f1_score(data['labels'], data['preds'], average='macro', zero_division=0)
            
            # 类别AUC计算
            class_auc = 0.0
            
            try:
                if len(set(data['labels'])) > 1:
                    # 对于二分类AUC，需要将当前类别作为正类
                    y_true_bin_class = (np.array(data['labels']) == class_label).astype(int)
                    probs_array = np.array(data['probs'])
                    
                    if probs_array.shape[1] > class_label:
                        class_probs = probs_array[:, class_label]
                        if len(np.unique(y_true_bin_class)) > 1:
                            class_auc = roc_auc_score(y_true_bin_class, class_probs)
                        else:
                            class_auc = 0.0
                    else:
                        class_auc = 0.0
            except Exception as e:
                print(f"类别{Config.LABEL_TO_CLASS[class_label]} AUC计算错误: {e}")
            
            class_metrics[class_label] = {
                '类别名称': Config.LABEL_TO_CLASS[class_label],
                '样本数量': len(data['labels']),
                '准确率': class_acc,
                '精确率': class_precision,
                '召回率': class_recall,
                'F1分数': class_f1,
                'AUC': class_auc
            }
        
        result['class_metrics'] = class_metrics
        
        # 保存类别指标到文件
        if class_metrics_file:
            df_class = pd.DataFrame.from_dict(class_metrics, orient='index')
            df_class.index.name = '类别'
            df_class.to_csv(class_metrics_file, encoding='utf-8-sig')
            print(f"\n各类别指标已保存至: {class_metrics_file}")

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

# ========== 主训练流程 ==========
def main():
    # 创建结果目录
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    # 初始化日志记录器
    logger = TrainingLogger(Config.RESULT_DIR)
    print(f"日志文件将保存至: {Config.RESULT_DIR}")

    # 准备PCA处理器
    print("准备高光谱PCA处理器...")
    hs_processor = prepare_hs_pca()

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
    ])

    # 构建数据集和加载器（传入PCA处理器）
    print("构建数据集...")
    train_dataset = TimeSeriesDataset(subset="train", transform=train_transform, hs_processor=hs_processor)
    val_dataset = TimeSeriesDataset(subset="val", transform=None, hs_processor=hs_processor)
    test_dataset = TimeSeriesDataset(subset="test", transform=None, hs_processor=hs_processor)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 使用手动下载函数替换原来的加载方式
    base_model = load_vit_simple()  # 使用手动下载的函数
    model = ViTWithProb(base_model, hs_dim=Config.HS_PCA_DIM, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 早停机制
    best_f1 = -1.0
    counter = 0
    best_model_state = None
    best_epoch = 0

    print(f"开始训练（设备：{Config.DEVICE}，类别数：{Config.NUM_CLASSES}）")
    for epoch in range(Config.EPOCHS):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        
        # 验证（不按类别分组）
        val_metrics = eval_one_epoch(model, val_loader, Config.DEVICE, record_errors=False, group_by_class=False)
        
        # 测试集评估（不记录错误，不按类别分组）
        test_metrics = eval_one_epoch(model, test_loader, Config.DEVICE, record_errors=False, group_by_class=False)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录epoch指标到日志
        train_metrics_dict = {'loss': train_loss, 'accuracy': train_acc}
        logger.log_epoch(epoch + 1, train_metrics_dict, val_metrics, test_metrics, current_lr)
        
        # 打印epoch信息
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print(f"训练：损失={train_loss:.4f}，准确率={train_acc:.4f}")
        print(f"验证：损失={val_metrics['loss']:.4f}，准确率={val_metrics['accuracy']:.4f}，"
              f"F1={val_metrics['f1']:.4f}，AUC_macro={val_metrics['auc_macro']:.4f}")
        print(f"测试：损失={test_metrics['loss']:.4f}，准确率={test_metrics['accuracy']:.4f}，"
              f"F1={test_metrics['f1']:.4f}，AUC_macro={test_metrics['auc_macro']:.4f}")
        print(f"学习率: {current_lr:.8f}")

        # 学习率调度和早停
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_f1 + Config.MIN_DELTA:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
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
        logger.log_best_model(best_epoch, best_f1)

    # 绘制训练曲线
    print("\n正在绘制训练曲线...")
    logger.plot_training_curves()

    # 测试集最终评估（使用最佳模型，按类别分组）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        error_file_path = os.path.join(Config.RESULT_DIR, "error_predictions.txt")
        class_metrics_file = os.path.join(Config.RESULT_DIR, "class_metrics.csv")
        
        test_metrics = eval_one_epoch(
            model, test_loader, Config.DEVICE, 
            record_errors=True, 
            error_file_path=error_file_path,
            group_by_class=True,
            class_metrics_file=class_metrics_file
        )
        
        # 打印各类别指标
        if 'class_metrics' in test_metrics:
            print("\n各类别详细指标：")
            for class_label, metrics in test_metrics['class_metrics'].items():
                print(f"\n类别 {Config.LABEL_TO_CLASS[class_label]}:")
                print(f"  样本数量: {metrics['样本数量']}")
                print(f"  准确率: {metrics['准确率']:.4f}")
                print(f"  精确率: {metrics['精确率']:.4f}")
                print(f"  召回率: {metrics['召回率']:.4f}")
                print(f"  F1分数: {metrics['F1分数']:.4f}")
                print(f"  AUC: {metrics['AUC']:.4f}")
        
        print("\n测试集总体指标：")
        print(f"准确率：{test_metrics['accuracy']:.4f}，"
              f"精确率：{test_metrics['precision']:.4f}，"
              f"召回率：{test_metrics['recall']:.4f}，"
              f"F1：{test_metrics['f1']:.4f}，"
              f"AUC_macro：{test_metrics['auc_macro']:.4f}，"
              f"AUC_micro：{test_metrics['auc_micro']:.4f}")
        
        # 保存测试指标
        with open(os.path.join(Config.RESULT_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
            f.write("测试集最终指标：\n")
            f.write(f"准确率：{test_metrics['accuracy']:.4f}\n")
            f.write(f"精确率：{test_metrics['precision']:.4f}\n")
            f.write(f"召回率：{test_metrics['recall']:.4f}\n")
            f.write(f"F1：{test_metrics['f1']:.4f}\n")
            f.write(f"AUC_macro：{test_metrics['auc_macro']:.4f}\n")
            f.write(f"AUC_micro：{test_metrics['auc_micro']:.4f}\n")
            if 'error_files' in test_metrics:
                f.write(f"\n预测错误文件数量：{len(test_metrics['error_files'])}\n")
            
            # 保存各类别指标
            if 'class_metrics' in test_metrics:
                f.write("\n各类别详细指标：\n")
                for class_label, metrics in test_metrics['class_metrics'].items():
                    f.write(f"\n类别 {Config.LABEL_TO_CLASS[class_label]}:\n")
                    f.write(f"  样本数量: {metrics['样本数量']}\n")
                    f.write(f"  准确率: {metrics['准确率']:.4f}\n")
                    f.write(f"  精确率: {metrics['精确率']:.4f}\n")
                    f.write(f"  召回率: {metrics['召回率']:.4f}\n")
                    f.write(f"  F1分数: {metrics['F1分数']:.4f}\n")
                    f.write(f"  AUC: {metrics['AUC']:.4f}\n")

    print(f"\n训练完成！所有结果已保存至: {Config.RESULT_DIR}")

if __name__ == '__main__':
    main()