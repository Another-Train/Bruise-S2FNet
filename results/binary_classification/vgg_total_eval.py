import os
import glob
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import spectral
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略无关警告
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ========== 配置参数（与训练代码保持一致） ==========
class Config:
    DATA_ROOT = "data2"  # 总数据集路径
    MODALITIES = ["MSI", "RGB"]
    TIME_POINTS = ['0', '01','6', '12', '18', '24', '30', '36']
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
    NUM_CLASSES = 2
    MSI_CHANNELS = 24
    RGB_CHANNELS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HS_PCA_DIM = 20
    PCA_WEIGHTS_FILE = "pca_weights.pkl"
    
    # 模型路径（根据训练结果修改）
    MODEL_PATH = "二分类results/vgg/vgg_total_savepca/best_model.pth"
    PCA_WEIGHTS_PATH = "二分类results/vgg/vgg_total_savepca/pca_weights.pkl"
    RESULT_DIR = "总数据集推理结果"

# ========== 高光谱PCA处理器（与训练代码一致） ==========
class HsProcessor:
    def __init__(self, pca_dim=Config.HS_PCA_DIM):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim)
        self.fitted = False

    def transform_hs(self, hs_data):
        """对高光谱数据进行PCA降维"""
        if not self.fitted:
            raise RuntimeError("需先调用load_pca_weights()加载PCA权重！")
        hs_mean = np.mean(hs_data, axis=(1, 2)).reshape(1, -1)
        return self.pca.transform(self.scaler.transform(hs_mean)).squeeze(0)
    
    def load_pca_weights(self, file_path):
        """从文件加载PCA权重"""
        with open(file_path, 'rb') as f:
            pca_weights = pickle.load(f)
        
        # 恢复scaler
        self.scaler.mean_ = pca_weights['scaler_mean']
        self.scaler.scale_ = pca_weights['scaler_scale']
        self.scaler.var_ = pca_weights['scaler_var']
        
        # 恢复PCA
        self.pca.components_ = pca_weights['pca_components']
        self.pca.explained_variance_ = pca_weights['pca_explained_variance']
        self.pca.explained_variance_ratio_ = pca_weights['pca_explained_variance_ratio']
        self.pca.mean_ = pca_weights['pca_mean']
        self.pca.n_components_ = pca_weights['pca_n_components']
        
        self.fitted = True
        print(f"PCA权重已从 {file_path} 加载")

# ========== VGG19模型（与训练代码一致） ==========
class MinimalVGG19(torch.nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, hs_dim=Config.HS_PCA_DIM):
        super().__init__()
        # 加载完整的预训练VGG19
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)

        # RGB分支
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])  # 输出4096维

        # 高光谱分支
        self.hs_processor = torch.nn.Sequential(
            torch.nn.Linear(hs_dim, 512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 4096),  # 与RGB分支特征维度对齐
            torch.nn.ReLU(True)
        )

        # 最终分类器
        self.final_classifier = torch.nn.Linear(4096 + 4096, num_classes)

    def forward(self, hs, rgb):
        # RGB特征提取
        x_rgb = self.features(rgb)
        x_rgb = self.avgpool(x_rgb)
        x_rgb = torch.flatten(x_rgb, 1)
        rgb_features = self.classifier(x_rgb)  # 输出4096维

        # 高光谱特征处理
        hs_features = self.hs_processor(hs)  # 输出4096维

        # 特征融合
        fused = torch.cat([rgb_features, hs_features], dim=1)
        output = self.final_classifier(fused)

        return output

# ========== 总数据集推理类 ==========
class TotalDatasetInference:
    def __init__(self, data_root, model_path, pca_weights_path):
        self.data_root = data_root
        self.model_path = model_path
        self.pca_weights_path = pca_weights_path
        self.device = Config.DEVICE
        
        # 创建结果目录
        os.makedirs(Config.RESULT_DIR, exist_ok=True)
        
        # 加载模型和PCA
        self.model = self._load_model()
        self.hs_processor = self._load_pca_processor()
        
        # 存储结果
        self.results = {
            'all_samples': [],
            'subset_results': {},
            'overall_metrics': {}
        }
    
    def _load_model(self):
        """加载训练好的模型"""
        model = MinimalVGG19()
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print(f"模型已从 {self.model_path} 加载")
            return model
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
    
    def _load_pca_processor(self):
        """加载PCA处理器"""
        hs_processor = HsProcessor()
        if os.path.exists(self.pca_weights_path):
            hs_processor.load_pca_weights(self.pca_weights_path)
            return hs_processor
        else:
            raise FileNotFoundError(f"PCA权重文件不存在: {self.pca_weights_path}")
    
    def _read_msi(self, msi_hdr_path):
        """读取高光谱数据并应用PCA降维"""
        try:
            msi_data_path = f"{os.path.splitext(msi_hdr_path)[0]}.float"
            img = spectral.io.envi.open(msi_hdr_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))  # [bands, height, width]

            # 应用PCA降维
            hs_features = self.hs_processor.transform_hs(msi_data)
            hs_tensor = torch.tensor(hs_features, dtype=torch.float32)
            return hs_tensor

        except Exception as e:
            print(f"读取MSI文件失败：{msi_hdr_path}\n错误详情：{str(e)}")
            return torch.zeros(Config.HS_PCA_DIM, dtype=torch.float32)
    
    def _read_rgb(self, rgb_path):
        """读取RGB图像"""
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
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
    
    def collect_all_samples(self):
        """收集总数据集中的所有样本"""
        print("开始收集总数据集样本...")
        subsets = ["train", "val", "test"]
        all_samples = []
        
        for subset in subsets:
            subset_dir = os.path.join(self.data_root, subset)
            print(f"处理 {subset} 子集...")
            
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
                        final_label = 0  # 健康样本
                    else:
                        final_label = label
                    
                    msi_path = os.path.join(msi_time_dir, f"{prefix}.hdr")
                    rgb_path = os.path.join(rgb_time_dir, f"{prefix}.png")
                    
                    sample_info = {
                        'subset': subset,
                        'time_point': time,
                        'true_label': final_label,
                        'msi_path': msi_path,
                        'rgb_path': rgb_path,
                        'sample_name': f"{prefix}_time{time}",
                        'predicted_label': None,
                        'confidence': None,
                        'probabilities': None
                    }
                    all_samples.append(sample_info)
        
        print(f"总共收集到 {len(all_samples)} 个样本")
        return all_samples
    
    def inference_single_sample(self, sample_info):
        """对单个样本进行推理"""
        try:
            # 读取数据
            hs_tensor = self._read_msi(sample_info['msi_path']).unsqueeze(0)  # 添加batch维度
            rgb_tensor = self._read_rgb(sample_info['rgb_path']).unsqueeze(0)  # 添加batch维度
            
            # 移动到设备
            hs_tensor = hs_tensor.to(self.device)
            rgb_tensor = rgb_tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.model(hs_tensor, rgb_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_label = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities).item()
                
            # 更新样本信息
            sample_info['predicted_label'] = predicted_label
            sample_info['confidence'] = confidence
            sample_info['probabilities'] = probabilities.cpu().numpy()[0]
            
            return sample_info
            
        except Exception as e:
            print(f"推理样本 {sample_info['sample_name']} 时出错: {e}")
            sample_info['predicted_label'] = -1  # 标记错误
            return sample_info
    
    def batch_inference(self, batch_size=16):
        """批量推理所有样本"""
        print("开始批量推理...")
        
        # 收集所有样本
        all_samples = self.collect_all_samples()
        
        # 分批处理
        results = []
        for i in tqdm(range(0, len(all_samples), batch_size)):
            batch = all_samples[i:i+batch_size]
            for sample in batch:
                result = self.inference_single_sample(sample)
                results.append(result)
        
        self.results['all_samples'] = results
        return results
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """计算二分类指标"""
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC计算（二分类）
        try:
            auc_score = roc_auc_score(y_true, y_probs[:, 1])  # 使用正类的概率
        except:
            auc_score = 0.5  # 如果计算失败，设为随机猜测
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def analyze_results(self):
        """分析推理结果并计算指标"""
        print("开始分析结果...")
        
        all_samples = self.results['all_samples']
        
        # 过滤推理失败的样本
        valid_samples = [s for s in all_samples if s['predicted_label'] != -1]
        print(f"有效样本数: {len(valid_samples)}/{len(all_samples)}")
        
        if len(valid_samples) == 0:
            print("没有有效样本可分析")
            return
        
        # 总体指标
        y_true_all = [s['true_label'] for s in valid_samples]
        y_pred_all = [s['predicted_label'] for s in valid_samples]
        y_probs_all = np.array([s['probabilities'] for s in valid_samples])
        
        overall_metrics = self.calculate_metrics(y_true_all, y_pred_all, y_probs_all)
        self.results['overall_metrics'] = overall_metrics
        
        # 按子集分析
        subsets = ["train", "val", "test"]
        for subset in subsets:
            subset_samples = [s for s in valid_samples if s['subset'] == subset]
            if len(subset_samples) > 0:
                y_true = [s['true_label'] for s in subset_samples]
                y_pred = [s['predicted_label'] for s in subset_samples]
                y_probs = np.array([s['probabilities'] for s in subset_samples])
                
                subset_metrics = self.calculate_metrics(y_true, y_pred, y_probs)
                self.results['subset_results'][subset] = {
                    'metrics': subset_metrics,
                    'samples': subset_samples
                }
        
        # 按时间点分析
        time_metrics = {}
        for time in Config.TIME_POINTS:
            time_samples = [s for s in valid_samples if s['time_point'] == time]
            if len(time_samples) > 0:
                y_true = [s['true_label'] for s in time_samples]
                y_pred = [s['predicted_label'] for s in time_samples]
                y_probs = np.array([s['probabilities'] for s in time_samples])
                
                time_metrics[time] = self.calculate_metrics(y_true, y_pred, y_probs)
        
        self.results['time_metrics'] = time_metrics
    
    def save_results(self):
        """保存推理结果"""
        print("保存结果...")
        
        # 保存详细样本结果
        df_samples = pd.DataFrame(self.results['all_samples'])
        df_samples.to_csv(os.path.join(Config.RESULT_DIR, "detailed_results.csv"), 
                         index=False, encoding='utf-8-sig')
        
        # 保存指标汇总
        metrics_data = []
        
        # 总体指标
        overall = self.results['overall_metrics']
        metrics_data.append({
            'subset': '总体',
            '样本数量': len([s for s in self.results['all_samples'] if s['predicted_label'] != -1]),
            '准确率': overall['accuracy'],
            '精确率': overall['precision'],
            '召回率': overall['recall'],
            'F1分数': overall['f1'],
            'AUC': overall['auc']
        })
        
        # 各子集指标
        for subset, result in self.results['subset_results'].items():
            metrics = result['metrics']
            metrics_data.append({
                'subset': subset,
                '样本数量': len(result['samples']),
                '准确率': metrics['accuracy'],
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1'],
                'AUC': metrics['auc']
            })
        
        # 各时间点指标
        for time, metrics in self.results.get('time_metrics', {}).items():
            time_samples = [s for s in self.results['all_samples'] 
                           if s['time_point'] == time and s['predicted_label'] != -1]
            metrics_data.append({
                'subset': f"时间点{time}h",
                '样本数量': len(time_samples),
                '准确率': metrics['accuracy'],
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1'],
                'AUC': metrics['auc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(os.path.join(Config.RESULT_DIR, "metrics_summary.csv"), 
                         index=False, encoding='utf-8-sig')
        
        # 保存错误预测样本
        error_samples = [s for s in self.results['all_samples'] 
                        if s['predicted_label'] != -1 and s['predicted_label'] != s['true_label']]
        df_errors = pd.DataFrame(error_samples)
        if len(error_samples) > 0:
            df_errors.to_csv(os.path.join(Config.RESULT_DIR, "error_predictions.csv"), 
                            index=False, encoding='utf-8-sig')
        
        print(f"结果已保存至: {Config.RESULT_DIR}")
    
    def plot_results(self):
        """绘制结果图表"""
        print("绘制图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 总体指标柱状图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        overall_metrics = self.results['overall_metrics']
        
        for i, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
            # 总体指标
            axes[i].bar(['总体'], [overall_metrics[metric_key]], color='skyblue', alpha=0.7)
            
            # 各子集指标
            subset_names = []
            subset_values = []
            for subset, result in self.results['subset_results'].items():
                subset_names.append(subset)
                subset_values.append(result['metrics'][metric_key])
            
            if subset_names:
                axes[i].bar(subset_names, subset_values, color=['lightgreen', 'lightcoral', 'gold'], alpha=0.7)
            
            axes[i].set_title(f'{metric_name}对比')
            axes[i].set_ylabel(metric_name)
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
        
        # 2. 混淆矩阵热力图
        cm = overall_metrics['confusion_matrix']
        ax_cm = axes[5]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=[Config.LABEL_TO_CLASS[i] for i in range(Config.NUM_CLASSES)],
                   yticklabels=[Config.LABEL_TO_CLASS[i] for i in range(Config.NUM_CLASSES)])
        ax_cm.set_title('混淆矩阵')
        ax_cm.set_xlabel('预测标签')
        ax_cm.set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULT_DIR, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 时间点指标趋势图
        if hasattr(self.results, 'time_metrics'):
            time_points = list(self.results['time_metrics'].keys())
            time_metrics_data = {metric: [] for metric in metric_keys}
            
            for time in time_points:
                for metric_key in metric_keys:
                    time_metrics_data[metric_key].append(
                        self.results['time_metrics'][time][metric_key]
                    )
            
            plt.figure(figsize=(12, 8))
            for metric_key, metric_name in zip(metric_keys, metrics):
                plt.plot(time_points, time_metrics_data[metric_key], 
                        marker='o', label=metric_name, linewidth=2)
            
            plt.xlabel('时间点 (小时)')
            plt.ylabel('指标值')
            plt.title('各时间点指标趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULT_DIR, 'time_metrics_trend.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"图表已保存至: {Config.RESULT_DIR}")
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("总数据集推理结果摘要")
        print("="*60)
        
        overall = self.results['overall_metrics']
        print(f"\n总体指标:")
        print(f"  样本总数: {len([s for s in self.results['all_samples'] if s['predicted_label'] != -1])}")
        print(f"  准确率: {overall['accuracy']:.4f}")
        print(f"  精确率: {overall['precision']:.4f}")
        print(f"  召回率: {overall['recall']:.4f}")
        print(f"  F1分数: {overall['f1']:.4f}")
        print(f"  AUC: {overall['auc']:.4f}")
        
        print(f"\n各子集指标:")
        for subset, result in self.results['subset_results'].items():
            metrics = result['metrics']
            print(f"  {subset}:")
            print(f"    样本数: {len(result['samples'])}")
            print(f"    准确率: {metrics['accuracy']:.4f}")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1分数: {metrics['f1']:.4f}")
            print(f"    AUC: {metrics['auc']:.4f}")
        
        # 错误统计
        error_count = len([s for s in self.results['all_samples'] 
                          if s['predicted_label'] != -1 and s['predicted_label'] != s['true_label']])
        total_valid = len([s for s in self.results['all_samples'] if s['predicted_label'] != -1])
        error_rate = error_count / total_valid if total_valid > 0 else 0
        
        print(f"\n错误统计:")
        print(f"  总错误数: {error_count}")
        print(f"  错误率: {error_rate:.4f}")
        print("="*60)

# ========== 主函数 ==========
def main():
    # 检查路径是否存在
    if not os.path.exists(Config.DATA_ROOT):
        print(f"错误: 数据根目录不存在: {Config.DATA_ROOT}")
        return
    
    # 初始化推理器
    inference = TotalDatasetInference(
        data_root=Config.DATA_ROOT,
        model_path=Config.MODEL_PATH,
        pca_weights_path=Config.PCA_WEIGHTS_PATH
    )
    
    # 批量推理
    inference.batch_inference(batch_size=8)
    
    # 分析结果
    inference.analyze_results()
    
    # 保存结果
    inference.save_results()
    
    # 绘制图表
    inference.plot_results()
    
    # 打印摘要
    inference.print_summary()
    
    print(f"\n推理完成！所有结果已保存至: {Config.RESULT_DIR}")

if __name__ == '__main__':
    import torchvision  # 需要在主程序中导入
    
    # 检查并更新模型路径
    if not os.path.exists(Config.MODEL_PATH):
        # 尝试查找最新的模型文件
        possible_dirs = [
            "二分类results/vgg/vgg_total_savepca",
            "results/vgg/vgg_total_savepca", 
            "./二分类results/vgg/vgg_total_savepca"
        ]
        
        for dir_path in possible_dirs:
            model_path = os.path.join(dir_path, "best_model.pth")
            pca_path = os.path.join(dir_path, "pca_weights.pkl")
            
            if os.path.exists(model_path) and os.path.exists(pca_path):
                Config.MODEL_PATH = model_path
                Config.PCA_WEIGHTS_PATH = pca_path
                Config.RESULT_DIR = os.path.join(dir_path, "总数据集推理")
                print(f"自动找到模型文件: {model_path}")
                break
    
    main()