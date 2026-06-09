import os
import time
import datetime
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# 假设这些是你的模型和数据加载模块（根据实际情况调整）
from models.GMA import BaseNet
from 二分类_B import load_msi_image, load_rgb_image, collect_data_paths

# 假设需要根据实际情况修改以下参数
model_name = "BaseNet"
RESULT_DIR = ("results/二分类/")  # 模型保存目录
MODEL_PATH = os.path.join(RESULT_DIR, "GMA_total.pth")  # 模型路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (64, 64)  # 需与训练时保持一致
BATCH_SIZE = 4  # 可根据设备调整

# 创建必要的目录
os.makedirs(RESULT_DIR, exist_ok=True)

# 解析文件名中的时间信息（复用训练时的解析逻辑）
def parse_time_from_filename(filename):
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]
    parts = basename.split('-')
    if len(parts) < 4:
        return None  # 无效格式
    try:
        time_str = parts[1]
        return int(time_str)  # 返回时间点（如0, 1, 3等）
    except (IndexError, ValueError):
        return None

# 自定义数据集（增加时间信息存储）
class TimeAwareDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, labels, img_size, msi_channels):
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.labels = labels
        self.img_size = img_size
        self.msi_channels = msi_channels
        # 提取时间信息
        self.times = [parse_time_from_filename(path) for path in rgb_paths]
        # 过滤无效时间样本
        valid_indices = [i for i, t in enumerate(self.times) if t is not None]
        self.rgb_paths = [self.rgb_paths[i] for i in valid_indices]
        self.msi_paths = [self.msi_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.times = [self.times[i] for i in valid_indices]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 这里复用原有数据加载逻辑，假设存在以下函数
        rgb = load_rgb_image(self.rgb_paths[idx], self.img_size)
        msi = load_msi_image(self.msi_paths[idx], self.img_size)
        label = self.labels[idx]
        time = self.times[idx]
        return rgb, msi, label, time

def plot_confusion_matrix(y_true, y_pred, time_point=None, save_path=None, title_prefix="混淆矩阵"):
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        time_point: 时间点（可选）
        save_path: 保存路径（可选）
        title_prefix: 标题前缀
    """
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(8, 6))

    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负类', '正类'],
                yticklabels=['负类', '正类'],
                cbar=True,
                annot_kws={"size": 14})

    # 设置标题和标签
    if time_point is not None:
        plt.title(f'{title_prefix} - 时间点 {time_point}', fontsize=16)
    else:
        plt.title(title_prefix, fontsize=16)

    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)

    # 调整刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图片
    plt.show()
    plt.close()

    return cm

def inference_by_time(model_path, result_dir):    # 1. 收集所有数据路径
    print("收集所有数据路径...")
    rgb_paths, msi_paths, labels = collect_data_paths()  # 复用原有数据收集函数
    if len(rgb_paths) == 0:
        print("错误: 没有找到任何数据文件")
        return

    # 2. 确定MSI通道数
    sample_msi = load_msi_image(msi_paths[0], IMG_SIZE)
    msi_channels = sample_msi.shape[0]
    print(f"MSI图像通道数: {msi_channels}")

    # 3. 创建带时间信息的数据集和数据加载器
    dataset = TimeAwareDataset(
        rgb_paths, msi_paths, labels,
        img_size=IMG_SIZE,
        msi_channels=msi_channels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 4. 加载模型
    print(f"加载模型: {model_name}")
    model = BaseNet(msi_channels=msi_channels, rgb_channels=3, num_classes=2).to(DEVICE)
    # 加载模型权重并过滤不需要的参数
    state_dict = torch.load(model_path, map_location=DEVICE)
    # 过滤掉旧权重中存在但当前模型已删除的pos_x和pos_y
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if "base_net.fusion.pos_enc.pos_x" not in k
           and "base_net.fusion.pos_enc.pos_y" not in k
    }
    # 加载过滤后的权重（strict=False确保忽略不匹配的键）
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # 计算模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_params = count_parameters(model)
    print(f"模型参数数量: {model_params:,} (P(G))")

    # 5. 推理并按时间点收集结果
    time_results = {}  # 格式: {时间点: {'preds': [], 'labels': []}}
    all_preds = []  # 存储所有预测结果（用于汇总混淆矩阵）
    all_labels = []  # 存储所有真实标签（用于汇总混淆矩阵）
    
    # 用于计算推理时间
    total_inference_time = 0
    total_samples = 0

    with torch.no_grad():
        for rgb, msi, labels, times in tqdm(dataloader, desc="推理中"):
            rgb, msi = rgb.to(DEVICE), msi.to(DEVICE)
            batch_size = rgb.size(0)
            total_samples += batch_size
            
            # 记录推理开始时间
            start_time = time.time()

            # 获取模型输出
            out, _ = model(msi, rgb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()  # 正类概率
            # 新增：处理NaN和极端值
            probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)  # NaN替换为0.5，极端值截断
            probs = np.clip(probs, 0.0, 1.0)  # 确保概率在[0,1]范围内
            
            # 记录推理结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time

            preds = np.round(probs).astype(int)  # 二分类预测结果

            # 按时间点分组存储
            for prob, pred, label, time in zip(probs, preds, labels.numpy(), times.numpy()):
                if time not in time_results:
                    time_results[time] = {'probs': [], 'preds': [], 'labels': []}
                time_results[time]['probs'].append(prob)
                time_results[time]['preds'].append(pred)
                time_results[time]['labels'].append(label)

                # 收集所有结果
                all_preds.append(pred)
                all_labels.append(label)

    # 6. 按时间点计算指标并展示
    sorted_times = sorted(time_results.keys())  # 按时间排序
    all_metrics = []

    print("\n各时间点推理指标:")
    print("时间点 | 准确率 | 精确率 | 召回率 | F1分数| AUC  ")
    print("-" * 50)

    # 创建混淆矩阵保存目录
    cm_dir = os.path.join(result_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for time in sorted_times:
        res = time_results[time]
        # 计算指标
        auc = roc_auc_score(res['labels'], res['probs'])
        acc = accuracy_score(res['labels'], res['preds'])
        precision = precision_score(res['labels'], res['preds'], zero_division=0)
        recall = recall_score(res['labels'], res['preds'], zero_division=0)
        f1 = f1_score(res['labels'], res['preds'], zero_division=0)

        # 存储指标
        all_metrics.append({
            'time': time,
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # 打印结果
        print(f"{time:4d}   | {auc:.4f} | {acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

        # 绘制并保存该时间点的混淆矩阵
        cm_path = os.path.join(cm_dir, f"confusion_matrix_time_{time}.png")
        cm = plot_confusion_matrix(
            res['labels'],
            res['preds'],
            time_point=time,
            save_path=cm_path,
            title_prefix=f"{model_name} 混淆矩阵"
        )
        print(f"时间点 {time} 混淆矩阵已保存至: {cm_path}")

    # 7. 绘制汇总混淆矩阵（所有时间点合并）
    total_cm_path = os.path.join(cm_dir, "confusion_matrix_total.png")
    total_cm = plot_confusion_matrix(
        all_labels,
        all_preds,
        time_point=None,
        save_path=total_cm_path,
        title_prefix=f"{model_name} 汇总混淆矩阵 (所有时间点)"
    )
    print(f"汇总混淆矩阵已保存至: {total_cm_path}")
    
    # 计算推理性能指标
    if total_inference_time > 0:
        # 平均推理时间（秒/样本）
        avg_inference_time = total_inference_time / total_samples
        # 推理速度（样本/秒）
        inference_speed = total_samples / total_inference_time
        # 吞吐量（imgs/s）
        throughput = inference_speed
        
        print("\n推理性能指标:")
        print("-" * 50)
        print(f"模型参数数量 (P(G)): {model_params:,}")
        print(f"平均推理时间: {avg_inference_time * 1000:.2f} ms/样本")
        print(f"推理速度 (F(G)): {inference_speed:.2f} 样本/秒")
        print(f"吞吐量 (Thr): {throughput:.2f} imgs/s")

    return all_metrics

if __name__ == "__main__":
    # 执行按时间点的推理与指标计算
    time_metrics = inference_by_time(MODEL_PATH, RESULT_DIR)
