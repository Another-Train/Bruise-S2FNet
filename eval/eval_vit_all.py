import os
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from spectral.io import envi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 替换为你的模型导入路径
from models.vit_total import ViTWithProb, load_vit_simple

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ========== 配置参数 ==========
model_name = "Vit"
RESULT_DIR = "results/二分类/"
MODEL_PATH = os.path.join(RESULT_DIR, "二分类_vit_total_best_model.pth")
PCA_DIR = os.path.join(RESULT_DIR, "pca_weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (64, 64)
BATCH_SIZE = 4
HS_PCA_DIM = 20  # PCA降维维度

# 核心：保留train/val/test目录结构，推理时遍历所有目录
DATA_ROOT = "data"  # 数据总根目录
DATA_SUBDIRS = ["train", "val", "test"]  # 需要遍历的子目录
RGB_SUBDIR = "RGB"
MSI_SUBDIR = "MSI"

# 0h样本复制配置
COPY_ZERO_HOUR_TIMES = 2  # 每个0h样本复制份数（可调整）


# ========== 混淆矩阵绘制函数 ==========
def plot_confusion_matrix(y_true, y_pred, time_point=None, save_path=None, title_prefix="混淆矩阵"):
    """
    绘制美观的混淆矩阵热力图

    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        time_point: 时间点（小时），用于标题显示
        save_path: 保存路径
        title_prefix: 标题前缀
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(8, 6))

    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0h (负类)', '其他时间 (正类)'],
                yticklabels=['0h (负类)', '其他时间 (正类)'],
                cbar=True,
                annot_kws={"size": 14})

    # 设置标题和标签
    if time_point is not None:
        plt.title(f'{title_prefix} - 时间点 {time_point}h', fontsize=16)
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
        print(f"混淆矩阵已保存至: {save_path}")

    # 显示图片（可选，批量处理时建议关闭）
    # plt.show()
    plt.close()

    return cm


# ========== PCA处理器 ==========
class HsProcessor:
    def __init__(self, pca_dim=20):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim)
        self.pca_dim = pca_dim
        self.fitted = False
        self.pca_components = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.pca_mean = None

    def load_pca_weights(self, pca_dir):
        required_files = [
            "pca_components.npy",
            "scaler_mean.npy",
            "scaler_scale.npy",
            "pca_mean.npy"
        ]

        for file in required_files:
            file_path = os.path.join(pca_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PCA权重文件不存在: {file_path}")

        self.pca_components = np.load(os.path.join(pca_dir, "pca_components.npy"))
        self.scaler_mean = np.load(os.path.join(pca_dir, "scaler_mean.npy"))
        self.scaler_scale = np.load(os.path.join(pca_dir, "scaler_scale.npy"))
        self.pca_mean = np.load(os.path.join(pca_dir, "pca_mean.npy"))

        if self.pca_components.shape[0] != self.pca_dim:
            raise ValueError(f"PCA组件维度不匹配: 预期{self.pca_dim}, 实际{self.pca_components.shape[0]}")

        self.pca.components_ = self.pca_components
        self.pca.mean_ = self.pca_mean
        self.scaler.mean_ = self.scaler_mean
        self.scaler.scale_ = self.scaler_scale
        self.fitted = True
        print(f"PCA权重加载完成: 输入维度={len(self.scaler_mean)}, 输出维度={self.pca_components.shape[0]}")

    def fit_pca(self, train_msi_paths):
        print("开始拟合PCA...")
        train_hs_data = []
        for msi_path in tqdm(train_msi_paths, desc="加载MSI数据拟合PCA"):
            try:
                msi_data_path = os.path.splitext(msi_path)[0] + ".float"
                if not os.path.exists(msi_data_path):
                    raise FileNotFoundError(f".float文件缺失: {msi_data_path}")
                img = envi.open(msi_path, image=msi_data_path)
                msi_data = img.load()
                msi_data = np.transpose(msi_data, (2, 0, 1))
                hs_mean = np.mean(msi_data, axis=(1, 2))
                train_hs_data.append(hs_mean)
            except Exception as e:
                print(f"\n跳过文件 {msi_path}: {str(e)[:100]}")
                continue

        if len(train_hs_data) == 0:
            raise RuntimeError("MSI数据为空，无法拟合PCA！")

        train_hs_data = np.array(train_hs_data)
        if np.isnan(train_hs_data).any() or np.isinf(train_hs_data).any():
            raise ValueError("MSI数据包含NaN/Inf，无法拟合PCA！")

        self.scaler.fit(train_hs_data)
        train_hs_scaled = self.scaler.transform(train_hs_data)
        self.pca.fit(train_hs_scaled)
        self.fitted = True

        self.pca_components = self.pca.components_
        self.scaler_mean = self.scaler.mean_
        self.scaler_scale = self.scaler.scale_
        self.pca_mean = self.pca.mean_

        print(f"PCA拟合完成！原始维度{train_hs_data.shape[1]}→降维后{self.pca_dim}")

    def transform_hs(self, hs_data):
        if not self.fitted:
            raise RuntimeError("需先拟合或加载PCA！")

        hs_mean = np.mean(hs_data, axis=(1, 2)).reshape(1, -1)
        hs_scaled = (hs_mean - self.scaler_mean) / self.scaler_scale
        hs_pca = np.dot(hs_scaled - self.pca_mean, self.pca_components.T)
        return torch.tensor(hs_pca.squeeze(0), dtype=torch.float32)

    def save_pca_weights(self, save_dir):
        if not self.fitted:
            raise RuntimeError("需先拟合PCA才能保存权重！")

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "pca_components.npy"), self.pca_components)
        np.save(os.path.join(save_dir, "scaler_mean.npy"), self.scaler_mean)
        np.save(os.path.join(save_dir, "scaler_scale.npy"), self.scaler_scale)
        np.save(os.path.join(save_dir, "pca_mean.npy"), self.pca_mean)
        print(f"PCA权重已保存至: {save_dir}")


# ========== 时间解析函数 ==========
def parse_time_from_filename(filename):
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]
    parts = basename.split('-')
    if len(parts) < 4:
        return None
    try:
        time_str = parts[1]
        time_num = int(time_str.replace('h', ''))
        return time_num
    except (IndexError, ValueError):
        return None


# ========== 收集所有目录下的数据（train/val/test） ==========
def collect_all_data_paths():
    """遍历train/val/test目录，收集所有RGB/MSI配对数据"""
    rgb_info = {}
    all_msi_paths = []

    # 遍历train/val/test三个子目录
    for subdir in DATA_SUBDIRS:
        rgb_root = os.path.join(DATA_ROOT, subdir, RGB_SUBDIR)
        msi_root = os.path.join(DATA_ROOT, subdir, MSI_SUBDIR)

        if not os.path.exists(rgb_root) or not os.path.exists(msi_root):
            print(f"警告：{subdir}目录不存在（RGB={rgb_root} / MSI={msi_root}），跳过")
            continue

        # 收集当前子目录下的RGB路径
        for root, dirs, files in os.walk(rgb_root):
            for file in files:
                if file.endswith('.png'):
                    rgb_path = os.path.join(root, file)
                    file_prefix = os.path.splitext(file)[0]
                    rgb_info[file_prefix] = (rgb_path, subdir)

        # 收集当前子目录下的MSI路径并匹配RGB
        for root, dirs, files in os.walk(msi_root):
            for file in files:
                if file.endswith('.hdr'):
                    msi_path = os.path.join(root, file)
                    file_prefix = os.path.splitext(file)[0]
                    if file_prefix in rgb_info:
                        rgb_path, subdir = rgb_info[file_prefix]
                        float_path = os.path.splitext(msi_path)[0] + ".float"
                        if os.path.exists(float_path):
                            all_msi_paths.append((rgb_path, msi_path, subdir))
                        else:
                            print(f"跳过{subdir}目录下MSI文件 {msi_path}：缺失.float文件")
                    else:
                        print(f"跳过{subdir}目录下MSI文件 {msi_path}：未找到匹配的RGB")

    # 分离RGB和MSI路径
    rgb_paths = [item[0] for item in all_msi_paths]
    msi_paths = [item[1] for item in all_msi_paths]
    sample_subdirs = [item[2] for item in all_msi_paths]

    # 输出收集结果
    print(f"\n全量数据收集结果：")
    print(f"- 总有效配对数：{len(rgb_paths)}")
    for subdir in DATA_SUBDIRS:
        count = sample_subdirs.count(subdir)
        print(f"- {subdir}目录样本数：{count}")
    if len(rgb_paths) > 0:
        print(f"- 示例配对：\n  RGB: {rgb_paths[0]}\n  MSI: {msi_paths[0]}\n  所属目录: {sample_subdirs[0]}")

    return rgb_paths, msi_paths


# ========== 数据集类（标签由时间决定：0h→0，其他→1） ==========
class TimeAwareDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, img_size, hs_processor):
        if not hs_processor.fitted:
            raise RuntimeError("传入的hs_processor未拟合PCA！请先调用fit_pca方法。")
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.img_size = img_size
        self.hs_processor = hs_processor
        self.valid_data = self._filter_and_generate_labels()
        self.rgb_paths = [d[0] for d in self.valid_data]
        self.msi_paths = [d[1] for d in self.valid_data]
        self.labels = [d[2] for d in self.valid_data]
        self.times = [d[3] for d in self.valid_data]

    def _filter_and_generate_labels(self):
        """过滤无效数据，根据时间生成二分类标签"""
        valid_data = []
        for rgb, msi in zip(self.rgb_paths, self.msi_paths):
            if not os.path.exists(rgb) or not os.path.exists(msi):
                print(f"跳过不存在的文件：RGB={rgb} | MSI={msi}")
                continue
            float_path = f"{os.path.splitext(msi)[0]}.float"
            if not os.path.exists(float_path):
                print(f"跳过缺少.float文件的MSI：{msi}")
                continue
            parsed_time = parse_time_from_filename(rgb)
            if parsed_time is None:
                print(f"跳过无法解析时间的文件：{rgb}")
                continue
            # 0h→0，其他→1
            time_based_label = 0 if parsed_time == 0 else 1
            valid_data.append((rgb, msi, time_based_label, parsed_time))
        return valid_data

    def _read_rgb(self, rgb_path):
        from PIL import Image
        from torchvision import transforms
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            rgb_img = rgb_img.resize(self.img_size)
            transform = transforms.Compose([transforms.ToTensor()])
            return transform(rgb_img)
        except Exception as e:
            print(f"读取RGB失败 {rgb_path}: {e}")
            return torch.zeros(3, *self.img_size)

    def _read_msi_with_pca(self, msi_path):
        try:
            msi_data_path = f"{os.path.splitext(msi_path)[0]}.float"
            img = envi.open(msi_path, image=msi_data_path)
            msi_data = img.load()
            msi_data = np.transpose(msi_data, (2, 0, 1))
            hs_pca = self.hs_processor.transform_hs(msi_data)
            return torch.tensor(hs_pca, dtype=torch.float32)
        except Exception as e:
            print(f"读取MSI失败 {msi_path}: {e}")
            return torch.zeros(HS_PCA_DIM, dtype=torch.float32)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = self._read_rgb(self.rgb_paths[idx])
        msi = self._read_msi_with_pca(self.msi_paths[idx])
        label = self.labels[idx]
        time = self.times[idx]
        return rgb, msi, label, time


# ========== 全量数据推理函数（含0h样本复制） ==========
def inference_all_time_data(model_path, result_dir, pca_dir):
    os.makedirs(result_dir, exist_ok=True)

    # 创建混淆矩阵保存目录
    cm_dir = os.path.join(result_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # 用于汇总所有时间点的预测结果
    all_global_labels = []
    all_global_preds = []

    # 1. 加载/拟合PCA
    print("第一步：加载PCA权重...")
    hs_processor = HsProcessor(pca_dim=HS_PCA_DIM)
    try:
        hs_processor.load_pca_weights(pca_dir)
    except Exception as e:
        print(f"加载PCA权重失败: {e}，将使用全量数据重新拟合PCA...")
        _, all_msi_paths = collect_all_data_paths()
        if len(all_msi_paths) == 0:
            raise RuntimeError("全量数据未收集到任何MSI文件！")
        hs_processor.fit_pca(all_msi_paths)
        hs_processor.save_pca_weights(pca_dir)

    if not hs_processor.fitted:
        raise RuntimeError("PCA拟合失败！")
    print("PCA准备完成。")

    # 2. 收集所有数据（train/val/test）
    print("\n第二步：收集全量数据（train/val/test）...")
    all_rgb_paths, all_msi_paths = collect_all_data_paths()
    if len(all_rgb_paths) == 0:
        print("警告: 无有效数据！")
        return

    # 3. 创建数据集
    dataset = TimeAwareDataset(
        all_rgb_paths, all_msi_paths,
        img_size=IMG_SIZE,
        hs_processor=hs_processor
    )
    if len(dataset) == 0:
        raise RuntimeError("有效数据为空！")

    # 4. 按时间分组 + 复制0h样本到其他时间点（核心解决类别不足）
    print("\n第三步：按时间分组数据并复制0h样本...")
    time_groups = {}
    zero_hour_data = []

    # 先分离0h样本和其他时间样本
    for i in range(len(dataset)):
        rgb, msi, label, time = dataset[i]
        if time == 0:
            zero_hour_data.append((rgb, msi, label))  # 收集0h样本
        else:
            if time not in time_groups:
                time_groups[time] = []
            time_groups[time].append((rgb, msi, label))  # 其他时间样本

    # 复制0h样本到其他时间点
    if zero_hour_data and time_groups:
        print(f"\n找到{len(zero_hour_data)}个0h样本，开始复制分配...")
        # 为每个非0h时间点添加复制的0h样本
        for time in time_groups.keys():
            # 复制指定份数的0h样本
            copied_zero_data = zero_hour_data * COPY_ZERO_HOUR_TIMES
            # 添加到当前时间点
            time_groups[time].extend(copied_zero_data)
            print(
                f"向{time}h时间点添加{len(copied_zero_data)}个0h样本副本（原样本数：{len(time_groups[time]) - len(copied_zero_data)}）")
    else:
        print("未找到0h样本或其他时间样本，跳过复制")

    # 补充0h自身的分组（保留原始0h数据）
    if zero_hour_data:
        time_groups[0] = zero_hour_data

    sorted_times = sorted(time_groups.keys())
    print(f"\n最终时间点分组: {sorted_times}")
    for time in sorted_times:
        print(f"- {time}h: 总样本数={len(time_groups[time])}")
    if not sorted_times:
        raise RuntimeError("未找到有效时间点！")

    # 5. 加载模型
    print("\n第四步：加载模型...")
    base_model = load_vit_simple()
    model = ViTWithProb(base_model, hs_dim=HS_PCA_DIM, num_classes=2)
    state_dict = torch.load(model_path, map_location=DEVICE)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if "base_net.fusion.pos_enc.pos_x" not in k
           and "base_net.fusion.pos_enc.pos_y" not in k
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    # 计算模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_params = count_parameters(model)
    print(f"模型参数数量: {model_params:,} (P(G))")

    # 6. 逐个时间点推理
    print("\n第五步：开始推理...")
    all_metrics = {}
    
    # 用于计算推理时间
    total_inference_time = 0
    total_samples = 0
    roc_dir = os.path.join(result_dir, "time_roc")
    os.makedirs(roc_dir, exist_ok=True)

    for time in sorted_times:
        print(f"\n===== 处理时间点: {time}h =====")
        group_data = time_groups[time]

        # 准备数据
        rgb_list = [item[0] for item in group_data]
        msi_list = [item[1] for item in group_data]
        labels_list = [item[2] for item in group_data]

        # 检查标签分布
        label_counts = {0: labels_list.count(0), 1: labels_list.count(1)}
        print(f"当前时间点标签分布: 0类(0h)={label_counts[0]}, 1类(其他)={label_counts[1]}")

        temp_dataset = torch.utils.data.TensorDataset(
            torch.stack(rgb_list),
            torch.stack(msi_list),
            torch.tensor(labels_list, dtype=torch.long)
        )
        temp_loader = DataLoader(
            temp_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0 if os.name == 'nt' else 2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # 推理
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for rgb, msi, labels in tqdm(temp_loader, desc=f"推理 {time}h"):
                rgb, msi = rgb.to(DEVICE), msi.to(DEVICE)
                batch_size = rgb.size(0)
                total_samples += batch_size
                
                # 记录推理开始时间
                start_time = time.time()
                
                out = model(msi, rgb)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                probs = np.clip(probs, 0.0, 1.0)
                
                # 记录推理结束时间
                end_time = time.time()
                inference_time = end_time - start_time
                total_inference_time += inference_time
                
                preds = np.round(probs).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # 收集全局结果用于汇总混淆矩阵
        all_global_labels.extend(all_labels)
        all_global_preds.extend(all_preds)

        # 计算指标（二分类专用）
        unique_labels = set(all_labels)
        if len(unique_labels) < 2:
            print(f"时间点 {time}h：仍仅包含一个类别（{unique_labels}）")
            auc = np.nan
            # 即使类别不足也绘制混淆矩阵（显示实际分布）
            cm_save_path = os.path.join(cm_dir, f"confusion_matrix_{time}h.png")
            plot_confusion_matrix(
                all_labels, all_preds,
                time_point=time,
                save_path=cm_save_path,
                title_prefix=f"{model_name} 混淆矩阵（含0h复制样本）"
            )
        else:
            auc = roc_auc_score(all_labels, all_probs)
            # 绘制并保存当前时间点的混淆矩阵
            cm_save_path = os.path.join(cm_dir, f"confusion_matrix_{time}h.png")
            cm = plot_confusion_matrix(
                all_labels, all_preds,
                time_point=time,
                save_path=cm_save_path,
                title_prefix=f"{model_name} 混淆矩阵（含0h复制样本）"
            )

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)

        # 保存指标
        all_metrics[time] = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'sample_count': len(all_labels),
            'label_distribution': f"0类(0h): {all_labels.count(0)}, 1类(其他): {all_labels.count(1)}"
        }

        # 输出指标
        print(f"时间点 {time}h 指标:")
        print("-" * 50)
        print(f"样本数: {len(all_labels)}")
        print(f"标签分布: {all_metrics[time]['label_distribution']}")
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # 绘制ROC曲线
        if len(unique_labels) >= 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title(f'时间点 {time}h ROC曲线（含0h复制样本）')
            plt.legend(loc="lower right")
            roc_path = os.path.join(roc_dir, f"roc_{time}h.png")
            plt.savefig(roc_path, dpi=300)
            plt.close()
            print(f"ROC曲线已保存: {roc_path}")
        else:
            print("未绘制ROC曲线（类别不足）")

    # 绘制所有时间点汇总的混淆矩阵
    if all_global_labels and all_global_preds:
        total_cm_path = os.path.join(cm_dir, "confusion_matrix_total.png")
        plot_confusion_matrix(
            all_global_labels, all_global_preds,
            time_point=None,
            save_path=total_cm_path,
            title_prefix=f"{model_name} 汇总混淆矩阵（所有时间点+0h复制样本）"
        )
        print(f"\n汇总混淆矩阵已保存至: {total_cm_path}")

    # 汇总指标
    print("\n" + "=" * 80)
    print("全量数据（train+val+test）各时间点指标汇总（含0h复制样本）:")
    print("-" * 80)
    header = f"{'时间(h)':<8} {'样本数':<8}  {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'AUC':<8}"
    print(header)
    print("-" * 80)
    for time in sorted_times:
        m = all_metrics[time]
        print(
            f"{time:<8} {m['sample_count']:<8} {m['acc']:.4f} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f}  {m['auc']:.4f}")

    # 保存汇总结果到文件
    metrics_path = os.path.join(result_dir, "all_time_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("全量数据（train+val+test）各时间点推理指标汇总（含0h复制样本）\n")
        f.write(f"0h样本复制份数: {COPY_ZERO_HOUR_TIMES}\n")
        f.write("=" * 80 + "\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for time in sorted_times:
            m = all_metrics[time]
            f.write(
                f"{time:<8} {m['sample_count']:<8} {m['acc']:.4f} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f}  {m['auc']:.4f}\n")
            f.write(f"标签分布: {m['label_distribution']}\n\n")
    print(f"\n指标汇总已保存至: {metrics_path}")
    
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


# ========== 执行推理 ==========
if __name__ == "__main__":
    # 安装seaborn（如果未安装）
    try:
        import seaborn as sns
    except ImportError:
        print("正在安装seaborn库...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns

    # 检查关键路径
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"数据总根目录不存在：{DATA_ROOT}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
    if not os.path.exists(PCA_DIR):
        print(f"警告：PCA目录不存在 {PCA_DIR}，将重新拟合并保存")

    # 执行全量数据推理
    try:
        all_metrics = inference_all_time_data(MODEL_PATH, RESULT_DIR, PCA_DIR)
    except Exception as e:
        print(f"\n推理出错：{e}")
        import traceback

        traceback.print_exc()