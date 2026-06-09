import glob
import os
import sys
import warnings
import logging
import time
from datetime import datetime
import numpy as np
import rasterio
import spectral
import torch
import torch.nn as nn
import torchvision.models as models
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
# 忽略 NotGeoreferencedWarning 警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 导入自定义模型
from models.BruiseSFNet_G import BruiseSTFNet  # GMA_changed.py中定义了BruiseSTFNet模型

# ========== 配置 ==========
DATA_DIR = "D://DATA//gtdata"
IMG_SIZE = (64, 64)
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 二分类：健康(0)和瘀伤(1)

# 创建结果文件夹
RESULT_DIR = f"results/model_performance/"
os.makedirs(RESULT_DIR, exist_ok=True)

# 配置日志
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(RESULT_DIR, f"performance_log_{timestamp}.txt")

# 配置日志同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()


# ========== 数据解析与处理 ==========
def parse_filename(filename):
    import os

    # 移除路径和扩展名，保留基础文件名
    basename = os.path.basename(filename)
    basename = basename.split('.')[0]

    # 处理可能的预处理前缀
    if basename.startswith('registered_'):
        basename = basename.replace('registered_', '')
    if '_band_' in basename:
        basename = basename.split('_band_')[0]

    # 按连字符分割文件名
    parts = basename.split('-')
    if len(parts) < 4:
        return {
            "fruit_type": "unknown",
            "fruit_label": 0,
            "drop_time": 0,
            "label_range": "0-0",
            "quality_label": 0
        }

    # 解析各部分
    fruit_type = parts[0]  # 瘀伤类型
    drop_time = int(parts[1])  # 掉落时间
    quality_label = int(parts[2])  # 质量标签

    # 二分类标签
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
        "label_range": f"{quality_label}-{quality_label}",
        "quality_label": quality_label
    }


def find_msi_data_file(hdr_path):
    # 首先尝试使用相同的基名但不同扩展名
    base_path = hdr_path.replace('.hdr', '')

    # 检查可能的扩展名
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
        img = img / (img.max() + 1e-8)
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


def collect_data_paths():
    rgb_paths = []
    msi_paths = []
    labels = []

    # 直接遍历RGB目录
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
    msi_basenames = {os.path.basename(f).replace('.hdr', '') for f in msi_files}
    logger.info(f"找到 {len(msi_files)} 个高光谱HDR文件")

    for rgb_file in rgb_files:
        # 获取基本文件名（不含扩展名）
        rgb_basename = os.path.basename(rgb_file).replace('.png', '')

        # 解析文件名获取标签信息
        label_info = parse_filename(rgb_file)

        # 检查是否存在对应的高光谱文件
        if rgb_basename in msi_basenames:
            msi_file = os.path.join(msi_dir, f"{rgb_basename}.hdr")

            # 验证数据文件是否存在
            if find_msi_data_file(msi_file):
                rgb_paths.append(rgb_file)
                msi_paths.append(msi_file)
                labels.append(label_info["bruise_label"])  # 使用瘀伤标签
            else:
                logger.warning(f"找到HDR文件但找不到对应的数据文件: {rgb_basename}")
        else:
            logger.warning(f"找不到与 {rgb_basename} 对应的高光谱文件")

    logger.info(f"总共匹配了 {len(rgb_paths)} 对RGB-高光谱图像")

    return rgb_paths, msi_paths, labels


# ========== PyTorch Dataset ==========
class MultiModalFruitDataset(Dataset):
    def __init__(self, rgb_paths, msi_paths, labels, img_size=IMG_SIZE, msi_channels=None):
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.labels = labels
        self.img_size = img_size
        self.msi_channels = msi_channels

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = load_rgb_image(self.rgb_paths[idx], self.img_size)
        msi = load_msi_image(self.msi_paths[idx], self.img_size, target_channels=self.msi_channels)
        label = self.labels[idx]
        return rgb, msi, label


# 从对比模型文件夹中的代码导入模型
from models.resnet_total import HSIRGBResNet18
from models.RNN_total import BruiseDetectionLSTM
from models.VGG_total import MinimalVGG19
from models.vit_total import ViTWithProb, load_vit_simple

# ========== 模型包装器 ==========
class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, msi_channels=10):
        super(ResNet18Wrapper, self).__init__()
        # 使用HSIRGBResNet18模型
        self.resnet = HSIRGBResNet18(hsi_channels=msi_channels)
    
    def forward(self, msi_data, rgb_img):
        # 同时使用MSI和RGB输入
        out = self.resnet(msi_data, rgb_img)
        return out, None


class RNNWrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, msi_channels=10):
        super(RNNWrapper, self).__init__()
        # 使用BruiseDetectionLSTM模型
        self.rnn = BruiseDetectionLSTM(msi_bands=msi_channels, rgb_channels=3, num_layers=2)
    
    def forward(self, msi_data, rgb_img):
        # 同时使用MSI和RGB输入
        out = self.rnn(msi_data, rgb_img)
        return out, None


class VGG19Wrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGG19Wrapper, self).__init__()
        # 使用MinimalVGG19模型
        self.vgg = MinimalVGG19()
    
    def forward(self, msi_data, rgb_img):
        # 同时使用MSI和RGB输入
        out = self.vgg(msi_data, rgb_img)
        return out, None


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, msi_channels=10):
        super(ViTWrapper, self).__init__()
        # 使用ViTWithProb模型
        base_model = load_vit_simple()
        self.vit = ViTWithProb(base_model, hs_dim=msi_channels, num_classes=num_classes)
    
    def forward(self, msi_data, rgb_img):
        # 同时使用MSI和RGB输入
        out = self.vit(msi_data, rgb_img)
        return out, None


class GMAWrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, msi_channels=10):
        super(GMAWrapper, self).__init__()
        # 使用BruiseSTFNet模型
        self.gma = BruiseSTFNet(msi_channels=msi_channels, num_classes=num_classes)
    
    def forward(self, msi_data, rgb_img):
        return self.gma(msi_data=msi_data, rgb_img=rgb_img)


# ========== 模型性能统计函数 ==========
def count_parameters(model):
    """计算模型参数量 P(M)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6  # 转换为百万(M)


def measure_inference_speed(model, device, test_loader, num_warmup=5, num_runs=20):
    """
    测量模型推理速度 F(G) 和吞吐量 Thr(imgs/s)
    F(G): GFLOPs (每秒十亿次浮点运算)
    Thr: 吞吐量 (imgs/s)
    """
    model.eval()
    
    # 获取第一个批次的真实数据作为测试输入
    for batch in test_loader:
        msi_data, rgb_img, _ = batch
        msi_data = msi_data.to(device)
        rgb_img = rgb_img.to(device)
        break
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(msi_data=msi_data, rgb_img=rgb_img)
    
    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(msi_data=msi_data, rgb_img=rgb_img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算平均推理时间 (秒)
    avg_inference_time = (end_time - start_time) / num_runs
    
    # 计算吞吐量 (imgs/s)
    batch_size = msi_data.size(0)
    throughput = batch_size / avg_inference_time
    
    # 估算FLOPs (使用thop库或简单估算)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(msi_data, rgb_img), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        # 如果没有thop库，使用简化的估算
        input_shape_rgb = rgb_img.shape
        input_shape_msi = msi_data.shape
        gflops = estimate_flops_simple(model, input_shape_rgb, input_shape_msi)
        logger.warning("未安装thop库，使用简化估算计算GFLOPs。")
    
    return gflops, throughput, avg_inference_time * 1000  # 返回GFLOPs, 吞吐量, 平均推理时间(ms)


def estimate_flops_simple(model, input_shape_rgb, input_shape_msi):
    """
    简化的FLOPs估算方法（当thop库不可用时）
    """
    total_flops = 0
    
    # 遍历所有模块进行估算
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs = 2 * Cin * Cout * K * K * Hout * Wout
            # 简化为: 2 * Cin * Cout * K * K * H * W (假设输出尺寸与输入相近)
            cin = module.in_channels
            cout = module.out_channels
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            # 假设特征图大小为输入的1/4 (经过若干下采样)
            h, w = input_shape_rgb[-2] // 4, input_shape_rgb[-1] // 4
            flops = 2 * cin * cout * k * k * h * w
            total_flops += flops
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = 2 * Cin * Cout
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm FLOPs ≈ 4 * C * H * W
            h, w = input_shape_rgb[-2] // 4, input_shape_rgb[-1] // 4
            flops = 4 * module.num_features * h * w
            total_flops += flops
    
    return total_flops / 1e9  # 转换为GFLOPs


# ========== 主测试函数 ==========
def test_model_performance():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 确认使用设备
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"使用GPU加速测试: {device_name}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
    else:
        logger.info("使用CPU测试（建议使用GPU以加速测试）")

    logger.info("收集数据路径...")
    rgb_paths, msi_paths, labels = collect_data_paths()
    if len(rgb_paths) == 0:
        logger.error("错误: 没有找到任何数据文件")
        return
    logger.info(f"找到 {len(rgb_paths)} 个样本")

    # 估算msi通道数
    sample_msi = load_msi_image(msi_paths[0], IMG_SIZE)
    msi_channels = sample_msi.shape[0]
    logger.info(f"MSI图像通道数: {msi_channels}")

    # 创建数据加载器
    test_dataset = MultiModalFruitDataset(rgb_paths, msi_paths, labels, IMG_SIZE, msi_channels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

    # 定义要测试的模型
    models_to_test = {
        "ResNet18": lambda: ResNet18Wrapper(msi_channels=msi_channels),
        "RNN": lambda: RNNWrapper(msi_channels=msi_channels),
        "VGG19": VGG19Wrapper,
        "ViT": lambda: ViTWrapper(msi_channels=msi_channels),
        "GMA": lambda: GMAWrapper(msi_channels=msi_channels)
    }

    # 存储所有模型的性能结果
    performance_results = {}

    # 测试每个模型
    for model_name, model_constructor in models_to_test.items():
        logger.info(f"\n===== 测试 {model_name} 模型 =====")
        
        # 构建模型
        model = model_constructor().to(DEVICE)
        
        # 计算模型参数量 P(M)
        total_params_m, trainable_params_m = count_parameters(model)
        logger.info(f"总参数量 P(M): {total_params_m:.2f}M")
        logger.info(f"可训练参数量: {trainable_params_m:.2f}M")
        
        # 测量推理速度和吞吐量
        try:
            gflops, throughput, avg_time_ms = measure_inference_speed(
                model, DEVICE, test_loader, num_warmup=5, num_runs=20
            )
            logger.info(f"推理速度 F(G): {gflops:.2f} GFLOPs")
            logger.info(f"吞吐量 Thr: {throughput:.2f} imgs/s")
            logger.info(f"平均推理时间: {avg_time_ms:.2f} ms")
        except Exception as e:
            logger.warning(f"性能测试时出错: {e}")
            gflops, throughput, avg_time_ms = 0.0, 0.0, 0.0
        
        # 保存结果
        performance_results[model_name] = {
            "total_params_m": total_params_m,
            "trainable_params_m": trainable_params_m,
            "gflops": gflops,
            "throughput": throughput,
            "avg_time_ms": avg_time_ms
        }

    # 保存所有模型的性能结果到文件
    results_file = os.path.join(RESULT_DIR, f"model_performance_results_{timestamp}.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("===== 模型性能测试结果 =====\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"设备: {DEVICE}\n")
        f.write(f"批量大小: {BATCH_SIZE}\n")
        f.write(f"输入尺寸: RGB={IMG_SIZE}, MSI通道数={msi_channels}\n\n")
        
        # 写入表头
        f.write("模型名称\t总参数量 P(M) (M)\t可训练参数量 (M)\t推理速度 F(G) (GFLOPs)\t吞吐量 Thr (imgs/s)\t平均推理时间 (ms)\n")
        
        # 写入每个模型的结果
        for model_name, results in performance_results.items():
            f.write(f"{model_name}\t{results['total_params_m']:.2f}\t{results['trainable_params_m']:.2f}\t{results['gflops']:.2f}\t{results['throughput']:.2f}\t{results['avg_time_ms']:.2f}\n")
    
    logger.info(f"\n所有模型性能测试完成！结果已保存到: {results_file}")
    
    # 打印汇总结果
    logger.info("\n===== 性能测试汇总 =====")
    for model_name, results in performance_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  参数量 P(M): {results['total_params_m']:.2f}M")
        logger.info(f"  推理速度 F(G): {results['gflops']:.2f} GFLOPs")
        logger.info(f"  吞吐量 Thr: {results['throughput']:.2f} imgs/s")


if __name__ == "__main__":
    test_model_performance()
