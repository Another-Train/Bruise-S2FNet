import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ===================== 1. 全局参数配置（适配14分类：7水果×fresh/rotten） =====================
TRAIN_DIR = "D://DATA/dataset/train"
TEST_DIR = "D://DATA/dataset/test"
IMG_SIZE = (64, 64)  # 保持原尺寸适配
BATCH_SIZE = 32
EPOCHS = 100
CHANNELS = 3
SEED = 42
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义7类水果（需和文件夹命名中的水果名完全匹配）
FRUIT_LIST = ["banana", "apples", "oranges", "cucumber", "okra", "potato", "tomato"]
# 状态映射（fresh=健康，rotten=损伤）
STATUS_MAP = {"fresh": "健康", "rotten": "损伤"}

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== 2. 数据加载+数据增强（适配PyTorch） =====================
# 清理空文件夹
def clean_empty_classes(dir_path):
    for class_dir in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_dir)
        if os.path.isdir(class_path) and len(os.listdir(class_path)) == 0:
            os.rmdir(class_path)
            print(f"删除空文件夹：{class_path}")


clean_empty_classes(TRAIN_DIR)
clean_empty_classes(TEST_DIR)

# 数据预处理（匹配BaseNet的预处理逻辑）
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 确认类别信息
class_names = train_dataset.classes
num_classes = len(class_names)
class_indices = train_dataset.class_to_idx
print("=" * 60)
print(f"✅ 训练集类别数量：{num_classes}")
print(f"✅ 测试集类别数量：{len(test_dataset.classes)}")
print(f"✅ 确认：训练/测试集类别数一致 = {num_classes == len(test_dataset.classes)}")
print(f"✅ 类别列表（fresh=健康/rotten=损伤）：{class_names}")
print("=" * 60)

# 构建「类别→水果+状态」的映射字典
class_to_fruit_status = {}
for cls in class_names:
    if cls.startswith("fresh"):
        status = "fresh"
        fruit = cls.replace("fresh", "")
    elif cls.startswith("rotten"):
        status = "rotten"
        fruit = cls.replace("rotten", "")
    else:
        print(f"⚠️  类别{cls}不符合fresh/rotten命名规则，跳过")
        continue
    class_to_fruit_status[cls] = {"fruit": fruit, "status": status}
print("✅ 类别→水果+状态映射：", class_to_fruit_status)


# ===================== 3. 复用BaseNet_N中的模型定义 =====================
class Multi_Attention(nn.Module):
    def __init__(self, channelNumber, patch_size):
        super(Multi_Attention, self).__init__()
        self.total_Channel_num = channelNumber
        self.patch_size = patch_size
        self.num_groups = min(3, channelNumber)  # RGB 3通道适配
        base_channels = channelNumber // self.num_groups
        remainder = channelNumber % self.num_groups
        self.group_channels = []
        for i in range(self.num_groups):
            self.group_channels.append(base_channels + 1 if i < remainder else base_channels)

    def forward(self, x, pixel_x):
        B, C, H, W = x.shape
        group_features = []
        start = 0
        for gc in self.group_channels:
            end = start + gc
            group = x[:, start:end, :, :]
            start = end
            attn = F.softmax(torch.matmul(group.flatten(2), group.flatten(2).transpose(1, 2)), dim=-1)
            group_feat = torch.matmul(attn, group.flatten(2)).reshape(B, gc, H, W)
            group_features.append(group_feat)
        return torch.cat(group_features, dim=1) + x


class GMABlock(nn.Module):
    def __init__(self, HSI_Data_Shape_C, patch_size):
        super(GMABlock, self).__init__()
        self.name = 'GMA_Net'
        self.band = HSI_Data_Shape_C  # RGB=3
        self.patch_size = patch_size
        self.gma = Multi_Attention(channelNumber=self.band, patch_size=self.patch_size)
        self.relu = nn.ReLU()

        # Pixel分支
        self.conv11 = nn.Sequential(nn.Conv2d(self.band, 32, kernel_size=(1, 1)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv12 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(1, 1)), nn.BatchNorm2d(32), nn.ReLU())

        # 修复BUG 核心代码
        self.channel_adapt = nn.Conv2d(32, self.band, kernel_size=(1, 1), bias=False)

    def forward(self, patchX, pixelX):
        spectral_features = self.gma(patchX, pixelX)  # 输出通道数=3
        pixel_feat = pixelX.unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]
        pixel_feat = self.conv12(self.conv11(pixel_feat))  # [B,32,1,1]
        pixel_feat = self.channel_adapt(pixel_feat)  # [B,3,1,1] → 通道数匹配
        pixel_feat = pixel_feat.expand_as(spectral_features)
        return self.relu(spectral_features + pixel_feat)


class SFE(nn.Module):
    def __init__(self, msi_channels, rgb_channels):
        super(SFE, self).__init__()
        self.msi_conv = nn.Sequential(nn.Conv2d(msi_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.rgb_conv = nn.Sequential(nn.Conv2d(rgb_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.fusion_conv = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())

    def forward(self, msi, rgb):
        msi_feat = self.msi_conv(msi)
        rgb_feat = self.rgb_conv(rgb)
        return self.fusion_conv(torch.cat([msi_feat, rgb_feat], dim=1))


class ES2CM(nn.Module):
    def __init__(self, input_channels, compression_ratio=0.5):
        super(ES2CM, self).__init__()
        self.compression = nn.Sequential(
            nn.Conv2d(input_channels, int(input_channels * compression_ratio), 1),
            nn.BatchNorm2d(int(input_channels * compression_ratio)),
            nn.ReLU()
        )
        self.output_channels = int(input_channels * compression_ratio)

    def forward(self, x):
        return self.compression(x)


class HFFG(nn.Module):
    def __init__(self, stm_dim, es2cm_dim, fusion_dim):
        super(HFFG, self).__init__()
        self.stm_proj = nn.Conv2d(stm_dim, fusion_dim, 1)
        self.es2cm_proj = nn.Conv2d(es2cm_dim, fusion_dim, 1)
        self.attn = nn.Sequential(nn.Conv2d(fusion_dim * 2, fusion_dim, 1), nn.Sigmoid())

    def forward(self, stm_feat, es2cm_feat):
        stm_proj = self.stm_proj(stm_feat)
        es2cm_proj = self.es2cm_proj(es2cm_feat)
        attn_map = self.attn(torch.cat([stm_proj, es2cm_proj], dim=1))
        fused = stm_proj * attn_map + es2cm_proj * (1 - attn_map)
        return fused, attn_map


class MSI_RGB_Fusion_Net(nn.Module):
    def __init__(self, msi_channels=3, rgb_channels=3, hidden_dim=32, fusion_dim=32, num_classes=num_classes):
        super().__init__()
        self.gma = GMABlock(HSI_Data_Shape_C=msi_channels, patch_size=IMG_SIZE[0])
        self.conv_1x1 = nn.Conv2d(msi_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.sfe = SFE(msi_channels=msi_channels, rgb_channels=rgb_channels)
        self.es2cm = ES2CM(input_channels=48, compression_ratio=32 / 48)
        self.fusion = HFFG(stm_dim=hidden_dim, es2cm_dim=self.es2cm.output_channels, fusion_dim=fusion_dim)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, rgb_data):
        B, C, H, W = rgb_data.shape
        center_h, center_w = H // 2, W // 2
        pixelX = rgb_data[:, :, center_h, center_w]
        spectral_features = self.gma(rgb_data, pixelX)
        spectral_features = self.conv_1x1(spectral_features)
        spatial_enhanced = self.sfe(rgb_data, rgb_data)
        es2cm_features = self.es2cm(spatial_enhanced)
        fused_features, attn_weights = self.fusion(spectral_features, es2cm_features)
        outputs = self.classifier(fused_features)
        return outputs


# 初始化模型、损失函数、优化器
model = MSI_RGB_Fusion_Net(msi_channels=3, rgb_channels=3, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 模型总参数量: {total_params / 1e6:.2f} M")
print(f"✅ 可训练参数量: {trainable_params / 1e6:.2f} M")
print("=" * 60)


# ===================== 4. 训练/测试函数（适配PyTorch） =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'训练中', leave=False)

    for batch_idx, (imgs, labels) in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

        # 进度条实时显示
        batch_loss = loss.item()
        batch_acc = torch.sum(preds == labels).item() / imgs.size(0)
        pbar.set_postfix({'批次损失': f'{batch_loss:.4f}', '批次准确率': f'{batch_acc:.4f}'})

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true, y_pred = [], []
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'测试中', leave=False)

    with torch.no_grad():
        for batch_idx, (imgs, labels) in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            total_correct += torch.sum(preds == labels).item()
            total_samples += imgs.size(0)

            # 进度条实时显示
            batch_loss = loss.item()
            pbar.set_postfix({'测试批次损失': f'{batch_loss:.4f}'})

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(y_true), np.array(y_pred)


# ===================== 5. 自定义指标统计（适配PyTorch） =====================
def calculate_fruit_metrics(y_true, y_pred, class_indices, class_to_fruit_status, FRUIT_LIST, STATUS_MAP):
    # 构建索引到类名的映射
    class_idx_to_name = {v: k for k, v in class_indices.items()}

    # 构建统计字典
    fruit_metrics = defaultdict(lambda: {"健康": {"correct": 0, "total": 0}, "损伤": {"correct": 0, "total": 0}})
    for true_idx, pred_idx in zip(y_true, y_pred):
        true_cls = class_idx_to_name[true_idx]
        pred_cls = class_idx_to_name[pred_idx]
        if true_cls not in class_to_fruit_status or pred_cls not in class_to_fruit_status:
            continue
        true_fruit = class_to_fruit_status[true_cls]["fruit"]
        true_status = STATUS_MAP[class_to_fruit_status[true_cls]["status"]]
        fruit_metrics[true_fruit][true_status]["total"] += 1
        if pred_idx == true_idx:
            fruit_metrics[true_fruit][true_status]["correct"] += 1

    # 计算指标
    fruit_stats = []
    for fruit in FRUIT_LIST:
        if fruit not in fruit_metrics:
            print(f"⚠️  未找到{fruit}的测试数据")
            continue

        fresh_total = fruit_metrics[fruit]["健康"]["total"]
        fresh_correct = fruit_metrics[fruit]["健康"]["correct"]
        fresh_acc = fresh_correct / fresh_total if fresh_total > 0 else 0.0

        rotten_total = fruit_metrics[fruit]["损伤"]["total"]
        rotten_correct = fruit_metrics[fruit]["损伤"]["correct"]
        rotten_acc = rotten_correct / rotten_total if rotten_total > 0 else 0.0

        fruit_total = fresh_total + rotten_total
        fruit_correct = fresh_correct + rotten_correct
        fruit_acc = fruit_correct / fruit_total if fruit_total > 0 else 0.0

        fruit_stats.append({
            "水果": fruit,
            "健康样本数": fresh_total,
            "健康分类准确率": f"{fresh_acc:.4f}",
            "损伤样本数": rotten_total,
            "损伤分类准确率": f"{rotten_acc:.4f}",
            "水果整体准确率": f"{fruit_acc:.4f}"
        })

        print(f"\n🍎 {fruit.upper()} 统计：")
        print(f"  健康样本数：{fresh_total} | 健康分类准确率：{fresh_acc:.4f}")
        print(f"  损伤样本数：{rotten_total} | 损伤分类准确率：{rotten_acc:.4f}")
        print(f"  整体准确率：{fruit_acc:.4f}")
    fruit_stats_df = pd.DataFrame(fruit_stats)
    print("\n✅ 所有水果分类指标汇总表：")
    print(fruit_stats_df)

    return fruit_stats


# ===================== 6. 模型训练主循环 =====================
print("\n✅ 开始训练MSI+RGB融合模型 (14分类：7水果×健康/损伤) ==========")
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    print(f"\n📌 第 [{epoch:02d}/{EPOCHS}] 轮训练")
    # 训练
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    # 测试
    test_loss, test_acc, y_true, y_pred = test_model(model, test_loader, criterion, DEVICE)
    # 学习率调度
    scheduler.step()

    # 记录指标
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # 打印轮次日志
    log_str = f"✅ 轮次 [{epoch:02d}/{EPOCHS}] "
    log_str += f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} "
    log_str += f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}"
    print(log_str)

    # 按水果维度统计指标
    print(f"\n{'=' * 80}")
    print(f"✅ 第 {epoch}/{EPOCHS} 轮训练结束 → 按7类水果统计分类指标")
    print(f"{'=' * 80}")
    calculate_fruit_metrics(y_true, y_pred, class_indices, class_to_fruit_status, FRUIT_LIST, STATUS_MAP)
    print(f"{'=' * 80}\n")

# ===================== 7. 最终指标统计 =====================
print("\n✅ 【最终轮次-按水果维度】统计准确率+损失率 ==========")
# 最终测试
final_test_loss, final_test_acc, y_true_final, y_pred_final = test_model(model, test_loader, criterion, DEVICE)
final_train_acc = train_accs[-1]
final_train_loss = train_losses[-1]

# 按水果维度统计
fruit_stats = calculate_fruit_metrics(y_true_final, y_pred_final, class_indices, class_to_fruit_status, FRUIT_LIST,
                                      STATUS_MAP)
fruit_stats_df = pd.DataFrame(fruit_stats)
print("\n✅ 所有水果统计表格：")
print(fruit_stats_df)

# ===================== 8. 可视化 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 10))

# 子图1：全局准确率曲线
plt.subplot(2, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_accs, label='训练准确率', c='blue')
plt.plot(range(1, EPOCHS + 1), test_accs, label='测试准确率', c='orange')
plt.title('MSI+RGB融合模型 全局准确率曲线')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：全局损失曲线
plt.subplot(2, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_losses, label='训练损失', c='green')
plt.plot(range(1, EPOCHS + 1), test_losses, label='测试损失', c='red')
plt.title('MSI+RGB融合模型 全局损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('损失率')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：按水果的健康/损伤准确率对比
plt.subplot(2, 2, 3)
fruits = [item["水果"] for item in fruit_stats]
fresh_accs = [float(item["健康分类准确率"]) for item in fruit_stats]
rotten_accs = [float(item["损伤分类准确率"]) for item in fruit_stats]

x = np.arange(len(fruits))
width = 0.35

plt.bar(x - width / 2, fresh_accs, width, label='健康（fresh）', color='limegreen')
plt.bar(x + width / 2, rotten_accs, width, label='损伤（rotten）', color='tomato')

plt.title('各水果健康/损伤分类准确率')
plt.xlabel('水果种类')
plt.ylabel('准确率')
plt.xticks(x, fruits, rotation=45)
plt.legend()
plt.grid(alpha=0.3)

# 子图4：各水果整体准确率
plt.subplot(2, 2, 4)
fruit_accs = [float(item["水果整体准确率"]) for item in fruit_stats]
plt.bar(fruits, fruit_accs, color='cornflowerblue')
plt.title('各水果整体分类准确率')
plt.xlabel('水果种类')
plt.ylabel('准确率')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===================== 9. 保存结果 =====================
# 保存模型
model_save_path = "D://DATA/dataset/BaseNet_14class_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 模型已保存至：{model_save_path}")

# 保存水果统计表格
stats_save_path = "D://DATA/dataset/水果分类统计结果BaseNet.xlsx"
fruit_stats_df.to_excel(stats_save_path, index=False)
print(f"✅ 水果统计表格已保存至：{stats_save_path}")

# 打印全局核心指标
print("\n✅ 全局核心指标汇总：")
print(f"最终训练准确率 : {final_train_acc:.4f}")
print(f"最终训练损失率 : {final_train_loss:.4f}")
print(f"最终测试准确率 : {final_test_acc:.4f}")
print(f"最终测试损失率 : {final_test_loss:.4f}")