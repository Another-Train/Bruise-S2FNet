import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
from collections import defaultdict

# ===================== 1. 全局参数配置（适配14分类：7水果×fresh/rotten） =====================
TRAIN_DIR = "D://DATA/dataset/train"
TEST_DIR = "D://DATA/dataset/test"
IMG_SIZE = (32, 32)  # RNN输入适配尺寸，保持和原代码一致
BATCH_SIZE = 16
EPOCHS = 30
CHANNELS = 3
CLASS_MODE = "categorical"
SEED = 42
LEARNING_RATE = 1e-4

# 定义7类水果（需和文件夹命名中的水果名完全匹配）
FRUIT_LIST = ["banana", "apples", "oranges", "cucumber", "okra", "potato", "tomato"]
# 状态映射（fresh=健康，rotten=损伤）
STATUS_MAP = {"fresh": "健康", "rotten": "损伤"}

# ===================== 2. 数据加载+数据增强（完全复用原代码） =====================
# 清理空文件夹
def clean_empty_classes(dir_path):
    for class_dir in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_dir)
        if os.path.isdir(class_path) and len(os.listdir(class_path)) == 0:
            os.rmdir(class_path)
            print(f"删除空文件夹：{class_path}")

clean_empty_classes(TRAIN_DIR)
clean_empty_classes(TEST_DIR)

# 训练集数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest"
)

# 测试集仅归一化
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 加载训练集
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    seed=SEED,
    shuffle=True
)

# 加载测试集
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    seed=SEED,
    shuffle=False
)

# 确认类别信息
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print("=" * 60)
print(f"✅ 训练集类别数量：{num_classes}")
print(f"✅ 测试集类别数量：{len(test_generator.class_indices)}")
print(f"✅ 确认：训练/测试集类别数一致 = {num_classes == len(test_generator.class_indices)}")
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

# ===================== 自定义回调函数（完全复用原代码） =====================
class FruitMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator, class_idx_to_name, class_to_fruit_status, FRUIT_LIST, STATUS_MAP):
        super().__init__()
        self.test_generator = test_generator
        self.class_idx_to_name = class_idx_to_name
        self.class_to_fruit_status = class_to_fruit_status
        self.FRUIT_LIST = FRUIT_LIST
        self.STATUS_MAP = STATUS_MAP

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n{'=' * 80}")
        print(f"✅ 第 {epoch + 1}/{EPOCHS} 轮训练结束 → 按7类水果统计分类指标")
        print(f"{'=' * 80}")

        # 获取测试集预测结果
        test_pred = self.model.predict(self.test_generator, verbose=0)
        test_pred_classes = np.argmax(test_pred, axis=1)
        test_true_classes = self.test_generator.classes

        # 构建统计字典
        fruit_metrics = defaultdict(lambda: {"健康": {"correct": 0, "total": 0}, "损伤": {"correct": 0, "total": 0}})
        for true_idx, pred_idx in zip(test_true_classes, test_pred_classes):
            true_cls = self.class_idx_to_name[true_idx]
            pred_cls = self.class_idx_to_name[pred_idx]
            if true_cls not in self.class_to_fruit_status or pred_cls not in self.class_to_fruit_status:
                continue
            true_fruit = self.class_to_fruit_status[true_cls]["fruit"]
            true_status = self.STATUS_MAP[self.class_to_fruit_status[true_cls]["status"]]
            fruit_metrics[true_fruit][true_status]["total"] += 1
            if pred_idx == true_idx:
                fruit_metrics[true_fruit][true_status]["correct"] += 1

        # 计算并打印指标
        fruit_stats_epoch = []
        for fruit in self.FRUIT_LIST:
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

            fruit_stats_epoch.append({
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

        df_epoch = pd.DataFrame(fruit_stats_epoch)
        print(f"\n📋 第{epoch + 1}轮 7类水果分类指标汇总表：")
        print(df_epoch.to_string(index=False))
        print(f"{'=' * 80}\n")

# ===================== 3. RNN模型搭建（替换原ResNet18） =====================
def build_RNN_model():
    """
    构建适用于图像分类的RNN模型：
    1. 先通过卷积层提取空间特征
    2. 将特征展平为序列输入到LSTM（RNN的一种）
    3. 全连接层输出分类结果
    """
    inputs = layers.Input(shape=(*IMG_SIZE, CHANNELS))

    # 步骤1：卷积特征提取（降低维度，提取关键空间特征）
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 步骤2：将卷积特征转换为序列（适配RNN输入）
    # 获取特征维度：(batch, height, width, channels) → 展平为 (batch, sequence_len, feature_dim)
    shape = x.shape  # shape[1]=height, shape[2]=width, shape[3]=channels
    sequence_len = shape[1] * shape[2]  # 序列长度 = 特征图的高×宽
    feature_dim = shape[3]  # 每个序列步的特征维度 = 通道数
    x = layers.Reshape((sequence_len, feature_dim))(x)  # 重塑为序列格式

    # 步骤3：RNN层（LSTM）提取序列特征
    x = layers.LSTM(128, return_sequences=False)(x)  # 单层LSTM，输出最终状态
    x = layers.Dropout(0.5)(x)  # 防过拟合

    # 步骤4：分类输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # 编译模型（保持和原代码一致的优化器、损失函数）
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 初始化RNN模型
model = build_RNN_model()
model.summary()

# ===================== 4. 模型训练（完全复用原代码） =====================
print("\n✅ 开始训练RNN模型 (14分类：7水果×健康/损伤) ==========")
# 构建索引到类名的映射
class_idx_to_name = {v: k for k, v in train_generator.class_indices.items()}
# 初始化自定义回调
fruit_metrics_callback = FruitMetricsCallback(test_generator, class_idx_to_name, class_to_fruit_status, FRUIT_LIST,
                                              STATUS_MAP)
# 训练模型
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1,
    callbacks=[fruit_metrics_callback]
)

# ===================== 5. 按水果维度统计指标（完全复用原代码） =====================
print("\n✅ 【最终轮次-按水果维度】统计准确率+损失率 ==========")

# 5.1 基础训练/测试指标（全局）
final_train_acc = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]

# 5.2 按水果+状态拆分指标
test_pred = model.predict(test_generator, verbose=0)
test_pred_classes = np.argmax(test_pred, axis=1)
test_true_classes = test_generator.classes

# 构建统计字典
fruit_metrics = defaultdict(lambda: {"健康": {"correct": 0, "total": 0}, "损伤": {"correct": 0, "total": 0}})
class_idx_to_name = {v: k for k, v in train_generator.class_indices.items()}
for true_idx, pred_idx in zip(test_true_classes, test_pred_classes):
    true_cls = class_idx_to_name[true_idx]
    pred_cls = class_idx_to_name[pred_idx]
    if true_cls not in class_to_fruit_status or pred_cls not in class_to_fruit_status:
        continue
    true_fruit = class_to_fruit_status[true_cls]["fruit"]
    true_status = STATUS_MAP[class_to_fruit_status[true_cls]["status"]]
    fruit_metrics[true_fruit][true_status]["total"] += 1
    if pred_idx == true_idx:
        fruit_metrics[true_fruit][true_status]["correct"] += 1

# 计算并打印每个水果的指标
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

# 转换为DataFrame
fruit_stats_df = pd.DataFrame(fruit_stats)
print("\n✅ 所有水果统计表格：")
print(fruit_stats_df)

# ===================== 6. 可视化（完全复用原代码） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 10))

# 子图1：全局准确率曲线
plt.subplot(2, 2, 1)
plt.plot(range(1, EPOCHS + 1), history.history['accuracy'], label='训练准确率', c='blue')
plt.plot(range(1, EPOCHS + 1), history.history['val_accuracy'], label='测试准确率', c='orange')
plt.title('RNN 全局准确率曲线')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：全局损失曲线
plt.subplot(2, 2, 2)
plt.plot(range(1, EPOCHS + 1), history.history['loss'], label='训练损失', c='green')
plt.plot(range(1, EPOCHS + 1), history.history['val_loss'], label='测试损失', c='red')
plt.title('RNN 全局损失曲线')
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

# ===================== 7. 保存结果（完全复用原代码） =====================
# 保存模型
model_save_path = "D://DATA/dataset/RNN_14class_model.h5"
model.save(model_save_path)
print(f"\n✅ 模型已保存至：{model_save_path}")

# 保存水果统计表格
stats_save_path = "D://DATA/dataset/水果分类统计结果_RNN.xlsx"
fruit_stats_df.to_excel(stats_save_path, index=False)
print(f"✅ 水果统计表格已保存至：{stats_save_path}")

# 打印全局核心指标
print("\n✅ 全局核心指标汇总：")
print(f"最终训练准确率 : {final_train_acc:.4f}")
print(f"最终训练损失率 : {final_train_loss:.4f}")
print(f"最终测试准确率 : {final_val_acc:.4f}")
print(f"最终测试损失率 : {final_val_loss:.4f}")