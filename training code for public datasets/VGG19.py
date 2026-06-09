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
IMG_SIZE = (32, 32)  # VGG19适配的输入尺寸，可改96/96节省显存
BATCH_SIZE = 16  # 显存不足改 16/8
EPOCHS = 30
CHANNELS = 3  # RGB彩色图片
CLASS_MODE = "categorical"  # 14分类必须用 categorical
SEED = 42  # 固定随机种子，结果可复现
LEARNING_RATE = 1e-4  # VGG19建议小学习率

# 定义7类水果（需和文件夹命名中的水果名完全匹配，如banana/apple等）
FRUIT_LIST = ["banana", "apples", "oranges", "cucumber", "okra", "potato", "tomato"]
# 状态映射（fresh=健康，rotten=损伤）
STATUS_MAP = {"fresh": "健康", "rotten": "损伤"}


# ===================== 2. 数据加载+数据增强（适配fresh/rotten命名） =====================
# 清理空文件夹（防止误识别为类别）
def clean_empty_classes(dir_path):
    for class_dir in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_dir)
        if os.path.isdir(class_path) and len(os.listdir(class_path)) == 0:
            os.rmdir(class_path)
            print(f"删除空文件夹：{class_path}")


clean_empty_classes(TRAIN_DIR)
clean_empty_classes(TEST_DIR)

# 训练集：数据增强（防止过拟合）
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

# 测试集：只归一化，不做任何增强（评估真实效果）
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
    shuffle=False  # 评估时必须关闭打乱，保证标签一一对应
)

# 确认类别信息 ✅ 适配fresh/rotten命名
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
    # 拆分fresh/rotten和水果名（如freshbanana → fresh + banana）
    if cls.startswith("fresh"):
        status = "fresh"
        fruit = cls.replace("fresh", "")
    elif cls.startswith("rotten"):
        status = "rotten"
        fruit = cls.replace("rotten", "")
    else:
        # 兼容异常命名（可选：抛出提示）
        print(f"⚠️  类别{cls}不符合fresh/rotten命名规则，跳过")
        continue
    class_to_fruit_status[cls] = {"fruit": fruit, "status": status}
print("✅ 类别→水果+状态映射：", class_to_fruit_status)


# ===================== 新增 START --- 仅新增这一段回调函数，无任何修改 =====================
class FruitMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator, class_idx_to_name, class_to_fruit_status, FRUIT_LIST, STATUS_MAP):
        super().__init__()
        self.test_generator = test_generator
        self.class_idx_to_name = class_idx_to_name
        self.class_to_fruit_status = class_to_fruit_status
        self.FRUIT_LIST = FRUIT_LIST
        self.STATUS_MAP = STATUS_MAP

    def on_epoch_end(self, epoch, logs=None):
        """每轮训练结束后执行 - 完全复用原代码的指标计算逻辑"""
        print(f"\n{'=' * 80}")
        print(f"✅ 第 {epoch + 1}/{EPOCHS} 轮训练结束 → 按7类水果统计分类指标")
        print(f"{'=' * 80}")

        # 获取测试集预测结果 - 复用原逻辑
        test_pred = self.model.predict(self.test_generator, verbose=0)
        test_pred_classes = np.argmax(test_pred, axis=1)
        test_true_classes = self.test_generator.classes

        # 构建统计字典 - 完全复用原逻辑
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

        # 计算并打印指标 - 完全复用原逻辑
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


# ===================== 3. VGG19_N网络搭建（替换原ResNet18） =====================
def vgg19_block(inputs, filters, num_convs, pool_size=2, pool_stride=2):
    """VGG19基础卷积块：num_convs个卷积层 + 1个池化层"""
    x = inputs
    for _ in range(num_convs):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization(axis=-1)(x)  # 新增BN层提升稳定性
    x = layers.MaxPooling2D((pool_size, pool_size), strides=pool_stride)(x)
    return x


def build_VGG19_N():
    """VGG19_N完整结构，输出适配14分类"""
    inputs = layers.Input(shape=(*IMG_SIZE, CHANNELS))

    # VGG19核心卷积层（5组卷积块）
    x = vgg19_block(inputs, 64, 4)    # 第1组：4个64卷积 + 池化
    x = vgg19_block(x, 128, 4)        # 第2组：4个128卷积 + 池化
    x = vgg19_block(x, 256, 4)        # 第3组：4个256卷积 + 池化
    x = vgg19_block(x, 512, 4)        # 第4组：4个512卷积 + 池化
    x = vgg19_block(x, 512, 3)        # 第5组：3个512卷积 + 池化（总计19层卷积）

    # 全连接层（适配小尺寸输入，减少参数）
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)        # Dropout防过拟合
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)        # 第二层Dropout

    # 14分类输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # 编译模型（保持和原ResNet18一致的优化器/损失函数）
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',  # 多分类必用损失函数
        metrics=['accuracy']
    )
    return model


# 初始化VGG19_N模型
model = build_VGG19_N()
model.summary()

# ===================== 4. 开始训练 - 完全复用原逻辑 =====================
print("\n✅ 开始训练VGG19_N (14分类：7水果×健康/损伤) ==========")
# 构建索引到类名的映射
class_idx_to_name = {v: k for k, v in train_generator.class_indices.items()}
# 初始化自定义回调
fruit_metrics_callback = FruitMetricsCallback(test_generator, class_idx_to_name, class_to_fruit_status, FRUIT_LIST,
                                              STATUS_MAP)
# 训练时加入回调
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

# 5.2 按水果+状态拆分指标（关键：通过预测结果计算每个水果的健康/损伤分类效果）
# 步骤1：获取测试集预测结果
test_pred = model.predict(test_generator, verbose=0)
test_pred_classes = np.argmax(test_pred, axis=1)  # 预测类别索引
test_true_classes = test_generator.classes  # 真实类别索引

# 步骤2：构建「水果→{健康/损伤: {acc: xx, loss: xx}}」的统计字典
fruit_metrics = defaultdict(lambda: {"健康": {"correct": 0, "total": 0}, "损伤": {"correct": 0, "total": 0}})

# 步骤3：遍历测试集样本，统计每个水果的健康/损伤分类正确数
class_idx_to_name = {v: k for k, v in train_generator.class_indices.items()}  # 索引→类别名
for true_idx, pred_idx in zip(test_true_classes, test_pred_classes):
    # 获取真实/预测的类别名
    true_cls = class_idx_to_name[true_idx]
    pred_cls = class_idx_to_name[pred_idx]

    # 跳过异常类别（不符合fresh/rotten命名）
    if true_cls not in class_to_fruit_status or pred_cls not in class_to_fruit_status:
        continue

    # 解析真实标签的水果+状态
    true_fruit = class_to_fruit_status[true_cls]["fruit"]
    true_status = STATUS_MAP[class_to_fruit_status[true_cls]["status"]]

    # 统计总数
    fruit_metrics[true_fruit][true_status]["total"] += 1
    # 统计正确数（预测类别=真实类别）
    if pred_idx == true_idx:
        fruit_metrics[true_fruit][true_status]["correct"] += 1

# 步骤4：计算每个水果的健康/损伤准确率，并打印
fruit_stats = []
for fruit in FRUIT_LIST:
    if fruit not in fruit_metrics:
        print(f"⚠️  未找到{fruit}的测试数据")
        continue

    # 健康（fresh）指标
    fresh_total = fruit_metrics[fruit]["健康"]["total"]
    fresh_correct = fruit_metrics[fruit]["健康"]["correct"]
    fresh_acc = fresh_correct / fresh_total if fresh_total > 0 else 0.0

    # 损伤（rotten）指标
    rotten_total = fruit_metrics[fruit]["损伤"]["total"]
    rotten_correct = fruit_metrics[fruit]["损伤"]["correct"]
    rotten_acc = rotten_correct / rotten_total if rotten_total > 0 else 0.0

    # 水果整体准确率
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

    # 打印单水果统计
    print(f"\n🍎 {fruit.upper()} 统计：")
    print(f"  健康样本数：{fresh_total} | 健康分类准确率：{fresh_acc:.4f}")
    print(f"  损伤样本数：{rotten_total} | 损伤分类准确率：{rotten_acc:.4f}")
    print(f"  整体准确率：{fruit_acc:.4f}")

# 转换为DataFrame，方便查看/导出
fruit_stats_df = pd.DataFrame(fruit_stats)
print("\n✅ 所有水果统计表格：")
print(fruit_stats_df)

# ===================== 6. 可视化优化（完全复用原代码） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# 6.1 原有全局准确率/损失曲线
plt.figure(figsize=(15, 10))

# 子图1：全局准确率曲线
plt.subplot(2, 2, 1)
plt.plot(range(1, EPOCHS + 1), history.history['accuracy'], label='训练准确率', c='blue')
plt.plot(range(1, EPOCHS + 1), history.history['val_accuracy'], label='测试准确率', c='orange')
plt.title('VGG19_N 全局准确率曲线')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：全局损失曲线
plt.subplot(2, 2, 2)
plt.plot(range(1, EPOCHS + 1), history.history['loss'], label='训练损失', c='green')
plt.plot(range(1, EPOCHS + 1), history.history['val_loss'], label='测试损失', c='red')
plt.title('VGG19_N 全局损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('损失率')
plt.legend()
plt.grid(alpha=0.3)

# 6.2 新增：按水果的健康/损伤准确率对比
plt.subplot(2, 2, 3)
fruits = [item["水果"] for item in fruit_stats]
fresh_accs = [float(item["健康分类准确率"]) for item in fruit_stats]
rotten_accs = [float(item["损伤分类准确率"]) for item in fruit_stats]

x = np.arange(len(fruits))
width = 0.35  # 柱状图宽度

plt.bar(x - width / 2, fresh_accs, width, label='健康（fresh）', color='limegreen')
plt.bar(x + width / 2, rotten_accs, width, label='损伤（rotten）', color='tomato')

plt.title('各水果健康/损伤分类准确率')
plt.xlabel('水果种类')
plt.ylabel('准确率')
plt.xticks(x, fruits, rotation=45)
plt.legend()
plt.grid(alpha=0.3)

# 6.3 新增：各水果整体准确率
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

# ===================== 7. 保存结果（模型+统计表格）（完全复用原代码） =====================
# 保存模型
model_save_path = "D://DATA/dataset/VGG19_N_14class_model.h5"
model.save(model_save_path)
print(f"\n✅ 模型已保存至：{model_save_path}")

# 保存水果统计表格
stats_save_path = "D://DATA/dataset/水果分类统计结果VGG19.xlsx"
fruit_stats_df.to_excel(stats_save_path, index=False)
print(f"✅ 水果统计表格已保存至：{stats_save_path}")

# 打印全局核心指标
print("\n✅ 全局核心指标汇总：")
print(f"最终训练准确率 : {final_train_acc:.4f}")
print(f"最终训练损失率 : {final_train_loss:.4f}")
print(f"最终测试准确率 : {final_val_acc:.4f}")
print(f"最终测试损失率 : {final_val_loss:.4f}")