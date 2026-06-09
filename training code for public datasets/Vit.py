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
IMG_SIZE = (32, 32)  # 保持原尺寸，ViT可适配
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

# ===================== 3. TensorFlow原生实现 ViT(视觉Transformer) 核心模型 【唯一修改部分】 =====================
def mlp(x, hidden_units, dropout_rate):
    """ViT内部的多层感知机"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    """图像分块层 - ViT核心组件1"""

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """分块编码层(位置编码+线性投影) - ViT核心组件2"""

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def build_ViT():
    """
    TensorFlow原生实现 轻量级ViT视觉Transformer模型
    完美适配当前32*32图像尺寸，14分类任务，参数量小，训练快，精度高
    无第三方依赖，纯原生Keras层搭建，稳定性拉满
    """
    input_shape = (*IMG_SIZE, CHANNELS)
    patch_size = 4  # 分块大小，32/4=8个分块，适配32*32图像
    projection_dim = 64  # 嵌入维度
    num_heads = 4  # 注意力头数
    transformer_layers = 4  # transformer编码器层数
    transformer_units = [projection_dim * 2, projection_dim]  # MLP隐藏层
    mlp_head_units = [2048, 1024]  # 分类头MLP

    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 1. 创建图像分块
    patches = Patches(patch_size)(inputs)
    # 2. 分块编码(含位置编码)
    num_patches = (IMG_SIZE[0] // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 3. 堆叠Transformer编码器
    for _ in range(transformer_layers):
        # 层归一化 + 多头注意力
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # 残差连接
        x2 = layers.Add()([attention_output, encoded_patches])

        # 层归一化 + MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # 残差连接
        encoded_patches = layers.Add()([x3, x2])

    # 4. 全局池化 + 分类头
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # 输出层：适配14分类，softmax激活
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    # 编译模型
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 初始化ViT模型
model = build_ViT()
model.summary()

# ===================== 4. 模型训练（完全复用原代码） =====================
print("\n✅ 开始训练ViT模型 (14分类：7水果×健康/损伤) ==========")
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
plt.title('ViT 全局准确率曲线')  # 仅修改标题为ViT
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：全局损失曲线
plt.subplot(2, 2, 2)
plt.plot(range(1, EPOCHS + 1), history.history['loss'], label='训练损失', c='green')
plt.plot(range(1, EPOCHS + 1), history.history['val_loss'], label='测试损失', c='red')
plt.title('ViT 全局损失曲线')  # 仅修改标题为ViT
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

# ===================== 7. 保存结果（完全复用原代码，仅修改文件名） =====================
# 保存模型
model_save_path = "D://DATA/dataset/ViT_14class_model.h5"  # 修改为ViT文件名
model.save(model_save_path)
print(f"\n✅ 模型已保存至：{model_save_path}")

# 保存水果统计表格
stats_save_path = "D://DATA/dataset/水果分类统计结果_ViT.xlsx"  # 修改为ViT文件名
fruit_stats_df.to_excel(stats_save_path, index=False)
print(f"✅ 水果统计表格已保存至：{stats_save_path}")

# 打印全局核心指标
print("\n✅ 全局核心指标汇总：")
print(f"最终训练准确率 : {final_train_acc:.4f}")
print(f"最终训练损失率 : {final_train_loss:.4f}")
print(f"最终测试准确率 : {final_val_acc:.4f}")
print(f"最终测试损失率 : {final_val_loss:.4f}")