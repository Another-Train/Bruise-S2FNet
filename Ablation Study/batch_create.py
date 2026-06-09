# 批量创建训练脚本
import os

# 模型配置
models = [
    ("base_PSAB", "models.base_PSAB"),
    ("full_ES2CM", "models.full_ES2CM"),
    ("full_HFFG", "models.full_HFFG"),
    ("full_PSAB", "models.full_PSAB"),
    ("full_over_foc", "models.full_over_foc"),
]

# 读取基础文件
with open("train_base_HFFG.py", "r", encoding="utf-8") as f:
    base_content = f.read()

for model_name, import_path in models:
    # 替换导入路径
    content = base_content.replace(
        "from models.base_HFFG import BruiseSTFNet",
        f"from {import_path} import BruiseSTFNet"
    )
    # 替换模型名称
    content = content.replace(
        'MODEL_NAME = "base_HFFG"',
        f'MODEL_NAME = "{model_name}"'
    )
    content = content.replace(
        "# 模型名称（用于结果目录）- 当前模型: base_HFFG",
        f"# 模型名称（用于结果目录）- 当前模型: {model_name}"
    )
    
    # 写入文件
    filename = f"train_{model_name}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filename}")

print("\nAll training scripts created successfully!")
