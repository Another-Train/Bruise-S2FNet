#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量创建训练脚本
"""

import shutil
import os

# 模型配置列表
models_config = [
    ("base_ES2CM", "models.base_ES2CM"),
    ("base_HFFG", "models.base_HFFG"),
    ("base_PSAB", "models.base_PSAB"),
    ("full_ES2CM", "models.full_ES2CM"),
    ("full_HFFG", "models.full_HFFG"),
    ("full_PSAB", "models.full_PSAB"),
    ("full_over_foc", "models.full_over_foc"),
]

base_file = "train_二分类.py"

for model_name, import_path in models_config:
    target_file = f"train_{model_name}.py"
    
    # 读取原文件
    with open(base_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换模型导入
    content = content.replace(
        'from models.base_ES2CM import BruiseSTFNet',
        f'from {import_path} import BruiseSTFNet'
    )
    
    # 替换模型名称
    content = content.replace(
        'MODEL_NAME = "base_ES2CM"',
        f'MODEL_NAME = "{model_name}"'
    )
    
    # 写入新文件
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已创建: {target_file}")

print("\n所有训练脚本创建完成！")
