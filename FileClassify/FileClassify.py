import os
import shutil
import re
from sklearn.model_selection import train_test_split

# --- 配置参数 ---
basedir = 'D:/DATA/gtdata1'  # 原始数据根目录（必须与实际路径完全一致）
output_dir = 'data'
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
# 修改：将有效时间点改为字符串，包括01
valid_times = {'0',
               '6',
               '12',
               '18',
               '24',
               '30',
               '36',
               '01'
                }

# 原始目录与输出目录的映射（必须与实际文件夹名一致！）
raw_to_output = {
    'hyperspectral_cropped': 'MSI',  # 高光谱原始目录 → 输出MSI
    'rgb_cropped': 'RGB'             # RGB原始目录 → 输出RGB
}
raw_dirs = list(raw_to_output.keys())  # 原始目录列表

# 各原始目录允许的文件格式（严格限制）
allowed_extensions = {
    'hyperspectral_cropped': {'.hdr', '.float'},
    'rgb_cropped': {'.png'}
}

valid_prefixes = {'bd', 'jk', 'ys','jjk'}


# --- 解析文件名提取时间点 ---
def get_time_from_filename(filename):
    # 修改：返回字符串而不是整数
    pattern = re.compile(r'^(bd|jk|jjk|ys)-(\d+)-.*z')
    match = pattern.match(filename)
    if match and match.group(1) in valid_prefixes:
        return match.group(2)  # 直接返回字符串，不转换为整数
    return None


# --- 验证文件是否属于当前原始目录 ---
def is_valid_file(raw_dir, filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions[raw_dir]


# --- 收集所有有效文件并标记原始目录 ---
def collect_files():
    file_info = []  # 存储元组：(前缀, 时间点, 原始目录, 文件路径)
    for raw_dir in raw_dirs:
        raw_path = os.path.join(basedir, raw_dir)
        # 检查原始目录是否存在（关键！）
        if not os.path.exists(raw_path):
            print(f"❌ 错误：原始目录不存在 → {raw_path}（请检查路径拼写）")
            continue
        if not os.path.isdir(raw_path):
            print(f"❌ 错误：{raw_path} 不是目录（请确认是文件夹）")
            continue

        # 遍历目录下的文件
        for filename in os.listdir(raw_path):
            file_path = os.path.join(raw_path, filename)
            if not os.path.isfile(file_path):
                continue  # 跳过子目录

            # 1. 验证文件格式是否符合当前原始目录
            if not is_valid_file(raw_dir, filename):
                print(f"⚠️ 跳过：{filename} 在 {raw_dir} 目录，但格式不符合（需{allowed_extensions[raw_dir]}）")
                continue

            # 2. 提取时间点并验证
            time_point = get_time_from_filename(filename)
            if time_point is None or time_point not in valid_times:
                print(f"⚠️ 跳过：{filename} 时间点无效（需{valid_times}）")
                continue

            # 3. 提取文件前缀（用于分组，确保同一样本的文件被同时划分）
            prefix = os.path.splitext(filename)[0]
            file_info.append((prefix, time_point, raw_dir, file_path))
            # 详细日志：确认文件所属原始目录
            print(f"✅ 收集：{filename} → 原始目录：{raw_dir} → 时间点：{time_point}")

    print(f"\n📊 共收集有效文件：{len(file_info)} 个")
    return file_info


# --- 按时间点和前缀分组（确保同一样本的文件在同一子集） ---
def group_files(file_info):
    # 按 (前缀, 时间点) 分组（同一前缀+时间点属于同一样本）
    groups = {}
    for prefix, time, raw_dir, path in file_info:
        key = (prefix, time)
        if key not in groups:
            groups[key] = {'time': time, 'raw_dirs': set(), 'paths': []}
        groups[key]['raw_dirs'].add(raw_dir)  # 记录该样本涉及的原始目录
        groups[key]['paths'].append(path)
    print(f"📦 共分为样本组：{len(groups)} 组")
    return groups


# --- 划分数据集并复制文件 ---
def split_and_copy(groups):
    # 按时间点整理样本组
    time_groups = {t: [] for t in valid_times}
    for key, group in groups.items():
        time_groups[group['time']].append(group)

    # 处理每个时间点
    for time in valid_times:
        current_groups = time_groups[time]
        if not current_groups:
            print(f"⚠️ 时间点 {time} 无有效样本组")
            continue

        # 划分训练/验证/测试集
        train, temp = train_test_split(current_groups, train_size=train_ratio, random_state=42)
        val, test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
        print(f"\n⏰ 处理时间点 {time}：训练组{len(train)}，验证组{len(val)}，测试组{len(test)}")

        # 复制文件到对应目录
        def copy_group(group, subset):
            for file_path in group['paths']:
                filename = os.path.basename(file_path)
                # 确定文件所属的原始目录（从路径反推，避免分组错误）
                raw_dir = None
                for rd in raw_dirs:
                    if rd in file_path:
                        raw_dir = rd
                        break
                if not raw_dir:
                    print(f"❌ 无法识别文件 {filename} 的原始目录，跳过")
                    continue

                # 映射到输出目录（MSI/RGB）
                output_modal = raw_to_output[raw_dir]
                # 目标路径：output_dir/子集/模态/时间点/文件名
                dst = os.path.join(output_dir, subset, output_modal, str(time), filename)
                shutil.copy2(file_path, dst)
                print(f"📋 复制：{filename} → {subset}/{output_modal}/{time}")

        # 复制到各子集
        for g in train:
            copy_group(g, 'train')
        for g in val:
            copy_group(g, 'val')
        for g in test:
            copy_group(g, 'test')


# --- 主函数 ---
if __name__ == '__main__':
    # 1. 创建输出目录结构
    for subset in ['train', 'val', 'test']:
        for output_modal in raw_to_output.values():
            for time in valid_times:
                os.makedirs(os.path.join(output_dir, subset, output_modal, str(time)), exist_ok=True)
    print("📌 输出目录结构创建完成：data/[train/val/test]/[MSI/RGB]/[时间点]")

    # 2. 收集并验证文件
    file_info = collect_files()
    if not file_info:
        print("❌ 未收集到任何有效文件，请检查原始数据路径和格式")
        exit(1)

    # 3. 分组并划分数据集
    groups = group_files(file_info)
    split_and_copy(groups)

    print("\n🎉 数据划分完成！PNG文件应在 data/*/RGB/* 目录下")