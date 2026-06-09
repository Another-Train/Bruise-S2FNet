import os
import shutil
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- 配置参数 ---
basedir = 'D:/DATA/gtdata1'  # 原始数据根目录（必须与实际路径完全一致）
output_dir = 'data0-36'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 其他类别（bd/jk/ys）的有效时间点（原配置）
need_valid_times = {'0', '36'}  # 可根据需要调整
# jjk类别配置：排除37h（时间点'37'），保留其他所有时间点
jjk_special_prefix = 'jjk'  # 标识jjk类的前缀
jjk_exclude_time = {'37'}  # 排除的时间点（37h）

# 原始目录与输出目录的映射（保持不变）
raw_to_output = {
    'hyperspectral_cropped': 'MSI',  # 高光谱原始目录 → 输出MSI
    'rgb_cropped': 'RGB'  # RGB原始目录 → 输出RGB
}
raw_dirs = list(raw_to_output.keys())  # 原始目录列表

# 各原始目录允许的文件格式（严格限制）
allowed_extensions = {
    'hyperspectral_cropped': {'.hdr', '.float'},
    'rgb_cropped': {'.png'}
}

valid_prefixes = {'bd', 'jk', 'ys', 'jjk'}  # 包含jjk


# --- 解析文件名提取关键信息（增强：区分jjk类别、过滤37h） ---
def parse_filename(filename):
    """解析文件名，返回 (前缀, 时间点, 是否为jjk类)"""
    pattern = re.compile(r'^(bd|jk|jjk|ys)-(\d+)-.*z')
    match = pattern.match(filename)
    if not match or match.group(1) not in valid_prefixes:
        return None, None, False  # 无效前缀
    prefix_type = match.group(1)  # 前缀类型（bd/jk/jjk/ys）
    time_point = match.group(2)  # 时间点（字符串）
    is_jjk = (prefix_type == jjk_special_prefix)  # 是否为jjk类
    return prefix_type, time_point, is_jjk


# --- 数据统计函数 ---
def print_statistics(jjk_file_info, other_file_info, jjk_valid_times, stage="收集"):
    """打印详细的数据统计信息"""
    print(f"\n{'=' * 60}")
    print(f"📊 {stage}阶段 - 数据统计详情")
    print(f"{'=' * 60}")

    # 按类别和时间点统计文件数量
    jjk_stats = defaultdict(lambda: defaultdict(int))
    other_stats = defaultdict(lambda: defaultdict(int))

    # 统计jjk类文件
    for prefix, time_point, raw_dir, path in jjk_file_info:
        jjk_stats[time_point][raw_dir] += 1

    # 统计其他类文件
    for prefix, time_point, raw_dir, path in other_file_info:
        other_stats[time_point][raw_dir] += 1

    # 打印jjk类统计
    if jjk_stats:
        print(f"\n🔬 jjk类统计（排除37h，实际包含时间点：{sorted(jjk_valid_times)}）:")
        total_jjk_files = 0
        for time_point in sorted(jjk_stats.keys()):
            time_files = sum(jjk_stats[time_point].values())
            total_jjk_files += time_files
            print(f"   时间点 {time_point}: {time_files} 个文件", end="")
            for modal, count in jjk_stats[time_point].items():
                print(f" | {raw_to_output[modal]}: {count}", end="")
            print()
        print(f"   jjk类总计: {total_jjk_files} 个文件")
    else:
        print(f"\n🔬 jjk类: 无有效文件")

    # 打印其他类统计
    if other_stats:
        print(f"\n🔬 非jjk类统计（时间点：{sorted(need_valid_times)}）:")
        total_other_files = 0
        for time_point in sorted(other_stats.keys()):
            time_files = sum(other_stats[time_point].values())
            total_other_files += time_files
            print(f"   时间点 {time_point}: {time_files} 个文件", end="")
            for modal, count in other_stats[time_point].items():
                print(f" | {raw_to_output[modal]}: {count}", end="")
            print()
        print(f"   非jjk类总计: {total_other_files} 个文件")
    else:
        print(f"\n🔬 非jjk类: 无有效文件")

    # 总体统计
    total_files = len(jjk_file_info) + len(other_file_info)
    print(f"\n📈 总体统计:")
    print(f"   总文件数: {total_files}")
    print(f"   总样本组数: {len(jjk_file_info) // 2 + len(other_file_info) // 2} (每个样本包含MSI和RGB文件)")

    # 模态统计
    modal_stats = defaultdict(int)
    for _, _, raw_dir, _ in jjk_file_info + other_file_info:
        modal_stats[raw_to_output[raw_dir]] += 1

    print(f"   模态分布:", end="")
    for modal, count in modal_stats.items():
        print(f" {modal}: {count}", end="")
    print()

    print(f"{'=' * 60}")


# --- 划分后统计函数 ---
def print_split_statistics(train_groups, val_groups, test_groups):
    """打印划分后的详细统计信息"""
    print(f"\n{'=' * 60}")
    print(f"📊 划分后数据集统计")
    print(f"{'=' * 60}")

    def analyze_split(groups, split_name):
        """分析单个划分的数据"""
        prefix_stats = defaultdict(int)
        time_stats = defaultdict(int)
        modal_stats = defaultdict(int)
        total_files = 0

        for group in groups:
            time_point = group['time']
            time_stats[time_point] += 1
            total_files += len(group['paths'])

            for file_path in group['paths']:
                # 识别类别前缀
                filename = os.path.basename(file_path)
                prefix_type, _, is_jjk = parse_filename(filename)
                if prefix_type:
                    if is_jjk:
                        prefix_stats['jjk'] += 1
                    else:
                        prefix_stats[prefix_type] += 1

                # 识别模态
                for rd in raw_dirs:
                    if rd in file_path:
                        modal_stats[raw_to_output[rd]] += 1
                        break

        print(f"\n🔹 {split_name}集:")
        print(f"   样本组数: {len(groups)}")
        print(f"   总文件数: {total_files}")

        # 类别分布
        if prefix_stats:
            print(f"   类别分布:", end="")
            for prefix in sorted(prefix_stats.keys()):
                print(f" {prefix}: {prefix_stats[prefix]}", end="")
            print()

        # 时间点分布
        if time_stats:
            print(f"   时间点分布:", end="")
            for time_point in sorted(time_stats.keys()):
                print(f" {time_point}h: {time_stats[time_point]}组", end="")
            print()

        # 模态分布
        if modal_stats:
            print(f"   模态分布:", end="")
            for modal in sorted(modal_stats.keys()):
                print(f" {modal}: {modal_stats[modal]}", end="")
            print()

        return len(groups), total_files

    # 分析各划分
    train_groups_count, train_files_count = analyze_split(train_groups, "训练")
    val_groups_count, val_files_count = analyze_split(val_groups, "验证")
    test_groups_count, test_files_count = analyze_split(test_groups, "测试")

    # 总体统计
    total_groups = train_groups_count + val_groups_count + test_groups_count
    total_files = train_files_count + val_files_count + test_files_count

    print(f"\n📈 划分汇总:")
    print(f"   总样本组数: {total_groups}")
    print(f"   总文件数: {total_files}")
    print(f"   训练集比例: {train_groups_count / total_groups:.1%} ({train_groups_count}组)")
    print(f"   验证集比例: {val_groups_count / total_groups:.1%} ({val_groups_count}组)")
    print(f"   测试集比例: {test_groups_count / total_groups:.1%} ({test_groups_count}组)")

    print(f"{'=' * 60}")


# --- 收集所有有效文件（分jjk和其他类别，jjk排除37h） ---
def collect_files():
    jjk_file_info = []  # jjk类（排除37h）的所有时间点文件
    other_file_info = []  # 其他类别（bd/jk/ys）按valid_times筛选的文件
    all_valid_jjk_times = set()  # 记录jjk类实际存在的有效时间点（排除37后）

    for raw_dir in raw_dirs:
        raw_path = os.path.join(basedir, raw_dir)
        # 检查原始目录是否存在
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

            # 1. 验证文件格式
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions[raw_dir]:
                print(f"⚠️ 跳过：{filename} 在 {raw_dir} 目录，但格式不符合（需{allowed_extensions[raw_dir]}）")
                continue

            # 2. 解析文件名（提取前缀、时间点、是否jjk）
            prefix_type, time_point, is_jjk = parse_filename(filename)
            if not prefix_type:
                print(f"⚠️ 跳过：{filename} 前缀无效（需{valid_prefixes}）")
                continue

            # 3. 按类别筛选文件
            if is_jjk:
                # jjk类：排除时间点'37'，其余所有时间点保留
                if time_point in jjk_exclude_time:
                    print(f"⚠️ 跳过[jjk类-37h]：{filename} → 时间点：{time_point}（已排除）")
                    continue
                file_prefix = os.path.splitext(filename)[0]
                jjk_file_info.append((file_prefix, time_point, raw_dir, file_path))
                all_valid_jjk_times.add(time_point)
                print(f"✅ 收集[jjk类]：{filename} → 时间点：{time_point} → 原始目录：{raw_dir}")
            else:
                # 其他类别：仅保留other_valid_times中的时间点
                if time_point not in need_valid_times:
                    print(f"⚠️ 跳过[非jjk类]：{filename} 时间点无效（需{need_valid_times}）")
                    continue
                file_prefix = os.path.splitext(filename)[0]
                other_file_info.append((file_prefix, time_point, raw_dir, file_path))
                print(f"✅ 收集[非jjk类]：{filename} → 时间点：{time_point} → 原始目录：{raw_dir}")

    # 打印收集阶段的统计信息
    print_statistics(jjk_file_info, other_file_info, all_valid_jjk_times, "收集")

    return jjk_file_info, other_file_info, all_valid_jjk_times


# --- 按 (前缀, 时间点) 分组（确保同一样本的文件被同时划分） ---
def group_files(file_info):
    groups = {}
    for prefix, time, raw_dir, path in file_info:
        key = (prefix, time)  # 同一前缀+时间点属于同一样本
        if key not in groups:
            groups[key] = {'time': time, 'paths': []}
        groups[key]['paths'].append(path)
    return groups


# --- 合并所有样本并统一划分数据集 ---
def split_and_copy(jjk_groups, other_groups, jjk_valid_times):
    # 合并所有样本组（jjk类 + 非jjk类）
    all_groups = list(other_groups.values()) + list(jjk_groups.values())
    print(f"\n📦 总样本组数量：{len(all_groups)}（非jjk类：{len(other_groups)} + jjk类：{len(jjk_groups)}）")

    # 统一划分训练/验证/测试集（所有类别一起划分）
    train_groups, temp_groups = train_test_split(all_groups, train_size=train_ratio, random_state=42)
    val_groups, test_groups = train_test_split(temp_groups, train_size=val_ratio / (val_ratio + test_ratio),
                                               random_state=42)

    print(f"\n📊 最终划分结果（所有类别统一划分）：")
    print(f"   训练集：{len(train_groups)} 个样本组")
    print(f"   验证集：{len(val_groups)} 个样本组")
    print(f"   测试集：{len(test_groups)} 个样本组")

    # 打印划分后的详细统计
    print_split_statistics(train_groups, val_groups, test_groups)

    # 复制文件的通用函数
    def copy_groups_to_subset(groups, subset):
        for group in groups:
            time_point = group['time']
            for file_path in group['paths']:
                filename = os.path.basename(file_path)
                # 识别文件所属原始目录（MSI/RGB）
                raw_dir = None
                for rd in raw_dirs:
                    if rd in file_path:
                        raw_dir = rd
                        break
                if not raw_dir:
                    print(f"❌ 无法识别文件 {filename} 的原始目录，跳过")
                    continue
                output_modal = raw_to_output[raw_dir]  # 映射为MSI/RGB

                # 目标路径：output_dir/子集/模态/时间点/文件名
                dst_dir = os.path.join(output_dir, subset, output_modal, str(time_point))
                os.makedirs(dst_dir, exist_ok=True)  # 确保目录存在
                dst_path = os.path.join(dst_dir, filename)
                shutil.copy2(file_path, dst_path)
                print(f"📋 复制：{filename} → {subset}/{output_modal}/{time_point}")

    # 复制到各子集
    print(f"\n🚀 开始复制训练集文件...")
    copy_groups_to_subset(train_groups, 'train')
    print(f"\n🚀 开始复制验证集文件...")
    copy_groups_to_subset(val_groups, 'val')
    print(f"\n🚀 开始复制测试集文件...")
    copy_groups_to_subset(test_groups, 'test')


# --- 主函数 ---
if __name__ == '__main__':
    # 1. 收集文件（jjk排除37h，非jjk按时间点筛选）
    jjk_file_info, other_file_info, jjk_valid_times = collect_files()
    if not jjk_file_info and not other_file_info:
        print("❌ 未收集到任何有效文件，请检查原始数据路径和格式")
        exit(1)

    # 2. 分组（按样本+时间点）
    jjk_groups = group_files(jjk_file_info) if jjk_file_info else {}
    other_groups = group_files(other_file_info) if other_file_info else {}

    # 3. 所有类别统一划分并复制文件
    split_and_copy(jjk_groups, other_groups, jjk_valid_times)

    print("\n🎉 数据划分完成！")
    print(f"📌 输出目录结构：{output_dir}/[train/val/test]/[MSI/RGB]/[时间点]")
    print(f"📌 jjk类包含时间点（排除37h）：{sorted(jjk_valid_times)}")
    print(f"📌 非jjk类包含时间点：{sorted(need_valid_times)}")
    print(f"📌 所有类别已统一划分到训练/验证/测试集")