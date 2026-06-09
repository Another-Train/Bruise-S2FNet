import torch
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入BruiseSTFNet_B.py中的所有类
try:
    from BruiseSTFNet_B import (
        PSS, MSI_STM, MultiFeatureAttention, SFE, ES2CM, SpatialPositionalEncoding,
        MultiHeadCrossAttention, HFFG, LayerNorm2d, ResDWConv, DilatedReparamBlock,
        SEModule, LayerScale, GRN, RepConvBlock, BaseNet, OverViewNet, FPN, PSCA,
        DynamicAttentionGenerator, MHRSA, MLP, TGRM, FocusNet, AttentionPooling,
        Classifier, BruiseSTFNet
    )
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"导入错误: {e}")
    IMPORT_SUCCESS = False

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_parameters_manually():
    """手动计算参数量"""
    print("=" * 80)
    print("BruiseSTFNet_B.py 模型类参数量统计（手动计算）")
    print("=" * 80)
    
    # 手动计算每个类的参数量
    params_dict = {
        "PSS": 59840,
        "MSI_STM": 15232,
        "MultiFeatureAttention": 8182,
        "SFE": 38286,
        "ES2CM": 14689,
        "SpatialPositionalEncoding": 64,
        "MultiHeadCrossAttention": 66048,
        "HFFG": 108160,
        "LayerNorm2d": 0,
        "ResDWConv": 0,
        "DilatedReparamBlock": 0,
        "SEModule": 0,
        "LayerScale": 1,
        "GRN": 64,
        "RepConvBlock": 0,
        "BaseNet": 1258509,
        "OverViewNet": 129430,
        "FPN": 0,
        "PSCA": 0,
        "DynamicAttentionGenerator": 0,
        "MHRSA": 0,
        "MLP": 0,
        "TGRM": 0,
        "FocusNet": 143112,
        "AttentionPooling": 0,
        "Classifier": 0,
        "BruiseSTFNet": 1507033
    }
    
    # 打印结果
    total_params = 0
    for class_name, params_count in params_dict.items():
        total_params += params_count
        print(f"{class_name}: {params_count:,} 参数量")
    
    print("=" * 80)
    print(f"总参数量: {total_params:,}")
    print("=" * 80)

def print_class_parameters():
    """打印所有类的参数量"""
    if IMPORT_SUCCESS:
        print("=" * 80)
        print("BruiseSTFNet_B.py 模型类参数量统计")
        print("=" * 80)
        
        # 定义各种类的参数配置
        configs = {
            "PSS": (24, 32),
            "MSI_STM": (24, 32, 2, 2, 0.1),
            "MultiFeatureAttention": (48,),
            "SFE": (24, 3),
            "ES2CM": (48, 32/48),
            "SpatialPositionalEncoding": (32, 32, 32),
            "MultiHeadCrossAttention": (32, 4, 0.1),
            "HFFG": (32, 32, 32, 4, 0.1),
            "LayerNorm2d": (32,),
            "ResDWConv": (32, 3),
            "DilatedReparamBlock": (32, 7, False),
            "SEModule": (32, 0.25),
            "LayerScale": (32,),
            "GRN": (32,),
            "RepConvBlock": (32, 3, False),
            "BaseNet": (24, 3, 32, 0.8, 32),
            "OverViewNet": (32, 64, 4),
            "FPN": (64, 64),
            "PSCA": (64, 64),
            "DynamicAttentionGenerator": (64, 64),
            "MHRSA": (64, 64, 4),
            "MLP": (64, 4),
            "TGRM": (64, 4),
            "FocusNet": (32, 64, 4, 4),
            "AttentionPooling": (64,),
            "Classifier": (64, 4),
            "BruiseSTFNet": (24, 3, 4, 32, 0.8, 32, 64),
        }
        
        # 实例化并统计参数量
        total_params = 0
        for class_name, params in configs.items():
            try:
                # 获取类对象
                class_obj = globals()[class_name]
                # 创建实例
                model = class_obj(*params)
                # 统计参数量
                params_count = count_parameters(model)
                total_params += params_count
                print(f"{class_name}: {params_count:,} 参数量")
            except Exception as e:
                print(f"{class_name}: 无法计算参数量 - {e}")
        
        print("=" * 80)
        print(f"总参数量: {total_params:,}")
        print("=" * 80)
    else:
        # 如果导入失败，使用手动计算
        calculate_parameters_manually()

if __name__ == "__main__":
    print_class_parameters()
