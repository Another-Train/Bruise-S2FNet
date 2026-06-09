import os
import time
import torch
import torch.nn as nn
import numpy as np
from models.vit_total import ViTWithProb, load_vit_simple
from thop import profile

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_vit_performance():
    """测试ViT模型的性能指标"""
    print("=" * 80)
    print("ViT模型性能测试")
    print("=" * 80)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型实例
    hs_dim = 20
    num_classes = 2
    
    # 加载ViT基础模型
    base_model = load_vit_simple()
    if base_model is None:
        print("错误: 无法加载ViT基础模型")
        return None
    
    model = ViTWithProb(base_model, hs_dim=hs_dim, num_classes=num_classes).to(device)
    model.eval()
    
    # 计算模型参数数量 P(G)
    model_params = count_parameters(model)
    model_params_m = model_params / 1e6  # 转换为百万单位
    print(f"\n模型参数数量 (P(G)): {model_params_m:.2f} M")
    
    # 创建模拟输入数据
    batch_size = 4
    img_size = 64
    hs_data = torch.randn(batch_size, hs_dim).to(device)
    rgb_data = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 计算FLOPS
    try:
        flops, params = profile(model, inputs=(hs_data, rgb_data))
        flops_g = flops / 1e9  # 转换为十亿单位
        print(f"FLOPS (F(G)): {flops_g:.2f} G")
    except Exception as e:
        print(f"计算FLOPS时出错: {e}")
        flops_g = 0
    
    # 预热运行
    print("\n进行预热运行...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(hs_data, rgb_data)
    
    # 性能测试
    num_iterations = 100
    print(f"\n开始性能测试，迭代次数: {num_iterations}")
    
    total_inference_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(num_iterations):
            # 记录推理开始时间
            start_time = time.time()
            
            # 执行推理
            logits = model(hs_data, rgb_data)
            
            # 记录推理结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_samples += batch_size
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{num_iterations} 次迭代")
    
    # 计算性能指标
    if total_inference_time > 0:
        # 平均推理时间（秒/样本）
        avg_inference_time = total_inference_time / total_samples
        # 吞吐量（imgs/s）
        throughput = total_samples / total_inference_time
        
        print("\n" + "=" * 80)
        print("ViT模型性能指标汇总")
        print("=" * 80)
        print(f"模型参数数量 (P(G)): {model_params_m:.2f} M")
        print(f"FLOPS (F(G)): {flops_g:.2f} G")
        print(f"平均推理时间: {avg_inference_time * 1000:.2f} ms/样本")
        print(f"吞吐量 (Thr): {throughput:.2f} imgs/s")
        print("=" * 80)
        
        return {
            'model_params_m': model_params_m,
            'flops_g': flops_g,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput': throughput
        }
    else:
        print("错误: 总推理时间为0")
        return None

if __name__ == "__main__":
    test_vit_performance()