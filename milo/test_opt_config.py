#!/usr/bin/env python3
"""
测试优化配置文件加载功能
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from yufu2mesh_new import load_optimization_config

def test_config(config_name: str):
    """测试加载指定的配置文件"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print('='*60)

    try:
        config = load_optimization_config(config_name)

        print("\n✓ 配置加载成功！")
        print("\n高斯参数配置:")
        print("-" * 40)
        for param_name, param_cfg in config["gaussian_params"].items():
            trainable = param_cfg.get("trainable", False)
            lr = param_cfg.get("lr", 0.0)
            status = "✓ 可训练" if trainable else "✗ 冻结"
            print(f"  {param_name:20s} {status:10s} lr={lr:.6f}")

        print("\nLoss权重配置:")
        print("-" * 40)
        for loss_name, weight in config["loss_weights"].items():
            print(f"  {loss_name:20s} {weight:.3f}")

        print("\n深度处理配置:")
        print("-" * 40)
        depth_cfg = config["depth_processing"]
        print(f"  clip_min: {depth_cfg.get('clip_min')}")
        print(f"  clip_max: {depth_cfg.get('clip_max')}")

        print("\nMesh正则化配置:")
        print("-" * 40)
        mesh_cfg = config["mesh_regularization"]
        print(f"  depth_weight:  {mesh_cfg.get('depth_weight')}")
        print(f"  normal_weight: {mesh_cfg.get('normal_weight')}")

        return True

    except Exception as e:
        print(f"\n✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """测试所有预设配置"""
    configs_to_test = [
        "default",
        "xyz_only",
        "xyz_geometry",
        "xyz_occupancy",
        "full"
    ]

    print("开始测试优化配置加载功能...")

    results = {}
    for config_name in configs_to_test:
        results[config_name] = test_config(config_name)

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for config_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {config_name:20s} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有配置测试通过！")
        return 0
    else:
        print("\n⚠️  部分配置测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
