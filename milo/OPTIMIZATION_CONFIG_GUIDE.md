# 优化配置系统使用指南

## 概述

我们为`yufu2mesh_new.py`实现了基于YAML的优化配置系统，大幅简化了超参数管理。现在只需要**一个参数**`--opt_config`就可以控制：

1. ✅ 高斯参数的训练/冻结状态
2. ✅ 每个参数的独立学习率
3. ✅ 所有loss权重
4. ✅ 深度处理参数（裁剪范围）
5. ✅ Mesh正则化权重

## 核心改进

### 改进前 (旧方式)

```bash
python yufu2mesh_new.py \
  --lr 5e-4 \
  --depth_loss_weight 1.0 \
  --normal_loss_weight 1.5 \
  --depth_clip_min 0.0 \
  --depth_clip_max 50.0 \
  --mesh_depth_weight 0.1 \
  --mesh_normal_weight 0.15 \
  # 只能优化xyz，其他参数硬编码冻结
```

**问题**：
- 超参数分散在多个命令行参数中
- 无法灵活控制哪些参数可训练
- 每个参数只能用统一学习率
- 配置难以复用和版本管理

### 改进后 (新方式)

```bash
python yufu2mesh_new.py --opt_config xyz_geometry
```

**优势**：
- 单一参数控制所有优化行为
- YAML配置文件清晰易读，带详细注释
- 支持版本控制和复用
- 预设多种常用配置

## 文件结构

```
milo/
├── yufu2mesh_new.py                      # 主程序（已修改）
├── test_opt_config.py                    # 配置测试脚本（新增）
├── OPTIMIZATION_CONFIG_GUIDE.md          # 本文档（新增）
└── configs/
    └── optimization/                      # 优化配置目录（新增）
        ├── README.md                      # 详细使用文档
        ├── default.yaml                   # 默认配置
        ├── xyz_only.yaml                  # 仅位置优化
        ├── xyz_geometry.yaml              # 位置+几何优化
        ├── xyz_occupancy.yaml             # 位置+占用优化
        └── full.yaml                      # 全参数优化
```

## 快速开始

### 1. 使用预设配置

```bash
# 方式1: 使用默认配置（只优化位置）
python yufu2mesh_new.py --opt_config default

# 方式2: 使用预设配置
python yufu2mesh_new.py --opt_config xyz_geometry
```

### 2. 创建自定义配置

```bash
# 复制模板
cd configs/optimization
cp default.yaml my_config.yaml

# 编辑配置文件
vim my_config.yaml

# 使用自定义配置
python yufu2mesh_new.py --opt_config my_config
```

### 3. 使用完整路径

```bash
python yufu2mesh_new.py --opt_config /path/to/my_config.yaml
```

## 预设配置对比

| 配置名称 | 可训练参数 | 适用场景 | 特点 |
|---------|-----------|---------|------|
| `default` | xyz | 已有良好初始化 | 默认选择，稳定 |
| `xyz_only` | xyz | 同default | 更保守，所有冻结参数lr=0 |
| `xyz_geometry` | xyz, scaling, rotation | 需要调整形状 | 增强法线loss |
| `xyz_occupancy` | xyz, occupancy_shift | 优化mesh质量 | 增强mesh正则化 |
| `full` | 所有（除base） | 初始化较差 | ⚠️ 容易过拟合 |

## 配置文件示例

```yaml
# configs/optimization/xyz_geometry.yaml

gaussian_params:
  _xyz:
    trainable: true      # 优化位置
    lr: 5.0e-4

  _scaling:
    trainable: true      # 优化尺度
    lr: 1.0e-4

  _rotation:
    trainable: true      # 优化旋转
    lr: 1.0e-4

  _features_dc:
    trainable: false     # 冻结颜色
    lr: 0.0

  # ... 其他参数

loss_weights:
  depth: 1.0            # 深度loss权重
  normal: 1.5           # 法线loss权重（增强）
  mesh_depth: 0.1
  mesh_normal: 0.15

depth_processing:
  clip_min: 0.0
  clip_max: null        # 不裁剪最大深度

mesh_regularization:
  depth_weight: 0.1
  normal_weight: 0.15   # 增强法线正则化
```

## 代码修改说明

### 新增函数

1. **`load_optimization_config(config_name: str) -> Dict`**
   - 加载并验证YAML配置文件
   - 支持配置名称或完整路径

2. **`setup_gaussian_optimization(gaussians, opt_config) -> (optimizer, loss_weights)`**
   - 根据配置设置参数的可训练性
   - 创建多参数组的Adam优化器
   - 返回优化器和loss权重字典

### 主要修改点

1. **命令行参数** (yufu2mesh_new.py:632-768)
   - 新增 `--opt_config` 参数
   - 旧参数标记为已弃用，保留向后兼容

2. **参数初始化** (yufu2mesh_new.py:817-822)
   - 移除硬编码的参数冻结逻辑
   - 使用 `setup_gaussian_optimization` 替代

3. **深度处理** (yufu2mesh_new.py:872-887)
   - 从YAML配置读取裁剪参数

4. **Mesh正则化** (yufu2mesh_new.py:854-858)
   - 从YAML配置读取权重覆盖

5. **Loss计算** (yufu2mesh_new.py:1024-1033)
   - 使用配置中的loss权重

6. **梯度计算** (yufu2mesh_new.py:1035-1047)
   - 支持多参数组的梯度范数计算

## 测试验证

### 运行配置测试

```bash
cd /home/zoyo/Desktop/MILo_rtx50/milo
python test_opt_config.py
```

预期输出：
```
🎉 所有配置测试通过！
```

### 查看参数设置

运行主程序时会打印配置信息：

```
[INFO] 加载优化配置：xyz_geometry
[INFO] 配置高斯参数优化...
[INFO] 参数 _xyz: trainable=True, lr=0.000500
[INFO] 参数 _features_dc: trainable=False
[INFO] 参数 _scaling: trainable=True, lr=0.000100
...
[INFO] Loss权重配置:
         > depth: 1.0
         > normal: 1.5
         > mesh_depth: 0.1
         > mesh_normal: 0.15
```

## 向后兼容性

旧的命令行参数仍然可以使用，但会被YAML配置覆盖并显示警告：

```bash
python yufu2mesh_new.py --opt_config default --lr 1e-3
# 输出: [WARNING] --lr 已弃用，将使用YAML配置中的学习率设置
```

建议尽快迁移到YAML配置方式。

## 高级使用

### 渐进式优化策略

```bash
# 阶段1: 快速收敛位置 (前1000次迭代)
python yufu2mesh_new.py \
  --opt_config xyz_only \
  --num_iterations 1000 \
  --heatmap_dir stage1_xyz

# 阶段2: 细化几何 (1000次迭代)
# 先将stage1的最终高斯复制为初始值，然后：
python yufu2mesh_new.py \
  --opt_config xyz_geometry \
  --num_iterations 1000 \
  --heatmap_dir stage2_geometry

# 阶段3: 优化mesh质量 (500次迭代)
python yufu2mesh_new.py \
  --opt_config xyz_occupancy \
  --num_iterations 500 \
  --heatmap_dir stage3_occupancy
```

### 调参建议

1. **学习率太大** → 训练不稳定，loss震荡
   - 解决：降低对应参数的lr，例如从5e-4降到1e-4

2. **学习率太小** → 收敛太慢
   - 解决：适当增加lr，或增加迭代次数

3. **过拟合** → 训练集loss很低但效果差
   - 解决：减少可训练参数，或降低loss权重

4. **欠拟合** → loss降不下来
   - 解决：增加可训练参数，或增加loss权重

## 常见问题

**Q: 如何查看所有可用的预设配置？**
```bash
ls configs/optimization/*.yaml
```

**Q: 如何知道某个配置具体训练哪些参数？**
```bash
python test_opt_config.py  # 查看所有配置
# 或查看YAML文件内容
cat configs/optimization/xyz_geometry.yaml
```

**Q: 配置文件修改后需要重启吗？**

不需要，每次运行时都会重新加载配置。

**Q: 可以在训练中切换配置吗？**

不建议。如需切换，请使用checkpoint机制，分阶段训练。

**Q: 如何备份我的配置？**

配置文件是纯文本，可以直接用git管理：
```bash
cd configs/optimization
git add my_config.yaml
git commit -m "Add custom optimization config"
```

## 相关文件

- 详细配置说明: `configs/optimization/README.md`
- 配置测试脚本: `test_opt_config.py`
- 主程序: `yufu2mesh_new.py`

## 技术细节

### 优化器实现

使用PyTorch的参数组（parameter groups）功能：

```python
optimizer = torch.optim.Adam([
    {"params": [gaussians._xyz], "lr": 5e-4, "name": "_xyz"},
    {"params": [gaussians._scaling], "lr": 1e-4, "name": "_scaling"},
    {"params": [gaussians._rotation], "lr": 1e-4, "name": "_rotation"},
])
```

每个参数可以有独立的学习率，Adam优化器会为每个参数组维护独立的动量和自适应学习率状态。

### 梯度范数计算

计算所有可训练参数的L2梯度范数：

```python
grad_norm = sqrt(sum(||param.grad||^2 for param in trainable_params))
```

用于监控训练稳定性和调试。

## 总结

通过YAML配置系统，我们实现了：

✅ **超参数管理简化**: 从8+个命令行参数减少到1个
✅ **灵活性提升**: 可以自由控制任意参数的训练状态
✅ **可复用性**: 配置文件易于分享和版本控制
✅ **可读性**: 清晰的YAML格式配合详细注释
✅ **向后兼容**: 不影响现有代码的使用

这个系统让实验配置更加模块化和易于管理，非常适合需要频繁调整优化策略的研究场景。
