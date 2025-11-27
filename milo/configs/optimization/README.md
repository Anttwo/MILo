# 优化配置文件说明 (Optimization Configuration)

本目录包含用于`yufu2mesh_new.py`的优化配置文件。通过YAML配置文件，你可以精细控制高斯参数的优化、损失权重和其他训练超参数。

## 快速开始 (Quick Start)

### 基本用法

```bash
# 使用默认配置
python yufu2mesh_new.py --opt_config default

# 使用预设配置
python yufu2mesh_new.py --opt_config xyz_only
python yufu2mesh_new.py --opt_config xyz_geometry

# 使用自定义配置文件（完整路径）
python yufu2mesh_new.py --opt_config /path/to/my_config.yaml
```

## 预设配置 (Available Presets)

### 1. `default.yaml` - 默认配置
- **优化参数**: 仅XYZ位置
- **适用场景**: 已有良好初始化的高斯，只需微调位置
- **特点**: 保守策略，稳定收敛

### 2. `xyz_only.yaml` - 纯位置优化
- **优化参数**: 仅XYZ位置
- **适用场景**: 与default类似，但所有冻结参数的学习率都为0
- **特点**: 最保守的优化策略

### 3. `xyz_geometry.yaml` - 位置+几何优化
- **优化参数**: XYZ位置 + Scaling + Rotation
- **适用场景**: 需要调整高斯形状以更好拟合深度和法线
- **特点**: 法线loss权重增加，mesh正则化权重也相应提高

### 4. `xyz_occupancy.yaml` - 位置+占用优化
- **优化参数**: XYZ位置 + Occupancy Shift
- **适用场景**: 需要精细调整mesh提取质量，改善mesh拓扑
- **特点**: 增强mesh正则化，较小的occupancy学习率

### 5. `full.yaml` - 全参数优化
- **优化参数**: 所有参数（除base_occupancy）
- **适用场景**: 初始化质量较差，需要全面优化
- **警告**: 可能导致过拟合，建议谨慎使用

## 配置文件结构 (Configuration Structure)

每个YAML配置文件包含以下4个主要部分：

### 1. `gaussian_params` - 高斯参数设置

控制哪些高斯参数需要被优化，以及各自的学习率。

```yaml
gaussian_params:
  _xyz:                    # 高斯中心位置
    trainable: true        # 是否训练
    lr: 5.0e-4            # 学习率

  _features_dc:            # 球谐0阶系数（主颜色）
    trainable: false
    lr: 1.0e-4

  _scaling:                # 高斯椭球的缩放
    trainable: false
    lr: 1.0e-4

  _rotation:               # 高斯椭球的旋转
    trainable: false
    lr: 1.0e-4

  # ... 其他参数
```

**可用参数列表**:
- `_xyz`: 高斯中心位置（3D坐标）
- `_features_dc`: 球谐函数0阶系数（基础颜色）
- `_features_rest`: 球谐函数高阶系数（视角相关颜色）
- `_scaling`: 高斯椭球的3轴缩放
- `_rotation`: 高斯椭球的旋转四元数
- `_opacity`: 不透明度
- `_base_occupancy`: SDF占用基础值（通常不训练）
- `_occupancy_shift`: SDF占用偏移量（可学习）

### 2. `loss_weights` - 损失权重

控制各个损失项在总损失中的比重。

```yaml
loss_weights:
  depth: 1.0           # 深度一致性损失
  normal: 1.0          # 法线一致性损失
  mesh_depth: 0.1      # Mesh深度损失
  mesh_normal: 0.1     # Mesh法线损失
```

### 3. `depth_processing` - 深度处理

控制深度图的加载和预处理。

```yaml
depth_processing:
  clip_min: 0.0        # 最小深度值，null表示不裁剪
  clip_max: null       # 最大深度值，null表示不裁剪
```

**常用设置**:
- 室内场景: `clip_max: 10.0` 或 `20.0`
- 室外场景: `clip_max: 50.0` 或 `100.0`
- 无裁剪: `clip_max: null`

### 4. `mesh_regularization` - Mesh正则化

覆盖mesh配置文件中的权重设置。

```yaml
mesh_regularization:
  depth_weight: 0.1    # Mesh深度项权重
  normal_weight: 0.1   # Mesh法线项权重
```

## 创建自定义配置 (Creating Custom Configuration)

### 方法1: 复制并修改预设配置

```bash
cd /home/zoyo/Desktop/MILo_rtx50/milo/configs/optimization
cp default.yaml my_custom.yaml
# 编辑 my_custom.yaml
```

### 方法2: 从模板开始

使用`default.yaml`作为模板，它包含所有参数的完整注释。

### 配置建议

1. **学习率设置**:
   - 位置参数 (_xyz): `5e-4` 到 `1e-3`
   - 几何参数 (_scaling, _rotation): `1e-4` 到 `5e-4`
   - 外观参数 (_features_*): `1e-4` 到 `2.5e-4`
   - 占用参数 (_occupancy_shift): `5e-5` 到 `1e-4`

2. **损失权重平衡**:
   - 深度和法线损失通常在 `1.0` 左右
   - Mesh损失通常为深度/法线损失的 `0.1` 到 `0.2` 倍
   - 如果优化几何参数，可适当增加法线权重

3. **渐进式优化策略**:
   - 第1阶段: 使用`xyz_only`快速收敛位置
   - 第2阶段: 使用`xyz_geometry`细化形状
   - 第3阶段: 使用`xyz_occupancy`优化mesh质量

## 调试技巧 (Debugging Tips)

### 验证配置文件

```bash
python test_opt_config.py
```

### 查看当前使用的配置

运行`yufu2mesh_new.py`时，会在开始打印所有参数设置：

```
[INFO] 加载优化配置：xyz_geometry
[INFO] 参数 _xyz: trainable=True, lr=0.0005
[INFO] 参数 _scaling: trainable=True, lr=0.0001
...
[INFO] Loss权重配置:
         > depth: 1.0
         > normal: 1.5
...
```

### 常见问题

**Q: 为什么优化后效果变差了？**
- A: 可能学习率过大，尝试降低学习率或减少可训练参数

**Q: 收敛太慢怎么办？**
- A: 可以适当增加学习率，或者增加可训练参数的数量

**Q: Mesh质量不好？**
- A: 尝试使用`xyz_occupancy`配置，或增加mesh正则化权重

**Q: 旧的命令行参数还能用吗？**
- A: 可以，但会被YAML配置覆盖，建议迁移到YAML配置

## 配置迁移 (Migration from Old Parameters)

如果你之前使用命令行参数：

```bash
# 旧方式
python yufu2mesh_new.py \
  --lr 5e-4 \
  --depth_loss_weight 1.0 \
  --normal_loss_weight 1.5 \
  --mesh_depth_weight 0.1 \
  --mesh_normal_weight 0.15
```

现在应该创建一个YAML配置文件：

```yaml
# my_config.yaml
gaussian_params:
  _xyz:
    trainable: true
    lr: 5.0e-4
  # ... 其他参数设为不可训练

loss_weights:
  depth: 1.0
  normal: 1.5
  mesh_depth: 0.1
  mesh_normal: 0.15

# ... 其他设置
```

然后使用：

```bash
python yufu2mesh_new.py --opt_config my_config
```

## 更多信息

- 配置测试脚本: `test_opt_config.py`
- 主程序: `yufu2mesh_new.py`
- 相关文档: 参见各配置文件内的详细注释
