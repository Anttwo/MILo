# 快速开始指南 (Quick Start)

## 基本用法

使用新的YAML配置系统，只需一个参数控制所有优化行为：

```bash
python yufu2mesh_new.py --opt_config <配置名称>
```

## 预设配置

| 配置名称 | 优化参数 | 使用场景 |
|---------|---------|---------|
| `default` | 仅位置(xyz) | 默认选择，适合大多数情况 |
| `xyz_only` | 仅位置(xyz) | 最保守策略 |
| `xyz_geometry` | 位置+形状 | 需要调整高斯形状 |
| `xyz_occupancy` | 位置+占用 | 改善mesh提取质量 |
| `full` | 所有参数 | 初始化较差时使用 ⚠️ |

## 常用命令

```bash
# 1. 使用默认配置
python yufu2mesh_new.py --opt_config default

# 2. 优化位置和几何形状
python yufu2mesh_new.py --opt_config xyz_geometry

# 3. 改善mesh质量
python yufu2mesh_new.py --opt_config xyz_occupancy

# 4. 指定迭代次数和输出目录
python yufu2mesh_new.py \
  --opt_config xyz_geometry \
  --num_iterations 200 \
  --heatmap_dir my_output

# 5. 使用自定义配置文件
python yufu2mesh_new.py --opt_config /path/to/custom.yaml
```

## 自定义配置

### 1. 复制模板

```bash
cd configs/optimization
cp default.yaml my_config.yaml
```

### 2. 编辑配置

打开 `my_config.yaml`，修改你需要的部分：

```yaml
gaussian_params:
  _xyz:
    trainable: true    # 是否训练
    lr: 5.0e-4        # 学习率

  _scaling:
    trainable: false   # 改为true可优化形状
    lr: 1.0e-4

loss_weights:
  depth: 1.0          # 深度loss权重
  normal: 1.5         # 法线loss权重
```

### 3. 使用自定义配置

```bash
python yufu2mesh_new.py --opt_config my_config
```

## 验证配置

测试所有配置文件是否正确：

```bash
python test_opt_config.py
```

## 配置文件位置

- 预设配置: `configs/optimization/*.yaml`
- 详细文档: `configs/optimization/README.md`
- 完整指南: `OPTIMIZATION_CONFIG_GUIDE.md`

## 与旧参数对比

### 旧方式（已弃用）
```bash
python yufu2mesh_new.py \
  --lr 5e-4 \
  --depth_loss_weight 1.0 \
  --normal_loss_weight 1.5 \
  --depth_clip_max 50.0 \
  --mesh_depth_weight 0.1
```

### 新方式（推荐）
```bash
python yufu2mesh_new.py --opt_config xyz_geometry
```

所有参数都在YAML文件中统一管理，更清晰易维护。

## 需要帮助？

- 查看预设配置: `ls configs/optimization/`
- 查看配置内容: `cat configs/optimization/default.yaml`
- 详细文档: `configs/optimization/README.md`
