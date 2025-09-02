# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 代码库概述

此代码库包含用于计算机视觉任务的数据处理工具，主要专注于：
1. 在不同标注格式（Cornell 和 DOTA）之间转换抓取检测数据集
2. 生成逼真的低光图像，用于在具有挑战性的光照条件下训练计算机视觉模型

## 代码库结构

```
data-tools/
├── label/
│   ├── convert_cornell_to_dota.py     # 将 Cornell 格式转换为 DOTA 格式
│   ├── convert_dota_to_cornell.py     # 将 DOTA 格式转换为 Cornell 格式
│   ├── visualize_grasps.py            # 在图像上可视化抓取标注
│   └── visualize_real_data.py         # 可视化真实数据集中的RGB图像及抓取标注
├── make_low_light_v2/
│   ├── low_light_generator.py         # 生成逼真低光图像的主程序
│   ├── batch_generate.py              # 批量处理多个配置的脚本
│   ├── dim_config.yaml                # 微暗光照配置
│   ├── dark_config.yaml               # 暗光照配置
│   ├── very_dark_config.yaml          # 极暗光照配置
│   └── INSTRUCTIONS.md                # 详细使用说明
└── README.md
```

## 主要组件

### 标注格式转换工具 (`label/`)

两个在 Cornell Grasping Dataset 格式和 DOTA 格式之间的双向转换工具：
- `convert_cornell_to_dota.py`：将 Cornell 格式（每行4个点）转换为 DOTA 格式（8个坐标 + 类别 + 难度）
- `convert_dota_to_cornell.py`：将 DOTA 格式转换回 Cornell 格式

两个工具都支持单文件和批量处理模式。

### 低光图像生成 (`make_low_light_v2/`)

一个复杂的图像处理管道，模拟逼真的低光条件：
- `low_light_generator.py`：应用多种照片级真实效果的核心引擎
- 效果包括：亮度降低、ISO 噪声、运动模糊、不均匀照明、色温偏移和阴影增强
- 三种预定义的光照条件配置（微暗、暗、极暗）
- 通过 YAML 文件配置参数

### 可视化工具

- `visualize_grasps.py`：用于检查图像上抓取标注的交互式查看器，支持键盘导航
- `visualize_real_data.py`：用于可视化真实数据集中RGB图像及抓取标注的交互式查看器，支持子目录导航

## 开发命令

### 环境设置
```bash
# 安装所需依赖
pip install opencv-python numpy scipy pyyaml imageio tqdm matplotlib pillow
```

### 运行工具

#### 标注转换
```bash
# 将 Cornell 转换为 DOTA 格式
python label/convert_cornell_to_dota.py --input grasp1.txt --output grasp1_dota.txt

# 批量转换目录
python label/convert_cornell_to_dota.py --src_dir cornell_labels --dst_dir dota_labels

# 将 DOTA 转换为 Cornell 格式
python label/convert_dota_to_cornell.py --input grasp_dota.txt --output grasp.txt
```

#### 低光图像生成
```bash
# 使用特定光照条件生成图像
python make_low_light_v2/low_light_generator.py --config make_low_light_v2/dim_config.yaml

# 批量生成所有光照条件
python make_low_light_v2/batch_generate.py
```

#### 可视化
```bash
# 可视化抓取标注
python label/visualize_grasps.py /path/to/directory/with/images/and/annotations

# 可视化真实数据集
python label/visualize_real_data.py H:/Datasets/real-data/Darker
```

## 架构说明

1. **模块化设计**：每个工具都是独立的，具有清晰的命令行接口
2. **配置驱动**：低光生成使用 YAML 配置文件以便轻松调整参数
3. **批量处理**：大多数工具支持单文件和批量操作
4. **质量控制**：内置验证和错误处理以确保稳健处理
5. **逼真模拟**：低光生成器精心模拟真实相机行为（ISO 噪声、运动模糊等）

## 常见开发任务

1. **添加新标注格式**：扩展现有 `label/` 目录中的转换工具
2. **调整低光效果**：修改 YAML 配置文件中的参数
3. **添加新光照条件**：基于现有模板创建新的 YAML 配置文件
4. **扩展可视化功能**：修改 `visualize_grasps.py` 或 `visualize_real_data.py` 以支持额外的标注类型或可视化功能