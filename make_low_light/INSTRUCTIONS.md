# 真实弱光环境图像生成器使用说明

## 概述

本工具用于将Cornell抓取检测数据集的RGB图像转换为模拟真实室内弱光环境的图像，专门用于训练在低光照条件下工作的抓取检测模型。

## 环境要求

### Python 版本
- Python 3.7+

### 依赖包安装

```bash
# 安装所需的Python包
pip install opencv-python numpy scipy pyyaml imageio tqdm

# 或者使用conda安装
conda install opencv python=3.10 numpy scipy pyyaml tqdm
conda install -c conda-forge imageio
```

### 依赖包说明
- `opencv-python`: 图像处理核心库
- `numpy`: 数值计算
- `scipy`: 科学计算（用于高斯滤波等）
- `pyyaml`: YAML配置文件解析
- `imageio`: 图像读写
- `tqdm`: 进度条显示

## 核心文件

```
make_low_light_albumentations/
├── low_light_generator.py   # 主生成器脚本
├── batch_generate.py        # 批量生成脚本
├── dim_config.yaml          # 较暗光照配置
├── dark_config.yaml         # 暗光照配置  
├── very_dark_config.yaml    # 非常暗光照配置
└── INSTRUCTIONS.md          # 本说明文件
```

## 快速开始

### 0. 环境准备

确保已安装所有依赖包：
```bash
# 检查Python版本
python --version

# 测试依赖包是否正确安装
python -c "import cv2, numpy, scipy, yaml, imageio, tqdm; print('所有依赖包安装成功!')"
```

### 1. 单配置文件生成

```bash
# 生成较暗光照图像（25-35%亮度）
python low_light_generator.py --config dim_config.yaml

# 生成暗光照图像（15-25%亮度）
python low_light_generator.py --config dark_config.yaml

# 生成非常暗光照图像（5-12%亮度）
python low_light_generator.py --config very_dark_config.yaml
```

### 2. 批量生成所有配置

```bash
python batch_generate.py
```

## 配置文件详解

### 基础设置

```yaml
# 数据集目录（相对于配置文件位置）
dataset_dir: "../01/"

# 文件扩展名过滤
extensions: "jpg,png"

# 是否递归搜索子目录
recursive: false

# 每张原图生成的变体数量
variations: 2

# 输出文件名前缀
prefix: "_dim_light_"

# 文件名模式过滤（只处理包含此模式的文件）
file_pattern: "r.png"
```

### 质量验证（已禁用）

```yaml
validation:
  enable: false    # 禁用验证，确保100%保存率
```

### 光照增强参数

#### 1. 基础调整
```yaml
brightness_range: [0.25, 0.35]    # 亮度范围（原图的25%-35%）
contrast_range: [0.75, 0.85]      # 对比度范围
saturation_range: [0.88, 0.93]    # 饱和度范围
```

#### 2. ISO噪声
```yaml
noise:
  iso_intensity: [3, 6]      # 噪声强度范围
  probability: 0.5           # 应用概率
```

#### 3. 运动模糊
```yaml
blur:
  sigma_range: [0.03, 0.06]  # 模糊程度范围
  probability: 0.08          # 应用概率
```

#### 4. 不均匀照明
```yaml
uneven_lighting:
  multiply_range: [0.96, 1.04]       # 光照变化范围
  add_range: [-1, 1]                 # 亮度偏移范围
  probability: 0.15                  # 应用概率
  vignetting_strength: [0.005, 0.012] # 边缘暗角强度
```

#### 5. 色温变化
```yaml
color_temperature:
  type: "mixed"                    # 混合暖光/冷光
  warm_range: [3460, 3540]         # 暖光色温范围（K）
  cool_range: [5960, 6040]         # 冷光色温范围（K）
  intensity_range: [0.008, 0.02]   # 色温影响强度
  probability: 0.06                # 应用概率
```

#### 6. 阴影增强
```yaml
shadow_enhancement:
  strength_range: [0.003, 0.01]   # 阴影强度范围
  probability: 0.03               # 应用概率
```

## 三种光照配置对比

| 配置 | 亮度范围 | 噪声强度 | 模糊程度 | 适用场景 |
|------|----------|----------|----------|----------|
| **dim_config** | 25-35% | 3-6 | 0.03-0.06 | 室内较暗环境，轻度挑战 |
| **dark_config** | 15-25% | 4-8 | 0.08-0.12 | 室内暗光环境，中等挑战 |
| **very_dark_config** | 5-12% | 6-12 | 0.1-0.15 | 极暗环境，高难度挑战 |

## 输出结果

### 文件命名规则
- 原文件：`pcd0100r.png`
- 生成文件：`pcd0100r_dim_light_1.png`, `pcd0100r_dim_light_2.png`

### 生成统计
生成器会显示：
- 发现的原图数量
- 需要生成的总图像数
- 实时进度条
- 成功/失败统计
- 最终成功率

## 技术特性

### 1. 基于真实数据优化
- 噪声特征匹配RealSense D415相机
- 光照效果模拟真实室内环境
- 参数范围经过大量实验验证

### 2. 稳定性保证
- 小范围参数变化确保结果一致性
- 禁用质量验证确保100%保存率
- 所有随机因子都有合理上下限

### 3. 真实性效果
- ISO噪声：模拟低光下的相机噪声
- 运动模糊：模拟手抖或物体运动
- 不均匀照明：模拟真实光源分布
- 色温变化：模拟不同光源类型
- 阴影增强：模拟物体遮挡效果
- 边缘暗角：模拟镜头特性

## 使用建议

### 1. 数据集准备
- 确保原图在`../01/`目录下
- 使用`r.png`文件（RGB图像）
- 建议图像分辨率一致

### 2. 配置选择
- **训练初期**：使用dim_config（较容易）
- **模型优化**：使用dark_config（中等难度）  
- **极限测试**：使用very_dark_config（最困难）

### 3. 批量处理
- 使用`batch_generate.py`一次生成所有配置
- 每种配置默认生成2个变体
- 总共生成原图数量×6张新图像

### 4. 结果验证
- 检查生成图像的亮度是否符合预期
- 验证噪声水平是否适中
- 确认边缘和细节信息仍然可识别

## 故障排除

### 1. 依赖包相关问题

```bash
# ImportError: No module named 'cv2'
pip install opencv-python

# ImportError: No module named 'yaml'
pip install pyyaml

# ImportError: No module named 'imageio'
pip install imageio

# 升级到最新版本
pip install --upgrade opencv-python numpy scipy pyyaml imageio tqdm
```

### 2. 文件不存在错误
- 检查`dataset_dir`路径是否正确
- 确认目标目录中有`r.png`文件

### 3. 内存不足
- 减少`variations`数量
- 分批处理大数据集

### 4. 生成图像过暗/过亮
- 调整对应配置文件中的`brightness_range`
- 修改`contrast_range`进行补偿

### 5. 处理速度慢
- 减少复杂效果的`probability`
- 使用更快的存储设备

## 扩展配置

如需自定义配置，可参考现有配置文件创建新的YAML文件：

```yaml
# 自定义配置示例
dataset_dir: "../01/"
extensions: "jpg,png" 
recursive: false
variations: 1
prefix: "_custom_"
file_pattern: "r.png"

validation:
  enable: false

augmentation:
  brightness_range: [0.4, 0.6]    # 自定义亮度范围
  contrast_range: [0.8, 0.9]      # 自定义对比度
  saturation_range: [0.9, 0.95]   # 自定义饱和度
  # ... 其他参数
```

然后使用：
```bash
python low_light_generator.py --config your_custom_config.yaml
```
