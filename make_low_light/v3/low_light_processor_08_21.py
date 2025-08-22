#!/usr/bin/env python3
"""
改进的低光图像处理模块
实现更真实的低光效果，包括噪声、颜色偏移和运动模糊
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import hashlib

def load_config(path: str) -> dict:
    """加载并设置配置文件的默认值"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("dataset_dir"):
        raise ValueError("'dataset_dir' 必须在配置中指定")

    cfg.setdefault("extensions", "jpg,jpeg,png,bmp")
    cfg.setdefault("recursive", False)
    cfg.setdefault("prefix", "_low_light_")
    cfg.setdefault("file_pattern", None)
    
    # 设置处理参数的默认值
    processing_defaults = {
        "brightness_factor": 0.3,
        "gamma": 1.5,
        "shot_noise_factor": 0.025,
        "read_noise_std": 0.012,
        "color_shift_factor": 0.05,
        "motion_blur_prob": 0.2,
        "max_blur_kernel": 5,
        "sensor_sensitivity": 0.8
    }
    
    if 'processing' not in cfg:
        cfg['processing'] = processing_defaults
    else:
        for key, value in processing_defaults.items():
            cfg['processing'].setdefault(key, value)
    
    return cfg

def apply_low_light_effect(img_path: str, output_path: str, config: dict = None):
    """
    应用改进的低光效果到图像
    
    参数:
    - img_path: 输入图像路径
    - output_path: 输出图像路径
    - config: 配置参数字典
    """
    # 使用默认参数或从配置中获取参数
    if config is not None and 'processing' in config:
        proc_cfg = config['processing']
        brightness_factor = proc_cfg.get('brightness_factor', 0.3)
        gamma = proc_cfg.get('gamma', 1.5)
        shot_noise_factor = proc_cfg.get('shot_noise_factor', 0.025)
        read_noise_std = proc_cfg.get('read_noise_std', 0.012)
        color_shift_factor = proc_cfg.get('color_shift_factor', 0.05)
        motion_blur_prob = proc_cfg.get('motion_blur_prob', 0.2)
        max_blur_kernel = proc_cfg.get('max_blur_kernel', 5)
        sensor_sensitivity = proc_cfg.get('sensor_sensitivity', 0.8)
    else:
        # 默认值
        brightness_factor = 0.3
        gamma = 1.5
        shot_noise_factor = 0.025
        read_noise_std = 0.012
        color_shift_factor = 0.05
        motion_blur_prob = 0.2
        max_blur_kernel = 5
        sensor_sensitivity = 0.8
    
    # 使用图像路径生成确定性但唯一的随机种子
    path_hash = int(hashlib.md5(img_path.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(path_hash)
    
    # 步骤1: 加载图像并归一化
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 转换为RGB（OpenCV默认是BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 步骤2: 模拟传感器灵敏度差异（不同颜色通道响应不同）
    sensitivity_matrix = np.array([
        [sensor_sensitivity, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0/sensor_sensitivity]
    ])
    img = np.dot(img, sensitivity_matrix)
    img = np.clip(img, 0, 1)

    # 步骤3: 降低亮度（模拟光照不足）
    low_light = img * brightness_factor

    # 步骤4: 添加颜色偏移（模拟低光白平衡问题）
    color_shift = 1.0 + rng.uniform(-color_shift_factor, color_shift_factor, 3)
    low_light = low_light * color_shift
    low_light = np.clip(low_light, 0, 1)

    # 步骤5: 在线性空间中添加传感器噪声
    # 5a: 添加光子散粒噪声（泊松噪声，与信号强度相关）
    if shot_noise_factor > 0:
        # 更真实的泊松噪声近似
        noise_std = np.sqrt(np.maximum(low_light, 0) + 1e-6) * shot_noise_factor
        shot_noise = rng.normal(0, 1, low_light.shape) * noise_std
        low_light = low_light + shot_noise

    # 5b: 添加读出噪声（固定高斯噪声，与信号无关）
    if read_noise_std > 0:
        read_noise = rng.normal(0, read_noise_std, low_light.shape)
        low_light = low_light + read_noise

    # 步骤6: 裁剪到合理范围
    low_light = np.clip(low_light, 0, 1)

    # 步骤7: 应用Gamma校正
    low_light = np.power(low_light, gamma)
    low_light = np.clip(low_light, 0, 1)

    # 步骤8: 随机添加运动模糊（模拟低光下长曝光）
    if rng.rand() < motion_blur_prob:
        # 随机选择模糊方向和大小
        blur_size = rng.randint(1, max_blur_kernel + 1)
        angle = rng.uniform(0, 180)
        
        # 创建运动模糊核
        kernel = np.zeros((blur_size, blur_size))
        kernel[int((blur_size-1)/2), :] = np.ones(blur_size)
        kernel = kernel / blur_size
        
        # 旋转核以模拟不同方向的运动
        M = cv2.getRotationMatrix2D((blur_size/2, blur_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (blur_size, blur_size))
        
        # 应用运动模糊
        low_light = cv2.filter2D(low_light, -1, kernel)

    # 步骤9: 最终裁剪并转换回uint8
    low_light = np.clip(low_light, 0, 1)
    
    # 转换回BGR格式（OpenCV格式）
    low_light = cv2.cvtColor((low_light * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 保存输出
    cv2.imwrite(output_path, low_light)

def process_image(src: Path, dst: Path, config: dict = None) -> bool:
    """处理单张图像"""
    try:
        apply_low_light_effect(str(src), str(dst), config)
        return True
    except Exception as e:
        tqdm.write(f"处理失败 {src.name}: {e}")
        return False

def process_dataset(cfg: dict):
    """处理整个数据集"""
    root = Path(cfg["dataset_dir"])
    exts = ["." + e.lower().lstrip(".") for e in cfg["extensions"].split(",")]
    iterator = root.rglob if cfg["recursive"] else root.glob
    files = sorted(set(f for ext in exts for f in iterator(f"*{ext}")))
    
    if cfg.get("file_pattern"):
        files = [f for f in files if cfg["file_pattern"] in f.name]

    total = len(files)
    print(f"发现 {len(files)} 张原图，需要生成 {total} 张低光样本")

    success = 0
    with tqdm(total=total, unit="img") as bar:
        for f in files:
            out_name = f"{f.stem}{cfg['prefix']}{f.suffix}"
            
            if process_image(f, f.parent / out_name, cfg):
                success += 1
            bar.update(1)
            bar.set_postfix(ok=success, fail=bar.n - success)
    
    rate = success / total * 100 if total else 0
    print(f"\n完成：{success}/{total}（成功率 {rate:.1f}%）")

def main():
    # 固定随机种子以保证结果可复现
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="低光图像生成器")
    parser.add_argument("--config", "-c", default="low_light_config.yaml", help="YAML配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"数据集目录: {cfg['dataset_dir']}")
    print(f"输出前缀: {cfg['prefix']}")
    print()
    
    # 显示处理参数
    if 'processing' in cfg:
        print("处理参数:")
        for key, value in cfg['processing'].items():
            print(f"  {key}: {value}")
        print()
    
    process_dataset(cfg)

if __name__ == "__main__":
    main()