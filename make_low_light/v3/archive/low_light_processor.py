#!/usr/bin/env python3
"""
低光图像处理模块
实现亮度降低、Gamma校正和噪声添加等功能
"""

import cv2
import numpy as np
from scipy.stats import poisson
from pathlib import Path
import yaml
from tqdm import tqdm

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
    return cfg

def apply_low_light_effect(img_path: str, output_path: str, config: dict = None):
    """
    应用低光效果到图像
    
    参数:
    - img_path: 输入图像路径
    - output_path: 输出图像路径
    - config: 配置参数字典
    """
    # 使用默认参数或从配置中获取参数
    if config is not None and 'processing' in config:
        brightness_factor = config['processing'].get('brightness_factor', 0.2)
        gamma = config['processing'].get('gamma', 2.0)
        gain = config['processing'].get('gain', 0.1)
        read_noise_std = config['processing'].get('read_noise_std', 0.05)
    else:
        brightness_factor = 0.2
        gamma = 2.0
        gain = 0.1
        read_noise_std = 0.05
    # 步骤1: 加载图像并归一化
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 步骤2: 降低亮度
    low_light = img * brightness_factor

    # 步骤3: 添加Gamma校正
    low_light = np.power(low_light, gamma)

    # 步骤4: 添加信号相关噪声（泊松噪声，模拟光子噪声）
    # 使用泊松分布：对每个像素应用Poisson采样（需缩放以匹配分布）
    # 先缩放像素值到计数域（假设最大光子数为1/gain）
    shot_noise = poisson.rvs(low_light / gain, size=low_light.shape) * gain
    low_light = np.clip(shot_noise, 0, 1)  # 应用噪声并裁剪

    # 步骤5: 添加信号无关噪声（高斯噪声，模拟读出噪声）
    read_noise = np.random.normal(0, read_noise_std, low_light.shape)
    low_light += read_noise

    # 步骤6: 裁剪到[0,1]并转换为uint8
    low_light = np.clip(low_light, 0, 1) * 255
    low_light = low_light.astype(np.uint8)

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

if __name__ == "__main__":
    # 示例用法
    # apply_low_light_effect('input.jpg', 'output.jpg')
    pass