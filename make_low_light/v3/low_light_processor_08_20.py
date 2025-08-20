#!/usr/bin/env python3
"""
低光图像处理模块
实现亮度降低、Gamma校正和噪声添加等功能
改进版本：修正噪声模型，使其更符合真实低光场景
"""

import cv2
import numpy as np
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
        brightness_factor = config['processing'].get('brightness_factor', 0.3)
        gamma = config['processing'].get('gamma', 1.5)
        shot_noise_factor = config['processing'].get('shot_noise_factor', 0.02)
        read_noise_std = config['processing'].get('read_noise_std', 0.01)
        color_shift = config['processing'].get('color_shift', True)
    else:
        brightness_factor = 0.3
        gamma = 1.5
        shot_noise_factor = 0.02
        read_noise_std = 0.01
        color_shift = True
    
    # 步骤1: 加载图像并归一化
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 转换为RGB（OpenCV默认是BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 步骤2: 降低亮度
    low_light = img * brightness_factor

    # 步骤3: 应用Gamma校正（更温和的gamma值）
    low_light = np.power(low_light, gamma)

    # 步骤4: 添加真实的光子散粒噪声
    # 光子噪声的标准差与信号强度的平方根成正比
    if shot_noise_factor > 0:
        # 计算每个像素的噪声标准差
        noise_std = np.sqrt(np.maximum(low_light, 0)) * shot_noise_factor
        shot_noise = np.random.normal(0, 1, low_light.shape) * noise_std
        low_light = low_light + shot_noise

    # 步骤5: 添加读出噪声（固定高斯噪声）
    if read_noise_std > 0:
        read_noise = np.random.normal(0, read_noise_std, low_light.shape)
        low_light = low_light + read_noise

    # 步骤6: 添加轻微的颜色偏移（模拟低光下的色彩失真）
    if color_shift:
        # 低光下通常蓝色通道噪声更明显
        color_noise_factors = [1.0, 1.1, 1.2]  # R, G, B通道的噪声因子
        for i in range(3):
            channel_noise = np.random.normal(0, 0.005 * color_noise_factors[i], low_light[:,:,i].shape)
            low_light[:,:,i] += channel_noise

    # 步骤7: 模拟传感器的非线性响应（可选）
    # 在极低光下，传感器响应可能不是完全线性的
    # 使用S曲线轻微调整
    def s_curve(x, strength=0.1):
        """应用S曲线调整"""
        # 使用sigmoid函数的变体
        return x + strength * (x - x**2) * (1 - x)
    
    low_light = s_curve(low_light, strength=0.05)

    # 步骤8: 裁剪到[0,1]并转换回uint8
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

if __name__ == "__main__":
    # 示例用法
    # apply_low_light_effect('input.jpg', 'output.jpg')
    pass