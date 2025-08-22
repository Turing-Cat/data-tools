#!/usr/bin/env python3
"""
低光图像处理模块
实现亮度降低、Gamma校正和噪声添加等功能
改进版本：修正噪声添加顺序，使其更符合真实低光场景的物理过程
"""

import argparse
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
    应用低光效果到图像（最终版，增加模糊、色彩损失和亮度随机扰动）
    """
    # 从配置中获取所有参数
    if config is not None and 'processing' in config:
        p_cfg = config['processing']
        brightness_factor = p_cfg.get('brightness_factor', 0.2)
        gamma = p_cfg.get('gamma', 1.8)
        shot_noise_factor = p_cfg.get('shot_noise_factor', 0.05)
        read_noise_std = p_cfg.get('read_noise_std', 0.03)
        blur_kernel_size = p_cfg.get('blur_kernel_size', 5)
        blur_sigma = p_cfg.get('blur_sigma', 1.5)
        desaturation_factor = p_cfg.get('desaturation_factor', 0.8)
    else:
        # 提供一组默认的挑战性参数
        brightness_factor = 0.2
        gamma = 1.8
        shot_noise_factor = 0.05
        read_noise_std = 0.03
        blur_kernel_size = 5
        blur_sigma = 1.5
        desaturation_factor = 0.8

    # [新增] 步骤1a: 为亮度添加随机波动
    brightness_factor *= np.random.uniform(0.95, 1.05)

    # 步骤1b: 加载图像并归一化
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # 步骤2: 降低亮度
    low_light = img * brightness_factor

    # 步骤2a: 降低色彩饱和度
    if desaturation_factor < 1.0:
        hls_img = cv2.cvtColor(low_light, cv2.COLOR_RGB2HLS)
        hls_img[:, :, 2] *= (1.0 - desaturation_factor)
        low_light = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)

    # 步骤2b: 应用高斯模糊
    if blur_kernel_size > 0 and blur_sigma > 0:
        kernel = (int(blur_kernel_size) // 2 * 2 + 1, int(blur_kernel_size) // 2 * 2 + 1)
        low_light = cv2.GaussianBlur(low_light, kernel, blur_sigma)

    # 步骤3: 添加高强度噪声
    if shot_noise_factor > 0:
        noise_std = np.sqrt(np.maximum(low_light, 0)) * shot_noise_factor
        shot_noise = np.random.normal(0, 1, low_light.shape) * noise_std
        low_light = low_light + shot_noise

    if read_noise_std > 0:
        read_noise = np.random.normal(0, read_noise_std, low_light.shape)
        low_light = low_light + read_noise

    # 步骤4: 裁剪
    low_light = np.clip(low_light, 0, 1)

    # 步骤5: 应用Gamma校正
    low_light = np.power(low_light, gamma)

    # 步骤6: 最终裁剪并转换
    low_light = np.clip(low_light, 0, 1)
    low_light = cv2.cvtColor((low_light * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, low_light)

# 别忘了在 process_image 函数中调用这个 apply_low_light_effect_revised 新版本

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