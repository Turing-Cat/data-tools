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
    else:
        brightness_factor = 0.3
        gamma = 1.5
        shot_noise_factor = 0.02
        read_noise_std = 0.01
    
    # 步骤1: 加载图像并归一化
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 转换为RGB（OpenCV默认是BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

    # 步骤2: 降低亮度（模拟光照不足）
    low_light = img * brightness_factor

    # 步骤3: 在线性空间中添加传感器噪声（更符合物理过程）
    
    # 3a: 添加光子散粒噪声（泊松噪声，与信号强度相关）
    if shot_noise_factor > 0:
        # 光子噪声的标准差与信号强度的平方根成正比
        noise_std = np.sqrt(np.maximum(low_light, 0)) * shot_noise_factor
        shot_noise = np.random.normal(0, 1, low_light.shape) * noise_std
        low_light = low_light + shot_noise

    # 3b: 添加读出噪声（固定高斯噪声，与信号无关）
    if read_noise_std > 0:
        read_noise = np.random.normal(0, read_noise_std, low_light.shape)
        low_light = low_light + read_noise



    # 步骤4: 裁剪到合理范围（避免负值影响Gamma校正）
    low_light = np.clip(low_light, 0, 1)

    # 步骤5: 应用Gamma校正（模拟ISP处理，进一步增强暗部效果）
    # Gamma > 1 会使图像更暗，符合低光合成的目标
    low_light = np.power(low_light, gamma)



    # 步骤6: 最终裁剪并转换回uint8
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