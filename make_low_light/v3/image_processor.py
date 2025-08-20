#!/usr/bin/env python3
"""
图像处理核心功能模块
包含配置文件解析、单张图像处理和目录处理功能
"""

import argparse
from pathlib import Path
import cv2, yaml, numpy as np
import imageio.v3 as iio
from tqdm import tqdm
import albumentations as A

# ==============================================================================
#  配置文件处理函数
# ==============================================================================

def load_config(path: str) -> dict:
    """加载并设置配置文件的默认值"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("dataset_dir"):
        raise ValueError("'dataset_dir' 必须在配置中指定")

    cfg.setdefault("extensions", "jpg,jpeg,png,bmp")
    cfg.setdefault("variations", 1)
    cfg.setdefault("recursive", False)
    cfg.setdefault("prefix", "_dark_hybrid_")
    cfg.setdefault("file_pattern", None)
    return cfg

# ==============================================================================
#  单张图像处理函数
# ==============================================================================

def process_image(src: Path, dst: Path, pipeline: A.Compose) -> bool:
    """处理单张图像，应用由Albumentations编排的混合流水线"""
    try:
        img = iio.imread(str(src))
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        # 应用完整的增强流水线
        result = pipeline(image=img)['image']
        
        iio.imwrite(str(dst), result)
        return True
    except Exception as e:
        tqdm.write(f"处理失败 {src.name}: {e}")
        return False

# ==============================================================================
#  目录处理函数
# ==============================================================================

def process_dataset(cfg: dict):
    """处理整个数据集"""
    root = Path(cfg["dataset_dir"])
    exts = ["." + e.lower().lstrip(".") for e in cfg["extensions"].split(",")]
    iterator = root.rglob if cfg["recursive"] else root.glob
    files = sorted(set(f for ext in exts for f in iterator(f"*{ext}")))
    
    if cfg.get("file_pattern"):
        files = [f for f in files if cfg["file_pattern"] in f.name]

    total = len(files) * cfg["variations"]
    print(f"发现 {len(files)} 张原图，需要生成 {total} 张真实弱光样本")

    # 构建一次流水线，供所有图像使用，效率更高
    pipeline = build_hybrid_augmentation_pipeline(cfg)

    success = 0
    with tqdm(total=total, unit="img") as bar:
        for f in files:
            for i in range(cfg["variations"]):
                suffix = f"_{i+1}" if cfg["variations"] > 1 else ""
                out_name = f"{f.stem}{cfg['prefix']}{suffix}{f.suffix}"
                
                if process_image(f, f.parent / out_name, pipeline):
                    success += 1
                bar.update(1)
                bar.set_postfix(ok=success, fail=bar.n - success)
    
    rate = success / total * 100 if total else 0
    print(f"\n完成：{success}/{total}（成功率 {rate:.1f}%）")

# ==============================================================================
#  主函数
# ==============================================================================

def main():
    # 固定随机种子以保证结果可复现
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Realistic low-light image generator (Hybrid Albumentations Version)")
    parser.add_argument("--config", "-c", default="dark_config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"数据集目录: {cfg['dataset_dir']}")
    print(f"变体数量: {cfg['variations']}")
    print(f"输出前缀: {cfg['prefix']}")
    print()
    
    process_dataset(cfg)

if __name__ == "__main__":
    main()