#!/usr/bin/env python3
"""
真实弱光环境图像生成器 - "混合动力"方案
使用 Albumentations 编排，并用其原生功能替换可标准化的部分（如模糊），
同时保留效果独特的定制函数。
"""

import argparse
from pathlib import Path
import cv2, yaml, numpy as np
import imageio.v3 as iio
from tqdm import tqdm
import albumentations as A

# ==============================================================================
#  第一部分：核心图像处理函数
#  保留了效果独特、无法被原生库完美替代的定制函数。
#  注意：原有的 apply_motion_blur 函数已被移除，因为它将被Albumentations原生实现替代。
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

def simple_darken(img: np.ndarray, factor_range: tuple) -> np.ndarray:
    """定制的乘性亮度调整"""
    factor = np.random.uniform(*factor_range)
    dtype = img.dtype
    max_val = 65535.0 if dtype == np.uint16 else 255.0
    img_float = img.astype(np.float32)
    darkened = img_float * factor
    return np.clip(darkened, 0, max_val).astype(dtype)

def add_iso_noise(img: np.ndarray, intensity_range: tuple) -> np.ndarray:
    """定制的、更逼真的Poisson-Gaussian噪声模型"""
    dtype = img.dtype
    max_val = 65535.0 if dtype == np.uint16 else 255.0
    intensity = np.random.uniform(*intensity_range)
    img_float = img.astype(np.float32)
    alpha = intensity * 0.01
    sigma = intensity * 0.5
    shot_noise = np.sqrt(np.maximum(alpha * img_float, 0)) * np.random.normal(0, 1, img.shape)
    read_noise = np.random.normal(0, sigma, img.shape)
    noisy = img_float + shot_noise + read_noise
    return np.clip(noisy, 0, max_val).astype(dtype)

# 光照相关函数已移除 - 对已拍摄图像有害且不必要
# 保留核心暗光化功能：亮度、对比度、噪声、模糊

def adjust_contrast_saturation(img: np.ndarray, contrast_range: tuple, saturation_range: tuple) -> np.ndarray:
    """定制的对比度和饱和度联合调整"""
    contrast = np.random.uniform(*contrast_range)
    saturation = np.random.uniform(*saturation_range)
    dtype = img.dtype
    max_val = 65535.0 if dtype == np.uint16 else 255.0
    img_float = img.astype(np.float32)
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    contrasted = np.clip((img_float - mean_val) * contrast + mean_val, 0, max_val)
    
    if len(img.shape) == 3 and saturation != 1.0:
        normalized = (contrasted * 255.0 / max_val).astype(np.uint8)
        hsv = cv2.cvtColor(normalized, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result_normalized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = (result_normalized.astype(np.float32) * max_val / 255.0)
    else:
        result = contrasted
    return np.clip(result, 0, max_val).astype(dtype)

# ==============================================================================
#  第二部分：构建“混合动力”增强流水线
# ==============================================================================

def build_hybrid_augmentation_pipeline(cfg: dict) -> A.Compose:
    """
    构建简化的暗光增强流水线，只保留核心功能：
    亮度降低、对比度调整、噪声增加、运动模糊
    """
    aug_cfg = cfg["augmentation"]
    pipeline_steps = []

    # 为核心函数创建包装器
    def simple_darken_wrapper(image, **kwargs):
        return simple_darken(image, factor_range=aug_cfg["brightness_range"])

    def adjust_contrast_saturation_wrapper(image, **kwargs):
        return adjust_contrast_saturation(image, contrast_range=aug_cfg["contrast_range"], saturation_range=aug_cfg["saturation_range"])

    def add_iso_noise_wrapper(image, **kwargs):
        return add_iso_noise(image, intensity_range=aug_cfg["noise"]["iso_intensity"])

    # 核心暗光效果流水线
    # 1. 亮度降低
    pipeline_steps.append(A.Lambda(name="simple_darken", image=simple_darken_wrapper, p=1.0))
    
    # 2. 对比度和饱和度调整
    pipeline_steps.append(A.Lambda(name="adjust_contrast_saturation", image=adjust_contrast_saturation_wrapper, p=1.0))

    # 3. 运动模糊 (使用Albumentations原生实现)
    sigma_min, sigma_max = aug_cfg["blur"]["sigma_range"]
    blur_limit_min = max(3, int(2 * sigma_min + 1))
    blur_limit_max = max(3, int(2 * sigma_max + 1))
    # 确保核大小是奇数
    if blur_limit_min % 2 == 0: blur_limit_min += 1
    if blur_limit_max % 2 == 0: blur_limit_max += 1
    
    pipeline_steps.append(
        A.OneOf([
            A.GaussianBlur(blur_limit=(blur_limit_min, blur_limit_max), p=0.8),
            A.MotionBlur(blur_limit=(blur_limit_min, blur_limit_max), p=0.2),
        ], p=aug_cfg["blur"]["probability"])
    )

    # 4. ISO噪声 (最后添加，模拟暗光下的高ISO噪声)
    pipeline_steps.append(A.Lambda(name="add_iso_noise", image=add_iso_noise_wrapper, p=aug_cfg["noise"]["probability"]))

    return A.Compose(pipeline_steps)


# ==============================================================================
#  第三部分：数据集处理与主函数
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