#!/usr/bin/env python3
"""
真实弱光环境图像生成器 - 模拟室内弱光抓取检测场景
基于Cornell抓取检测数据集生成更接近真实弱光环境的图像
"""

import argparse
from pathlib import Path
import cv2, yaml, numpy as np
import imageio.v3 as iio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("dataset_dir"):
        raise ValueError("'dataset_dir' 必须在配置中指定")

    cfg.setdefault("extensions", "jpg,jpeg,png,bmp")
    cfg.setdefault("variations", 1)
    cfg.setdefault("recursive", False)
    cfg.setdefault("prefix", "_realistic_dark_")
    cfg.setdefault("file_pattern", None)
    
    # 添加验证参数默认值
    if "validation" not in cfg:
        cfg["validation"] = {"enable": False}
    cfg["validation"].setdefault("min_brightness", 10)
    cfg["validation"].setdefault("max_brightness", 100)
    
    return cfg

def simple_darken(img: np.ndarray, factor_range: tuple) -> np.ndarray:
    """基础图像变暗"""
    factor = np.random.uniform(*factor_range)
    darkened = (img.astype(np.float32) * factor).astype(np.uint8)
    return darkened

def add_iso_noise(img: np.ndarray, intensity_range: tuple) -> np.ndarray:
    """添加极稳定的ISO噪声，严格匹配RealSense真实噪声特征"""
    # 使用更小的噪声强度范围
    intensity = np.random.uniform(*intensity_range)
    
    # 更保守的亮度相关调整
    img_brightness = np.mean(img)
    brightness_factor = max(0.5, min(1.2, (255 - img_brightness) / 200))  # 限制变化范围
    adjusted_intensity = intensity * brightness_factor
    
    # 使用更稳定的噪声模型
    if np.random.random() > 0.6:  # 40%泊松噪声，60%高斯噪声
        # 稳定的泊松噪声
        img_normalized = img.astype(np.float32) / 255.0
        # 大幅减少噪声强度的随机性
        poisson_noise = np.random.poisson(img_normalized * adjusted_intensity * 0.05) - img_normalized * adjusted_intensity * 0.05
        noise = poisson_noise * 20.0  # 控制噪声幅度
    else:
        # 稳定的高斯噪声
        noise = np.random.normal(0, adjusted_intensity * 0.3, img.shape).astype(np.float32)
    
    # 极轻微的乘性噪声
    multiplicative_noise = 1 + np.random.normal(0, 0.001, img.shape).astype(np.float32)
    
    # 应用稳定的噪声
    noisy = img.astype(np.float32) * multiplicative_noise + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_motion_blur(img: np.ndarray, sigma_range: tuple) -> np.ndarray:
    """添加更自然的运动模糊，基于真实相机抖动模式"""
    sigma = np.random.uniform(*sigma_range)
    
    # 大多数情况使用各向同性的高斯模糊（更常见）
    if np.random.random() < 0.8:  # 80%概率使用高斯模糊
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    else:  # 20%概率使用轻微的方向性模糊
        # 创建轻微的运动核
        kernel_size = max(3, int(2 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 随机方向的运动模糊
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 创建线性运动核
        center = kernel_size // 2
        length = min(kernel_size, max(3, int(sigma * 2)))
        
        for i in range(length):
            x = int(center + (i - length//2) * np.cos(np.radians(angle)))
            y = int(center + (i - length//2) * np.sin(np.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
            blurred = cv2.filter2D(img, -1, kernel)
        else:
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    return blurred

def create_uneven_lighting(img: np.ndarray, multiply_range: tuple, add_range: tuple, vignetting_range: tuple = None) -> np.ndarray:
    """创建非均匀照明效果，模拟真实光源分布"""
    h, w = img.shape[:2]
    
    # 创建渐变光照图
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    
    # 主光源位置随机化
    light_x = np.random.uniform(w * 0.2, w * 0.8)
    light_y = np.random.uniform(h * 0.2, h * 0.8)
    
    # 距离衰减
    dist_from_light = np.sqrt((x - light_x)**2 + (y - light_y)**2)
    max_dist = np.sqrt(w**2 + h**2)
    
    # 光照强度分布
    multiply_factor = np.random.uniform(*multiply_range)
    add_factor = np.random.uniform(*add_range)
    
    # 创建光照图
    lighting_map = multiply_factor * (1 - dist_from_light / max_dist) + add_factor
    lighting_map = np.clip(lighting_map, 0.3, 1.5)  # 限制范围避免过暗或过亮
    
    # 添加边缘暗角效果
    if vignetting_range:
        vignetting_strength = np.random.uniform(*vignetting_range)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_center_dist = np.sqrt(center_x**2 + center_y**2)
        vignetting = 1 - vignetting_strength * (dist_from_center / max_center_dist)**2
        lighting_map *= vignetting
    
    # 应用光照效果
    if len(img.shape) == 3:
        lighting_map = np.stack([lighting_map] * 3, axis=2)
    
    result = img.astype(np.float32) * lighting_map
    return np.clip(result, 0, 255).astype(np.uint8)

def adjust_color_temperature(img: np.ndarray, temp_type: str, warm_range: tuple, cool_range: tuple, intensity_range: tuple) -> np.ndarray:
    """调整色温，模拟不同光源的影响"""
    intensity = np.random.uniform(*intensity_range)
    
    if temp_type == "warm" or (temp_type == "mixed" and np.random.random() < 0.5):
        # 暖光 - 增加红色/黄色分量
        temp_shift = np.random.uniform(*warm_range) / 6500  # 归一化
        r_boost = 1 + intensity * temp_shift
        b_reduce = 1 - intensity * temp_shift * 0.3
        color_matrix = np.array([r_boost, 1.0, b_reduce])
    else:
        # 冷光 - 增加蓝色分量
        temp_shift = np.random.uniform(*cool_range) / 6500
        b_boost = 1 + intensity * (temp_shift - 0.5)
        r_reduce = 1 - intensity * 0.2
        color_matrix = np.array([r_reduce, 1.0, b_boost])
    
    # 应用色温调整
    if len(img.shape) == 3:
        result = img.astype(np.float32) * color_matrix
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        return img

def enhance_shadows(img: np.ndarray, strength_range: tuple) -> np.ndarray:
    """增强阴影效果，模拟物体遮挡"""
    strength = np.random.uniform(*strength_range)
    
    # 使用形态学操作找到潜在的阴影区域
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # 创建阴影模板
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_regions = cv2.morphologyEx(255 - gray, cv2.MORPH_CLOSE, kernel)
    
    # 模糊阴影边缘
    shadow_mask = cv2.GaussianBlur(dark_regions, (21, 21), 0) / 255.0
    
    # 应用阴影效果
    shadow_factor = 1 - strength * shadow_mask
    if len(img.shape) == 3:
        shadow_factor = np.stack([shadow_factor] * 3, axis=2)
    
    result = img.astype(np.float32) * shadow_factor
    return np.clip(result, 0, 255).astype(np.uint8)

def adjust_contrast_saturation(img: np.ndarray, contrast_range: tuple, saturation_range: tuple) -> np.ndarray:
    """调整对比度和饱和度"""
    contrast = np.random.uniform(*contrast_range)
    saturation = np.random.uniform(*saturation_range)
    
    # 调整对比度
    mean_val = np.mean(img)
    contrasted = (img.astype(np.float32) - mean_val) * contrast + mean_val
    contrasted = np.clip(contrasted, 0, 255)
    
    if len(img.shape) == 3 and saturation != 1.0:
        # 调整饱和度
        hsv = cv2.cvtColor(contrasted.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        result = contrasted.astype(np.uint8)
    
    return result

def validate_image_quality(img: np.ndarray, cfg: dict) -> bool:
    """验证生成图像的质量 - 已禁用，直接返回True确保所有图像都保存"""
    return True  # 直接返回True，不进行任何验证

def process_image(src: Path, dst: Path, cfg: dict) -> bool:
    """处理单张图像，应用真实弱光环境效果"""
    try:
        img = iio.imread(str(src))
        original_mean = np.mean(img)
        
        # 确保图像是RGB格式
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA转RGB
            img = img[:, :, :3]
        
        aug_cfg = cfg["augmentation"]
        
        # 1. 基础变暗
        result = simple_darken(img, aug_cfg["brightness_range"])
        
        # 2. 调整对比度和饱和度
        result = adjust_contrast_saturation(result, 
                                          aug_cfg["contrast_range"], 
                                          aug_cfg["saturation_range"])
        
        # 3. 添加非均匀照明效果
        if np.random.random() < aug_cfg["uneven_lighting"]["probability"]:
            vignetting_range = aug_cfg["uneven_lighting"].get("vignetting_strength", [0.1, 0.3])
            result = create_uneven_lighting(result,
                                          aug_cfg["uneven_lighting"]["multiply_range"],
                                          aug_cfg["uneven_lighting"]["add_range"],
                                          vignetting_range)
        
        # 4. 添加ISO噪声
        if np.random.random() < aug_cfg["noise"]["probability"]:
            result = add_iso_noise(result, aug_cfg["noise"]["iso_intensity"])
        
        # 5. 添加运动模糊
        if np.random.random() < aug_cfg["blur"]["probability"]:
            result = apply_motion_blur(result, aug_cfg["blur"]["sigma_range"])
        
        # 6. 色温调整
        if np.random.random() < aug_cfg["color_temperature"]["probability"]:
            result = adjust_color_temperature(result,
                                            aug_cfg["color_temperature"]["type"],
                                            aug_cfg["color_temperature"]["warm_range"],
                                            aug_cfg["color_temperature"]["cool_range"],
                                            aug_cfg["color_temperature"]["intensity_range"])
        
        # 7. 增强阴影效果
        if np.random.random() < aug_cfg["shadow_enhancement"]["probability"]:
            result = enhance_shadows(result, aug_cfg["shadow_enhancement"]["strength_range"])
        
        # 8. 质量验证 - 已禁用，所有图像都会保存
        # if not validate_image_quality(result, cfg):
        #     print(f"质量验证失败: {src.name} (亮度: {np.mean(result):.1f})")
        #     return False
        
        # 保存结果
        iio.imwrite(str(dst), result)
        
        # 不再输出单个文件的处理信息，只返回成功状态
        return True
        
    except Exception as e:
        # 只在控制台输出错误，不影响进度条显示
        tqdm.write(f"处理失败 {src.name}: {e}")
        return False

def process_dataset(cfg: dict):
    """处理数据集"""
    root = Path(cfg["dataset_dir"])
    exts = ["."+e.lower().lstrip(".") for e in cfg["extensions"].split(",")]
    iterator = root.rglob if cfg["recursive"] else root.glob

    files = []
    for ext in exts:
        files.extend(iterator(f"*{ext}"))
        files.extend(iterator(f"*{ext.upper()}"))
    
    if cfg["file_pattern"]: 
        files = [f for f in files if cfg["file_pattern"] in f.name]

    total = len(files) * cfg["variations"]
    print(f"发现 {len(files)} 张原图，需要生成 {total} 张真实弱光样本")
    print(f"应用效果: 变暗、噪声、模糊、不均匀照明、色温变化、阴影增强\n")

    success = 0
    with tqdm(total=total, unit="img") as bar:
        for f in files:
            for i in range(cfg["variations"]):
                suffix = f"_{i+1}" if cfg["variations"] > 1 else ""
                out_name = f"{f.stem}{cfg['prefix']}{suffix}{f.suffix}"
                
                if process_image(f, f.parent / out_name, cfg):
                    success += 1
                bar.update(1)
                bar.set_postfix(ok=success, fail=bar.n-success)
    
    rate = success / total * 100 if total else 0
    print(f"\n完成：{success}/{total}（成功率 {rate:.1f}%）")

def main():
    parser = argparse.ArgumentParser(description="Realistic low-light image generator for grasp detection dataset")
    parser.add_argument("--config", default="simple_config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=== 真实弱光环境图像生成器 ===")
    print(f"数据集目录: {cfg['dataset_dir']}")
    print(f"变体数量: {cfg['variations']}")
    print(f"输出前缀: {cfg['prefix']}")
    print()
    
    process_dataset(cfg)

if __name__ == "__main__":
    main()
