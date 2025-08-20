#!/usr/bin/env python3
"""
低光图像生成器主程序
"""

import argparse
import numpy as np
from low_light_processor import load_config, process_dataset

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