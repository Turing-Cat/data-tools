#!/usr/bin/env python3
"""
批量生成不同光照强度的弱光图像
"""

import subprocess
import sys
import time
from pathlib import Path

def run_generation(config_file, description):
    """运行单个配置的生成"""
    print(f"\n{'='*60}")
    print(f"开始生成 {description} 图像")
    print(f"使用配置: {config_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "low_light_generator.py", 
            "--config", config_file
        ], capture_output=True, text=True, check=True)
        
        # 输出生成结果
        print(result.stdout)
        if result.stderr:
            print("警告信息:", result.stderr)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} 图像生成完成！耗时: {elapsed_time:.1f}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 图像生成失败!")
        print("错误输出:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ 生成过程出现异常: {e}")
        return False

def main():
    """主函数：批量生成三种光照强度的图像"""
    print("🌙 弱光图像批量生成器")
    print("将生成三种不同光照强度的图像：较暗、暗、非常暗")
    
    # 检查配置文件是否存在
    configs = [
        ("dim_config.yaml", "较暗光照"),
        ("dark_config.yaml", "暗光照"), 
        ("very_dark_config.yaml", "非常暗光照")
    ]
    
    missing_configs = []
    for config_file, _ in configs:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"❌ 缺少配置文件: {', '.join(missing_configs)}")
        return
    
    # 询问用户是否继续
    response = input("\n是否开始批量生成? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("已取消生成")
        return
    
    # 记录整体开始时间
    total_start_time = time.time()
    successful_generations = 0
    
    # 逐个运行配置
    for config_file, description in configs:
        if run_generation(config_file, description):
            successful_generations += 1
        else:
            # 询问是否继续下一个
            if len(configs) > 1:
                continue_response = input(f"\n{description} 生成失败，是否继续下一个配置? (y/n): ").lower().strip()
                if continue_response not in ['y', 'yes', '是']:
                    break
    
    # 总结
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"批量生成完成!")
    print(f"成功生成: {successful_generations}/{len(configs)} 种光照类型")
    print(f"总耗时: {total_elapsed:.1f}秒")
    print(f"{'='*60}")
    
    # 显示生成的文件统计
    print("\n📊 生成文件统计:")
    data_dir = Path("../01")
    for prefix in ["_dim_light_", "_dark_light_", "_very_dark_light_"]:
        files = list(data_dir.glob(f"*{prefix}*"))
        print(f"  {prefix:<18} {len(files):>3} 个文件")

if __name__ == "__main__":
    main()
