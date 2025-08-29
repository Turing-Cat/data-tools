import os
import re
import shutil
import argparse

# --- 脚本主逻辑 ---

def parse_filename(filename):
    """解析复杂的文件名，提取关键信息。"""
    pattern = re.compile(
        r"(\d{8}T\d{9})_RealSense_(\d+)_frame_\d+_([\w_]+)\.(tiff|png|txt)"
    )
    match = pattern.match(filename)
    if match:
        return {
            "full_ts": match.group(1),
            "serial": match.group(2),
            "type": match.group(3),
            "ext": match.group(4)
        }
    return None

def run_conversion_single_dir(source_dir, target_dir):
    """将所有文件重命名并放入单个目标目录中。"""
    if not os.path.isdir(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在。请检查路径是否正确。")
        return False

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    try:
        filenames = os.listdir(source_dir)
        print(f"在 '{source_dir}' 中找到 {len(filenames)} 个文件，开始处理...")
    except FileNotFoundError:
        print(f"错误: 无法访问源目录 '{source_dir}'。请检查路径和权限。")
        return False

    # --- 直接遍历所有文件，进行重命名和复制 ---
    processed_count = 0
    skipped_count = 0
    
    for filename in filenames:
        parsed = parse_filename(filename)
        if not parsed:
            print(f"  - 格式不匹配，已忽略文件: {filename}")
            skipped_count += 1
            continue

        # 使用您最终确认的命名格式
        simplified_ts = parsed["full_ts"]
        short_serial = parsed["serial"][-4:]    # 取序列号后四位

        if parsed["type"] == "rgb" and parsed["ext"] == "txt":
            data_type = "grasps"
            new_ext = "txt"
        else:
            data_type = parsed["type"]
            new_ext = parsed["ext"]
        
        # 构建新的文件名 -> {timestamp}c{short_serial}_{data_type}.ext
        new_filename = f"{simplified_ts}c{short_serial}_{data_type}.{new_ext}"
        
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, new_filename)

        print(f"  复制: {filename}  ->  {new_filename}")
        shutil.copy2(source_path, target_path)
        processed_count += 1

    print(f"\n转换完成！处理了 {processed_count} 个文件，跳过了 {skipped_count} 个文件。")
    print(f"所有重命名后的文件已保存在 '{target_dir}' 目录中。")
    return True

def main():
    """主函数：解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description='批量重命名数据集文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python rename_dataset.py ./original_files ./renamed_files
  python rename_dataset.py /path/to/source /path/to/target
        '''
    )
    
    parser.add_argument(
        'source_dir',
        help='源目录路径，包含原始文件'
    )
    
    parser.add_argument(
        'target_dir',
        help='目标目录路径，用于存放重命名后的文件'
    )
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.source_dir):
        print(f"错误: 源目录 '{args.source_dir}' 不存在")
        return 1
    
    if not os.path.isdir(args.source_dir):
        print(f"错误: 源路径 '{args.source_dir}' 不是目录")
        return 1
    
    # 执行转换
    success = run_conversion_single_dir(args.source_dir, args.target_dir)
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())