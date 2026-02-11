import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def parse_renamed_filename(filename):
    """
    解析重命名后的文件名，检查是否符合格式
    格式: {timestamp}c{serial}_{type}.{ext}
    例如: 20241201T123456789c5678_rgb.png
    """
    pattern = re.compile(r'^(\d{8}T\d{9})c(\d{4})_([\w-]+)\.(png|tiff|txt|json)$')
    match = pattern.match(filename)
    if match:
        return {
            "timestamp": match.group(1),
            "serial": match.group(2),
            "type": match.group(3),
            "ext": match.group(4)
        }
    return None


def check_directory(base_path):
    """检查目录中所有文件的命名格式，返回是否合格"""
    if not os.path.exists(base_path):
        print(f"❌ {base_path} (目录不存在)")
        return False

    files = [f for f in Path(base_path).iterdir() if f.is_file()]
    if not files:
        print(f"❌ {base_path} (目录为空)")
        return False

    invalid_files = []
    file_groups = defaultdict(set)  # key -> {(type, ext), ...}

    for file_path in files:
        parsed = parse_renamed_filename(file_path.name)
        if parsed:
            key = f"{parsed['timestamp']}c{parsed['serial']}"
            file_groups[key].add((parsed["type"], parsed["ext"]))
        else:
            invalid_files.append(file_path.name)

    required = {("rgb", "png"), ("depth", "tiff"), ("grasps", "txt"), ("rgb", "json")}
    allowed = required | {("depth_raw", "tiff")}
    incomplete_groups = 0

    for _, entries in file_groups.items():
        has_required = required.issubset(entries)
        has_only_allowed = entries.issubset(allowed)
        if not (has_required and has_only_allowed):
            incomplete_groups += 1

    is_valid = len(invalid_files) == 0 and incomplete_groups == 0
    print(f"{'✅' if is_valid else '❌'} {base_path}")
    return is_valid


def check_multiple_directories(base_path):
    """检查基础目录下所有子目录"""
    if not os.path.exists(base_path):
        print(f"❌ {base_path} (目录不存在)")
        return

    subdirs = [d for d in Path(base_path).iterdir() if d.is_dir()]
    if not subdirs:
        check_directory(base_path)
        return

    subdirs.sort(key=lambda x: x.name)
    for subdir in subdirs:
        check_directory(str(subdir))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="检查文件名是否符合 rename_dataset.py 的命名格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
命名格式说明:
  格式: {timestamp}c{serial}_{type}.{ext}
  例如: 20241201T123456789c5678_rgb.png

  - timestamp: 8位日期 + T + 9位时间
  - serial: 4位序列号（前缀c）
  - type: 文件类型（rgb, depth, grasps等）
    - ext: 扩展名（png, tiff, txt, json）

使用示例:
  python check_rename_format.py "D:\\Datasets\\processed"
  python check_rename_format.py /path/to/dataset --single
        '''
    )

    parser.add_argument(
        'path',
        nargs='?',
        default=r"D:\Datasets\rechecked",
        help='要检查的目录路径'
    )

    parser.add_argument(
        '--single', '-s',
        action='store_true',
        help='只检查指定目录，不检查子目录'
    )

    args = parser.parse_args()

    if args.single:
        check_directory(args.path)
    else:
        check_multiple_directories(args.path)


if __name__ == "__main__":
    main()
