import os
import argparse
from pathlib import Path
from PIL import Image


def quick_check_file(file_path):
    """快速检查文件是否损坏"""
    try:
        # 检查文件是否存在且不为空
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False, "文件不存在或为空"

        # 根据文件类型进行检查
        file_ext = Path(file_path).suffix.lower()

        if file_ext in ['.png', '.tiff']:
            with Image.open(file_path) as img:
                img.verify()
            return True, "正常"
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # 只读取一部分来检查
            return True, "正常"
        else:
            return True, "正常"

    except Exception as e:
        return False, str(e)


def check_all_subdirectories(base_path):
    """检查指定目录下所有子目录中的文件是否损坏"""
    
    print(f"检查目录: {base_path}")
    print("=" * 60)

    # 检查基础目录是否存在
    if not os.path.exists(base_path):
        print(f"❌ 错误: 目录 '{base_path}' 不存在")
        return

    # 获取所有子目录
    subdirs = [d for d in Path(base_path).iterdir() if d.is_dir()]

    if not subdirs:
        print(f"❌ 错误: '{base_path}' 中没有找到子目录")
        return

    # 按目录名排序
    subdirs.sort(key=lambda x: x.name)

    print(f"找到 {len(subdirs)} 个子目录\n")

    total_files = 0
    corrupted_files = 0
    corrupted_details = []

    # 遍历所有子目录
    for subdir in subdirs:
        directory_path = subdir
        print(f"检查目录 {subdir.name}: ", end="")
        
        if not os.path.exists(directory_path):
            print("❌ 目录不存在")
            continue
        
        # 获取该目录中的所有文件
        files = list(directory_path.iterdir())
        files = [f for f in files if f.is_file()]
        
        dir_total = len(files)
        dir_corrupted = 0
        total_files += dir_total
        
        if dir_total == 0:
            print("目录为空")
            continue
        
        # 检查每个文件
        for file_path in files:
            is_ok, message = quick_check_file(file_path)
            if not is_ok:
                dir_corrupted += 1
                corrupted_files += 1
                corrupted_details.append((f"{subdir.name}/{file_path.name}", message))
        
        # 显示该目录状态
        dir_status = "✅" if dir_corrupted == 0 else "❌"
        print(f"{dir_status} (损坏: {dir_corrupted}/{dir_total})")
    
    # 显示总结
    print("\n" + "=" * 60)
    print("完整性检查总结:")
    print(f"总文件数: {total_files}")
    print(f"损坏文件数: {corrupted_files}")
    print(f"正常文件数: {total_files - corrupted_files}")
    
    if total_files > 0:
        success_rate = ((total_files - corrupted_files) / total_files) * 100
        print(f"完好率: {success_rate:.2f}%")
    
    if corrupted_files > 0:
        print(f"\n❌ 发现损坏文件:")
        for filename, error in corrupted_details[:10]:  # 只显示前10个
            print(f"  • {filename}: {error}")
        if len(corrupted_details) > 10:
            print(f"  ... 还有 {len(corrupted_details) - 10} 个损坏文件")
    else:
        print("🎉 所有文件都正常！")

def main():
    """主函数：解析命令行参数并执行检查"""
    parser = argparse.ArgumentParser(
        description="检查目录下所有子目录的文件完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python check.py "D:\\Datasets\\rechecked"
  python check.py /path/to/dataset
        '''
    )

    parser.add_argument(
        'base_path',
        nargs='?',
        default=r"D:\Datasets\rechecked",
        help='要检查的基础目录路径（默认: D:\\Datasets\\rechecked）'
    )

    args = parser.parse_args()

    # 执行检查
    check_all_subdirectories(args.base_path)


if __name__ == "__main__":
    main()