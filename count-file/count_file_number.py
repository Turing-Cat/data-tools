import os

def count_files_by_suffix(root_dir, suffix=".png", show_files=False):
    """
    递归统计某个目录下所有以指定后缀结尾的文件数量。
    
    参数:
        root_dir (str): 根目录路径
        suffix (str): 文件后缀（例如 ".png"）
        show_files (bool): 是否打印出文件路径列表

    返回:
        total_count (int): 匹配的文件总数
    """
    matched_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(suffix.lower()):
                matched_files.append(os.path.join(dirpath, file))

    if show_files:
        for f in matched_files:
            print(f)

    print(f"\n📊 总共有 {len(matched_files)} 个以 '{suffix}' 结尾的文件。")
    return len(matched_files)

# 示例用法（替换成你自己的路径）
directory = r'D:\Datasets\DATASET\mutiple-objects\the-dataset'  # 请修改为你的实际目录
count_files_by_suffix(directory, suffix=".png", show_files=False)
