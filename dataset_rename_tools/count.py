import os
from pathlib import Path

def check_your_specific_directories():
    """
    专门检查你的目录结构
    """
    base_path = r"D:\Datasets\real-data\processed\Darker"
    
    print("检查目录结构...")
    print("=" * 50)
    
    total_png = 0
    total_depth_tiff = 0
    total_depth_raw_tiff = 0
    total_txt = 0
    
    all_good = True
    
    # 检查目录1到10
    for i in range(1, 11):
        directory_path = os.path.join(base_path, str(i))
        print(f"目录 {i}: ", end="")
        
        if not os.path.exists(directory_path):
            print("❌ 目录不存在")
            all_good = False
            continue
        
        # 统计文件数量
        png_count = len(list(Path(directory_path).glob("*.png")))
        depth_tiff_count = len(list(Path(directory_path).glob("*_depth.tiff")))
        depth_raw_tiff_count = len(list(Path(directory_path).glob("*_depth_raw.tiff")))
        txt_count = len(list(Path(directory_path).glob("*.txt")))
        
        # 显示单个目录结果
        dir_ok = (png_count == 100 and depth_tiff_count == 100 and 
                 depth_raw_tiff_count == 100 and txt_count == 100)
        status = "✅" if dir_ok else "❌"
        print(f"{status} (PNG:{png_count}, Depth:{depth_tiff_count}, Raw:{depth_raw_tiff_count}, TXT:{txt_count})")
        
        # 累加总数
        total_png += png_count
        total_depth_tiff += depth_tiff_count
        total_depth_raw_tiff += depth_raw_tiff_count
        total_txt += txt_count
        
        if not dir_ok:
            all_good = False
    
    # 显示总计
    print("\n" + "=" * 50)
    print("总计 (期望: 每项1000个):")
    print(f"PNG文件: {total_png}/1000 {'✅' if total_png == 1000 else '❌'}")
    print(f"_depth.tiff文件: {total_depth_tiff}/1000 {'✅' if total_depth_tiff == 1000 else '❌'}")
    print(f"_depth_raw.tiff文件: {total_depth_raw_tiff}/1000 {'✅' if total_depth_raw_tiff == 1000 else '❌'}")
    print(f"TXT文件: {total_txt}/1000 {'✅' if total_txt == 1000 else '❌'}")
    
    if all_good:
        print("\n🎉 完美！所有目录都符合要求！")
    else:
        print("\n❌ 需要检查文件数量")
    
    return all_good

# 运行检查
if __name__ == "__main__":
    check_your_specific_directories()