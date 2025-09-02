"""
visualize_real_data.py

可视化真实数据集目录及其子目录下的RGB图像及抓取标注

示例：
  # 可视化真实数据集目录及其子目录中的图像和标注
  python visualize_real_data.py H:/Datasets/real-data/Darker

使用说明：
  - 使用左右箭头键或上下箭头键切换图片
  - 按 's' 键保存当前可视化结果到图像文件所在目录的 visualized_grasps 子目录
  - 按 'q' 键或关闭窗口退出程序
  - 按 'Home' 键跳转到第一张图片
  - 按 'End' 键跳转到最后一张图片
  - 按 'd' 键切换到下一个子目录
  - 按 'a' 键切换到上一个子目录
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from matplotlib import cm

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RealDataVisualizer:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        # 递归搜索所有子目录中的RGB图像文件
        self.image_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # 查找PNG或JPG文件（不再限制必须在rgb或color目录中）
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        self.image_files = sorted(self.image_files)
        self.current_index = 0
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 设置窗口最大化
        manager = plt.get_current_fig_manager()
        try:
            # 尝试使用不同的后端方法最大化窗口
            if hasattr(manager, 'window'):
                # TkAgg 后端
                manager.window.wm_state('zoomed')
        except Exception:
            try:
                # Qt 后端
                manager.window.showMaximized()
            except Exception:
                try:
                    # 其他后端
                    manager.frame.Maximize(True)
                except Exception:
                    pass  # 如果所有方法都失败，忽略错误
        
        if not self.image_files:
            print(f"目录中未找到 RGB 图像：{directory_path}")
            plt.close()
            return
        
        # 获取所有子目录
        self.subdirectories = sorted(list(set(os.path.dirname(os.path.dirname(f)) for f in self.image_files)))
        
        self.load_and_show_image()
        plt.tight_layout()
        plt.show()

    def load_grasp_rectangles(self, file_path):
        if not os.path.exists(file_path):
            return [], False
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            lines = [list(map(float, line.strip().split())) for line in lines]
            grasp_rects = []
            format_error = False

            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    rect = np.array(lines[i:i+4])
                    # 检查矩形坐标格式是否正确（每个点应该有x,y两个坐标）
                    if rect.shape == (4, 2):
                        grasp_rects.append(rect)
                    else:
                        format_error = True
                        print(f"警告: 标注文件 {file_path} 格式不正确，跳过该抓取框")
            return grasp_rects, format_error
        except Exception as e:
            print(f"警告: 无法正确解析标注文件 {file_path}: {e}")
            return [], True

    def load_and_show_image(self):
        self.ax.clear()
        
        if self.current_index >= len(self.image_files):
            self.current_index = 0
        elif self.current_index < 0:
            self.current_index = len(self.image_files) - 1

        image_path = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # 对于_rgb.png文件，对应的标注文件应该是_grasps.txt
        if base_name.endswith('_rgb'):
            label_base_name = base_name[:-4] + '_grasps'
        else:
            label_base_name = base_name
        # 标签文件应该在图像文件相同的目录中
        label_path = os.path.join(os.path.dirname(image_path), label_base_name + ".txt")
        
        img = Image.open(image_path)
        self.ax.imshow(img)
        
        grasp_rects, format_error = self.load_grasp_rectangles(label_path)
        colors = cm.rainbow(np.linspace(0, 1, len(grasp_rects)))
        
        for idx, rect in enumerate(grasp_rects):
            polygon = patches.Polygon(rect, closed=True, 
                                    edgecolor=colors[idx], 
                                    linewidth=2, 
                                    fill=False,
                                    label=f'Grasp {idx+1}')
            self.ax.add_patch(polygon)
        
        if grasp_rects:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 显示标题和抓取框数量或错误信息
        if format_error:
            title = f"{os.path.relpath(image_path, self.directory_path)} - 标注文件格式不正确，无法显示抓取框 (Image {self.current_index + 1}/{len(self.image_files)})"
            # 在图像上显示红色警告
            self.ax.text(0.5, 0.95, "标注文件格式不正确，无法显示抓取框", 
                        transform=self.ax.transAxes, 
                        horizontalalignment='center', 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                        fontsize=12, fontweight='bold', color='white')
        else:
            title = f"{os.path.relpath(image_path, self.directory_path)} - {len(grasp_rects)} grasps (Image {self.current_index + 1}/{len(self.image_files)})"
        
        self.ax.set_title(title)
        
        # 添加快捷键说明到右侧 grasp 颜色框的下方
        help_text = ("快捷键:\n"
                    "→/↓ : 下一张\n"
                    "←/↑ : 上一张\n"
                    "Home : 第一张\n"
                    "End : 最后一张\n"
                    "d : 下一个子目录\n"
                    "a : 上一个子目录\n"
                    "s : 保存图像\n"
                    "q : 退出")
        
        # 将快捷键说明放在右侧 grasp 颜色框的下方
        self.ax.text(1.05, 0.5, help_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
        
        self.ax.axis('off')
        self.fig.canvas.draw()

    def save_current_image(self):
        if not self.image_files:
            return
        image_path = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # 在图像文件相同的目录中创建 visualized_grasps 子目录
        image_dir = os.path.dirname(image_path)
        save_dir = os.path.join(image_dir, "visualized_grasps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base_name}_visualized.png")
        
        self.fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"已保存可视化结果到: {save_path}")

    def navigate_to_next_subdirectory(self):
        if len(self.subdirectories) <= 1:
            return
            
        current_image_path = self.image_files[self.current_index]
        current_dir = os.path.dirname(current_image_path)  # rgb目录
        parent_dir = os.path.dirname(current_dir)  # 子目录
        current_dir_index = self.subdirectories.index(parent_dir)
        
        # 计算下一个子目录索引
        next_dir_index = (current_dir_index + 1) % len(self.subdirectories)
        next_dir = self.subdirectories[next_dir_index]
        
        # 找到下一个子目录中的第一张图片
        for i, image_path in enumerate(self.image_files):
            # 获取图像的父目录的父目录（子目录）
            img_parent_dir = os.path.dirname(os.path.dirname(image_path))
            if img_parent_dir == next_dir:
                self.current_index = i
                break
                
        self.load_and_show_image()
        print(f"已切换到子目录: {os.path.relpath(next_dir, self.directory_path)}")

    def navigate_to_previous_subdirectory(self):
        if len(self.subdirectories) <= 1:
            return
            
        current_image_path = self.image_files[self.current_index]
        current_dir = os.path.dirname(current_image_path)  # rgb目录
        parent_dir = os.path.dirname(current_dir)  # 子目录
        current_dir_index = self.subdirectories.index(parent_dir)
        
        # 计算上一个子目录索引
        prev_dir_index = (current_dir_index - 1) % len(self.subdirectories)
        prev_dir = self.subdirectories[prev_dir_index]
        
        # 找到上一个子目录中的第一张图片
        for i, image_path in enumerate(self.image_files):
            # 获取图像的父目录的父目录（子目录）
            img_parent_dir = os.path.dirname(os.path.dirname(image_path))
            if img_parent_dir == prev_dir:
                self.current_index = i
                break
                
        self.load_and_show_image()
        print(f"已切换到子目录: {os.path.relpath(prev_dir, self.directory_path)}")

    def on_key_press(self, event):
        if event.key == 'right' or event.key == 'down':
            self.current_index += 1
            self.load_and_show_image()
        elif event.key == 'left' or event.key == 'up':
            self.current_index -= 1
            self.load_and_show_image()
        elif event.key == 'q':
            plt.close()
            print("退出可视化")
        elif event.key == 'home':
            self.current_index = 0
            self.load_and_show_image()
        elif event.key == 'end':
            self.current_index = len(self.image_files) - 1
            self.load_and_show_image()
        elif event.key == 's':
            self.save_current_image()
        # 添加子目录导航功能
        elif event.key == 'd':
            self.navigate_to_next_subdirectory()
        elif event.key == 'a':
            self.navigate_to_previous_subdirectory()

def visualize_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"目录不存在：{directory_path}")
        return
    
    RealDataVisualizer(directory_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="可视化真实数据集目录下的RGB图像及抓取标注")
    parser.add_argument("directory", type=str, help="包含RGB图像和TXT标注文件的目录路径")
    args = parser.parse_args()
    visualize_directory(args.directory)