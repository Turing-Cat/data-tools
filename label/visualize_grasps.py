"""
visualize_grasps.py

可视化指定目录下的PNG图片及抓取标注

示例：
  # 可视化目录中的图片和标注
  python visualize_grasps.py /path/to/directory

使用说明：
  - 使用左右箭头键或上下箭头键切换图片
  - 按 's' 键保存当前可视化结果到目录中的 visualized_grasps 子目录
  - 按 'q' 键或关闭窗口退出程序
  - 按 'Home' 键跳转到第一张图片
  - 按 'End' 键跳转到最后一张图片
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from matplotlib import cm

class GraspVisualizer:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.image_files = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        self.current_index = 0
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        if not self.image_files:
            print(f"目录中未找到 PNG 图片：{directory_path}")
            plt.close()
            return
        
        # 创建保存目录
        self.save_dir = os.path.join(directory_path, "visualized_grasps")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.load_and_show_image()
        plt.tight_layout()
        plt.show()

    def load_grasp_rectangles(self, file_path):
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        lines = [list(map(float, line.strip().split())) for line in lines]
        grasp_rects = []

        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                rect = np.array(lines[i:i+4])
                grasp_rects.append(rect)
        return grasp_rects

    def load_and_show_image(self):
        self.ax.clear()
        
        if self.current_index >= len(self.image_files):
            self.current_index = 0
        elif self.current_index < 0:
            self.current_index = len(self.image_files) - 1

        image_path = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.directory_path, base_name + ".txt")

        img = Image.open(image_path)
        self.ax.imshow(img)
        
        grasp_rects = self.load_grasp_rectangles(label_path)
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
        
        title = f"{base_name}.png - {len(grasp_rects)} grasps (Image {self.current_index + 1}/{len(self.image_files)})"
        self.ax.set_title(title)
        self.ax.axis('off')
        self.fig.canvas.draw()

    def save_current_image(self):
        if not self.image_files:
            return
        
        image_path = self.image_files[self.current_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.save_dir, f"{base_name}_visualized.png")
        
        self.fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"已保存可视化结果到: {save_path}")

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

def visualize_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"目录不存在：{directory_path}")
        return
    
    GraspVisualizer(directory_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="可视化指定目录下的PNG图片及抓取标注")
    parser.add_argument("directory", type=str, help="包含PNG图片和TXT标注文件的目录路径")
    args = parser.parse_args()
    visualize_directory(args.directory)