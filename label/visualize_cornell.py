"""
visualize_cornell.py

可视化Cornell数据集中的PNG图片及抓取标注

示例：
  # 可视化目录中的图片和标注
  python visualize_cornell.py /path/to/directory

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
        
        # 先读取标注文件，再查找对应的图像文件
        # Cornell数据集的标注文件有两种：cpos.txt（正样本）和cneg.txt（负样本）
        self.label_files_cpos = sorted(glob.glob(os.path.join(directory_path, "*cpos.txt")))
        self.label_files_cneg = sorted(glob.glob(os.path.join(directory_path, "*cneg.txt")))
        
        # 获取基础文件名（去掉cpos/cneg后缀）
        base_names_cpos = [os.path.join(directory_path, os.path.basename(f).replace('cpos.txt', '')) for f in self.label_files_cpos]
        base_names_cneg = [os.path.join(directory_path, os.path.basename(f).replace('cneg.txt', '')) for f in self.label_files_cneg]
        
        # 找到同时有正负样本标注的文件
        common_base_names = set(base_names_cpos) & set(base_names_cneg)
        
        # 根据标注文件找到对应的图像文件
        self.image_files = []
        self.label_files_matched_cpos = []  # 匹配的正样本标注文件
        self.label_files_matched_cneg = []  # 匹配的负样本标注文件
        
        for base_name in sorted(common_base_names):
            # Cornell数据集的图像文件名格式为 pcd0105r.png
            image_file = base_name + "r.png"
            if os.path.exists(image_file):
                self.image_files.append(image_file)
                self.label_files_matched_cpos.append(base_name + "cpos.txt")
                self.label_files_matched_cneg.append(base_name + "cneg.txt")
        
        # 如果没有同时有正负样本标注的文件，则分别查找
        if not self.image_files:
            # 单独查找正样本标注文件
            for i, base_name in enumerate(base_names_cpos):
                image_file = base_name + "r.png"
                if os.path.exists(image_file):
                    self.image_files.append(image_file)
                    self.label_files_matched_cpos.append(self.label_files_cpos[i])
                    self.label_files_matched_cneg.append("")  # 没有负样本标注
            
            # 单独查找负样本标注文件
            for i, base_name in enumerate(base_names_cneg):
                image_file = base_name + "r.png"
                if os.path.exists(image_file) and image_file not in self.image_files:
                    self.image_files.append(image_file)
                    self.label_files_matched_cpos.append("")  # 没有正样本标注
                    self.label_files_matched_cneg.append(self.label_files_cneg[i])
        
        self.current_index = 0
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        if not self.image_files:
            print(f"目录中未找到匹配的 PNG 图片：{directory_path}")
            plt.close()
            return
        
        # 创建保存目录
        self.save_dir = os.path.join(directory_path, "visualized_grasps")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.load_and_show_image()
        plt.tight_layout()
        plt.show()

    def load_grasp_rectangles(self, file_path):
        """加载抓取矩形标注"""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 处理可能的注释行和空行
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        lines = [list(map(float, line.split())) for line in lines]
        
        grasp_rects = []
        
        # Cornell数据集的标注格式：每4行为一个抓取矩形
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
        label_path_cpos = self.label_files_matched_cpos[self.current_index]
        label_path_cneg = self.label_files_matched_cneg[self.current_index]

        img = Image.open(image_path)
        self.ax.imshow(img)
        
        # 加载正样本抓取标注
        grasp_rects_cpos = self.load_grasp_rectangles(label_path_cpos)
        # 加载负样本抓取标注
        grasp_rects_cneg = self.load_grasp_rectangles(label_path_cneg)
        
        # 绘制正样本抓取（绿色）
        for rect in grasp_rects_cpos:
            polygon = patches.Polygon(rect, closed=True, 
                                    edgecolor='green', 
                                    linewidth=2, 
                                    fill=False)
            self.ax.add_patch(polygon)
        
        # 绘制负样本抓取（红色）
        for rect in grasp_rects_cneg:
            polygon = patches.Polygon(rect, closed=True, 
                                    edgecolor='red', 
                                    linewidth=2, 
                                    fill=False)
            self.ax.add_patch(polygon)
        
        # 添加简单的图例显示抓取框颜色和数量
        if grasp_rects_cpos or grasp_rects_cneg:
            # 创建自定义图例
            legend_elements = []
            if grasp_rects_cpos:
                legend_elements.append(patches.Patch(facecolor='none', edgecolor='green', label=f'Positive: {len(grasp_rects_cpos)}'))
            if grasp_rects_cneg:
                legend_elements.append(patches.Patch(facecolor='none', edgecolor='red', label=f'Negative: {len(grasp_rects_cneg)}'))
            
            self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        base_name = os.path.splitext(os.path.basename(image_path))[0][:-1]  # 去掉最后的'r'
        title = f"{base_name}.png - {len(grasp_rects_cpos) + len(grasp_rects_cneg)} grasps (Image {self.current_index + 1}/{len(self.image_files)})"
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
    parser = argparse.ArgumentParser(description="可视化Cornell数据集中的PNG图片及抓取标注")
    parser.add_argument("directory", type=str, help="包含PNG图片和TXT标注文件的目录路径")
    args = parser.parse_args()
    visualize_directory(args.directory)