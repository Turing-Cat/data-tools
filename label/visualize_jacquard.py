"""
visualize_jacquard.py

Visualize PNG images and grasp annotations from the Jacquard dataset

Example:
  # Visualize images and annotations in a directory
  python visualize_jacquard.py /path/to/directory

Usage:
  - Use left/right arrow keys or up/down arrow keys to navigate images
  - Press 's' key to save current visualization to visualized_grasps subdirectory
  - Press 'q' key or close window to exit program
  - Press 'Home' key to jump to first image
  - Press 'End' key to jump to last image
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from matplotlib import cm

class JacquardVisualizer:
    def __init__(self, directory_path):
        self.directory_path = directory_path

        # Find image files (RGB.png) and corresponding annotation files (_grasps.txt)
        self.image_files = sorted(glob.glob(os.path.join(directory_path, "*_RGB.png")))
        self.annotation_files = []

        # Match annotation files to image files
        for image_file in self.image_files:
            base_name = image_file.replace('_RGB.png', '')
            annotation_file = base_name + '_grasps.txt'
            if os.path.exists(annotation_file):
                self.annotation_files.append(annotation_file)
            else:
                self.annotation_files.append(None)  # No annotation file

        self.current_index = 0
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        if not self.image_files:
            print(f"No PNG images found in directory: {directory_path}")
            plt.close()
            return

        # Create save directory
        self.save_dir = os.path.join(directory_path, "visualized_grasps")


        self.load_and_show_image()
        plt.tight_layout()
        plt.show()

    def load_grasp_rectangles(self, file_path):
        """Load grasp rectangles from Jacquard annotation file"""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Process lines, removing comments and empty lines
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        grasp_rects = []

        # Parse Jacquard format: x; y; angle; width; depth
        for line in lines:
            try:
                values = list(map(float, line.split(';')))
                if len(values) == 5:
                    x, y, angle, width, depth = values

                    # Convert to rectangle coordinates
                    # Jacquard specifies a rectangle with center (x,y), angle, width and depth
                    rect = self.create_rectangle_coords(x, y, angle, width, depth)
                    grasp_rects.append(rect)
            except ValueError:
                # Skip lines that can't be parsed
                continue

        return grasp_rects

    def create_rectangle_coords(self, x, y, angle, width, depth):
        """Create rectangle coordinates from Jacquard format parameters"""
        # Convert angle from degrees to radians
        angle_rad = np.deg2rad(angle)

        # Calculate rectangle corners relative to center
        # Width is the horizontal dimension, depth is the vertical dimension
        corners = np.array([
            [-width/2, -depth/2],  # Bottom left
            [ width/2, -depth/2],  # Bottom right
            [ width/2,  depth/2],  # Top right
            [-width/2,  depth/2]   # Top left
        ])

        # Rotate corners by angle
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle]
        ])

        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate to actual position
        coords = rotated_corners + np.array([x, y])

        return coords

    def load_and_show_image(self):
        self.ax.clear()

        if self.current_index >= len(self.image_files):
            self.current_index = 0
        elif self.current_index < 0:
            self.current_index = len(self.image_files) - 1

        image_path = self.image_files[self.current_index]
        annotation_path = self.annotation_files[self.current_index]

        img = Image.open(image_path)
        self.ax.imshow(img)

        # Load grasp annotations
        grasp_rects = []
        if annotation_path:
            grasp_rects = self.load_grasp_rectangles(annotation_path)

        # Draw grasp rectangles (blue color)
        for rect in grasp_rects:
            polygon = patches.Polygon(rect, closed=True,
                                    edgecolor='blue',
                                    linewidth=2,
                                    fill=False)
            self.ax.add_patch(polygon)

        # Add legend
        if grasp_rects:
            legend_elements = [patches.Patch(facecolor='none', edgecolor='blue',
                                           label=f'Grasps: {len(grasp_rects)}')]
            self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        base_name = os.path.basename(image_path).replace('_RGB.png', '')
        title = f"{base_name}_RGB.png - {len(grasp_rects)} grasps (Image {self.current_index + 1}/{len(self.image_files)})"
        self.ax.set_title(title)
        self.ax.axis('off')
        self.fig.canvas.draw()

    def save_current_image(self):
        if not self.image_files:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        image_path = self.image_files[self.current_index]
        base_name = os.path.basename(image_path).replace('_RGB.png', '')
        save_path = os.path.join(self.save_dir, f"{base_name}_RGB_visualized.png")

        self.fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to: {save_path}")

    def on_key_press(self, event):
        if event.key == 'right' or event.key == 'down':
            self.current_index += 1
            self.load_and_show_image()
        elif event.key == 'left' or event.key == 'up':
            self.current_index -= 1
            self.load_and_show_image()
        elif event.key == 'q':
            plt.close()
            print("Exiting visualization")
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
        print(f"Directory does not exist: {directory_path}")
        return

    JacquardVisualizer(directory_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize PNG images and grasp annotations from the Jacquard dataset")
    parser.add_argument("directory", type=str, help="Path to directory containing PNG images and TXT annotation files")
    args = parser.parse_args()
    visualize_directory(args.directory)
