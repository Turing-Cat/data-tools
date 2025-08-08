#!/usr/bin/env python3
"""
dota_to_cornell.py

把 DOTA 标注（x1 y1 x2 y2 x3 y3 x4 y4 cls diff）转换为
Cornell Grasping Dataset 标注（每行一个点，4 行为一组）。

示例：
  # 转换单个 DOTA 文件
  python dota_to_cornell.py --input grasp_dota.txt --output grasp.txt

  # 批量转换整个文件夹
  python dota_to_cornell.py --src_dir dota_labels --dst_dir cornell_labels
"""

import argparse
import glob
import os
from typing import List, Tuple


def read_dota_file(path: str) -> List[List[Tuple[float, float]]]:
    """读取 DOTA 标注文件，返回四点列表"""
    quads: List[List[Tuple[float, float]]] = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                raise ValueError(f"{path}:{ln} - 坐标少于 8 个")
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                raise ValueError(f"{path}:{ln} - 非法数字")
            quad = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            quads.append(quad)
    return quads


def write_cornell(quads: List[List[Tuple[float, float]]], out_path: str) -> None:
    """把四边形列表写成 Cornell 格式"""
    with open(out_path, "w") as f:
        for quad in quads:
            for x, y in quad:
                f.write(f"{x:.3f} {y:.3f}\n")


def convert_file(in_path: str, out_path: str) -> None:
    quads = read_dota_file(in_path)
    write_cornell(quads, out_path)


def batch_convert(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for in_path in glob.glob(os.path.join(src_dir, "*.txt")):
        out_path = os.path.join(dst_dir, os.path.basename(in_path))
        convert_file(in_path, out_path)
    print(f"已将 {src_dir} 中的标注全部保存到 {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DOTA → Cornell 标注转换器")
    parser.add_argument("--input", help="单个 DOTA 标注文件")
    parser.add_argument("--output", help="输出 Cornell 标注文件")
    parser.add_argument("--src_dir", help="批量转换：输入文件夹")
    parser.add_argument("--dst_dir", help="批量转换：输出文件夹")
    args = parser.parse_args()

    if args.input:
        out_path = args.output or os.path.splitext(args.input)[0] + "_cornell.txt"
        convert_file(args.input, out_path)
        print(f"转换完成：{out_path}")
    elif args.src_dir and args.dst_dir:
        batch_convert(args.src_dir, args.dst_dir)
    else:
        parser.error("必须指定 --input，或同时指定 --src_dir 与 --dst_dir")


if __name__ == "__main__":
    main()
