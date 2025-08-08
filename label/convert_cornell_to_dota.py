#!/usr/bin/env python3
"""
convert_cornell_to_dota.py

示例：
  # 转换单个文件
  python convert_cornell_to_dota.py --input grasp1.txt --output grasp1_dota.txt

  # 批量转换文件夹
  python convert_cornell_to_dota.py --src_dir cornell_labels --dst_dir dota_labels

  # 自定义类别与难度
  python convert_cornell_to_dota.py --input a.txt --cls myobj --diff 1
"""

import argparse
import glob
import os
from typing import List, Tuple


def read_cornell_file(path: str) -> List[List[Tuple[float, float]]]:
    """读取 Cornell 标注文件，返回按四点分组后的列表。"""
    points: List[Tuple[float, float]] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            x_str, y_str, *rest = line.strip().split()
            if rest:
                raise ValueError(f"{path} 中出现非 2 列格式：{line}")
            points.append((float(x_str), float(y_str)))

    if len(points) % 4 != 0:
        raise ValueError(f"{path} 的点数不是 4 的倍数，无法组成矩形")

    return [points[i : i + 4] for i in range(0, len(points), 4)]


def write_dota(quads: List[List[Tuple[float, float]]],
               out_path: str,
               cls: str,
               diff: str) -> None:
    """把四边形列表写成 DOTA 标注文件。"""
    with open(out_path, "w") as f:
        for quad in quads:
            coord_str = " ".join(f"{x:.3f} {y:.3f}" for x, y in quad)
            f.write(f"{coord_str} {cls} {diff}\n")


def convert_file(in_path: str,
                 out_path: str,
                 cls: str,
                 diff: str) -> None:
    quads = read_cornell_file(in_path)
    write_dota(quads, out_path, cls, diff)


def batch_convert(src_dir: str,
                  dst_dir: str,
                  cls: str,
                  diff: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for in_path in glob.glob(os.path.join(src_dir, "*.txt")):
        out_path = os.path.join(dst_dir, os.path.basename(in_path))
        convert_file(in_path, out_path, cls, diff)
    print(f"已将 {src_dir} 中的标注全部保存到 {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Cornell Grasping Dataset 标注转换为 DOTA 格式")
    parser.add_argument("--input", help="单个 Cornell 标注文件")
    parser.add_argument("--output", help="输出 DOTA 标注文件")
    parser.add_argument("--src_dir", help="批量转换：输入文件夹")
    parser.add_argument("--dst_dir", help="批量转换：输出文件夹")
    parser.add_argument("--cls", default="grasp",
                        help="DOTA 类别名，默认 grasp")
    parser.add_argument("--diff", default="0",
                        help="DOTA 难度，默认 0")
    args = parser.parse_args()

    if args.input:
        # 单文件模式
        out_path = args.output or os.path.splitext(args.input)[0] + "_dota.txt"
        convert_file(args.input, out_path, args.cls, args.diff)
        print(f"转换完成：{out_path}")
    elif args.src_dir and args.dst_dir:
        # 批量模式
        batch_convert(args.src_dir, args.dst_dir, args.cls, args.diff)
    else:
        parser.error("必须指定 --input，或同时指定 --src_dir 与 --dst_dir")


if __name__ == "__main__":
    main()
