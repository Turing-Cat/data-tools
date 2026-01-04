#!/usr/bin/env python3
"""
convert_cornell_to_dota.py

将 Cornell Grasping Dataset 标注转换为 DOTA 格式

示例：
  # 转换单个文件
  python convert_cornell_to_dota.py --input grasp1.txt --output grasp1_dota.txt
  python convert_cornell_to_dota.py -i grasp1.txt -o grasp1_dota.txt

  # 批量转换文件夹
  python convert_cornell_to_dota.py --src_dir cornell_labels --dst_dir dota_labels
  python convert_cornell_to_dota.py -s cornell_labels -d dota_labels

  # 自定义类别与难度
  python convert_cornell_to_dota.py --input a.txt --cls myobj --diff 1
  python convert_cornell_to_dota.py -i a.txt -c myobj --diff 1
"""

import argparse
import glob
import os
from typing import List, Tuple
import shutil

def looks_like_dota_file(path: str) -> bool:
    """
    判断一个 txt 是否像 DOTA: 第一条非空行 token >= 8 且前 8 个可转 float，
    同时不是 Cornell 的 2 列格式。
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()

            # Cornell 通常是 2 列
            if len(parts) == 2:
                return False

            if len(parts) >= 8:
                try:
                    _ = list(map(float, parts[:8]))
                    return True
                except ValueError:
                    return False

            return False
    return False

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


def batch_convert(src_dir: str, dst_dir: str, cls: str, diff: int) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    same_dir = os.path.abspath(src_dir) == os.path.abspath(dst_dir)

    for in_path in glob.glob(os.path.join(src_dir, "*.txt")):
        out_path = os.path.join(dst_dir, os.path.basename(in_path))

        # ✅ 已经是 DOTA：不做转换
        if looks_like_dota_file(in_path):
            if same_dir:
                print(f"[SKIP] 已是 DOTA，跳过：{in_path}")
            else:
                # 目标目录不同：直接拷贝过去（等价于“跳过转换”）
                shutil.copy2(in_path, out_path)
                print(f"[COPY] 已是 DOTA，直接拷贝：{in_path} -> {out_path}")
            continue

        convert_file(in_path, out_path, cls, diff)

    print(f"已处理完成：{src_dir} -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Cornell Grasping Dataset 标注转换为 DOTA 格式")
    parser.add_argument("--input", "-i", help="单个 Cornell 标注文件")
    parser.add_argument("--output", "-o", help="输出 DOTA 标注文件")
    parser.add_argument("--src_dir", "-s", help="批量转换：输入文件夹")
    parser.add_argument("--dst_dir", "-d", help="批量转换：输出文件夹")
    parser.add_argument("--cls", "-c", default="0",
                        help="DOTA 类别名，默认 0")
    parser.add_argument("--diff", "-f", default="0",
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
