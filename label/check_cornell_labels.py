#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_cornell_labels.py

Recursively validate whether *.txt files follow Cornell Grasping label format:
- Each line represents one vertex: x y
- Every 4 lines represent one rectangle (quadrilateral)
- Total valid lines must be a multiple of 4

Usage:
  python check_cornell_labels.py --root "D:\\Datasets\\real-data\\Darker-reannotation"
  python check_cornell_labels.py -r . --csv out.csv

Exit code:
  0: all good
  1: has invalid files/lines
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Issue:
    file: str
    line: int
    reason: str
    text: str


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def iter_txt_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.txt")


def check_file(
    path: Path,
    allow_comment: bool = True,
    comment_prefix: str = "#",
) -> List[Issue]:
    issues: List[Issue] = []
    try:
        # utf-8-sig: 兼容带 BOM 的 UTF-8
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        issues.append(Issue(str(path), 0, f"ReadError: {e}", ""))
        return issues

    valid_vertex_count = 0

    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if allow_comment and line.startswith(comment_prefix):
            continue

        parts = line.split()
        
        # Cornell 格式检查：每行必须严格是 2 列 (x, y)
        if len(parts) != 2:
            issues.append(
                Issue(str(path), idx, f"ColumnCount != 2 (got {len(parts)})", raw)
            )
            # 即使列数不对，也可能影响后续的模4检查，这里先不计数或根据需求处理
            # 为了严谨，格式错误的行不算作有效顶点
            continue

        # 检查是否为浮点数
        if not is_float(parts[0]) or not is_float(parts[1]):
            issues.append(
                Issue(str(path), idx, f"Coord not float: '{parts[0]}, {parts[1]}'", raw)
            )
            continue

        # 这是一个有效的顶点行
        valid_vertex_count += 1

    # 文件级检查：总行数必须是 4 的倍数（因为 4 个点组成一个框）
    if valid_vertex_count % 4 != 0:
        issues.append(
            Issue(
                str(path), 
                0, # 0 表示文件级别的错误
                f"Total vertices {valid_vertex_count} is not a multiple of 4 (Incomplete Rectangle)", 
                ""
            )
        )

    return issues


def write_csv(issues: List[Issue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "reason", "text"])
        for it in issues:
            w.writerow([it.file, it.line, it.reason, it.text])


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether txt files are in Cornell Grasping format (4 lines per rect).")
    ap.add_argument("--root", "-r", default=".", help="Root directory to scan recursively.")
    ap.add_argument("--csv", default=None, help="Path to save CSV report (default: <root>/cornell_check_report.csv)")
    ap.add_argument("--no-comment", action="store_true", help="Do NOT treat lines starting with # as comments.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] root not found: {root}")
        return 1

    report_path = Path(args.csv).resolve() if args.csv else (root / "cornell_check_report.csv")

    all_issues: List[Issue] = []
    files = list(iter_txt_files(root))

    print(f"Scanning {len(files)} files in: {root} ...")

    for p in files:
        all_issues.extend(
            check_file(
                p,
                allow_comment=not args.no_comment,
                comment_prefix="#",
            )
        )

    bad_files = {it.file for it in all_issues if it.line != 0} # 统计具体的行错误
    # 包含文件级错误（如行数不是4的倍数）
    all_bad_files = {it.file for it in all_issues}
    read_errors = [it for it in all_issues if "ReadError" in it.reason]

    print(f"-" * 40)
    print(f"Root: {root}")
    print(f"Total txt files: {len(files)}")
    print(f"Files with issues: {len(all_bad_files)}")
    print(f"Read errors: {len(read_errors)}")
    print(f"Total issues found: {len(all_issues)}")

    if all_issues:
        write_csv(all_issues, report_path)
        print(f"CSV report saved to: {report_path}")

        print("\nFirst 10 issues:")
        for it in all_issues[:10]:
            # 如果是文件级错误(line=0)，不显示text
            if it.line == 0:
                print(f"- {it.file} | {it.reason}")
            else:
                print(f"- {it.file}:{it.line} | {it.reason} | {it.text}")

        return 1

    print("-" * 40)
    print("All txt files look like valid Cornell format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())