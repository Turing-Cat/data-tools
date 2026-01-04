#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dota_labels.py

Recursively validate whether *.txt files follow DOTA label format:
x1 y1 x2 y2 x3 y3 x4 y4 class difficulty

Usage:
  python check_dota_labels.py --root "D:\\Datasets\\real-data\\Darker-reannotation"
  python check_dota_labels.py -r . --csv out.csv

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


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def iter_txt_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.txt")


def check_file(
    path: Path,
    require_cols: int = 10,
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

    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if allow_comment and line.startswith(comment_prefix):
            continue

        parts = line.split()
        if len(parts) != require_cols:
            issues.append(
                Issue(str(path), idx, f"ColumnCount != {require_cols} (got {len(parts)})", raw)
            )
            continue

        # first 8 floats
        for i in range(8):
            if not is_float(parts[i]):
                issues.append(
                    Issue(str(path), idx, f"Coord[{i}] not float: '{parts[i]}'", raw)
                )
                break
        else:
            # class non-empty
            if not parts[8]:
                issues.append(Issue(str(path), idx, "Class is empty", raw))
                continue

            # diff int
            if not is_int(parts[9]):
                issues.append(Issue(str(path), idx, f"Diff not int: '{parts[9]}'", raw))

    return issues


def write_csv(issues: List[Issue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "reason", "text"])
        for it in issues:
            w.writerow([it.file, it.line, it.reason, it.text])


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether txt files are in DOTA annotation format.")
    ap.add_argument("--root", "-r", default=".", help="Root directory to scan recursively.")
    ap.add_argument("--csv", default=None, help="Path to save CSV report (default: <root>/dota_check_report.csv)")
    ap.add_argument("--cols", type=int, default=10, help="Required columns per line (default: 10)")
    ap.add_argument("--no-comment", action="store_true", help="Do NOT treat lines starting with # as comments.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] root not found: {root}")
        return 1

    report_path = Path(args.csv).resolve() if args.csv else (root / "dota_check_report.csv")

    all_issues: List[Issue] = []
    files = list(iter_txt_files(root))

    for p in files:
        all_issues.extend(
            check_file(
                p,
                require_cols=args.cols,
                allow_comment=not args.no_comment,
                comment_prefix="#",
            )
        )

    bad_files = {it.file for it in all_issues if it.line != 0}
    read_errors = [it for it in all_issues if it.line == 0]

    print(f"Root: {root}")
    print(f"Total txt files: {len(files)}")
    print(f"Files with format issues: {len(bad_files)}")
    print(f"Read errors: {len(read_errors)}")
    print(f"Total issues: {len(all_issues)}")

    if all_issues:
        write_csv(all_issues, report_path)
        print(f"CSV report saved to: {report_path}")

        # show first few issues
        print("\nFirst 10 issues:")
        for it in all_issues[:10]:
            print(f"- {it.file}:{it.line} | {it.reason} | {it.text}")

        return 1

    print("All txt files look like DOTA format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
