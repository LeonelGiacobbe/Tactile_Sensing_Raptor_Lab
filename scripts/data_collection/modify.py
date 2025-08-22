#!/usr/bin/env python3
"""
rename_gp_from_images.py

Recursively scan target directories and rename subfolders so that gpown/gpother
values match the maximum found in the images inside each folder.

Expected folder name pattern:
tr_{num1}_dp_{num2}_{material}_y_{num3}_z_{num4}_gpown_{num5}_gpother_{num6}

Expected image filename patterns (inside each folder):
  1_gp_{num}_frameXXX.jpg  -> agent 1 ("own")
  2_gp_{num}_frameXXX.jpg  -> agent 2 ("other")
"""

import os
import re
import argparse
from decimal import Decimal, ROUND_HALF_UP

# Regex to match folder names
FOLDER_PATTERN = re.compile(
    r"^tr_(?P<num1>[^_]+)_dp_(?P<num2>[^_]+)_(?P<material>.+?)_y_(?P<y>-?\d+(?:\.\d+)?)_z_(?P<z>-?\d+(?:\.\d+)?)_gpown_(?P<gpown>-?\d+(?:\.\d+)?)_gpother_(?P<gpother>-?\d+(?:\.\d+)?)$"
)

# Regex to match image files
IMAGE_PATTERN = re.compile(r"^(?P<agent>[12])_gp_(?P<val>-?\d+(?:\.\d+)?)_frame\d+\.jpg$")

DEFAULT_TARGETS = ["wood_block", "gel", "hard_rubber", "soft_foam"]

def get_max_values(folder_path):
    """
    Scan image files inside folder_path and return the max gpown, gpother.
    Returns (gpown, gpother) or (None, None) if no valid images.
    """
    max_own = None
    max_other = None

    for fname in os.listdir(folder_path):
        m = IMAGE_PATTERN.match(fname)
        if not m:
            continue
        agent = m.group("agent")
        val = Decimal(m.group("val"))
        if agent == "1":
            if max_own is None or val > max_own:
                max_own = val
        elif agent == "2":
            if max_other is None or val > max_other:
                max_other = val

    return max_own, max_other

def format_decimal(val: Decimal, original_str: str) -> str:
    """
    Format Decimal 'val' with the same number of decimal places as 'original_str'.
    """
    if "." in original_str:
        decimals = len(original_str.split(".", 1)[1])
    else:
        decimals = 3
    quant = Decimal('1').scaleb(-decimals)
    val_q = val.quantize(quant, rounding=ROUND_HALF_UP)
    return f"{val_q:.{decimals}f}"

def scan_and_rename(root_dir: str, targets, do_apply: bool):
    total_planned = 0
    total_applied = 0
    total_skipped = 0

    for t in targets:
        top = os.path.join(root_dir, t)
        if not os.path.isdir(top):
            print(f"[skip] target not found or not a dir: {top}")
            continue

        for root, dirs, files in os.walk(top, topdown=False):
            for d in dirs:
                m = FOLDER_PATTERN.match(d)
                if not m:
                    continue
                gd = m.groupdict()
                folder_path = os.path.join(root, d)

                max_own, max_other = get_max_values(folder_path)
                if max_own is None or max_other is None:
                    print(f"[WARNING] no valid images found in {folder_path}, skipping")
                    continue

                new_gpown = format_decimal(max_own, gd["gpown"])
                new_gpother = format_decimal(max_other - Decimal(0.5), gd["gpother"])

                new_name = (
                    f"tr_{gd['num1']}_dp_{gd['num2']}_{gd['material']}_y_{gd['y']}_z_{gd['z']}"
                    f"_gpown_{new_gpown}_gpother_{new_gpother}"
                )
                old_path = os.path.join(root, d)
                new_path = os.path.join(root, new_name)

                if old_path == new_path:
                    continue

                total_planned += 1
                if os.path.exists(new_path):
                    print(f"[WARNING] target already exists; skipping:\n  FROM: {old_path}\n  TO:   {new_path}")
                    total_skipped += 1
                    continue

                if do_apply:
                    try:
                        os.rename(old_path, new_path)
                        total_applied += 1
                        print(f"[RENAMED] {old_path} -> {new_path}")
                    except Exception as e:
                        print(f"[ERROR] failed to rename {old_path} -> {new_path}: {e}")
                else:
                    print(f"[DRY RUN] would rename:\n  {old_path}\n  -> {new_path}")

    print(f"\nSummary: planned={total_planned}, applied={total_applied}, skipped={total_skipped}")

def main():
    p = argparse.ArgumentParser(description="Rename trial folders based on maximum gp values in contained images.")
    p.add_argument("--root", default=".", help="Parent directory containing the material folders (default: current dir)")
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                   help=f"Top-level material folders to operate on (default: {', '.join(DEFAULT_TARGETS)})")
    p.add_argument("--apply", action="store_true", help="Actually perform renames. Without this, just prints planned changes.")
    args = p.parse_args()

    scan_and_rename(args.root, args.targets, args.apply)

if __name__ == "__main__":
    main()
