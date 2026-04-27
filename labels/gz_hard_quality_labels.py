"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ============================================================
# 输入路径
# ============================================================

catalog_path = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue.csv"

output_csv = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv"

output_fits = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.fits"


# ============================================================
# Hayat HQ label function
# ============================================================

def hayat_hq_label(p):

    if np.isnan(p):
        return -1, False

    if p > 0.8:
        return 1, True

    if p < 0.2:
        return 0, True

    return -1, False


# ============================================================
# 读取 catalogue
# ============================================================

print("Loading morphology catalogue...")

df = pd.read_csv(catalog_path)

print("Number of galaxies:", len(df))


# ============================================================
# 自动寻找 fraction 列
# ============================================================

fraction_cols = [c for c in df.columns if c.endswith("_fraction")]

print("Detected Galaxy Zoo fraction columns:")

for c in fraction_cols:
    print("  ", c)


# ============================================================
# 生成 soft + HQ labels
# ============================================================

for col in fraction_cols:

    soft_col = col + "_soft_label"
    hq_col = col + "_hq_label"
    hq_flag = col + "_is_hq"

    soft_values = df[col].values

    hq_labels = []
    hq_mask = []

    for p in soft_values:

        label, is_hq = hayat_hq_label(p)

        hq_labels.append(label)
        hq_mask.append(is_hq)

    df[soft_col] = soft_values
    df[hq_col] = hq_labels
    df[hq_flag] = hq_mask


print("HQ labels generated.")


# ============================================================
# 生成综合 morphology label
# ============================================================

def morphology_class(row):

    smooth = row.get("smooth-or-featured_smooth_fraction", np.nan)
    featured = row.get("smooth-or-featured_featured_fraction", np.nan)
    spiral = row.get("spiral-arms_yes_fraction", np.nan)
    edgeon = row.get("disk-edge-on_yes_fraction", np.nan)

    if smooth > 0.8:
        return "elliptical"

    if featured > 0.8 and spiral > 0.3:
        return "spiral"

    if edgeon > 0.8:
        return "edge_on_disk"

    return "uncertain"


df["morphology_label"] = df.apply(morphology_class, axis=1)


label_map = {
    "elliptical": 0,
    "spiral": 1,
    "edge_on_disk": 2,
    "uncertain": 3
}

df["morphology_label_id"] = df["morphology_label"].map(label_map)


# ============================================================
# 输出统计
# ============================================================

print("\nMorphology label statistics:")

print(df["morphology_label"].value_counts())


# ============================================================
# 保存 CSV
# ============================================================

df.to_csv(output_csv, index=False)

print("\nSaved CSV to:")

print(output_csv)


# ============================================================
# 保存 FITS
# ============================================================

table = Table.from_pandas(df)

table.write(output_fits, overwrite=True)

print("\nSaved FITS to:")

print(output_fits)