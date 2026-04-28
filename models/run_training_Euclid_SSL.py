# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import json
import time
import platform
import random
import re
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Euclid_DINOv2_VIT'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Warning: tensorboard not found, skipping TensorBoard logging")
    SummaryWriter = None

from euclid_dino.datasets.euclid_vis import EuclidVISDataset
from euclid_dino.datasets.multicrop import EuclidMultiCrop
from euclid_dino.models.dino_model import DINOModel
from euclid_dino.utils.dino_loss import DINOLoss
from euclid_dino.utils.ema import update_teacher


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, msg: str):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def dump_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_ignore_labels(text: str):
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_torch_checkpoint(path: str, map_location=None, trusted: bool = True):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if trusted and ("Weights only load failed" in msg or "weights_only" in msg):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def expected_num_tokens(img_size: int, patch_size: int) -> int:
    if img_size % patch_size != 0:
        raise ValueError(f"img_size={img_size} 不能被 patch_size={patch_size} 整除")
    n = img_size // patch_size
    return n * n + 1


def resize_pos_embedding(pos_ckpt: torch.Tensor, pos_model: torch.Tensor, logger: TeeLogger | None = None):
    if pos_ckpt.ndim != 3 or pos_model.ndim != 3:
        raise ValueError(f"pos_embedding 维度异常: ckpt={tuple(pos_ckpt.shape)}, model={tuple(pos_model.shape)}")
    if pos_ckpt.shape[-1] != pos_model.shape[-1]:
        raise ValueError(f"pos_embedding hidden dim 不一致: ckpt={tuple(pos_ckpt.shape)}, model={tuple(pos_model.shape)}")
    if pos_ckpt.shape == pos_model.shape:
        return pos_ckpt

    cls_ckpt = pos_ckpt[:, :1, :]
    patch_ckpt = pos_ckpt[:, 1:, :]

    old_n = patch_ckpt.shape[1]
    new_n = pos_model.shape[1] - 1

    old_hw = int(round(old_n ** 0.5))
    new_hw = int(round(new_n ** 0.5))

    if old_hw * old_hw != old_n or new_hw * new_hw != new_n:
        raise ValueError(f"pos_embedding patch token 不是平方数: old_n={old_n}, new_n={new_n}")

    if logger:
        logger.log(f"插值位置嵌入: {tuple(pos_ckpt.shape)} -> {tuple(pos_model.shape)} "
                   f"(grid {old_hw}x{old_hw} -> {new_hw}x{new_hw})")

    patch_ckpt = patch_ckpt.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
    patch_ckpt = F.interpolate(patch_ckpt, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
    patch_ckpt = patch_ckpt.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)

    return torch.cat([cls_ckpt, patch_ckpt], dim=1)


def adapt_state_dict_for_model(loaded_state_dict: dict, model_state_dict: dict, logger: TeeLogger | None = None):
    adapted = {}
    skipped = []

    for k, v in loaded_state_dict.items():
        if k not in model_state_dict:
            adapted[k] = v
            continue

        model_v = model_state_dict[k]
        if tuple(v.shape) == tuple(model_v.shape):
            adapted[k] = v
            continue

        if "pos_embedding" in k:
            adapted[k] = resize_pos_embedding(v, model_v, logger=logger)
            continue

        skipped.append((k, tuple(v.shape), tuple(model_v.shape)))

    if logger and skipped:
        logger.log(f"以下参数 shape 不一致，已跳过加载（前20项）: {skipped[:20]}")

    return adapted


def log_model_token_info(model: DINOModel, args, logger: TeeLogger, prefix: str):
    pos = None
    try:
        pos = model.backbone.pos_embedding
    except Exception:
        pass

    expected = expected_num_tokens(args.img_size, args.patch_size)
    if pos is not None:
        logger.log(f"[{prefix}] model.backbone.pos_embedding.shape = {tuple(pos.shape)}")
        logger.log(f"[{prefix}] expected tokens from img_size={args.img_size}, patch_size={args.patch_size} -> {expected}")
        if pos.shape[1] != expected:
            logger.log(
                f"[{prefix}] WARNING: 当前模型 token 数 {pos.shape[1]} 与理论期望 {expected} 不一致，"
                f"这通常说明 backbone 实例化参数和训练输入设置不一致。"
            )
    else:
        logger.log(f"[{prefix}] WARNING: 未能读取 model.backbone.pos_embedding")


OBJ_RE = re.compile(r'^(-?\d+)_')


def parse_object_id_from_filename(path_or_name: str) -> int:
    name = os.path.basename(path_or_name)
    m = OBJ_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse OBJECT_ID from filename: {name}")
    return int(m.group(1))


def _find_best_fits_table_hdu(hdul):
    best_idx = None
    fallback_idx = None
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        names = None
        if hasattr(data, "dtype") and getattr(data.dtype, "names", None) is not None:
            names = list(data.dtype.names)
        elif hasattr(data, "columns") and hasattr(data.columns, "names"):
            names = list(data.columns.names)
        if names:
            lower_names = {str(x).lower() for x in names}
            if fallback_idx is None and len(data) > 0:
                fallback_idx = idx
            if ("object_id" in lower_names or "morphology_label" in lower_names or "morphology_main" in lower_names) and len(data) > 0:
                best_idx = idx
                break
    if best_idx is not None:
        return best_idx
    if fallback_idx is not None:
        return fallback_idx
    if len(hdul) > 1:
        return 1
    return 0


def load_catalog(catalog_path: str):
    if catalog_path is None or catalog_path == "":
        return None
    if catalog_path.endswith(".fits"):
        try:
            from astropy.io import fits
        except Exception as e:
            raise ImportError("Reading .fits catalog requires astropy. Please install astropy.") from e
        with fits.open(catalog_path) as hdul:
            hdu_idx = _find_best_fits_table_hdu(hdul)
            data = hdul[hdu_idx].data
            if data is None:
                raise ValueError(f"No table data found in FITS catalog: {catalog_path}")
            return data
    elif catalog_path.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(catalog_path)
    else:
        raise ValueError(f"Unsupported catalog format: {catalog_path}")


def _get_df_col_case_insensitive(df, candidates):
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_to_real:
            return lower_to_real[key]
    return None


def _get_rec_col_case_insensitive(rec, candidates):
    names = list(rec.dtype.names) if getattr(rec.dtype, "names", None) is not None else []
    lower_to_real = {str(c).lower(): c for c in names}
    for cand in candidates:
        key = str(cand).lower()
        if key in lower_to_real:
            return lower_to_real[key]
    return None


def get_object_id_column_name(catalog):
    import pandas as pd
    if catalog is None:
        return None
    if isinstance(catalog, pd.DataFrame):
        return _get_df_col_case_insensitive(catalog, ["OBJECT_ID", "object_id"])
    return _get_rec_col_case_insensitive(catalog, ["OBJECT_ID", "object_id"])


def build_object_id_index_from_catalog_df(catalog_df):
    import pandas as pd
    obj_col = get_object_id_column_name(catalog_df)
    if obj_col is None:
        raise KeyError("Catalog missing OBJECT_ID/object_id")
    s = pd.Series(catalog_df.index.values, index=pd.to_numeric(catalog_df[obj_col], errors="coerce").astype("Int64"))
    s = s[s.index.notna()]
    s.index = s.index.astype(np.int64)
    return s[~s.index.duplicated(keep="first")]


def build_object_id_index_from_fits_rec(catalog_rec):
    obj_col = get_object_id_column_name(catalog_rec)
    if obj_col is None:
        raise KeyError("Catalog missing OBJECT_ID/object_id")
    obj = np.array(catalog_rec[obj_col], dtype=np.int64)
    d = {}
    for i, oid in enumerate(obj):
        if oid not in d:
            d[int(oid)] = int(i)
    return d


def get_catalog_column_values(catalog, col_candidates, dtype=None):
    import pandas as pd
    if catalog is None:
        return None, None
    if isinstance(catalog, pd.DataFrame):
        col = _get_df_col_case_insensitive(catalog, col_candidates)
        if col is None:
            return None, None
        values = catalog[col].to_numpy()
    else:
        col = _get_rec_col_case_insensitive(catalog, col_candidates)
        if col is None:
            return None, None
        values = np.array(catalog[col])
    if dtype is not None:
        if dtype == "str":
            values = np.array([None if x is None else str(x) for x in values], dtype=object)
        else:
            values = values.astype(dtype)
    return col, values


def align_dataset_files_to_catalog(singleview_dataset, catalog, logger: TeeLogger | None = None):
    import pandas as pd
    files = getattr(singleview_dataset, "files", None)
    if files is None:
        raise AttributeError("EuclidVISDataset has no attribute .files; cannot align by OBJECT_ID")

    obj_ids = np.array([parse_object_id_from_filename(p) for p in files], dtype=np.int64)

    if isinstance(catalog, pd.DataFrame):
        idx_map = build_object_id_index_from_catalog_df(catalog)
        cat_idx = np.array([int(idx_map.get(oid, -1)) for oid in obj_ids], dtype=np.int64)
        obj_col = get_object_id_column_name(catalog)
        dup_cnt = int(catalog[obj_col].duplicated().sum()) if obj_col is not None else 0
    else:
        idx_map = build_object_id_index_from_fits_rec(catalog)
        cat_idx = np.array([int(idx_map.get(int(oid), -1)) for oid in obj_ids], dtype=np.int64)
        dup_cnt = 0

    keep = cat_idx >= 0
    if logger:
        logger.log(f"[Align] dataset files: {len(files)}")
        logger.log(f"[Align] matched to catalog by OBJECT_ID: {int(keep.sum())}/{len(keep)} ({keep.mean()*100:.2f}%)")
        if dup_cnt > 0:
            logger.log(f"[Align] WARNING: catalog has duplicated OBJECT_ID rows: {dup_cnt} (kept first occurrence)")
        if (~keep).any():
            bad_idx = np.where(~keep)[0][:10]
            bad_names = [os.path.basename(files[i]) for i in bad_idx]
            logger.log(f"[Align] unmatched examples (first 10): {bad_names}")
    return keep, cat_idx


class SimpleSingleViewTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, img):
        if hasattr(img, "convert"):
            img = np.array(img)
        x = img.astype(np.float32)
        mx = np.max(x)
        x = x / mx if mx > 0 else x
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    embs = []
    model.eval()
    for batch in loader:
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device, non_blocking=True)
        _, emb, _ = model(batch)
        embs.append(emb.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


def get_morphology_labels(catalog, label_col_candidates, ignore_labels=None, logger=None):
    ignore_labels = set(ignore_labels or [])
    col, labels = get_catalog_column_values(catalog, label_col_candidates, dtype="str")
    if col is None:
        if logger:
            logger.log(f"[MorphologyProbe] WARNING: missing label column. tried={label_col_candidates}")
        return None, None, None

    clean = []
    valid = np.ones(len(labels), dtype=bool)
    for i, x in enumerate(labels):
        if x is None:
            valid[i] = False
            clean.append(None)
            continue
        s = str(x).strip()
        s_lower = s.lower()
        if s == "" or s_lower in {"nan", "none", "null"}:
            valid[i] = False
            clean.append(None)
            continue
        if s_lower in {v.lower() for v in ignore_labels}:
            valid[i] = False
            clean.append(s)
            continue
        clean.append(s)

    labels = np.array(clean, dtype=object)
    if logger:
        uniq, cnt = np.unique(labels[valid], return_counts=True) if valid.any() else ([], [])
        preview = ", ".join([f"{u}:{c}" for u, c in zip(uniq[:10], cnt[:10])])
        logger.log(f"[MorphologyProbe] label_col={col}, valid_labels={int(valid.sum())}/{len(valid)}, unique_valid={len(uniq)}")
        if preview:
            logger.log(f"[MorphologyProbe] class distribution preview: {preview}")
    return col, labels, valid


def run_morphology_probe_epoch(
    student,
    device,
    singleview_dataset,
    train_idx,
    val_idx,
    catalog_path,
    batch_size=256,
    max_iter=2000,
    label_col="morphology_label",
    ignore_labels=None,
    logger=None
):
    if logger:
        logger.log("[MorphologyProbe] 开始执行 Morphology Probe")
    catalog = load_catalog(catalog_path)
    if logger:
        logger.log(f"[MorphologyProbe] 加载 catalog 完成，类型: {type(catalog)}")

    candidates = [x.strip() for x in str(label_col).split(",") if x.strip()]
    if "morphology_label" not in [x.lower() for x in candidates]:
        candidates = candidates + ["morphology_label", "morphology_main"]
    label_col_real, y_text_all, valid_label_mask = get_morphology_labels(
        catalog,
        candidates,
        ignore_labels=ignore_labels,
        logger=logger
    )
    if y_text_all is None or valid_label_mask is None:
        if logger:
            logger.log("[MorphologyProbe] WARNING: label column not found; skip.")
        return None

    keep, cat_idx = align_dataset_files_to_catalog(singleview_dataset, catalog, logger=logger)
    matched_and_labeled = keep.copy()
    matched_rows = np.where(keep)[0]
    matched_and_labeled[matched_rows] = valid_label_mask[cat_idx[matched_rows]]

    train_idx = [i for i in train_idx if i < len(matched_and_labeled) and matched_and_labeled[i]]
    val_idx = [i for i in val_idx if i < len(matched_and_labeled) and matched_and_labeled[i]]

    if logger:
        logger.log(f"[MorphologyProbe] 过滤后训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        if logger:
            logger.log("[MorphologyProbe] WARNING: no matched labeled samples in train/val after OBJECT_ID alignment; skip.")
        return None

    y_train_text = np.array([y_text_all[cat_idx[i]] for i in train_idx], dtype=object)
    y_val_text = np.array([y_text_all[cat_idx[i]] for i in val_idx], dtype=object)

    train_classes, train_counts = np.unique(y_train_text, return_counts=True)
    if len(train_classes) < 2:
        if logger:
            logger.log(f"[MorphologyProbe] WARNING: train classes < 2 after filtering: {train_classes.tolist()}")
        return None
    too_small = train_classes[train_counts < 2]
    if len(too_small) > 0 and logger:
        logger.log(f"[MorphologyProbe] WARNING: classes with <2 samples in train: {too_small.tolist()}")

    allowed = set(train_classes.tolist())
    val_keep = np.array([y in allowed for y in y_val_text], dtype=bool)
    if not np.all(val_keep):
        dropped = int((~val_keep).sum())
        if logger:
            logger.log(f"[MorphologyProbe] val 中有 {dropped} 个样本类别未出现在训练集中，已跳过。")
        y_val_text = y_val_text[val_keep]
        val_idx = [idx for idx, m in zip(val_idx, val_keep) if m]

    if len(val_idx) == 0:
        if logger:
            logger.log("[MorphologyProbe] WARNING: val set empty after removing unseen classes; skip.")
        return None

    label_to_int = {lab: i for i, lab in enumerate(sorted(allowed))}
    y_train = np.array([label_to_int[y] for y in y_train_text], dtype=int)
    y_val = np.array([label_to_int[y] for y in y_val_text], dtype=int)

    train_ds = Subset(singleview_dataset, train_idx)
    val_ds = Subset(singleview_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    X_train = extract_embeddings(student, train_loader, device)
    X_val = extract_embeddings(student, val_loader, device)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        multi_class="auto"
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_val, y_pred, average=None, labels=list(range(len(label_to_int))), zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(label_to_int)))).tolist()
    int_to_label = {v: k for k, v in label_to_int.items()}

    res = {
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "labels": [int_to_label[i] for i in range(len(int_to_label))],
        "per_class": [
            {
                "label": int_to_label[i],
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i in range(len(int_to_label))
        ],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "embedding_dim": int(X_train.shape[1]),
        "label_col": label_col_real,
    }
    if logger:
        logger.log(
            f"[MorphologyProbe] acc={acc:.4f} balanced_acc={bal_acc:.4f} macro_f1={macro_f1:.4f} "
            f"weighted_f1={weighted_f1:.4f} (n_train={len(y_train)}, n_val={len(y_val)}, classes={len(label_to_int)})"
        )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\datasets\gz_morphology_catalogue', help='数据根目录')
    parser.add_argument('--catalog-path', type=str, default=r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\morphology_catalogue.fits', help='星表文件路径')
    parser.add_argument('--output-dir', type=str, default=r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs', help='输出目录')
    parser.add_argument('--model-type', type=str, default='s', choices=['s', 'b'], help='模型类型')
    parser.add_argument('--patch-size', type=int, default=16, choices=[8, 16], help='Patch大小')
    parser.add_argument('--img-size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')

    parser.add_argument('--epochs', type=int, default=8, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')

    parser.add_argument('--teacher-momentum', type=float, default=0.996, help='Teacher动量')
    parser.add_argument('--teacher-temp', type=float, default=0.04, help='Teacher温度')
    parser.add_argument('--student-temp', type=float, default=0.1, help='Student温度')

    parser.add_argument('--global-crops', type=int, default=2, help='全局裁剪数量')
    parser.add_argument('--local-crops', type=int, default=2, help='局部裁剪数量')
    parser.add_argument('--global-size', type=int, default=224, help='全局裁剪尺寸')
    parser.add_argument('--local-size', type=int, default=96, help='局部裁剪尺寸')

    parser.add_argument('--val-split', type=float, default=0.05, help='验证集比例')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce_on_plateau'], help='学习率调度器类型')
    parser.add_argument('--lr-step-size', type=int, default=10, help='学习率步进大小（仅用于step调度器）')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--checkpoint-freq', type=int, default=1, help='模型保存频率')
    parser.add_argument('--resume', type=str, default=None, help='从哪个检查点继续训练，例如 "checkpoint_epoch_3.pth"')

    parser.add_argument('--seed', type=int, default=42, help='随机种子（可复现）')
    parser.add_argument('--save-config', action='store_true', help='是否在 output-dir 落盘 config.json + train.log')
    parser.add_argument('--eval-linear-probe', type=str2bool, default=True, help='默认开启 Morphology Probing（保留原参数名以兼容旧命令）')
    parser.add_argument('--probe-batch-size', type=int, default=256, help='线性探测抽embedding的batch size')
    parser.add_argument('--probe-max-iter', type=int, default=2000, help='线性探测LogReg最大迭代')
    parser.add_argument('--probe-every', type=int, default=1, help='每多少个epoch跑一次morphology probe')
    parser.add_argument('--probe-label-col', type=str, default='morphology_label,morphology_main', help='形态学硬标签列名，支持逗号分隔候选列')
    parser.add_argument('--probe-ignore-labels', type=str, default='uncertain', help='probe时忽略的标签，逗号分隔，例如 uncertain,problem')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = TeeLogger(os.path.join(log_dir, "train.log"))
    logger.log(f"创建输出目录: {args.output_dir}")
    logger.log(f"创建日志目录: {log_dir}")

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_dir)
        logger.log(f"TensorBoard日志目录: {log_dir}")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'使用设备: {device}')

    logger.log('加载数据...')
    transform = EuclidMultiCrop(
        global_crops=args.global_crops,
        local_crops=args.local_crops,
        global_size=args.global_size,
        local_size=args.local_size
    )

    actual_data_root = args.data_root
    if os.path.exists(os.path.join(args.data_root, 'VIS')):
        actual_data_root = os.path.join(args.data_root, 'VIS')
    logger.log(f"data_root={args.data_root}")
    logger.log(f"actual_data_root={actual_data_root}")

    full_dataset = EuclidVISDataset(
        root=actual_data_root,
        transform=transform,
        img_size=args.img_size
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    logger.log(f'总数据集大小: {len(full_dataset)}')
    logger.log(f'训练集大小: {len(train_dataset)}')
    logger.log(f'验证集大小: {len(val_dataset)}')
    logger.log(f'训练批次数量: {len(train_loader)}')
    logger.log(f'验证批次数量: {len(val_loader)}')

    singleview_dataset = EuclidVISDataset(
        root=actual_data_root,
        transform=SimpleSingleViewTransform(args.img_size),
        img_size=args.img_size
    )
    train_indices = list(getattr(train_dataset, "indices", []))
    val_indices = list(getattr(val_dataset, "indices", []))
    logger.log(f"train_indices length: {len(train_indices)}")
    logger.log(f"val_indices length: {len(val_indices)}")

    logger.log('构建模型...')
    student = DINOModel(
        model_type=args.model_type,
        patch_size=args.patch_size,
        input_channels=1,
        img_size=args.img_size
    ).to(device)

    teacher = DINOModel(
        model_type=args.model_type,
        patch_size=args.patch_size,
        input_channels=1,
        img_size=args.img_size
    ).to(device)

    for param in teacher.parameters():
        param.requires_grad = False

    teacher.load_state_dict(student.state_dict())
    logger.log('模型构建完成：teacher 已从 student 初始化')

    log_model_token_info(student, args, logger, prefix="Student")
    log_model_token_info(teacher, args, logger, prefix="Teacher")

    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.img_size, args.img_size, device=device)
        _, emb_dummy, _ = student(dummy)
        embedding_dim = int(emb_dummy.shape[-1])
    logger.log(f"student embedding_dim = {embedding_dim}")

    dino_loss = DINOLoss(
        out_dim=65536,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=5,
            verbose=True
        )

    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "data": {
            "data_root": args.data_root,
            "actual_data_root": actual_data_root,
            "catalog_path": args.catalog_path,
            "dataset_len": len(full_dataset),
            "train_len": len(train_dataset),
            "val_len": len(val_dataset),
            "val_split": args.val_split,
        },
        "model": {
            "model_type": args.model_type,
            "patch_size": args.patch_size,
            "img_size": args.img_size,
            "input_channels": 1,
            "embedding_dim": embedding_dim,
            "expected_tokens": expected_num_tokens(args.img_size, args.patch_size),
            "student_pos_embedding_shape": tuple(student.backbone.pos_embedding.shape) if hasattr(student.backbone, "pos_embedding") else None,
        },
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_scheduler": args.lr_scheduler,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "teacher_momentum": args.teacher_momentum,
            "teacher_temp": args.teacher_temp,
            "student_temp": args.student_temp,
            "checkpoint_freq": args.checkpoint_freq,
            "resume": args.resume,
        },
        "augment": {
            "global_crops": args.global_crops,
            "local_crops": args.local_crops,
            "global_size": args.global_size,
            "local_size": args.local_size,
        },
        "morphology_probe": {
            "enabled": bool(args.eval_linear_probe),
            "probe_every": args.probe_every,
            "probe_batch_size": args.probe_batch_size,
            "probe_max_iter": args.probe_max_iter,
            "probe_label_col": args.probe_label_col,
            "probe_ignore_labels": parse_ignore_labels(args.probe_ignore_labels),
            "alignment_rule": "OBJECT_ID parsed from cutout filename -> join with catalog.OBJECT_ID/object_id",
            "filename_example": "-506074769276397166_VIS_BGSUB.fits",
        }
    }
    config_path = os.path.join(args.output_dir, "config.json")
    dump_json(config, config_path)
    logger.log(f"已写入训练配置: {config_path}")

    train_losses, val_losses, lrs = [], [], []
    morphology_probe_hist = []

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, args.resume)
        if os.path.exists(checkpoint_path):
            logger.log(f'从检查点恢复训练: {checkpoint_path}')
            checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device, trusted=True)

            student_state_dict = adapt_state_dict_for_model(checkpoint['student_state_dict'], student.state_dict(), logger=logger)
            teacher_state_dict = adapt_state_dict_for_model(checkpoint['teacher_state_dict'], teacher.state_dict(), logger=logger)

            student.load_state_dict(student_state_dict, strict=False)
            teacher.load_state_dict(teacher_state_dict, strict=False)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.log(f'加载 optimizer_state_dict 失败，将重新初始化优化器: {e}')

            start_epoch = checkpoint['epoch']
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            lrs = checkpoint.get('lrs', [])
            morphology_probe_hist = checkpoint.get('morphology_probe_hist', checkpoint.get('linear_probe_hist', []))
            logger.log(f'成功恢复训练状态，从epoch {start_epoch+1}开始训练')
        else:
            logger.log(f'警告: 检查点文件不存在 {checkpoint_path}，将从头开始训练')

    logger.log(f'训练循环参数: start_epoch={start_epoch}, epochs={args.epochs}')
    logger.log(f'计划训练 {args.epochs} 个epoch，每 {args.checkpoint_freq} 个epoch保存一次模型')

    for epoch in range(start_epoch, args.epochs):
        logger.log(f'\n=== 开始 Epoch {epoch+1}/{args.epochs} ===')
        student.train()
        teacher.eval()

        total_train_loss = 0.0

        for step, views in enumerate(train_loader):
            views = [v.to(device, non_blocking=True) for v in views]

            student_out = []
            teacher_out = []

            for v in views:
                out_s, _, _ = student(v)
                student_out.append(out_s)

            with torch.no_grad():
                for v in views[:args.global_crops]:
                    out_t, _, _ = teacher(v)
                    teacher_out.append(out_t)

            loss = dino_loss(student_out, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher(student, teacher, args.teacher_momentum)

            total_train_loss += float(loss.item())

            if (step + 1) % 50 == 0:
                avg_loss = total_train_loss / (step + 1)
                current_lr = optimizer.param_groups[0]['lr']
                logger.log(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], '
                           f'Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        student.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for views in val_loader:
                views = [v.to(device, non_blocking=True) for v in views]

                student_out = []
                teacher_out = []

                for v in views:
                    out_s, _, _ = student(v)
                    student_out.append(out_s)

                for v in views[:args.global_crops]:
                    out_t, _, _ = teacher(v)
                    teacher_out.append(out_t)

                loss = dino_loss(student_out, teacher_out)
                total_val_loss += float(loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)

        logger.log(f"Epoch [{epoch+1}/{args.epochs}] 总结: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} lr={current_lr:.6f}")

        if args.lr_scheduler == 'reduce_on_plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        logger.log(f"[MorphologyProbe] 检查是否执行 Morphology Probe: epoch={epoch+1}, probe_every={args.probe_every}")
        if bool(args.eval_linear_probe) and (epoch + 1) % args.probe_every == 0:
            probe_res = run_morphology_probe_epoch(
                student=student,
                device=device,
                singleview_dataset=singleview_dataset,
                train_idx=train_indices,
                val_idx=val_indices,
                catalog_path=args.catalog_path,
                batch_size=args.probe_batch-size if False else args.probe_batch_size,
                max_iter=args.probe_max_iter,
                label_col=args.probe_label_col,
                ignore_labels=parse_ignore_labels(args.probe_ignore_labels),
                logger=logger
            )
            if probe_res is not None:
                probe_res["epoch"] = int(epoch + 1)
                morphology_probe_hist.append(probe_res)

                if writer is not None:
                    writer.add_scalar("MorphologyProbe/acc", probe_res["acc"], epoch + 1)
                    writer.add_scalar("MorphologyProbe/balanced_acc", probe_res["balanced_acc"], epoch + 1)
                    writer.add_scalar("MorphologyProbe/macro_f1", probe_res["macro_f1"], epoch + 1)
                    writer.add_scalar("MorphologyProbe/weighted_f1", probe_res["weighted_f1"], epoch + 1)

                try:
                    import pandas as pd
                    hist_for_csv = []
                    for item in morphology_probe_hist:
                        row = {k: v for k, v in item.items() if k not in {"confusion_matrix", "per_class", "labels"}}
                        hist_for_csv.append(row)
                    pd.DataFrame(hist_for_csv).to_csv(os.path.join(args.output_dir, "morphology_probe_history.csv"), index=False)
                    logger.log(f"已保存形态学探针历史: {os.path.join(args.output_dir, 'morphology_probe_history.csv')}")
                except Exception as e:
                    logger.log(f"保存morphology_probe_history.csv失败: {e}")

                try:
                    dump_json({"history": morphology_probe_hist}, os.path.join(args.output_dir, "morphology_probe_history.json"))
                    logger.log(f"已保存形态学探针历史: {os.path.join(args.output_dir, 'morphology_probe_history.json')}")
                except Exception as e:
                    logger.log(f"保存morphology_probe_history.json失败: {e}")

                try:
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    df = pd.DataFrame([{k: v for k, v in item.items() if isinstance(v, (int, float))} for item in morphology_probe_hist])
                    plt.figure()
                    plt.plot(df["epoch"], df["acc"], label="Accuracy")
                    plt.plot(df["epoch"], df["balanced_acc"], label="Balanced Accuracy")
                    plt.plot(df["epoch"], df["macro_f1"], label="Macro F1")
                    plt.plot(df["epoch"], df["weighted_f1"], label="Weighted F1")
                    plt.xlabel("Epoch")
                    plt.ylabel("Score")
                    plt.legend()
                    plt.title("Morphology Probe Performance")
                    plt.savefig(os.path.join(args.output_dir, "morphology_probe_curve.png"))
                    plt.close()
                    logger.log("已生成 morphology_probe_curve.png")
                except Exception as e:
                    logger.log(f"绘制Morphology Probe曲线失败: {e}")
            else:
                logger.log("[MorphologyProbe] Morphology Probe执行失败，跳过保存步骤")

        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'lrs': lrs,
                'embedding_dim': embedding_dim,
                'args': vars(args),
                'morphology_probe_hist': morphology_probe_hist,
            }, checkpoint_path)
            logger.log(f'模型已保存到 {checkpoint_path}')

    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': lrs
    }
    loss_path = os.path.join(args.output_dir, 'loss_data.npy')
    try:
        np.save(loss_path, loss_data)
        logger.log(f'损失曲线数据已保存到 {loss_path}')
    except Exception as e:
        logger.log(f'保存loss_data.npy失败: {e}')

    if writer is not None:
        writer.close()

    logger.log('训练完成！')
    logger.log(f'最终训练损失: {train_losses[-1]:.4f}')
    logger.log(f'最终验证损失: {val_losses[-1]:.4f}')
    logger.log(f'所有输出文件保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
