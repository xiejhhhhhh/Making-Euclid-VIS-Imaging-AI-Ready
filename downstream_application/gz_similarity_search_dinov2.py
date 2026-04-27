# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import os, re, glob, math, json, time, pickle, hashlib, warnings, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch"
PROJECT_DIR = os.path.join(ROOT_DIR, "Euclid_DINOv2_VIT")
VIS_DIR = os.path.join(PROJECT_DIR, r"datasets\gz_morphology_catalogue\VIS")
MODEL_PATH = os.path.join(PROJECT_DIR, r"gz_outputs_official_dinov2\dinov2_vits14_official_backbone.pth")
MODEL_FAMILY = "official"
OFFICIAL_MODEL_NAME = "dinov2_vits14"
CATALOG_CANDIDATES = [
    os.path.join(ROOT_DIR, r"catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv"),
    os.path.join(ROOT_DIR, r"catalogs\gz_euclid_q1\morphology_catalogue_with_labels.fits"),
]
OUTPUT_DIR = os.path.join(ROOT_DIR, r"datasets\Morphology_retrieval_improved_official_dinov2_fixed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
TOPK = 4
CANDIDATE_POOL = 300

QUALITY_THRESHOLDS = {
    "spurious_prob": 0.20,
    "blended_prob": 0.90,
    "point_like_prob_upper": 0.85,
    "snr_min": 3.0,
}
PHYSICAL_FEATURES = ["mu_max", "mumax_minus_mag", "kron_radius", "ellipticity"]

TASK_CONFIGS = {
    "spiral":     {"target_value": 1, "emb_w": 0.70, "label_w": 0.22, "phys_w": 0.06, "quality_w": 0.02},
    "nonspiral":  {"target_value": 0, "emb_w": 0.70, "label_w": 0.22, "phys_w": 0.06, "quality_w": 0.02},
    "smooth":     {"target_value": 1, "emb_w": 0.68, "label_w": 0.24, "phys_w": 0.06, "quality_w": 0.02},
    "featured":   {"target_value": 1, "emb_w": 0.68, "label_w": 0.24, "phys_w": 0.06, "quality_w": 0.02},
    "round":      {"target_value": 1, "emb_w": 0.62, "label_w": 0.24, "phys_w": 0.12, "quality_w": 0.02},
    "cigar":      {"target_value": 1, "emb_w": 0.62, "label_w": 0.24, "phys_w": 0.12, "quality_w": 0.02},
    "edgeon":     {"target_value": 1, "emb_w": 0.62, "label_w": 0.24, "phys_w": 0.12, "quality_w": 0.02},
    "nonedgeon":  {"target_value": 0, "emb_w": 0.62, "label_w": 0.24, "phys_w": 0.12, "quality_w": 0.02},
}
TASKS_TO_RUN = ["spiral", "nonspiral", "featured", "smooth", "round", "cigar", "edgeon", "nonedgeon"]

LABEL_ALIASES = {
    "spiral": ["spiral", "is_spiral", "label_spiral", "p_spiral", "prob_spiral", "spiral_prob", "spiral_fraction", "spiral_yes", "spiral_yes_fraction"],
    "smooth": ["smooth", "is_smooth", "label_smooth", "p_smooth", "prob_smooth", "smooth_prob", "smooth_fraction", "smooth_yes", "smooth_yes_fraction"],
    "featured": ["featured", "features", "disk", "featured_or_disk", "is_featured", "label_featured", "p_featured", "prob_featured", "featured_prob", "features_or_disk", "features_yes", "disk_yes"],
    "round": ["round", "completely_round", "is_round", "label_round", "p_round", "prob_round", "round_prob", "round_fraction"],
    "cigar": ["cigar", "cigar_shaped", "is_cigar", "label_cigar", "p_cigar", "prob_cigar", "cigar_prob", "cigar_shaped_yes"],
    "edgeon": ["edgeon", "edge_on", "is_edgeon", "label_edgeon", "p_edgeon", "prob_edgeon", "edgeon_prob", "edge_on_yes", "edgeon_yes"],
}
TEXT_LABEL_ALIASES = ["morphology_label", "morphology_main", "hard_label", "label", "class_label", "gz_label"]
ID_ALIASES = ["OBJECT_ID", "object_id", "SOURCE_ID", "source_id", "ID", "id"]
PHYS_ALIASES = {
    "mu_max": ["MU_MAX", "mu_max"],
    "mumax_minus_mag": ["MUMAX_MINUS_MAG", "MUMINUSMAG", "mumax_minus_mag", "mumax-mag", "mu_minus_mag"],
    "kron_radius": ["KRON_RADIUS", "kron_radius"],
    "ellipticity": ["ELLIPTICITY", "ellipticity", "ELLIP", "ellip"],
    "semimajor_axis": ["SEMIMAJOR_AXIS", "A_IMAGE", "semimajor_axis"],
    "segmentation_area": ["SEGMENTATION_AREA", "segmentation_area", "ISOAREA_IMAGE"],
    "blended_prob": ["BLENDED_PROB", "blended_prob"],
    "spurious_prob": ["SPURIOUS_PROB", "spurious_prob"],
    "point_like_prob": ["POINT_LIKE_PROB", "point_like_prob"],
    "snr": ["SNR", "snr", "signal_to_noise"],
}

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def log(msg: str): print(f"[{now()}] {msg}")
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def normalize_colname(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())
def basename_noext(path: str) -> str: return os.path.splitext(os.path.basename(path))[0]

def extract_object_id_from_filename(file_path: str) -> Optional[int]:
    name = os.path.basename(file_path)
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None

def safe_torch_load(path: str, map_location=None, trusted: bool = True):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if trusted and ("Weights only load failed" in msg or "weights_only" in msg):
            log("torch.load 默认 weights_only=True 加载失败，回退为 weights_only=False")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise

def smart_extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["student_state_dict", "teacher_state_dict", "backbone_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
    if isinstance(ckpt, dict):
        tensor_like = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if len(tensor_like) > 0:
            return tensor_like, "raw_dict"
    raise ValueError("Unsupported checkpoint format")

def load_catalog(candidates: List[str]) -> pd.DataFrame:
    for p in candidates:
        if os.path.exists(p):
            log(f"加载星表: {p}")
            if p.lower().endswith(".csv"):
                return pd.read_csv(p)
            elif p.lower().endswith(".fits"):
                with fits.open(p) as hdul:
                    data = hdul[1].data
                    try:
                        return pd.DataFrame(np.array(data).byteswap().newbyteorder())
                    except Exception:
                        return pd.DataFrame(np.array(data))
    raise FileNotFoundError(f"未找到星表，候选路径: {candidates}")

def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_colname(c): c for c in df.columns}

def find_column(df: pd.DataFrame, aliases: List[str], exact_only: bool = False) -> Optional[str]:
    cmap = build_column_map(df)
    norm_aliases = [normalize_colname(a) for a in aliases]
    for a in norm_aliases:
        if a in cmap:
            return cmap[a]
    if exact_only:
        return None
    for a in norm_aliases:
        for norm_c, orig_c in cmap.items():
            if norm_c.startswith(a) or norm_c.endswith(a):
                return orig_c
    return None

def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_probability_like_series(s: pd.Series) -> bool:
    x = safe_numeric_series(s).to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return False
    q01, q99 = np.percentile(x, [1, 99])
    if q01 >= -1e-6 and q99 <= 1.000001:
        return True
    if q01 >= -1e-6 and q99 <= 100.000001:
        return True
    uniq = np.unique(np.round(x, 6))
    if len(uniq) <= 3 and np.all(np.isin(uniq, [0.0, 1.0])):
        return True
    return False

def coerce_prob_like_series(s: pd.Series) -> pd.Series:
    x = safe_numeric_series(s).astype(float)
    valid = x[np.isfinite(x)]
    if len(valid) > 0 and np.percentile(valid, 99) > 1.000001 and np.percentile(valid, 99) <= 100.000001:
        x = x / 100.0
    return x.clip(lower=0.0, upper=1.0)

def resolve_feature_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {key: find_column(df, aliases, exact_only=False) for key, aliases in PHYS_ALIASES.items()}

def resolve_id_column(df: pd.DataFrame) -> str:
    col = find_column(df, ID_ALIASES, exact_only=False)
    if col is None:
        raise ValueError(f"星表里找不到 ID 列，候选: {ID_ALIASES}")
    return col

def infer_text_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in TEXT_LABEL_ALIASES:
        if col in df.columns:
            return col
    cmap = build_column_map(df)
    for alias in TEXT_LABEL_ALIASES:
        n = normalize_colname(alias)
        if n in cmap:
            return cmap[n]
    return None

def build_binary_from_text_labels(text_series: pd.Series, task: str) -> pd.Series:
    s = text_series.fillna("").astype(str).str.lower()
    def contains_any(series, words):
        mask = pd.Series(False, index=series.index)
        for w in words:
            mask = mask | series.str.contains(w, regex=False)
        return mask
    if task == "spiral":
        pos = contains_any(s, ["spiral"]); neg = contains_any(s, ["smooth", "elliptical", "round", "cigar", "nonspiral"])
    elif task == "featured":
        pos = contains_any(s, ["featured", "features", "disk"]); neg = contains_any(s, ["smooth", "elliptical"])
    elif task == "smooth":
        pos = contains_any(s, ["smooth", "elliptical"]); neg = contains_any(s, ["featured", "features", "disk", "spiral"])
    elif task == "round":
        pos = contains_any(s, ["round"]); neg = contains_any(s, ["cigar"])
    elif task == "cigar":
        pos = contains_any(s, ["cigar"]); neg = contains_any(s, ["round"])
    elif task == "edgeon":
        pos = contains_any(s, ["edgeon", "edge_on", "edge-on"]); neg = contains_any(s, ["nonedgeon", "non-edgeon"])
    else:
        return pd.Series(np.nan, index=s.index)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[pos] = 1.0
    out[neg & (~pos)] = 0.0
    return out

def resolve_label_columns(df: pd.DataFrame):
    resolved_cols, task_series, task_source = {}, {}, {}
    text_col = infer_text_label_column(df)
    for task, aliases in LABEL_ALIASES.items():
        chosen_col, chosen_series = None, None
        for alias in aliases:
            col = find_column(df, [alias], exact_only=True)
            if col is not None and is_probability_like_series(df[col]):
                chosen_col, chosen_series = col, coerce_prob_like_series(df[col]); break
        if chosen_col is None:
            for alias in aliases:
                col = find_column(df, [alias], exact_only=False)
                if col is not None and is_probability_like_series(df[col]):
                    chosen_col, chosen_series = col, coerce_prob_like_series(df[col]); break
        if chosen_col is None and text_col is not None:
            derived = build_binary_from_text_labels(df[text_col], task)
            if np.isfinite(derived.to_numpy(dtype=float)).sum() > 0:
                chosen_series = derived; task_source[task] = f"text-derived:{text_col}"
        if chosen_series is None:
            resolved_cols[task] = None; task_series[task] = pd.Series(np.nan, index=df.index, dtype=float); task_source[task] = "missing"
        else:
            resolved_cols[task] = chosen_col; task_series[task] = chosen_series.astype(float)
            if task not in task_source: task_source[task] = f"numeric:{chosen_col}"
    return resolved_cols, task_series, task_source

def robust_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn, mx = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - mn) / (mx - mn), 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)

def preprocess_fits_data(data: np.ndarray, img_size: int = 224) -> np.ndarray:
    import cv2
    data = np.array(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim > 2: data = np.squeeze(data)
    if data.ndim != 2: raise ValueError(f"FITS 数据不是 2D 图像，shape={data.shape}")
    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return robust_minmax(data).astype(np.float32)

def read_fits_first_image(file_path: str) -> np.ndarray:
    with fits.open(file_path, lazy_load_hdus=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                return np.array(hdu.data)
    raise ValueError(f"FITS 没有图像数据: {file_path}")

def save_fits_png(fits_path: str, png_path: str):
    data = np.nan_to_num(read_fits_first_image(fits_path), nan=0.0, posinf=0.0, neginf=0.0)
    norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.03), stretch=AsinhStretch(a=0.2))
    plt.figure(figsize=(3, 3), facecolor="black")
    plt.imshow(data, origin="lower", cmap="gray", norm=norm)
    plt.axis("off")
    plt.gca().set_facecolor("black")
    plt.tight_layout(pad=0)
    plt.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close()

class EmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
    def _key(self, vis_dir: str, model_path: str, img_size: int, model_family: str, official_model_name: str):
        return hashlib.md5(f"{vis_dir}|{model_path}|{img_size}|{model_family}|{official_model_name}".encode("utf-8")).hexdigest()
    def load(self, vis_dir: str, model_path: str, img_size: int, model_family: str, official_model_name: str):
        key = self._key(vis_dir, model_path, img_size, model_family, official_model_name)
        emb_path = os.path.join(self.cache_dir, f"{key}_emb.npy")
        meta_path = os.path.join(self.cache_dir, f"{key}_meta.pkl")
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            with open(meta_path, "rb") as f: meta = pickle.load(f)
            return np.load(emb_path), meta
        return None, None
    def save(self, vis_dir: str, model_path: str, img_size: int, model_family: str, official_model_name: str, emb: np.ndarray, meta: dict):
        key = self._key(vis_dir, model_path, img_size, model_family, official_model_name)
        np.save(os.path.join(self.cache_dir, f"{key}_emb.npy"), emb)
        with open(os.path.join(self.cache_dir, f"{key}_meta.pkl"), "wb") as f: pickle.dump(meta, f)

class FitsFolderDataset(Dataset):
    def __init__(self, file_list: List[str], img_size: int = 224, model_family: str = "official"):
        self.file_list, self.img_size, self.model_family = file_list, img_size, model_family
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = preprocess_fits_data(read_fits_first_image(path), self.img_size)
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        return x, path

class OfficialDINOv2BackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__(); self.model = model
    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats: return feats["x_norm_clstoken"]
                if "cls_token" in feats: return feats["cls_token"]
                if "x_prenorm" in feats and torch.is_tensor(feats["x_prenorm"]) and feats["x_prenorm"].ndim == 3: return feats["x_prenorm"][:, 0]
                if "x_norm_patchtokens" in feats and torch.is_tensor(feats["x_norm_patchtokens"]) and feats["x_norm_patchtokens"].ndim == 3: return feats["x_norm_patchtokens"].mean(dim=1)
            if torch.is_tensor(feats): return feats[:, 0] if feats.ndim == 3 else feats
        out = self.model(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out: return out["x_norm_clstoken"]
            for _, v in out.items():
                if torch.is_tensor(v): return v[:, 0] if v.ndim == 3 else v
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v): return v[:, 0] if v.ndim == 3 else v
        if torch.is_tensor(out): return out[:, 0] if out.ndim == 3 else out
        raise RuntimeError("Cannot extract embedding from official DINOv2 output.")

def build_official_model(device: torch.device, model_name: str):
    log(f"构建官方模型: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False, force_reload=False)
    return model.to(device).eval()

def load_official_model(model_path: str, device: torch.device, model_name: str):
    model = build_official_model(device, model_name)
    ckpt = safe_torch_load(model_path, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(ckpt)
    log(f"official checkpoint source_key = {source_key}")
    msg = model.load_state_dict(state, strict=False)
    log(f"official 权重加载完成 missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return OfficialDINOv2BackboneWrapper(model).to(device).eval()

def extract_embeddings_for_vis(vis_dir: str, model_path: str, img_size: int = 224, model_family: str = "official", official_model_name: str = "dinov2_vits14") -> Tuple[np.ndarray, List[str]]:
    cache = EmbeddingCache(os.path.join(OUTPUT_DIR, "cache"))
    emb, meta = cache.load(vis_dir, model_path, img_size, model_family, official_model_name)
    if emb is not None:
        log("命中 embedding 缓存")
        return emb, meta["files"]
    file_list = sorted(glob.glob(os.path.join(vis_dir, "*.fits")))
    if len(file_list) == 0: raise FileNotFoundError(f"VIS 目录下没有 fits: {vis_dir}")
    log(f"共发现 {len(file_list)} 个 FITS 文件")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")
    model = load_official_model(model_path, device, official_model_name)
    ds = FitsFolderDataset(file_list=file_list, img_size=img_size, model_family=model_family)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    embs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dl):
            x = x.to(device, non_blocking=(device.type == "cuda"))
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    e = model(x)
            else:
                e = model(x)
            embs.append(e.detach().cpu().numpy())
            if (i + 1) % 20 == 0: log(f"embedding 批次 {i+1}/{len(dl)}")
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    meta = {"files": file_list, "time": now(), "shape": emb.shape, "model_family": model_family, "official_model_name": official_model_name}
    cache.save(vis_dir, model_path, img_size, model_family, official_model_name, emb, meta)
    log(f"embedding 已保存, shape={emb.shape}")
    return emb, file_list

def build_master_table(catalog_df: pd.DataFrame, vis_files: List[str], embeddings: np.ndarray):
    id_col = resolve_id_column(catalog_df)
    label_cols, task_series, task_source = resolve_label_columns(catalog_df)
    phys_cols = resolve_feature_columns(catalog_df)
    work = catalog_df.copy(); work["_object_id"] = pd.to_numeric(work[id_col], errors="coerce").astype("Int64")
    for task, series in task_series.items(): work[f"_task_{task}"] = series.values
    vis_df = pd.DataFrame({"file_path": vis_files, "file_name": [os.path.basename(p) for p in vis_files], "_object_id": [extract_object_id_from_filename(p) for p in vis_files], "_emb_idx": np.arange(len(vis_files))})
    master = vis_df.merge(work, on="_object_id", how="left", suffixes=("", "_cat"))
    emb = np.array(embeddings, dtype=np.float32); norms = np.linalg.norm(emb, axis=1, keepdims=True); emb_norm = emb / np.clip(norms, 1e-12, None)
    master["_emb_idx"] = master["_emb_idx"].astype(int)
    for task in LABEL_ALIASES.keys():
        c = f"_task_{task}"
        master[f"label_{task}"] = safe_numeric_series(master[c]).astype(float).clip(lower=0.0, upper=1.0) if c in master.columns else np.nan
    for key, col in phys_cols.items():
        if col is not None: master[col] = safe_numeric_series(master[col])
    for feat, col in phys_cols.items(): master[f"phys_{feat}"] = np.nan if col is None else master[col].astype(float)
    if "phys_ellipticity" not in master.columns: master["phys_ellipticity"] = np.nan
    master["quality_score"] = compute_quality_score(master); master["_has_emb"] = master["_emb_idx"].notna()
    phys_mat, phys_scaler = build_physical_matrix(master); master["_phys_valid"] = np.isfinite(phys_mat).all(axis=1)
    return master, emb_norm, phys_mat, phys_scaler, label_cols, phys_cols, task_source

def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    score = np.ones(len(df), dtype=np.float32)
    if "phys_spurious_prob" in df.columns:
        x = df["phys_spurious_prob"].astype(float).to_numpy(); mask = np.isfinite(x); score[mask] *= (1.0 - np.clip(x[mask], 0, 1))
    if "phys_blended_prob" in df.columns:
        x = df["phys_blended_prob"].astype(float).to_numpy(); mask = np.isfinite(x); score[mask] *= (1.0 - 0.5 * np.clip(x[mask], 0, 1))
    if "phys_point_like_prob" in df.columns:
        x = df["phys_point_like_prob"].astype(float).to_numpy(); mask = np.isfinite(x); score[mask] *= (1.0 - 0.7 * np.clip(x[mask], 0, 1))
    if "phys_snr" in df.columns:
        x = df["phys_snr"].astype(float).to_numpy(); mask = np.isfinite(x); score[mask] *= (0.3 + 0.7 * np.clip(x[mask] / 20.0, 0, 1))
    return pd.Series(np.clip(score, 0, 1), index=df.index)

def build_physical_matrix(master: pd.DataFrame):
    cols = [f"phys_{f}" for f in PHYSICAL_FEATURES]
    X = master[cols].copy()
    for c in cols:
        if c not in X.columns: X[c] = np.nan
        med = np.nanmedian(X[c].to_numpy(dtype=float))
        if not np.isfinite(med): med = 0.0
        X[c] = X[c].fillna(med)
    X_np = X.to_numpy(dtype=np.float32); X_np[~np.isfinite(X_np)] = np.nan
    for i in range(X_np.shape[1]):
        col = X_np[:, i]; med = np.nanmedian(col)
        X_np[np.isnan(col), i] = med if np.isfinite(med) else 0.0
    scaler = RobustScaler()
    return scaler.fit_transform(X_np).astype(np.float32), scaler

def apply_quality_filters(master: pd.DataFrame) -> np.ndarray:
    keep = np.ones(len(master), dtype=bool)
    if "phys_spurious_prob" in master.columns:
        x = master["phys_spurious_prob"].to_numpy(dtype=float); keep &= (~np.isfinite(x)) | (x <= QUALITY_THRESHOLDS["spurious_prob"])
    if "phys_blended_prob" in master.columns:
        x = master["phys_blended_prob"].to_numpy(dtype=float); keep &= (~np.isfinite(x)) | (x <= QUALITY_THRESHOLDS["blended_prob"])
    if "phys_point_like_prob" in master.columns:
        x = master["phys_point_like_prob"].to_numpy(dtype=float); keep &= (~np.isfinite(x)) | (x <= QUALITY_THRESHOLDS["point_like_prob_upper"])
    if "phys_snr" in master.columns:
        x = master["phys_snr"].to_numpy(dtype=float); keep &= (~np.isfinite(x)) | (x >= QUALITY_THRESHOLDS["snr_min"])
    return keep

def score_label_match(task: str, task_values: np.ndarray, target_value: int) -> np.ndarray:
    x = np.nan_to_num(task_values.astype(np.float32), nan=0.5)
    x = np.clip(x, 0, 1)
    return x if target_value == 1 else (1.0 - x)

def counterpart_penalty(master: pd.DataFrame, task: str) -> np.ndarray:
    if task in ["spiral", "nonspiral"]:
        smooth = np.nan_to_num(master.get("label_smooth", pd.Series(np.nan, index=master.index)).to_numpy(dtype=float), nan=0.5)
        featured = np.nan_to_num(master.get("label_featured", pd.Series(np.nan, index=master.index)).to_numpy(dtype=float), nan=0.5)
        return 0.6 * featured + 0.4 * (1.0 - smooth) if task == "spiral" else 0.6 * smooth + 0.4 * (1.0 - featured)
    if task in ["featured", "smooth"]:
        spiral = np.nan_to_num(master.get("label_spiral", pd.Series(np.nan, index=master.index)).to_numpy(dtype=float), nan=0.5)
        smooth = np.nan_to_num(master.get("label_smooth", pd.Series(np.nan, index=master.index)).to_numpy(dtype=float), nan=0.5)
        return 0.6 * spiral + 0.4 * (1.0 - smooth) if task == "featured" else 0.6 * (1.0 - spiral) + 0.4 * smooth
    if task in ["round", "cigar"]:
        ell = np.nan_to_num(master.get("phys_ellipticity", pd.Series(np.nan, index=master.index)).to_numpy(dtype=float), nan=0.5)
        ell = np.clip(ell, 0, 1)
        return (1.0 - ell) if task == "round" else ell
    return np.zeros(len(master), dtype=np.float32)

def choose_typical_query(master: pd.DataFrame, emb_norm: np.ndarray, task: str, target_value: int, forbidden_indices: Optional[set] = None) -> Optional[int]:
    forbidden_indices = set() if forbidden_indices is None else forbidden_indices
    label_col = "label_spiral" if task == "nonspiral" else ("label_edgeon" if task == "nonedgeon" else f"label_{task}")
    if label_col not in master.columns: return None
    label_values = master[label_col].to_numpy(dtype=float)
    label_match = score_label_match(task, label_values, target_value)
    valid = np.isfinite(label_match) & apply_quality_filters(master) & master["_has_emb"].to_numpy(dtype=bool)
    if len(forbidden_indices) > 0:
        mask_forbidden = np.zeros(len(master), dtype=bool)
        idxs = [i for i in forbidden_indices if 0 <= i < len(master)]
        if len(idxs) > 0:
            mask_forbidden[np.array(idxs, dtype=int)] = True
            valid &= (~mask_forbidden)
    candidate_idx = np.where(valid & (label_match >= 0.75))[0]
    if len(candidate_idx) == 0: candidate_idx = np.where(valid & (label_match >= 0.55))[0]
    if len(candidate_idx) == 0: candidate_idx = np.where(valid)[0]
    if len(candidate_idx) == 0: return None
    E = emb_norm[master.loc[candidate_idx, "_emb_idx"].to_numpy(dtype=int)]
    mean_emb = E.mean(axis=0, keepdims=True)
    centrality = (E @ mean_emb.T).squeeze()
    qual = master.loc[candidate_idx, "quality_score"].to_numpy(dtype=float)
    counterpart = counterpart_penalty(master, task)[candidate_idx]
    final = 0.48 * label_match[candidate_idx] + 0.22 * centrality + 0.18 * qual + 0.12 * counterpart
    return int(candidate_idx[np.argmax(final)])

@dataclass
class RetrievalResult:
    query_index: int
    query_file: str
    task: str
    results: pd.DataFrame

def retrieve_similar(master: pd.DataFrame, emb_norm: np.ndarray, phys_mat: np.ndarray, query_index: int, task: str, topk: int = 8, candidate_pool: int = 300, hard_filter: bool = False) -> RetrievalResult:
    cfg = TASK_CONFIGS[task]; target_value = cfg["target_value"]
    label_col = "label_spiral" if task in ["spiral", "nonspiral"] else ("label_edgeon" if task in ["edgeon", "nonedgeon"] else f"label_{task}")
    q_emb_idx = int(master.iloc[query_index]["_emb_idx"]); q_emb = emb_norm[q_emb_idx:q_emb_idx + 1]
    all_emb_idx = master["_emb_idx"].to_numpy(dtype=int)
    emb_scores = (emb_norm[all_emb_idx] @ q_emb.T).reshape(-1); emb_scores[query_index] = -1e9
    keep = apply_quality_filters(master) & master["_has_emb"].to_numpy(dtype=bool)
    if hard_filter and label_col in master.columns:
        lv = master[label_col].to_numpy(dtype=float)
        keep &= (np.nan_to_num(lv, nan=0.0) >= 0.5) if target_value == 1 else (np.nan_to_num(lv, nan=0.0) < 0.5)
    emb_scores_masked = emb_scores.copy(); emb_scores_masked[~keep] = -1e9
    candidate_idx = np.argsort(emb_scores_masked)[::-1][:candidate_pool]
    candidate_idx = candidate_idx[emb_scores_masked[candidate_idx] > -1e8]
    if len(candidate_idx) == 0: raise RuntimeError(f"任务 {task}: 没有可用候选，请放松过滤条件")
    lv = master[label_col].to_numpy(dtype=float) if label_col in master.columns else np.full(len(master), 0.5, dtype=float)
    label_scores = score_label_match(task, np.nan_to_num(lv, nan=0.5), target_value)
    q_phys = phys_mat[query_index:query_index + 1]; cand_phys = phys_mat[candidate_idx]
    phys_scores = 1.0 / (1.0 + np.linalg.norm(cand_phys - q_phys, axis=1))
    qual_scores = master.loc[candidate_idx, "quality_score"].to_numpy(dtype=float)
    cand_emb_scores = np.clip((emb_scores[candidate_idx] + 1.0) / 2.0, 0.0, 1.0)
    cand_label_scores = label_scores[candidate_idx]
    final_scores = cfg["emb_w"] * cand_emb_scores + cfg["label_w"] * cand_label_scores + cfg["phys_w"] * phys_scores + cfg["quality_w"] * qual_scores
    order = np.argsort(final_scores)[::-1][:topk]; sel = candidate_idx[order]
    out = master.loc[sel, ["file_path", "file_name", "_object_id", "quality_score"]].copy()
    out["task"] = task; out["emb_score"] = cand_emb_scores[order]; out["label_score"] = cand_label_scores[order]; out["phys_score"] = phys_scores[order]; out["final_score"] = final_scores[order]
    if label_col in master.columns: out[label_col] = master.loc[sel, label_col].to_numpy()
    for f in PHYSICAL_FEATURES:
        c = f"phys_{f}"
        if c in master.columns: out[c] = master.loc[sel, c].to_numpy()
    return RetrievalResult(query_index=query_index, query_file=master.iloc[query_index]["file_path"], task=task, results=out.reset_index(drop=True))

def create_retrieval_grid(query_file: str, result_df: pd.DataFrame, out_png: str, title: str):
    n = len(result_df); ncols = min(5, n + 1); nrows = math.ceil((n + 1) / ncols)
    plt.figure(figsize=(3.0 * ncols, 3.2 * nrows))
    items = [("Query", query_file, None)] + [(f"Rank {i+1}\n{row['final_score']:.3f}", row["file_path"], row) for i, row in result_df.iterrows()]
    for i, (ttl, fpath, row) in enumerate(items, 1):
        ax = plt.subplot(nrows, ncols, i)
        data = np.nan_to_num(read_fits_first_image(fpath), nan=0.0, posinf=0.0, neginf=0.0)
        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.03), stretch=AsinhStretch(a=0.2))
        ax.imshow(data, origin="lower", cmap="gray", norm=norm); ax.axis("off")
        if row is None:
            ax.set_title(ttl, fontsize=11)
        else:
            extra = []
            for c in ["label_spiral", "label_smooth", "label_featured", "label_edgeon", "label_round", "label_cigar"]:
                if c in row.index and pd.notna(row.get(c, np.nan)): extra.append(f"{c.replace('label_','')}={row.get(c):.2f}")
            ax.set_title("\n".join([ttl] + extra[:2]), fontsize=9)
    plt.suptitle(title, fontsize=14); plt.tight_layout(); plt.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close()

def main():
    log("========== Improved morphology similarity search (official DINOv2, fixed queries) ==========")
    log(f"VIS_DIR = {VIS_DIR}"); log(f"MODEL_PATH = {MODEL_PATH}"); log(f"OFFICIAL_MODEL_NAME = {OFFICIAL_MODEL_NAME}"); log(f"OUTPUT_DIR = {OUTPUT_DIR}")
    catalog_df = load_catalog(CATALOG_CANDIDATES)
    embeddings, vis_files = extract_embeddings_for_vis(VIS_DIR, MODEL_PATH, IMG_SIZE, model_family=MODEL_FAMILY, official_model_name=OFFICIAL_MODEL_NAME)
    master, emb_norm, phys_mat, phys_scaler, label_cols, phys_cols, task_source = build_master_table(catalog_df, vis_files, embeddings)
    log(f"master shape = {master.shape}")
    meta = {
        "resolved_label_columns": label_cols,
        "resolved_label_sources": task_source,
        "resolved_physical_columns": phys_cols,
        "id_column": resolve_id_column(catalog_df),
        "tasks": TASKS_TO_RUN,
        "model_path": MODEL_PATH,
        "model_family": MODEL_FAMILY,
        "official_model_name": OFFICIAL_MODEL_NAME,
    }
    with open(os.path.join(OUTPUT_DIR, "resolved_columns.json"), "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    for task in TASKS_TO_RUN:
        label_col = f"label_{task.replace('non', '')}" if task.startswith("non") else f"label_{task}"
        vals = master[label_col].to_numpy(dtype=float) if label_col in master.columns else np.array([])
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            log(f"[LabelCheck] {task}: source={task_source.get(task)} min={finite.min():.4f} max={finite.max():.4f} mean={finite.mean():.4f}")
        else:
            log(f"[LabelCheck] {task}: source={task_source.get(task)} no finite values")
    summary_rows, used_query_indices = [], set()
    for task in TASKS_TO_RUN:
        cfg = TASK_CONFIGS[task]
        qidx = choose_typical_query(master, emb_norm, task, cfg["target_value"], forbidden_indices=used_query_indices)
        if qidx is None:
            log(f"[跳过] 任务 {task}: 找不到合适 query"); continue
        used_query_indices.add(qidx)
        qfile = master.iloc[qidx]["file_path"]
        log(f"[{task}] 选中 query: idx={qidx}, file={os.path.basename(qfile)}")
        rr = retrieve_similar(master, emb_norm, phys_mat, qidx, task, topk=TOPK, candidate_pool=CANDIDATE_POOL, hard_filter=False)
        task_dir = os.path.join(OUTPUT_DIR, task); ensure_dir(task_dir)
        save_fits_png(qfile, os.path.join(task_dir, "query.png"))
        query_info = {"task": task, "query_index": int(qidx), "query_file": os.path.basename(qfile), "object_id": None if pd.isna(master.iloc[qidx]["_object_id"]) else int(master.iloc[qidx]["_object_id"])}
        for c in ["label_spiral", "label_smooth", "label_featured", "label_edgeon", "label_round", "label_cigar"]:
            if c in master.columns:
                v = master.iloc[qidx][c]; query_info[c] = None if pd.isna(v) else float(v)
        with open(os.path.join(task_dir, "query_info.json"), "w", encoding="utf-8") as f: json.dump(query_info, f, indent=2, ensure_ascii=False)
        rr.results.to_csv(os.path.join(task_dir, f"{task}_retrieval.csv"), index=False, encoding="utf-8-sig")
        create_retrieval_grid(qfile, rr.results, os.path.join(task_dir, f"{task}_grid.png"), f"{task} retrieval")
        for i, row in rr.results.iterrows():
            try:
                save_fits_png(row["file_path"], os.path.join(task_dir, f"rank_{i+1}_{basename_noext(row['file_name'])}.png"))
            except Exception as e:
                log(f"保存 {row['file_name']} PNG 失败: {e}")
        for i, row in rr.results.iterrows():
            summary_rows.append({
                "task": task, "query_index": int(qidx), "query_file": os.path.basename(qfile), "rank": i + 1,
                "result_file": row["file_name"], "object_id": row["_object_id"], "final_score": row["final_score"],
                "emb_score": row["emb_score"], "label_score": row["label_score"], "phys_score": row["phys_score"], "quality_score": row["quality_score"],
            })
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, "retrieval_summary.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        log(f"汇总已保存: {summary_csv}")
    log("========== Done ==========")

if __name__ == "__main__":
    main()
