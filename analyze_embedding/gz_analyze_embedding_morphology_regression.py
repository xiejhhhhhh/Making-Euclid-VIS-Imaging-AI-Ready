# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import os
import sys
import time
import json
import pickle
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import umap

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None

plt.switch_backend("Agg")

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Euclid_DINOv2_VIT"))
from euclid_dino.datasets.euclid_vis import EuclidVISDataset
from euclid_dino.models.dino_model import DINOModel


def sanitize_embeddings(emb):
    nan_count = np.isnan(emb).sum()
    inf_count = np.isinf(emb).sum()
    print(f"Checking embeddings... NaN={nan_count}, Inf={inf_count}")
    if nan_count > 0 or inf_count > 0:
        emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    return emb


def safe_torch_load(path, map_location=None, trusted=True):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if trusted and ("Weights only load failed" in msg or "weights_only" in msg):
            print("torch.load 默认 weights_only=True 加载失败，回退为 weights_only=False（仅对可信 checkpoint 使用）")
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


def resize_pos_embedding(pos_ckpt, pos_model):
    if pos_ckpt.shape == pos_model.shape:
        return pos_ckpt
    if pos_ckpt.ndim != 3 or pos_model.ndim != 3 or pos_ckpt.shape[-1] != pos_model.shape[-1]:
        return pos_model
    cls_ckpt = pos_ckpt[:, :1, :]
    patch_ckpt = pos_ckpt[:, 1:, :]
    old_n = patch_ckpt.shape[1]
    new_n = pos_model.shape[1] - 1
    old_hw = int(round(old_n ** 0.5))
    new_hw = int(round(new_n ** 0.5))
    if old_hw * old_hw != old_n or new_hw * new_hw != new_n:
        return pos_model
    patch_ckpt = patch_ckpt.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
    patch_ckpt = torch.nn.functional.interpolate(patch_ckpt, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
    patch_ckpt = patch_ckpt.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
    return torch.cat([cls_ckpt, patch_ckpt], dim=1)


def adapt_state_dict_for_custom_model(loaded_state_dict, model_state_dict):
    adapted = {}
    for k, v in loaded_state_dict.items():
        if k not in model_state_dict:
            adapted[k] = v
            continue
        mv = model_state_dict[k]
        if tuple(v.shape) == tuple(mv.shape):
            adapted[k] = v
        elif "pos_embedding" in k:
            print(f"插值位置嵌入: {k} {tuple(v.shape)} -> {tuple(mv.shape)}")
            adapted[k] = resize_pos_embedding(v, mv)
        else:
            print(f"跳过 shape 不一致参数: {k} {tuple(v.shape)} -> {tuple(mv.shape)}")
    return adapted


class OfficialDINOv2BackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict):
                for key in ["x_norm_clstoken", "cls_token"]:
                    if key in feats:
                        return feats[key]
                if "x_prenorm" in feats and torch.is_tensor(feats["x_prenorm"]) and feats["x_prenorm"].ndim == 3:
                    return feats["x_prenorm"][:, 0]
                if "x_norm_patchtokens" in feats and torch.is_tensor(feats["x_norm_patchtokens"]) and feats["x_norm_patchtokens"].ndim == 3:
                    return feats["x_norm_patchtokens"].mean(dim=1)
            if torch.is_tensor(feats):
                return feats[:, 0] if feats.ndim == 3 else feats
        out = self.model(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            for _, v in out.items():
                if torch.is_tensor(v):
                    return v[:, 0] if v.ndim == 3 else v
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    return v[:, 0] if v.ndim == 3 else v
        if torch.is_tensor(out):
            return out[:, 0] if out.ndim == 3 else out
        raise RuntimeError("Cannot extract embedding from official DINOv2 output.")


def infer_model_family(model_path, explicit_family="auto"):
    if explicit_family in {"custom", "official"}:
        return explicit_family
    name = os.path.basename(str(model_path)).lower()
    if any(k in name for k in ["official", "dinov2", "vits14", "vitb14", "vitl14"]):
        return "official"
    return "custom"


def build_custom_model(device, img_size=224):
    return DINOModel(model_type="s", patch_size=16, input_channels=1, img_size=img_size).to(device)


def build_official_model(device, model_name="dinov2_vits14"):
    print(f"通过 torch.hub 构建官方模型: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False, force_reload=False)
    return model.to(device).eval()


def load_custom_model_weights(model, model_path, device):
    ckpt = safe_torch_load(model_path, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(ckpt)
    print(f"解析自定义 state_dict 成功，source_key={source_key}")
    state = adapt_state_dict_for_custom_model(state, model.state_dict())
    msg = model.load_state_dict(state, strict=False)
    print(f"模型权重加载成功 missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model.eval()


def load_official_model_weights(model, model_path, device):
    ckpt = safe_torch_load(model_path, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(ckpt)
    print(f"解析 official state_dict 成功，source_key={source_key}")
    msg = model.load_state_dict(state, strict=False)
    print(f"官方模型权重加载成功 missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model.eval()


class EmbeddingCache:
    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, data_dir, model_path, batch_size, img_size, model_family="auto", official_model_name="dinov2_vits14"):
        s = f"{data_dir}_{model_path}_{batch_size}_{img_size}_{model_family}_{official_model_name}"
        return hashlib.md5(s.encode()).hexdigest()

    def get_cache_path(self, key, suffix):
        return os.path.join(self.cache_dir, key + suffix)

    def load_embeddings(self, data_dir, model_path, batch_size, img_size, model_family="auto", official_model_name="dinov2_vits14"):
        key = self._get_cache_key(data_dir, model_path, batch_size, img_size, model_family, official_model_name)
        emb_path = self.get_cache_path(key, "_embeddings.npy")
        file_path = self.get_cache_path(key, "_filenames.pkl")
        if os.path.exists(emb_path) and os.path.exists(file_path):
            print("Loading cached embeddings")
            emb = np.load(emb_path)
            with open(file_path, "rb") as f:
                files = pickle.load(f)
            return emb, files, key
        return None, None, key

    def save_embeddings(self, emb, files, key):
        np.save(self.get_cache_path(key, "_embeddings.npy"), emb)
        with open(self.get_cache_path(key, "_filenames.pkl"), "wb") as f:
            pickle.dump(files, f)
        print("Embeddings cached")


def load_catalog(catalog_csv, catalog_fits):
    if catalog_csv and os.path.exists(catalog_csv):
        print("Loading CSV catalog")
        return pd.read_csv(catalog_csv)
    print("Loading FITS catalog")
    with fits.open(catalog_fits) as hdul:
        return pd.DataFrame(hdul[1].data)


def find_id_column(cat):
    for col in ["OBJECT_ID", "object_id", "Object_ID"]:
        if col in cat.columns:
            return col
    raise KeyError("catalog 中没有 OBJECT_ID / object_id 列")


def parse_object_ids_from_files(file_names):
    out = []
    for f in file_names:
        try:
            out.append(int(os.path.basename(f).split("_")[0]))
        except Exception:
            out.append(None)
    return out


def build_catalog_index(cat):
    id_col = find_id_column(cat)
    idx = cat.drop_duplicates(subset=[id_col], keep="first").set_index(id_col, drop=False)
    return idx, id_col


def robust_read_image(data_root, file_name):
    p = file_name if os.path.isabs(file_name) else os.path.join(data_root, file_name)
    if not os.path.exists(p):
        alt = os.path.join(data_root, "VIS", file_name)
        if os.path.exists(alt):
            p = alt
    with fits.open(p, memmap=False) as hdul:
        x = np.asarray(hdul[0].data, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"图像不是二维: {file_name}, shape={x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_image_for_display(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo):
        lo = float(np.nanmin(x))
    if not np.isfinite(hi):
        hi = float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def get_morphology_features(cat, file_names):
    object_ids = parse_object_ids_from_files(file_names)
    features = []
    id_column = "OBJECT_ID" if "OBJECT_ID" in cat.columns else ("object_id" if "object_id" in cat.columns else None)
    if id_column is None:
        for _ in object_ids:
            features.append(dict(ellipticity=np.nan, kron_radius=np.nan, mumax_minus_mag=np.nan, mu_max=np.nan))
        return features
    for obj in object_ids:
        row = cat[cat[id_column] == obj]
        if len(row) == 0:
            features.append(dict(ellipticity=np.nan, kron_radius=np.nan, mumax_minus_mag=np.nan, mu_max=np.nan))
            continue
        r = row.iloc[0]
        def get_value(name):
            for case in [name, name.lower(), name.upper()]:
                if case in r:
                    return r[case]
            return np.nan
        features.append(dict(
            ellipticity=get_value("ELLIPTICITY"),
            kron_radius=get_value("KRON_RADIUS"),
            mumax_minus_mag=get_value("mumax_minus_mag"),
            mu_max=get_value("MU_MAX"),
        ))
    return features


def get_morphology_labels(cat, file_names, label_cols=("morphology_label", "morphology_main"), ignore_labels=("uncertain",)):
    object_ids = parse_object_ids_from_files(file_names)
    idx_cat, _ = build_catalog_index(cat)
    label_col = None
    for c in label_cols:
        if c in idx_cat.columns:
            label_col = c
            break
    if label_col is None:
        raise KeyError(f"catalog 中缺少形态标签列，候选列: {label_cols}")
    labels = []
    matched = 0
    for oid in object_ids:
        if oid is None or oid not in idx_cat.index:
            labels.append(None); continue
        v = idx_cat.loc[oid, label_col]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        if pd.isna(v):
            labels.append(None); continue
        s = str(v).strip()
        if s in ignore_labels:
            labels.append(None); continue
        labels.append(s); matched += 1
    print(f"Morphology labels matched: {matched}/{len(file_names)} using column={label_col}")
    return np.array(labels, dtype=object), label_col


def make_custom_transform():
    def transform(x):
        x = np.asarray(x, dtype=np.float32)
        mx = np.max(x)
        x = x / mx if np.isfinite(mx) and mx > 0 else np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    return transform


def make_official_transform():
    def transform(x):
        x = np.asarray(x, dtype=np.float32)
        mx = np.max(x)
        x = x / mx if np.isfinite(mx) and mx > 0 else np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return transform


def extract_embeddings(data_root, model_path, img_size, batch_size, model_family="auto", official_model_name="dinov2_vits14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model_family = infer_model_family(model_path, model_family)
    print(f"推断模型类型: {model_family}")
    if model_family == "official":
        model = load_official_model_weights(build_official_model(device=device, model_name=official_model_name), model_path, device)
        transform = make_official_transform()
        backbone_wrapper = OfficialDINOv2BackboneWrapper(model)
        print("官方 DINOv2 使用灰度图复制为 3 通道输入")
    else:
        model = load_custom_model_weights(build_custom_model(device=device, img_size=img_size), model_path, device)
        transform = make_custom_transform()
        backbone_wrapper = None
        print("自定义 DINOModel 使用单通道输入")
    dataset = EuclidVISDataset(root=data_root, transform=transform, img_size=img_size)
    print(f"数据集创建成功，包含 {len(dataset)} 个样本")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, file_names = [], []
    with torch.no_grad():
        print("开始提取嵌入...")
        for i, imgs in enumerate(loader):
            print(f"处理批次 {i+1}/{len(loader)}")
            imgs = imgs.to(device)
            emb = backbone_wrapper(imgs) if model_family == "official" else model(imgs)[1]
            embeddings.append(emb.cpu().numpy())
            if hasattr(dataset, "files"):
                for f in dataset.files[i * batch_size : (i + 1) * batch_size]:
                    file_names.append(os.path.basename(f))
    if len(embeddings) == 0:
        return np.array([]), []
    embeddings = sanitize_embeddings(np.concatenate(embeddings))
    print(f"嵌入提取完成，形状: {embeddings.shape}")
    return embeddings, file_names


def cosine_similarity_matrix(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x = x / norms
    return x @ x.T


def perform_kmeans_clustering(embeddings, output_dir=None, n_clusters=5):
    print(f"Running K-means clustering in high-dim embedding space with {n_clusters} clusters")
    X = StandardScaler().fit_transform(embeddings)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    if output_dir is not None:
        emb2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42, verbose=True).fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb2[:, 0], emb2[:, 1], c=cluster_labels, s=15, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Cluster Label')
        plt.title(f'K-means Clustering (k={n_clusters})')
        plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'kmeans_clusters_{n_clusters}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        np.savez(os.path.join(output_dir, "kmeans_clusters.npz"), cluster_labels=cluster_labels, umap_embeddings=emb2, n_clusters=n_clusters)
    return cluster_labels


def plot_umap(embeddings, props, output_dir):
    embeddings = sanitize_embeddings(embeddings)
    physical_params = [
        ('ELLIPTICITY', np.array([p["ellipticity"] for p in props], dtype=float), 'viridis', 'ELLIPTICITY'),
        ('KRON_RADIUS', np.array([p["kron_radius"] for p in props], dtype=float), 'plasma', 'KRON_RADIUS'),
        ('MU_MAX', np.array([p["mu_max"] for p in props], dtype=float), 'inferno', 'MU_MAX'),
        ('MUMAX_MINUS_MAG', np.array([p["mumax_minus_mag"] for p in props], dtype=float), 'magma', 'MUMAX_MINUS_MAG')
    ]
    emb2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42, verbose=True).fit_transform(embeddings)
    emb3 = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42, verbose=True).fit_transform(embeddings)
    np.savez(os.path.join(output_dir, "umap_embeddings.npz"), umap_2d=emb2, umap_3d=emb3)
    view_angles = [(20, 45), (60, 45), (20, 135), (20, 225)]
    ranges = {}
    for name, values, cmap, _ in physical_params:
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            ranges[name] = (0.0, 1.0, cmap); continue
        q1, q3 = np.percentile(valid, [25, 75]); iqr = q3 - q1
        vmin = max(q1 - 1.5 * iqr, np.min(valid)); vmax = min(q3 + 1.5 * iqr, np.max(valid))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.min(valid)); vmax = float(np.max(valid) + 1e-8)
        ranges[name] = (vmin, vmax, cmap)
    for name, values, cmap, title in physical_params:
        plt.figure(figsize=(10, 8))
        plt.scatter(emb2[:, 0], emb2[:, 1], c=values, s=1, cmap=cmap, vmin=ranges[name][0], vmax=ranges[name][1])
        plt.colorbar(label=title); plt.title(f'UMAP - {title}'); plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
        plt.savefig(os.path.join(output_dir, f"umap_{name.lower()}.png")); plt.close()
    for name, values, cmap, title in physical_params:
        fig = plt.figure(figsize=(20, 15))
        for i, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            sc = ax.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], c=values, s=10, alpha=0.7, cmap=cmap, vmin=ranges[name][0], vmax=ranges[name][1])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'3D UMAP (elev={elev}, azim={azim})')
            ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); ax.set_zlabel('UMAP 3')
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]); fig.colorbar(sc, cax=cax).set_label(title)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(output_dir, f'umap_3d_{name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return emb2, emb3


def plot_umap_with_morphology_labels(embeddings, labels_raw, output_dir, title_suffix="morphology"):
    valid = np.array([x is not None and str(x).strip() != "" for x in labels_raw], dtype=bool)
    if valid.sum() < 10:
        print("跳过 morphology-labeled UMAP：有效标签太少"); return None
    X = embeddings[valid]; y = np.array(labels_raw, dtype=object)[valid].astype(str)
    emb2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42, verbose=True).fit_transform(X)
    classes = sorted(pd.Series(y).unique().tolist())
    plt.figure(figsize=(10, 8))
    for cls in classes:
        m = (y == cls)
        plt.scatter(emb2[m, 0], emb2[m, 1], s=6, alpha=0.65, label=cls)
    plt.title(f"UMAP colored by {title_suffix}"); plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.legend(markerscale=2, fontsize=8, frameon=True); plt.tight_layout()
    out = os.path.join(output_dir, f"umap_labels_{title_suffix}.png")
    plt.savefig(out, dpi=260, bbox_inches="tight"); plt.close()
    print(f"标签着色 UMAP 已保存到 {out}")
    return emb2


def safe_corr(y_true, y_pred):
    out = {"pearson": None, "spearman": None}
    if pearsonr is not None:
        try: out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception: pass
    if spearmanr is not None:
        try: out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
        except Exception: pass
    return out


def run_physical_regression_probe(emb, props, output_dir):
    feature_map = {
        "ELLIPTICITY": np.array([p["ellipticity"] for p in props], dtype=float),
        "KRON_RADIUS": np.array([p["kron_radius"] for p in props], dtype=float),
        "MU_MAX": np.array([p["mu_max"] for p in props], dtype=float),
        "MUMAX_MINUS_MAG": np.array([p["mumax_minus_mag"] for p in props], dtype=float),
    }
    all_results = []
    for feature_name, y in feature_map.items():
        valid = np.isfinite(y)
        X_valid, y_valid = emb[valid], y[valid]
        if len(y_valid) < 50:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
        scaler = StandardScaler(); X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        reg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)
        corr = safe_corr(y_test, y_pred)
        result = {"target_feature": feature_name, "n_samples": int(len(y_valid)), "alpha": float(reg.alpha_), "metrics": {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "pearson": corr["pearson"], "spearman": corr["spearman"],
        }}
        all_results.append(result)
        plt.figure(figsize=(7, 7))
        plt.scatter(y_test, y_pred, s=6, alpha=0.35)
        y_min = float(np.nanmin([np.nanmin(y_test), np.nanmin(y_pred)]))
        y_max = float(np.nanmax([np.nanmax(y_test), np.nanmax(y_pred)]))
        plt.plot([y_min, y_max], [y_min, y_max], linestyle="--")
        plt.xlabel("True"); plt.ylabel("Pred"); plt.title(f"{feature_name} regression probe")
        plt.savefig(os.path.join(output_dir, f"physical_regression_scatter_{feature_name.lower()}.png"), dpi=200, bbox_inches="tight")
        plt.close()
    if all_results:
        with open(os.path.join(output_dir, "physical_regression_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        pd.DataFrame([{
            "feature": item["target_feature"], "n_samples": item["n_samples"], "alpha": item["alpha"],
            "r2": item["metrics"]["r2"], "mae": item["metrics"]["mae"], "rmse": item["metrics"]["rmse"],
            "pearson": item["metrics"]["pearson"], "spearman": item["metrics"]["spearman"],
            "embedding_dim": int(emb.shape[1]),
        } for item in all_results]).to_csv(os.path.join(output_dir, "physical_regression_results.csv"), index=False)
    return all_results


def plot_confusion_matrix(cm, labels, out_path, normalize=False):
    cm_plot = cm.astype(float)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True); row_sum[row_sum == 0] = 1.0; cm_plot = cm_plot / row_sum
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label"); plt.xlabel("Pred label")
    plt.title("Normalized confusion matrix" if normalize else "Confusion matrix")
    fmt = ".2f" if normalize else "d"; thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            plt.text(j, i, format(cm_plot[i, j], fmt), ha="center", color="white" if cm_plot[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()


def run_morphology_probe(emb, cat, file_names, output_dir, label_col_candidates=("morphology_label", "morphology_main"), ignore_labels=("uncertain",)):
    labels_raw, used_label_col = get_morphology_labels(cat=cat, file_names=file_names, label_cols=label_col_candidates, ignore_labels=ignore_labels)
    valid = np.array([x is not None and str(x).strip() != "" for x in labels_raw], dtype=bool)
    X, y_text = emb[valid], labels_raw[valid].astype(str)
    if len(y_text) < 50:
        return None
    label_counts = pd.Series(y_text).value_counts().sort_index()
    if label_counts.shape[0] < 2:
        return None
    le = LabelEncoder(); y = le.fit_transform(y_text); class_names = list(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", multi_class="auto").fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s); y_prob = clf.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred); bacc = balanced_accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    macro_ovr_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        macro_ovr_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    result = {
        "label_column": used_label_col, "ignore_labels": list(ignore_labels), "n_samples": int(len(y)),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)), "n_classes": int(len(class_names)),
        "classes": class_names, "class_counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
        "metrics": {"accuracy": float(acc), "balanced_accuracy": float(bacc), "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1), "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1), "macro_ovr_auc": macro_ovr_auc},
        "confusion_matrix": cm.tolist(), "classification_report": report,
    }
    with open(os.path.join(output_dir, "morphology_probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame([{
        "label_column": used_label_col, "n_samples": int(len(y)), "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_classes": int(len(class_names)), "accuracy": float(acc), "balanced_accuracy": float(bacc), "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r), "weighted_f1": float(weighted_f1), "macro_ovr_auc": macro_ovr_auc, "embedding_dim": int(emb.shape[1]),
    }]).to_csv(os.path.join(output_dir, "morphology_probe_results.csv"), index=False)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report_morphology.csv"), encoding="utf-8-sig")
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix_morphology.png"), normalize=False)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix_morphology_norm.png"), normalize=True)
    return result


def select_query_indices(file_names, labels_raw=None, n_queries=8):
    n = len(file_names)
    if n == 0:
        return []
    if labels_raw is None:
        step = max(n // n_queries, 1)
        return list(range(0, n, step))[:n_queries]
    valid = [(i, labels_raw[i]) for i in range(n) if labels_raw[i] is not None]
    if len(valid) == 0:
        step = max(n // n_queries, 1)
        return list(range(0, n, step))[:n_queries]
    by_class = {}
    for i, lab in valid:
        by_class.setdefault(str(lab), []).append(i)
    out = []
    for cls in sorted(by_class.keys()):
        out.append(by_class[cls][0])
        if len(out) >= n_queries:
            break
    if len(out) < n_queries:
        remaining = [i for i in range(n) if i not in out]
        step = max(len(remaining) // max(n_queries - len(out), 1), 1)
        out.extend(remaining[::step][: max(0, n_queries - len(out))])
    return out[:n_queries]


def create_retrieval_examples(embeddings, file_names, data_root, output_dir, labels_raw=None, topk=5, n_queries=8):
    if len(file_names) == 0 or len(embeddings) == 0:
        return None
    sim = cosine_similarity_matrix(embeddings)
    query_indices = select_query_indices(file_names, labels_raw=labels_raw, n_queries=n_queries)
    retrieval_dir = os.path.join(output_dir, "retrieval_examples"); os.makedirs(retrieval_dir, exist_ok=True)
    summary_rows = []
    for q_idx in query_indices:
        sims = sim[q_idx].copy(); sims[q_idx] = -1.0
        nn_idx = np.argsort(-sims)[:topk]
        fig, axes = plt.subplots(1, topk + 1, figsize=(3 * (topk + 1), 3.2))
        indices = [q_idx] + nn_idx.tolist(); titles = ["query"] + [f"top{i}" for i in range(1, topk + 1)]
        for ax, idx, title in zip(axes, indices, titles):
            img = normalize_image_for_display(robust_read_image(data_root, file_names[idx]))
            ax.imshow(img, cmap="gray")
            label_txt = ""
            if labels_raw is not None and idx < len(labels_raw) and labels_raw[idx] is not None:
                label_txt = f"\n{labels_raw[idx]}"
            ax.set_title(f"{title}{label_txt}" if title == "query" else f"{title} s={sim[q_idx, idx]:.3f}{label_txt}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(retrieval_dir, f"retrieval_query_{q_idx:05d}.png"), dpi=220, bbox_inches="tight")
        plt.close()
        row = {"query_index": int(q_idx), "query_file": file_names[q_idx], "neighbors": [
            {"rank": int(r+1), "index": int(idx), "file": file_names[idx], "similarity": float(sim[q_idx, idx]),
             "label": None if labels_raw is None or labels_raw[idx] is None else str(labels_raw[idx])}
            for r, idx in enumerate(nn_idx.tolist())]}
        if labels_raw is not None and labels_raw[q_idx] is not None:
            row["query_label"] = str(labels_raw[q_idx])
        summary_rows.append(row)
    with open(os.path.join(retrieval_dir, "retrieval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"Retrieval 示例图已保存到 {retrieval_dir}")
    return summary_rows


def create_anomaly_examples(embeddings, file_names, data_root, output_dir, labels_raw=None, n_examples=25):
    if len(file_names) == 0 or len(embeddings) == 0:
        return None
    sim = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -1.0)
    max_neighbor_sim = np.max(sim, axis=1)
    anomaly_score = 1.0 - max_neighbor_sim
    order = np.argsort(-anomaly_score)[:n_examples]
    n_cols = 5; n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows))
    axes = np.array(axes).reshape(-1)
    rows = []
    for ax_i, idx in enumerate(order):
        ax = axes[ax_i]
        img = normalize_image_for_display(robust_read_image(data_root, file_names[idx]))
        ax.imshow(img, cmap="gray")
        title = f"score={anomaly_score[idx]:.3f}"
        if labels_raw is not None and labels_raw[idx] is not None:
            title += f"\n{labels_raw[idx]}"
        ax.set_title(title, fontsize=8); ax.axis("off")
        rows.append({"index": int(idx), "file": file_names[idx], "anomaly_score": float(anomaly_score[idx]), "nearest_neighbor_similarity": float(max_neighbor_sim[idx]), "label": None if labels_raw is None or labels_raw[idx] is None else str(labels_raw[idx])})
    for j in range(len(order), len(axes)):
        axes[j].axis("off")
    out_path = os.path.join(output_dir, "anomaly_gallery.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()
    with open(os.path.join(output_dir, "anomaly_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Anomaly 示例图已保存到 {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\datasets\gz_morphology_catalogue\VIS")
    parser.add_argument("--model-path", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_official_dinov2\dinov2_vits14_official_backbone.pth")
    parser.add_argument("--catalog-csv", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv")
    parser.add_argument("--catalog-fits", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue.fits")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="gz_outputs_analysis")
    parser.add_argument("--morph-label-cols", type=str, default="morphology_label,morphology_main")
    parser.add_argument("--ignore-labels", type=str, default="uncertain")
    parser.add_argument("--model-family", type=str, default="auto", choices=["auto", "custom", "official"])
    parser.add_argument("--official-model-name", type=str, default="dinov2_vits14")
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-nqueries", type=int, default=8)
    parser.add_argument("--anomaly-nexamples", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache = EmbeddingCache()
    emb, files, key = cache.load_embeddings(args.data_root, args.model_path, args.batch_size, args.img_size, args.model_family, args.official_model_name)
    if emb is not None:
        emb = sanitize_embeddings(emb)
    if emb is None:
        emb, files = extract_embeddings(args.data_root, args.model_path, args.img_size, args.batch_size, model_family=args.model_family, official_model_name=args.official_model_name)
        cache.save_embeddings(emb, files, key)

    cat = load_catalog(args.catalog_csv, args.catalog_fits)
    props = get_morphology_features(cat, files)
    plot_umap(emb, props, args.output_dir)

    labels_raw = None
    used_label_col = None
    try:
        label_cols = tuple([x.strip() for x in args.morph_label_cols.split(",") if x.strip()])
        ignore_labels = tuple([x.strip() for x in args.ignore_labels.split(",") if x.strip()])
        labels_raw, used_label_col = get_morphology_labels(cat, files, label_cols=label_cols, ignore_labels=ignore_labels)
        plot_umap_with_morphology_labels(emb, labels_raw, args.output_dir, title_suffix=used_label_col)
    except Exception as e:
        print("标签着色 UMAP 失败或跳过：", str(e))

    try:
        run_physical_regression_probe(emb, props, args.output_dir)
    except Exception as e:
        print("physical regression probe 失败或跳过：", str(e))

    try:
        run_morphology_probe(
            emb=emb,
            cat=cat,
            file_names=files,
            output_dir=args.output_dir,
            label_col_candidates=tuple([x.strip() for x in args.morph_label_cols.split(",") if x.strip()]),
            ignore_labels=tuple([x.strip() for x in args.ignore_labels.split(",") if x.strip()]),
        )
    except Exception as e:
        print("morphology probe 失败或跳过：", str(e))

    perform_kmeans_clustering(emb, args.output_dir, n_clusters=5)

    try:
        create_retrieval_examples(emb, files, args.data_root, args.output_dir, labels_raw=labels_raw, topk=args.retrieval_topk, n_queries=args.retrieval_nqueries)
    except Exception as e:
        print("retrieval 示例图生成失败或跳过：", str(e))

    try:
        create_anomaly_examples(emb, files, args.data_root, args.output_dir, labels_raw=labels_raw, n_examples=args.anomaly_nexamples)
    except Exception as e:
        print("anomaly 示例图生成失败或跳过：", str(e))

    workflow_summary = {
        "model_path": args.model_path,
        "model_family": infer_model_family(args.model_path, args.model_family),
        "official_model_name": args.official_model_name,
        "n_embeddings": int(len(emb)),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else None,
        "output_dir": args.output_dir,
        "artifacts": {
            "umap_embeddings": os.path.join(args.output_dir, "umap_embeddings.npz"),
            "kmeans_clusters": os.path.join(args.output_dir, "kmeans_clusters.npz"),
            "retrieval_dir": os.path.join(args.output_dir, "retrieval_examples"),
            "anomaly_gallery": os.path.join(args.output_dir, "anomaly_gallery.png"),
            "morphology_probe_results": os.path.join(args.output_dir, "morphology_probe_results.json"),
            "physical_regression_results": os.path.join(args.output_dir, "physical_regression_results.json"),
        },
        "note": "AI-ready workflow: embedding extraction + UMAP + retrieval + anomaly analysis",
    }
    with open(os.path.join(args.output_dir, "ai_ready_workflow_summary.json"), "w", encoding="utf-8") as f:
        json.dump(workflow_summary, f, indent=2, ensure_ascii=False)

    print("Now run:")
    print("process_outliers.py")
    print("copy_outlier_files.py")


if __name__ == "__main__":
    main()
