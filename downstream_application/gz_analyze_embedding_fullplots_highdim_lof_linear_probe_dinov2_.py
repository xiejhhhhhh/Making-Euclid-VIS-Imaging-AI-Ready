# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import sys
import os
import time
import json
import pickle
import hashlib
import argparse
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from torch.utils.data import DataLoader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    r2_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Euclid_DINOv2_VIT'
    )
)

plt.switch_backend('Agg')

from euclid_dino.datasets.euclid_vis import EuclidVISDataset

try:
    import seaborn as sns
except Exception:
    sns = None


DEFAULT_OUTPUT_DIR = r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\outputs_official_dinov2_analysis'
DEFAULT_DATA_ROOT = r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\datasets\gz_morphology_catalogue\VIS'
DEFAULT_MODEL_PATH = r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_official_dinov2\dinov2_vits14_official_backbone.pth'
DEFAULT_CATALOG_PATH = r'C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv'
DEFAULT_OFFICIAL_MODEL_NAME = 'dinov2_vits14'


class EmbeddingCache:
    def __init__(self, cache_dir='./embedding_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, data_dir, model_path, batch_size=16, img_size=224, model_family='official', official_model_name='dinov2_vits14'):
        cache_str = f"{data_dir}_{model_path}_{batch_size}_{img_size}_{model_family}_{official_model_name}"
        try:
            mtime = os.path.getmtime(data_dir)
            cache_str += f"_{mtime}"
        except Exception:
            pass
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        return cache_key

    def get_cache_path(self, cache_key, suffix=''):
        return os.path.join(self.cache_dir, f"{cache_key}{suffix}")

    def load_embeddings(self, data_dir, model_path, batch_size=16, img_size=224, model_family='official', official_model_name='dinov2_vits14'):
        cache_key = self._get_cache_key(data_dir, model_path, batch_size, img_size, model_family, official_model_name)
        emb_cache_path = self.get_cache_path(cache_key, '_embeddings.npy')
        files_cache_path = self.get_cache_path(cache_key, '_filenames.pkl')
        meta_cache_path = self.get_cache_path(cache_key, '_metadata.pkl')

        if os.path.exists(emb_cache_path) and os.path.exists(files_cache_path):
            print(f"加载缓存的嵌入数据: {cache_key}")
            if os.path.exists(meta_cache_path):
                with open(meta_cache_path, 'rb') as f:
                    metadata = pickle.load(f)
                current_metadata = {
                    'data_dir': data_dir,
                    'model_path': model_path,
                    'batch_size': batch_size,
                    'img_size': img_size,
                    'model_family': model_family,
                    'official_model_name': official_model_name,
                }
                for key in current_metadata.keys():
                    if metadata.get(key) != current_metadata[key]:
                        print(f"参数 {key} 已更改，重新计算嵌入...")
                        return None, None, None

            embeddings = np.load(emb_cache_path)
            with open(files_cache_path, 'rb') as f:
                file_names = pickle.load(f)
            print(f"  嵌入形状: {embeddings.shape}, 文件数: {len(file_names)}")
            return embeddings, file_names, cache_key

        print("未找到缓存，需要重新计算嵌入")
        return None, None, cache_key

    def save_embeddings(self, embeddings, file_names, cache_key, metadata):
        emb_cache_path = self.get_cache_path(cache_key, '_embeddings.npy')
        files_cache_path = self.get_cache_path(cache_key, '_filenames.pkl')
        meta_cache_path = self.get_cache_path(cache_key, '_metadata.pkl')

        np.save(emb_cache_path, embeddings)
        with open(files_cache_path, 'wb') as f:
            pickle.dump(file_names, f)
        metadata['timestamp'] = time.time()
        with open(meta_cache_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"嵌入已保存到缓存: {cache_key}")


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


class OfficialDINOv2BackboneWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    return feats["x_norm_clstoken"]
                if "cls_token" in feats:
                    return feats["cls_token"]
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


def build_official_model(device, model_name='dinov2_vits14'):
    print(f"通过 torch.hub 构建官方模型: {model_name}")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=False,
        force_reload=False,
    )
    model = model.to(device).eval()
    return model


def load_official_backbone(model_path, device, official_model_name='dinov2_vits14'):
    model = build_official_model(device, official_model_name)
    checkpoint = safe_torch_load(model_path, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(checkpoint)
    print(f"解析 official checkpoint 成功，source_key={source_key}")
    msg = model.load_state_dict(state, strict=False)
    print(f"官方模型权重加载成功 missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("missing_keys sample:", msg.missing_keys[:20])
    if len(msg.unexpected_keys) > 0:
        print("unexpected_keys sample:", msg.unexpected_keys[:20])
    return OfficialDINOv2BackboneWrapper(model).to(device).eval()


def load_catalog_data(catalog_path):
    print(f"加载星表数据: {catalog_path}")
    if catalog_path.endswith('.fits'):
        with fits.open(catalog_path) as hdul:
            data = hdul[1].data
        return data
    elif catalog_path.endswith('.csv'):
        return pd.read_csv(catalog_path)
    else:
        raise ValueError(f"不支持的星表格式: {catalog_path}")


def parse_object_id_from_filename(file_name):
    base = os.path.basename(file_name)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    for p in parts:
        try:
            return int(p)
        except Exception:
            continue
    digits = ''.join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except Exception:
            return None
    return None


def build_vis_object_id_map(vis_dir):
    mapping = {}
    for path in sorted(os.listdir(vis_dir)):
        if not path.lower().endswith('.fits'):
            continue
        obj_id = parse_object_id_from_filename(path)
        if obj_id is not None:
            mapping[obj_id] = os.path.join(vis_dir, path)
    print(f"VIS object_id 映射数量: {len(mapping)}")
    return mapping


def read_first_image_from_fits(fits_path):
    with fits.open(fits_path, lazy_load_hdus=False, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                x = np.array(hdu.data, dtype=np.float32)
                x = np.squeeze(x)
                if x.ndim == 2:
                    return x
    raise ValueError(f"无法从 FITS 中读取二维图像: {fits_path}")


def normalize_for_display(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if not np.isfinite(lo):
        lo = np.nanmin(x)
    if not np.isfinite(hi):
        hi = np.nanmax(x)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y


def get_catalog_value(row, candidates, is_record):
    for col in candidates:
        if is_record:
            if col in row.dtype.names:
                return getattr(row, col)
        else:
            if col in row:
                return row[col]
    return np.nan


def get_catalog_row_by_object_id(catalog_data, obj_id):
    if isinstance(catalog_data, pd.DataFrame):
        for id_col in ['OBJECT_ID', 'SOURCE_ID', 'source_id', 'object_id', 'ID', 'id']:
            if id_col in catalog_data.columns:
                row = catalog_data[catalog_data[id_col] == obj_id]
                if not row.empty:
                    return row.iloc[0], False
    else:
        for id_col in ['OBJECT_ID', 'SOURCE_ID', 'source_id', 'object_id', 'ID', 'id']:
            if id_col in catalog_data.dtype.names:
                mask = catalog_data[id_col] == obj_id
                if np.any(mask):
                    return catalog_data[mask][0], True
    return None, None


def get_physical_properties(catalog_data, file_names):
    print(f"从星表中获取物理属性，共处理 {len(file_names)} 个文件")
    object_ids = [parse_object_id_from_filename(f) for f in file_names]
    print(f"前5个提取的OBJECT_ID: {object_ids[:5]}")

    physical_properties = []

    for obj_id in object_ids:
        if obj_id is None:
            physical_properties.append({
                'segmentation_area': np.nan,
                'semimajor_axis': np.nan,
                'kron_radius': np.nan,
                'mu_max': np.nan,
                'mumax_minus_mag': np.nan,
                'psf_dominance': np.nan,
                'growth': np.nan,
                'blended_prob': np.nan,
                'spurious_prob': np.nan,
                'SNR': np.nan
            })
            continue

        row, is_record = get_catalog_row_by_object_id(catalog_data, obj_id)
        if row is None:
            physical_properties.append({
                'segmentation_area': np.nan,
                'semimajor_axis': np.nan,
                'kron_radius': np.nan,
                'mu_max': np.nan,
                'mumax_minus_mag': np.nan,
                'psf_dominance': np.nan,
                'growth': np.nan,
                'blended_prob': np.nan,
                'spurious_prob': np.nan,
                'SNR': np.nan
            })
            continue

        segmentation_area = get_catalog_value(row, ['SEGMENTATION_AREA', 'segmentation_area', 'ISOAREA_IMAGE'], is_record)
        semimajor_axis = get_catalog_value(row, ['SEMIMAJOR_AXIS', 'semimajor_axis', 'A_IMAGE'], is_record)
        kron_radius = get_catalog_value(row, ['KRON_RADIUS', 'kron_radius'], is_record)
        mu_max = get_catalog_value(row, ['MU_MAX', 'mu_max'], is_record)
        mumax_minus_mag = get_catalog_value(row, ['MUMAX_MINUS_MAG', 'mumax_minus_mag', 'MUMINUSMAG'], is_record)

        flux_1fwhm = get_catalog_value(row, ['FLUX_VIS_1FWHM_APER'], is_record)
        flux_4fwhm = get_catalog_value(row, ['FLUX_VIS_4FWHM_APER'], is_record)

        psf_dominance = np.nan
        growth = np.nan
        if np.isfinite(flux_4fwhm) and flux_4fwhm > 0:
            psf_dominance = flux_1fwhm / flux_4fwhm
        if np.isfinite(flux_1fwhm) and flux_1fwhm > 0:
            growth = flux_4fwhm / flux_1fwhm

        blended_prob = get_catalog_value(row, ['BLENDED_PROB', 'blended_prob'], is_record)
        spurious_prob = get_catalog_value(row, ['SPURIOUS_PROB', 'spurious_prob'], is_record)
        snr = get_catalog_value(row, ['SNR', 'snr'], is_record)

        physical_properties.append({
            'segmentation_area': segmentation_area,
            'semimajor_axis': semimajor_axis,
            'kron_radius': kron_radius,
            'mu_max': mu_max,
            'mumax_minus_mag': mumax_minus_mag,
            'psf_dominance': psf_dominance,
            'growth': growth,
            'blended_prob': blended_prob,
            'spurious_prob': spurious_prob,
            'SNR': snr
        })

    return physical_properties


def safe_fill_nan(arr, default):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, default, dtype=float)
    med = np.nanmedian(arr)
    return np.nan_to_num(arr, nan=med)


def save_outlier_catalog_csv(output_dir, file_names, both_methods_outlier, both_methods_outlier_indices, lof_scores_positive, outlier_clusters):
    object_ids = [parse_object_id_from_filename(f) for f in file_names]
    rows = []
    for idx in both_methods_outlier_indices:
        rows.append({
            'index': int(idx),
            'object_id': None if object_ids[idx] is None else int(object_ids[idx]),
            'file_name': file_names[idx],
            'lof_score': float(lof_scores_positive[idx]),
            'outlier_cluster': int(outlier_clusters[idx]),
            'is_outlier': bool(both_methods_outlier[idx]),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'outlier_catalog.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"异常点 CSV 已保存到 {csv_path}")
    return csv_path


def visualize_outliers_from_csv(outlier_csv_path, vis_dir, output_dir, max_examples=100):
    print(f"根据异常点 CSV 可视化 FITS: {outlier_csv_path}")
    df = pd.read_csv(outlier_csv_path)
    if 'object_id' not in df.columns:
        print("CSV 中没有 object_id 列，跳过 FITS 可视化")
        return

    vis_map = build_vis_object_id_map(vis_dir)
    vis_out_dir = os.path.join(output_dir, 'outlier_fits_gallery')
    os.makedirs(vis_out_dir, exist_ok=True)

    df = df.dropna(subset=['object_id']).copy()
    if len(df) == 0:
        print("异常点 CSV 中没有有效 object_id")
        return

    df['object_id'] = df['object_id'].astype(int)
    df = df.sort_values(by='lof_score', ascending=False).head(max_examples).reset_index(drop=True)

    found_rows = []
    panel_paths = []

    for _, row in df.iterrows():
        oid = int(row['object_id'])
        fits_path = vis_map.get(oid)
        if fits_path is None:
            continue

        img = read_first_image_from_fits(fits_path)
        img = normalize_for_display(img)

        png_name = f"object_{oid}_cluster_{int(row.get('outlier_cluster', 0))}.png"
        png_path = os.path.join(vis_out_dir, png_name)

        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray')
        plt.title(f"OBJECT_ID={oid}\nLOF={row['lof_score']:.3f}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(png_path, dpi=180, bbox_inches='tight')
        plt.close()

        found_rows.append({
            'object_id': oid,
            'fits_path': fits_path,
            'png_path': png_path,
            'lof_score': float(row['lof_score']),
            'outlier_cluster': int(row.get('outlier_cluster', 0)),
        })
        panel_paths.append((oid, png_path, float(row['lof_score']), int(row.get('outlier_cluster', 0))))

    found_csv = os.path.join(output_dir, 'outlier_catalog_with_vis_paths.csv')
    pd.DataFrame(found_rows).to_csv(found_csv, index=False, encoding='utf-8-sig')
    print(f"带 VIS 路径的异常点 CSV 已保存到 {found_csv}")

    if len(panel_paths) > 0:
        n = len(panel_paths)
        ncols = 5
        nrows = math.ceil(n / ncols)
        fig = plt.figure(figsize=(3.1 * ncols, 3.2 * nrows))
        for i, (oid, png_path, lof_score, cluster_id) in enumerate(panel_paths, 1):
            ax = fig.add_subplot(nrows, ncols, i)
            img = plt.imread(png_path)
            ax.imshow(img)
            ax.set_title(f"{oid}\nLOF={lof_score:.2f} C={cluster_id}", fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        panel_out = os.path.join(output_dir, 'outlier_fits_gallery_panel.png')
        plt.savefig(panel_out, dpi=220, bbox_inches='tight')
        plt.close()
        print(f"异常点总图已保存到 {panel_out}")


def analyze_embedding_physical_correlation(output_dir, data_root, model_path, img_size=224, batch_size=16, catalog_path=DEFAULT_CATALOG_PATH, official_model_name=DEFAULT_OFFICIAL_MODEL_NAME):
    print("分析embedding与物理量的相关性...")
    os.makedirs(output_dir, exist_ok=True)

    cache_manager = EmbeddingCache()
    embeddings, file_names, cache_key = cache_manager.load_embeddings(
        data_root, model_path, batch_size, img_size, model_family='official', official_model_name=official_model_name
    )

    if embeddings is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        model = load_official_backbone(model_path, device, official_model_name=official_model_name)
        print(f"官方模型已加载: {model_path}")

        class GrayToRGBTransform:
            def __init__(self, img_size):
                self.img_size = img_size

            def __call__(self, img):
                if hasattr(img, 'convert'):
                    img = np.array(img)
                img = np.asarray(img, dtype=np.float32)
                mx = np.max(img)
                img = img / mx if np.isfinite(mx) and mx > 0 else np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                t = t.repeat(3, 1, 1)
                return t

        transform = GrayToRGBTransform(img_size)
        dataset = EuclidVISDataset(root=data_root, transform=transform, img_size=img_size)
        file_names = [os.path.basename(file_path) for file_path in dataset.files]
        sample_dataset = dataset
        print(f"使用完整数据集，样本数量: {len(sample_dataset)}")

        dataloader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

        embeddings = []
        with torch.no_grad():
            for i, images in enumerate(dataloader):
                if isinstance(images, list):
                    images = images[0]
                images = images.to(device, non_blocking=(device.type == 'cuda'))
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        emb = model(images)
                else:
                    emb = model(images)
                embeddings.append(emb.cpu().numpy())
                if i % 10 == 0:
                    print(f"处理批次 {i+1}/{len(dataloader)}")

        embeddings = np.concatenate(embeddings, axis=0)
        print(f"嵌入提取完成，形状: {embeddings.shape}")

        if cache_key:
            metadata = {
                'data_dir': data_root,
                'model_path': model_path,
                'batch_size': batch_size,
                'img_size': img_size,
                'model_family': 'official',
                'official_model_name': official_model_name,
                'num_files': len(file_names),
                'embedding_dim': embeddings.shape[1]
            }
            cache_manager.save_embeddings(embeddings, file_names, cache_key, metadata)
    else:
        print("使用缓存的嵌入数据，跳过模型加载和嵌入提取步骤")

    catalog_data = load_catalog_data(catalog_path)
    physical_props = get_physical_properties(catalog_data, file_names)

    try:
        if isinstance(catalog_data, pd.DataFrame):
            if 'POINT_LIKE_PROB' not in catalog_data.columns:
                raise KeyError('Catalog missing POINT_LIKE_PROB')
            point_like_prob = catalog_data['POINT_LIKE_PROB'].to_numpy(dtype=float)
        else:
            if 'POINT_LIKE_PROB' not in catalog_data.dtype.names:
                raise KeyError('Catalog missing POINT_LIKE_PROB')
            point_like_prob = np.array(catalog_data['POINT_LIKE_PROB'], dtype=float)

        n_lp = min(len(point_like_prob), embeddings.shape[0])
        y_lp = (point_like_prob[:n_lp] >= 0.5).astype(int)
        X_lp = embeddings[:n_lp]

        X_train, X_test, y_train, y_test = train_test_split(X_lp, y_lp, test_size=0.2, random_state=42, stratify=y_lp)
        scaler_lp = StandardScaler()
        X_train_s = scaler_lp.fit_transform(X_train)
        X_test_s = scaler_lp.transform(X_test)
        clf_lp = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf_lp.fit(X_train_s, y_train)
        y_prob = clf_lp.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None
        cm = confusion_matrix(y_test, y_pred).tolist()
        linear_probe_results = {
            'label_rule': 'POINT_LIKE_PROB >= 0.5 => star-like (1), else extended (0)',
            'n_aligned': int(n_lp),
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'roc_auc': None if auc is None else float(auc),
                'confusion_matrix': cm
            }
        }
        lp_prob_all = clf_lp.predict_proba(scaler_lp.transform(X_lp))[:, 1]
        with open(os.path.join(output_dir, 'linear_probe_results.json'), 'w', encoding='utf-8') as f:
            json.dump(linear_probe_results, f, indent=2, ensure_ascii=False)
        pd.DataFrame([{
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
            'embedding_dim': int(X_lp.shape[1]), 'n_aligned': int(n_lp)
        }]).to_csv(os.path.join(output_dir, 'linear_probe_results.csv'), index=False)
        print('Linear probing 完成：', linear_probe_results['metrics'])
    except Exception as e:
        lp_prob_all = None
        print('Linear probing 失败或跳过：', str(e))

    segmentation_area = safe_fill_nan([p['segmentation_area'] for p in physical_props], 10.0)
    semimajor_axis = safe_fill_nan([p['semimajor_axis'] for p in physical_props], 3.0)
    kron_radius = safe_fill_nan([p['kron_radius'] for p in physical_props], 2.0)
    mu_max = safe_fill_nan([p['mu_max'] for p in physical_props], 22.0)
    mumax_minus_mag = safe_fill_nan([p['mumax_minus_mag'] for p in physical_props], 5.0)
    psf_dominance = safe_fill_nan([p['psf_dominance'] for p in physical_props], 0.8)
    growth = safe_fill_nan([p['growth'] for p in physical_props], 1.5)
    blended_prob = safe_fill_nan([p['blended_prob'] for p in physical_props], 0.1)
    spurious_prob = safe_fill_nan([p['spurious_prob'] for p in physical_props], 0.05)
    snr = np.array([p.get('SNR', np.nan) for p in physical_props], dtype=float)
    if np.all(np.isnan(snr)):
        snr = psf_dominance
    else:
        snr = safe_fill_nan(snr, 5.0)

    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    embeddings_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(embeddings)
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'pca_embeddings.npy'), embeddings_pca)
    np.save(os.path.join(output_dir, 'umap_embeddings.npy'), embeddings_umap)

    embeddings_umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(embeddings)
    np.save(os.path.join(output_dir, 'umap_3d_embeddings.npy'), embeddings_umap_3d)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    physical_params = [
        ('SEGMENTATION_AREA', segmentation_area, 'viridis', 'SEGMENTATION_AREA'),
        ('SEMIMAJOR_AXIS', semimajor_axis, 'plasma', 'SEMIMAJOR_AXIS'),
        ('PSF_DOMINANCE', psf_dominance, 'inferno', 'PSF DOMINANCE'),
        ('GROWTH', growth, 'magma', 'GROWTH'),
        ('SNR', snr, 'cividis', 'SNR')
    ]
    view_angles = [(20, 45), (60, 45), (20, 135), (20, 225)]
    param_color_ranges = {}
    for param_name, param_values, cmap, title in physical_params:
        q1, q3 = np.percentile(param_values, [25, 75])
        iqr = q3 - q1
        vmin = max(q1 - 1.5 * iqr, np.min(param_values))
        vmax = min(q3 + 1.5 * iqr, np.max(param_values))
        param_color_ranges[param_name] = (vmin, vmax, cmap)
    for param_name, param_values, cmap, title in physical_params:
        fig = plt.figure(figsize=(20, 15))
        vmin, vmax, cmap = param_color_ranges[param_name]
        for i, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            scatter = ax.scatter(embeddings_umap_3d[:, 0], embeddings_umap_3d[:, 1], embeddings_umap_3d[:, 2], c=param_values, s=10, alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'3D UMAP (elev={elev}, azim={azim})')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(title)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(output_dir, f'umap_3d_embedding_{param_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    lof = LocalOutlierFactor(n_neighbors=100, contamination=0.1, novelty=False)
    lof_labels = lof.fit_predict(embeddings)
    is_outlier = lof_labels == -1
    lof_scores_positive = -lof.negative_outlier_factor_
    umap_centroid = np.mean(embeddings_umap, axis=0)
    umap_distances = np.linalg.norm(embeddings_umap - umap_centroid, axis=1)
    centroid_is_outlier = umap_distances > np.percentile(umap_distances, 98)
    both_methods_outlier = is_outlier & centroid_is_outlier
    both_methods_outlier_indices = np.where(both_methods_outlier)[0]

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(embeddings_umap[~both_methods_outlier, 0], embeddings_umap[~both_methods_outlier, 1], c='blue', s=15, alpha=0.5, label='Normal Points')
    plt.scatter(embeddings_umap[both_methods_outlier, 0], embeddings_umap[both_methods_outlier, 1], c='purple', s=30, edgecolors='black', label='Combined Outliers')
    plt.legend()
    plt.title(f'Combined Outliers: {len(both_methods_outlier_indices)} points')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(lof_scores_positive, bins=50, alpha=0.5, label='All Points')
    plt.hist(lof_scores_positive[both_methods_outlier], bins=50, color='purple', alpha=0.8, label='Combined Outliers')
    plt.axvline(np.percentile(lof_scores_positive, 98), color='red', linestyle='--')
    plt.title('LOF Score Distribution')
    plt.xlabel('LOF Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_outliers_lof.png'), dpi=300, bbox_inches='tight')
    plt.close()

    outlier_clusters = np.zeros(len(embeddings), dtype=int)
    if len(both_methods_outlier_indices) > 0:
        outlier_umap = embeddings_umap[both_methods_outlier_indices]
        cluster_labels = KMeans(n_clusters=2, random_state=42).fit_predict(outlier_umap)
        outlier_clusters[both_methods_outlier_indices] = cluster_labels + 1
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_umap[~both_methods_outlier, 0], embeddings_umap[~both_methods_outlier, 1], c='blue', s=15, alpha=0.5, label='Normal Points')
        cluster1_indices = both_methods_outlier_indices[cluster_labels == 0]
        cluster2_indices = both_methods_outlier_indices[cluster_labels == 1]
        plt.scatter(embeddings_umap[cluster1_indices, 0], embeddings_umap[cluster1_indices, 1], c='red', s=30, edgecolors='black', label='Outlier Cluster 1')
        plt.scatter(embeddings_umap[cluster2_indices, 0], embeddings_umap[cluster2_indices, 1], c='orange', s=30, edgecolors='black', label='Outlier Cluster 2')
        plt.legend()
        plt.title(f'Outlier Clusters: {len(cluster1_indices)} + {len(cluster2_indices)} points')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'outlier_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()

    np.savez(os.path.join(output_dir, 'outlier_info.npz'), is_outlier=both_methods_outlier, outlier_indices=both_methods_outlier_indices, lof_scores=lof_scores_positive, umap_embeddings=embeddings_umap, outlier_clusters=outlier_clusters)

    outlier_csv_path = save_outlier_catalog_csv(output_dir, file_names, both_methods_outlier, both_methods_outlier_indices, lof_scores_positive, outlier_clusters)
    visualize_outliers_from_csv(outlier_csv_path, data_root, output_dir, max_examples=100)

    if lp_prob_all is not None:
        umap_color = lp_prob_all[:len(embeddings_umap)]
        umap_cmap = 'viridis'
    else:
        umap_color = None
        umap_cmap = None

    plt.figure(figsize=(16, 20))
    plt.subplot(3, 2, 1)
    plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=umap_color if umap_color is not None else 'blue', s=15, alpha=0.7, cmap=umap_cmap)
    plt.scatter(embeddings_umap[is_outlier, 0], embeddings_umap[is_outlier, 1], c='black', s=25, marker='x', alpha=0.8)
    plt.title('UMAP Embedding Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)

    props_for_panels = [
        ('SEGMENTATION_AREA', segmentation_area, 'viridis'),
        ('SEMIMAJOR_AXIS', semimajor_axis, 'plasma'),
        ('PSF_DOMINANCE', psf_dominance, 'inferno'),
        ('GROWTH', growth, 'magma'),
        ('SNR', snr, 'cividis'),
    ]
    for idx, (name, values, cmap) in enumerate(props_for_panels, start=2):
        plt.subplot(3, 2, idx)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        vmin = max(q1 - 1.5 * iqr, np.min(values))
        vmax = min(q3 + 1.5 * iqr, np.max(values))
        sc = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=values, s=15, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.scatter(embeddings_umap[is_outlier, 0], embeddings_umap[is_outlier, 1], c='black', s=25, marker='x', alpha=0.8)
        plt.colorbar(sc, label=name)
        plt.title(f'UMAP vs {name} (Q1-Q3: {q1:.3f}-{q3:.3f})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_physical_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    pca3 = PCA(n_components=3, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(18, 5))
    pairs = [(0,1),(0,2),(1,2)]
    for i,(a,b) in enumerate(pairs,1):
        plt.subplot(1,3,i)
        sc = plt.scatter(pca3[:, a], pca3[:, b], c=segmentation_area, cmap='viridis', s=15, alpha=0.7)
        plt.title(f'PCA {a+1}-{b+1} Colored by SEGMENTATION_AREA')
        plt.xlabel(f'PCA {a+1}')
        plt.ylabel(f'PCA {b+1}')
        plt.colorbar(sc, label='SEGMENTATION_AREA')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_physical_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()

    physical_prop_list = [('SEGMENTATION_AREA', segmentation_area), ('SEMIMAJOR_AXIS', semimajor_axis), ('PSF_DOMINANCE', psf_dominance), ('GROWTH', growth)]
    for prop_name, prop_values in physical_prop_list:
        correlations = []
        for i in range(embeddings.shape[1]):
            corr = np.corrcoef(embeddings[:, i], prop_values)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
            correlations.append(abs(corr))
        top_5_dims = np.argsort(correlations)[-5:][::-1]
        plt.figure(figsize=(20, 3))
        for i, dim in enumerate(top_5_dims):
            plt.subplot(1, 5, i + 1)
            reg = LinearRegression()
            X = embeddings[:, dim].reshape(-1, 1)
            reg.fit(X, prop_values)
            y_pred = reg.predict(X)
            r2 = r2_score(prop_values, y_pred)
            plt.scatter(embeddings[:, dim], prop_values, s=10, alpha=0.5)
            plt.plot(embeddings[:, dim], y_pred, color='red', linewidth=2)
            plt.title(f'Dim {dim+1}\nr²={r2:.3f}')
            plt.xlabel(f'Embedding Dim {dim+1}')
            if i == 0:
                plt.ylabel(prop_name)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'embedding_vs_{prop_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    physical_props_array = np.array([segmentation_area, semimajor_axis, psf_dominance, growth, snr]).T
    combined_data = np.hstack([pca3, physical_props_array])
    corr_matrix = np.corrcoef(combined_data.T)
    labels = [f'PCA {i+1}' for i in range(3)] + ['SEGMENTATION_AREA', 'SEMIMAJOR_AXIS', 'PSF_DOMINANCE', 'GROWTH', 'SNR']
    plt.figure(figsize=(12, 10))
    if sns is not None:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, xticklabels=labels, yticklabels=labels, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    else:
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, shrink=0.8)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', fontsize=8)
    plt.title('Correlation between PCA Components and Physical Properties')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Embedding 物理相关性分析完成！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT, help='VIS 数据根目录')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='官方 DINOv2 checkpoint 路径')
    parser.add_argument('--catalog-path', type=str, default=DEFAULT_CATALOG_PATH, help='星表文件路径')
    parser.add_argument('--img-size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--official-model-name', type=str, default=DEFAULT_OFFICIAL_MODEL_NAME, help='torch.hub 官方模型名')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"错误: 未找到模型检查点: {args.model_path}")
        return
    if not os.path.exists(args.data_root):
        print(f"错误: 未找到数据目录: {args.data_root}")
        return
    if not os.path.exists(args.catalog_path):
        print(f"错误: 未找到星表文件: {args.catalog_path}")
        return

    analyze_embedding_physical_correlation(args.output_dir, args.data_root, args.model_path, args.img_size, args.batch_size, args.catalog_path, official_model_name=args.official_model_name)


if __name__ == '__main__':
    main()
