# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import argparse
import json
import math
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Euclid_DINOv2_VIT"))

from euclid_dino.datasets.euclid_vis import EuclidVISDataset
from euclid_dino.models.dino_model import DINOModel

OBJ_RE = re.compile(r"^(-?\d+)_")


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def parse_csv_floats(s: str):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_ints(s: str):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_strs(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_torch_checkpoint(path: str, map_location=None, trusted: bool = True):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if trusted and ("Weights only load failed" in msg or "weights_only" in msg):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def actual_root(data_root):
    vis_root = os.path.join(data_root, "VIS")
    return vis_root if os.path.exists(vis_root) else data_root


def parse_object_id_from_filename(path_or_name: str) -> int:
    name = os.path.basename(path_or_name)
    m = OBJ_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse OBJECT_ID from filename: {name}")
    return int(m.group(1))


def load_catalog(catalog_path: str):
    if not catalog_path:
        return None
    if catalog_path.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(catalog_path)
    if catalog_path.endswith(".fits"):
        from astropy.io import fits
        import pandas as pd
        with fits.open(catalog_path) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                names = getattr(getattr(data, "dtype", None), "names", None)
                if data is not None and names is not None and len(data) > 0:
                    return pd.DataFrame(data)
        raise ValueError("No tabular HDU found in FITS")
    raise ValueError(f"Unsupported catalog format: {catalog_path}")


def find_id_column(df):
    for col in ["OBJECT_ID", "object_id", "Object_ID"]:
        if col in df.columns:
            return col
    raise KeyError("catalog missing OBJECT_ID/object_id")


def find_label_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"catalog missing label columns: {candidates}")


class SingleViewGrayTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, img):
        if hasattr(img, "convert"):
            img = np.array(img)
        x = img.astype(np.float32)
        mx = np.max(x)
        x = x / mx if mx > 0 else x
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


class SingleViewGrayToRGBTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, img):
        if hasattr(img, "convert"):
            img = np.array(img)
        x = img.astype(np.float32)
        mx = np.max(x)
        x = x / mx if mx > 0 else x
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return x.repeat(3, 1, 1)


class IndexedLabelDataset(Dataset):
    def __init__(self, base_dataset, indices, labels):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = self.base_dataset[self.indices[idx]]
        y = int(self.labels[idx])
        return x, y


def build_labeled_records(files, catalog_df, label_col_candidates, ignore_labels, logger=None):
    import pandas as pd

    id_col = find_id_column(catalog_df)
    label_col = find_label_column(catalog_df, label_col_candidates)

    df = catalog_df.drop_duplicates(subset=[id_col], keep="first").copy()
    df[id_col] = df[id_col].astype(np.int64)
    df = df.set_index(id_col, drop=False)

    records, miss, ignored = [], 0, 0
    for i, f in enumerate(files):
        try:
            oid = parse_object_id_from_filename(f)
        except Exception:
            miss += 1
            continue
        if oid not in df.index:
            miss += 1
            continue
        row = df.loc[oid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        label = row[label_col]
        if pd.isna(label):
            ignored += 1
            continue
        label = str(label).strip()
        if label in ignore_labels or label == "":
            ignored += 1
            continue
        records.append({"dataset_index": i, "object_id": oid, "label_text": label})

    if logger:
        logger.log(f"[Labels] matched labeled records={len(records)} missing={miss} ignored={ignored} label_col={label_col}")
        logger.log(f"[Labels] class counts={dict(Counter([r['label_text'] for r in records]))}")
    return records, label_col


def sample_fraction_per_class(records, labeled_fraction=1.0, max_per_class=None, seed=42, logger=None):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for r in records:
        grouped[r["label_text"]].append(r)

    sampled, sampled_counts = [], {}
    for cls, items in grouped.items():
        items = items[:]
        rng.shuffle(items)
        take = len(items) if labeled_fraction >= 1.0 else max(1, int(round(len(items) * labeled_fraction)))
        if max_per_class is not None:
            take = min(take, int(max_per_class))
        sampled.extend(items[:take])
        sampled_counts[cls] = take

    rng.shuffle(sampled)
    if logger:
        logger.log(f"[FewLabelSample] fraction={labeled_fraction:.4f} sampled={len(sampled)}/{len(records)} counts={sampled_counts}")
    return sampled


def stratified_split(records, val_frac=0.2, seed=42, min_train_per_class=1, min_val_per_class=1):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for r in records:
        grouped[r["label_text"]].append(r)

    train_records, val_records, dropped_classes = [], [], []
    for cls, items in grouped.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        if n < (min_train_per_class + min_val_per_class):
            dropped_classes.append((cls, n))
            continue
        n_val = max(min_val_per_class, int(round(n * val_frac)))
        n_train = n - n_val
        if n_train < min_train_per_class:
            n_train = min_train_per_class
            n_val = n - n_train
        if n_val < min_val_per_class or n_train < min_train_per_class:
            dropped_classes.append((cls, n))
            continue
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])
    return train_records, val_records, dropped_classes


def encode_labels(train_records, val_records):
    classes = sorted(set([r["label_text"] for r in train_records] + [r["label_text"] for r in val_records]))
    c2i = {c: i for i, c in enumerate(classes)}
    train_indices = [r["dataset_index"] for r in train_records]
    val_indices = [r["dataset_index"] for r in val_records]
    train_y = [c2i[r["label_text"]] for r in train_records]
    val_y = [c2i[r["label_text"]] for r in val_records]
    return classes, train_indices, train_y, val_indices, val_y


def morphology_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = None
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "macro_ovr_auc": None if auc is None else float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_class_weights(labels, num_classes):
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class CustomBackboneWrapper(nn.Module):
    def __init__(self, backbone: DINOModel):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        _, emb, _ = self.backbone(x)
        return emb


class OfficialDINOv2BackboneWrapper(nn.Module):
    def __init__(self, model: nn.Module):
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


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GenericMorphologyClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, num_classes: int, dropout: float = 0.0,
                 probe_head: str = "linear", mlp_hidden_dim: Optional[int] = None,
                 normalize_embeddings: bool = True, l2_normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedding_norm = nn.LayerNorm(embedding_dim) if normalize_embeddings else nn.Identity()
        self.l2_normalize_embeddings = l2_normalize_embeddings
        if probe_head == "linear":
            self.head = nn.Linear(embedding_dim, num_classes)
        elif probe_head == "mlp":
            hidden_dim = int(mlp_hidden_dim or embedding_dim)
            self.head = MLPHead(embedding_dim, hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown probe_head={probe_head}")

    def forward(self, x):
        emb = self.backbone(x)
        emb = self.embedding_norm(emb)
        if self.l2_normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        logits = self.head(self.dropout(emb))
        return logits, emb


def set_backbone_trainable(model: GenericMorphologyClassifier, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_total_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def get_embedding_dim(backbone, img_size, channels, device):
    with torch.no_grad():
        dummy = torch.zeros(1, channels, img_size, img_size, device=device)
        emb = backbone(dummy)
        return int(emb.shape[-1])


def smart_extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["student_state_dict", "teacher_state_dict", "backbone_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
    if isinstance(ckpt, dict):
        return ckpt, "raw_dict"
    raise ValueError("Unsupported checkpoint format")


def interpolate_pos_embedding(checkpoint_pos_emb, model_pos_emb, logger=None):
    if checkpoint_pos_emb.shape == model_pos_emb.shape:
        return checkpoint_pos_emb
    cls_token = checkpoint_pos_emb[:, :1, :]
    patch_pos = checkpoint_pos_emb[:, 1:, :]
    num_patches_old = patch_pos.shape[1]
    num_patches_new = model_pos_emb.shape[1] - 1
    size_old = int(round(num_patches_old ** 0.5))
    size_new = int(round(num_patches_new ** 0.5))
    if size_old * size_old != num_patches_old or size_new * size_new != num_patches_new:
        if logger:
            logger.log("[Init][euclid_ssl] pos_embedding patch tokens not square; fallback to model pos_embedding.")
        return model_pos_emb
    if logger:
        logger.log(f"[Init][euclid_ssl] interpolate pos_embedding {size_old}x{size_old} -> {size_new}x{size_new}")
    patch_pos = patch_pos.reshape(1, size_old, size_old, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(size_new, size_new), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, size_new * size_new, -1)
    return torch.cat([cls_token, patch_pos], dim=1)


def safe_load_custom_backbone_from_ssl(backbone: DINOModel, ckpt_path: str, device, logger):
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(ckpt)
    if 'pos_embedding' in state and 'pos_embedding' in backbone.state_dict():
        checkpoint_pos_emb = state['pos_embedding']
        model_pos_emb = backbone.state_dict()['pos_embedding']
        if checkpoint_pos_emb.shape != model_pos_emb.shape:
            logger.log(f"[Init][euclid_ssl] 修复位置嵌入大小: {checkpoint_pos_emb.shape} -> {model_pos_emb.shape}")
            state['pos_embedding'] = interpolate_pos_embedding(checkpoint_pos_emb, model_pos_emb, logger=logger)
    msg = backbone.load_state_dict(state, strict=False)
    logger.log(f"[Init][euclid_ssl] source_key={source_key} from {ckpt_path}")
    logger.log(f"[Init][euclid_ssl] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        logger.log(f"[Init][euclid_ssl] missing_keys sample: {msg.missing_keys[:20]}")
    if len(msg.unexpected_keys) > 0:
        logger.log(f"[Init][euclid_ssl] unexpected_keys sample: {msg.unexpected_keys[:20]}")
    target_keys = set(backbone.state_dict().keys())
    state_keys = set(state.keys())
    overlap = len(target_keys & state_keys)
    coverage = overlap / max(len(target_keys), 1)
    logger.log(f"[Init][euclid_ssl] overlap={overlap}/{len(target_keys)} coverage={coverage:.4f}")
    if coverage < 0.80:
        raise RuntimeError(f"Euclid SSL checkpoint coverage too low: {coverage:.4f}. Likely wrong checkpoint or wrong architecture.")


def build_official_dinov2_backbone(args, device, logger):
    logger.log(f"[Init][imagenet_dinov2] building official model via torch.hub: {args.imagenet_model_name}")
    model = torch.hub.load("facebookresearch/dinov2", args.imagenet_model_name, pretrained=False, force_reload=False)
    ckpt = load_torch_checkpoint(args.imagenet_init, map_location=device, trusted=True)
    state, source_key = smart_extract_state_dict(ckpt)
    logger.log(f"[Init][imagenet_dinov2] source_key={source_key} from {args.imagenet_init}")
    msg = model.load_state_dict(state, strict=False)
    logger.log(f"[Init][imagenet_dinov2] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        logger.log(f"[Init][imagenet_dinov2] missing_keys sample: {msg.missing_keys[:20]}")
    if len(msg.unexpected_keys) > 0:
        logger.log(f"[Init][imagenet_dinov2] unexpected_keys sample: {msg.unexpected_keys[:20]}")
    target_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    overlap = len(target_keys & state_keys)
    coverage = overlap / max(len(target_keys), 1)
    logger.log(f"[Init][imagenet_dinov2] overlap={overlap}/{len(target_keys)} coverage={coverage:.4f}")
    if coverage < 0.80:
        raise RuntimeError(f"Official DINOv2 checkpoint coverage too low: {coverage:.4f}. Please check whether --imagenet-init matches --imagenet-model-name.")
    model = model.to(device)
    model.eval()
    return OfficialDINOv2BackboneWrapper(model).to(device)


def build_custom_backbone(args, device):
    model = DINOModel(model_type=args.model_type, patch_size=args.patch_size, input_channels=1, img_size=args.img_size).to(device)
    return CustomBackboneWrapper(model).to(device)


def build_backbone_and_dataset_for_method(args, method, root, device, logger):
    if method == "random":
        dataset = EuclidVISDataset(root=root, transform=SingleViewGrayTransform(args.img_size), img_size=args.img_size)
        backbone = build_custom_backbone(args, device)
        logger.log("[Init][random] random initialization, no checkpoint loaded.")
        return backbone, dataset, 1, "random"
    if method == "euclid_ssl":
        if not args.euclid_init:
            raise ValueError("method=euclid_ssl requires --euclid-init")
        dataset = EuclidVISDataset(root=root, transform=SingleViewGrayTransform(args.img_size), img_size=args.img_size)
        backbone = build_custom_backbone(args, device)
        safe_load_custom_backbone_from_ssl(backbone.backbone, args.euclid_init, device, logger)
        return backbone, dataset, 1, "euclid_ssl"
    if method == "imagenet_dinov2":
        if not args.imagenet_init:
            raise ValueError("method=imagenet_dinov2 requires --imagenet-init")
        dataset = EuclidVISDataset(root=root, transform=SingleViewGrayToRGBTransform(args.img_size), img_size=args.img_size)
        backbone = build_official_dinov2_backbone(args, device, logger)
        return backbone, dataset, 3, "official_dinov2_rgb_replicated_from_gray"
    raise ValueError(f"Unknown method={method}")


@torch.no_grad()
def compute_embedding_health(backbone, loader, device, max_batches=4):
    backbone.eval()
    embs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        emb = backbone(x).detach().float().cpu()
        embs.append(emb)
        ys.append(y.cpu())
    if len(embs) == 0:
        return None
    emb = torch.cat(embs, dim=0)
    y = torch.cat(ys, dim=0) if len(ys) > 0 else None
    if emb.ndim != 2 or emb.shape[0] < 2:
        return None
    feat_std_per_dim = emb.std(dim=0)
    mean_feature_std = float(feat_std_per_dim.mean().item())
    mean_abs_activation = float(emb.abs().mean().item())
    norms = emb.norm(dim=1)
    mean_norm = float(norms.mean().item())
    std_norm = float(norms.std().item())
    emb_norm = F.normalize(emb, dim=1)
    sim = emb_norm @ emb_norm.t()
    off_diag_mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    mean_pairwise_cosine = float(sim[off_diag_mask].mean().item())
    result = {
        "n_samples": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "mean_feature_std": mean_feature_std,
        "mean_abs_activation": mean_abs_activation,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "mean_pairwise_cosine": mean_pairwise_cosine,
    }
    if y is not None and len(torch.unique(y)) > 1:
        centers = []
        for cls in torch.unique(y):
            centers.append(emb[y == cls].mean(dim=0, keepdim=True))
        if len(centers) > 1:
            c = F.normalize(torch.cat(centers, dim=0), dim=1)
            cs = c @ c.t()
            off = ~torch.eye(cs.shape[0], dtype=torch.bool)
            result["class_center_mean_pairwise_cosine"] = float(cs[off].mean().item())
    return result


def grad_norm_of_params(params):
    sq = 0.0
    has_grad = False
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += float(torch.sum(g * g).item())
            has_grad = True
    if not has_grad:
        return 0.0
    return math.sqrt(max(sq, 0.0))


def evaluate_model(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * x.size(0)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            y_prob.append(prob.detach().cpu().numpy())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    y_prob = np.concatenate(y_prob, axis=0) if y_prob else None
    metrics = morphology_metrics(np.array(y_true), np.array(y_pred), y_prob=y_prob)
    metrics["loss"] = float(avg_loss)
    return metrics


def run_one_finetune(args, dataset, backbone, input_channels, train_records, val_records, method, seed, probe_head,
                     device, logger, writer=None, exp_tag=""):
    classes, tr_idx, tr_y, va_idx, va_y = encode_labels(train_records, val_records)
    num_classes = len(classes)
    train_ds = IndexedLabelDataset(dataset, tr_idx, tr_y)
    val_ds = IndexedLabelDataset(dataset, va_idx, va_y)
    train_loader = DataLoader(train_ds, batch_size=args.finetune_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.finetune_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    embedding_dim = get_embedding_dim(backbone, args.img_size, input_channels, device)
    logger.log(f"[{exp_tag}] embedding_dim={embedding_dim} input_channels={input_channels} probe_head={probe_head}")
    mlp_hidden_dim = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else embedding_dim
    model = GenericMorphologyClassifier(backbone=backbone, embedding_dim=embedding_dim, num_classes=num_classes,
                                        dropout=args.finetune_dropout, probe_head=probe_head,
                                        mlp_hidden_dim=mlp_hidden_dim,
                                        normalize_embeddings=args.normalize_embeddings,
                                        l2_normalize_embeddings=args.l2_normalize_embeddings).to(device)
    if args.finetune_strategy in {"linear", "linear_then_full"}:
        set_backbone_trainable(model, False)
    else:
        set_backbone_trainable(model, True)
    logger.log(f"[{exp_tag}] backbone trainable params at start = {count_trainable_params(model.backbone)}/{count_total_params(model.backbone)}")
    class_weights = compute_class_weights(tr_y, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def build_optimizer(backbone_trainable: bool):
        params = []
        if backbone_trainable:
            bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
            if len(bb_params) > 0:
                params.append({"params": bb_params, "lr": args.finetune_lr_backbone})
        params.append({"params": model.head.parameters(), "lr": args.finetune_lr_head})
        return torch.optim.AdamW(params, weight_decay=args.finetune_weight_decay)

    optimizer = build_optimizer(backbone_trainable=any(p.requires_grad for p in model.backbone.parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run_dir = os.path.join(args.output_dir, "runs", exp_tag)
    os.makedirs(run_dir, exist_ok=True)
    health_before = compute_embedding_health(model.backbone, train_loader, device, max_batches=args.embedding_health_batches)
    if health_before is not None:
        logger.log(f"[{exp_tag}] embedding health before training: {health_before}")
        dump_json(health_before, os.path.join(run_dir, "embedding_health_before.json"))
    history, best_metric, best_epoch, best_summary = [], -1.0, -1, None
    for epoch in range(args.finetune_epochs):
        just_unfroze = False
        if args.finetune_strategy == "linear_then_full" and epoch == args.head_warmup_epochs:
            logger.log(f"[{exp_tag}] warmup finished -> unfreezing backbone")
            set_backbone_trainable(model, True)
            logger.log(f"[{exp_tag}] backbone trainable params after unfreeze = {count_trainable_params(model.backbone)}/{count_total_params(model.backbone)}")
            if count_trainable_params(model.backbone) == 0:
                raise RuntimeError(f"[{exp_tag}] backbone still has 0 trainable params after unfreeze.")
            optimizer = build_optimizer(backbone_trainable=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.finetune_epochs - epoch, 1))
            just_unfroze = True
        model.train()
        train_true, train_pred = [], []
        total_train_loss, backbone_grad_norm_sum, head_grad_norm_sum, num_steps = 0.0, 0.0, 0.0, 0
        for x, y in tqdm(train_loader, desc=f"{exp_tag} Epoch {epoch+1}/{args.finetune_epochs}", unit="batch", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            bb_grad = grad_norm_of_params(model.backbone.parameters())
            hd_grad = grad_norm_of_params(model.head.parameters())
            backbone_grad_norm_sum += bb_grad
            head_grad_norm_sum += hd_grad
            num_steps += 1
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += float(loss.item()) * x.size(0)
            train_true.extend(y.detach().cpu().numpy().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        train_loss = total_train_loss / max(len(train_ds), 1)
        train_metrics = morphology_metrics(np.array(train_true), np.array(train_pred), y_prob=None)
        val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp=args.amp)
        scheduler.step()
        mean_backbone_grad_norm = backbone_grad_norm_sum / max(num_steps, 1)
        mean_head_grad_norm = head_grad_norm_sum / max(num_steps, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_macro_ovr_auc": val_metrics["macro_ovr_auc"],
            "lr_head": float(optimizer.param_groups[-1]["lr"]),
            "mean_backbone_grad_norm": float(mean_backbone_grad_norm),
            "mean_head_grad_norm": float(mean_head_grad_norm),
            "backbone_trainable_params": int(count_trainable_params(model.backbone)),
            "probe_head": probe_head,
        }
        history.append(row)
        if writer is not None:
            writer.add_scalar(f"{exp_tag}/train_loss", train_loss, epoch + 1)
            writer.add_scalar(f"{exp_tag}/val_loss", val_metrics["loss"], epoch + 1)
            writer.add_scalar(f"{exp_tag}/val_accuracy", val_metrics["accuracy"], epoch + 1)
            writer.add_scalar(f"{exp_tag}/val_macro_f1", val_metrics["macro_f1"], epoch + 1)
            writer.add_scalar(f"{exp_tag}/mean_backbone_grad_norm", mean_backbone_grad_norm, epoch + 1)
            writer.add_scalar(f"{exp_tag}/mean_head_grad_norm", mean_head_grad_norm, epoch + 1)
        logger.log(f"[{exp_tag}] epoch={epoch+1}/{args.finetune_epochs} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} bb_grad={mean_backbone_grad_norm:.6e} head_grad={mean_head_grad_norm:.6e}")
        if (epoch >= args.head_warmup_epochs or args.finetune_strategy == "full" or just_unfroze) and count_trainable_params(model.backbone) > 0 and mean_backbone_grad_norm < args.backbone_grad_warn_threshold:
            logger.log(f"[{exp_tag}] WARNING: backbone grad norm very small after unfreeze ({mean_backbone_grad_norm:.6e} < {args.backbone_grad_warn_threshold:.6e}). Backbone may still be effectively frozen or embedding may be degenerate.")
        score = val_metrics[args.model_select_metric]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch + 1
            best_summary = {
                "best_epoch": best_epoch,
                "best_score_metric": args.model_select_metric,
                "best_score": float(best_metric),
                "classes": classes,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "train_class_counts": dict(Counter(tr_y)),
                "val_class_counts": dict(Counter(va_y)),
                "last_val_metrics": deepcopy(val_metrics),
                "method": method,
                "seed": seed,
                "embedding_dim": embedding_dim,
                "input_channels": input_channels,
                "probe_head": probe_head,
                "health_before": health_before,
            }
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "backbone_state_dict": model.backbone.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "history": history,
                "summary": best_summary,
                "args": vars(args),
            }, os.path.join(run_dir, "best.pth"))
    health_after = compute_embedding_health(model.backbone, train_loader, device, max_batches=args.embedding_health_batches)
    if health_after is not None:
        logger.log(f"[{exp_tag}] embedding health after training: {health_after}")
        dump_json(health_after, os.path.join(run_dir, "embedding_health_after.json"))
    try:
        import pandas as pd
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    except Exception:
        pass
    dump_json({"history": history, "best_summary": best_summary, "embedding_health_before": health_before, "embedding_health_after": health_after}, os.path.join(run_dir, "history.json"))
    _plot_single_run_diagnostics(history, run_dir, exp_tag)
    return best_summary


def _plot_single_run_diagnostics(history, run_dir, exp_tag):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    df = pd.DataFrame(history)
    for ycols, fname, title in [
        (["train_loss", "val_loss"], "diag_loss_curve.png", f"{exp_tag} Loss"),
        (["val_accuracy", "val_macro_f1"], "diag_val_metrics_curve.png", f"{exp_tag} Validation metrics"),
        (["mean_backbone_grad_norm", "mean_head_grad_norm"], "diag_grad_norm_curve.png", f"{exp_tag} Gradient norms"),
    ]:
        cols = [c for c in ycols if c in df.columns]
        if not cols:
            continue
        plt.figure(figsize=(7.0, 5.0))
        for c in cols:
            plt.plot(df["epoch"], df[c], marker="o", label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()


def _plot_embedding_health_summary(raw_df, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    if len(raw_df) == 0:
        return []
    metrics = [m for m in [
        "health_before_mean_feature_std",
        "health_before_mean_abs_activation",
        "health_before_mean_norm",
        "health_before_std_norm",
        "health_before_mean_pairwise_cosine",
    ] if m in raw_df.columns]
    created = []
    for metric in metrics:
        agg = raw_df.groupby(["probe_head", "method", "fraction"])[metric].mean().reset_index()
        for probe_head in sorted(agg["probe_head"].unique()):
            sub_probe = agg[agg["probe_head"] == probe_head]
            plt.figure(figsize=(7.2, 5.4))
            for method in ["euclid_ssl", "imagenet_dinov2", "random"]:
                sub = sub_probe[sub_probe["method"] == method].sort_values("fraction")
                if len(sub) == 0:
                    continue
                plt.plot((sub["fraction"] * 100).astype(int), sub[metric], marker="o", linewidth=2.0, label=method)
            plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
            plt.xlabel("Label fraction")
            plt.ylabel(metric.replace("health_before_", ""))
            plt.title(f"{probe_head.upper()} probe: embedding health ({metric.replace('health_before_', '')})")
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=True)
            plt.tight_layout()
            out = os.path.join(output_dir, f"embedding_health_{metric}_{probe_head}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            created.append(out)
    return created


def fraction_to_percent_label(frac: float) -> str:
    return f"{int(round(frac * 100))}%"


def aggregate_results(all_rows):
    import pandas as pd
    df = pd.DataFrame(all_rows)
    agg = df.groupby(["probe_head", "method", "fraction"]).agg(
        accuracy_mean=("best_accuracy", "mean"),
        accuracy_std=("best_accuracy", "std"),
        macro_f1_mean=("best_macro_f1", "mean"),
        macro_f1_std=("best_macro_f1", "std"),
        balanced_accuracy_mean=("best_balanced_accuracy", "mean"),
        balanced_accuracy_std=("best_balanced_accuracy", "std"),
        n_runs=("seed", "count"),
        n_train_mean=("n_train", "mean"),
        n_val_mean=("n_val", "mean"),
    ).reset_index()
    for c in ["accuracy_std", "macro_f1_std", "balanced_accuracy_std"]:
        agg[c] = agg[c].fillna(0.0)
    agg["fraction_percent"] = agg["fraction"].apply(lambda x: int(round(float(x) * 100)))
    agg = agg.sort_values(["probe_head", "method", "fraction"])
    return df, agg


def save_result_tables(raw_df, agg_df, output_dir, metric_for_table="macro_f1_mean"):
    raw_csv = os.path.join(output_dir, "fewlabel_results_raw.csv")
    agg_csv = os.path.join(output_dir, "fewlabel_results_aggregated.csv")
    raw_df.to_csv(raw_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    table_paths = {"raw_csv": raw_csv, "agg_csv": agg_csv}
    fractions_sorted = sorted(agg_df["fraction"].unique().tolist())
    for probe_head in sorted(agg_df["probe_head"].unique()):
        sub = agg_df[agg_df["probe_head"] == probe_head].copy()
        show_cols = [fraction_to_percent_label(x) for x in fractions_sorted]
        pivot = sub.pivot(index="method", columns="fraction", values=metric_for_table)
        pivot = pivot.reindex(columns=fractions_sorted)
        pivot.columns = show_cols
        pivot = pivot.reset_index()
        suffix = f"_{probe_head}"
        table_csv = os.path.join(output_dir, f"table2_fewlabel_comparison{suffix}.csv")
        table_md = os.path.join(output_dir, f"table2_fewlabel_comparison{suffix}.md")
        table_tex = os.path.join(output_dir, f"table2_fewlabel_comparison{suffix}.tex")
        pivot.to_csv(table_csv, index=False)
        with open(table_md, "w", encoding="utf-8") as f:
            f.write(f"Table 2 ({probe_head}). Few-label classification performance comparison\n\n")
            f.write(pivot.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n")
        with open(table_tex, "w", encoding="utf-8") as f:
            f.write(f"% Table 2 ({probe_head}). Few-label classification performance comparison\n")
            f.write(pivot.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
        table_paths[f"table_csv_{probe_head}"] = table_csv
        table_paths[f"table_md_{probe_head}"] = table_md
        table_paths[f"table_tex_{probe_head}"] = table_tex
    return table_paths


def plot_fewlabel_curves(agg_df, output_dir, title_suffix=""):
    import matplotlib.pyplot as plt
    method_order = ["euclid_ssl", "imagenet_dinov2", "random"]
    pretty = {"euclid_ssl": "Euclid SSL", "imagenet_dinov2": "Official DINOv2", "random": "Random init"}
    fig_paths = []
    for probe_head in sorted(agg_df["probe_head"].unique()):
        sub_head = agg_df[agg_df["probe_head"] == probe_head].copy()
        for metric_mean, metric_std, ylabel, fname in [
            ("accuracy_mean", "accuracy_std", "Accuracy", f"figure5_fewlabel_accuracy_{probe_head}.png"),
            ("macro_f1_mean", "macro_f1_std", "Macro-F1", f"figure5_fewlabel_macro_f1_{probe_head}.png"),
        ]:
            plt.figure(figsize=(7.2, 5.4))
            for method in method_order:
                sub = sub_head[sub_head["method"] == method].sort_values("fraction")
                if len(sub) == 0:
                    continue
                x = sub["fraction_percent"].to_numpy()
                y = sub[metric_mean].to_numpy()
                e = sub[metric_std].to_numpy()
                plt.plot(x, y, marker="o", linewidth=2.2, label=pretty.get(method, method))
                if np.any(e > 0):
                    plt.fill_between(x, y - e, y + e, alpha=0.15)
            plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
            plt.xlabel("Label fraction")
            plt.ylabel(ylabel)
            plt.title(f"{probe_head.upper()} probe: Performance vs label fraction{title_suffix}")
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=True)
            plt.tight_layout()
            png_path = os.path.join(output_dir, fname)
            pdf_path = png_path.replace(".png", ".pdf")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.close()
            fig_paths.extend([png_path, pdf_path])
    if len(set(agg_df["probe_head"])) > 1:
        for metric in ["accuracy_mean", "macro_f1_mean"]:
            ylabel = "Accuracy" if metric == "accuracy_mean" else "Macro-F1"
            plt.figure(figsize=(7.2, 5.4))
            for probe_head in sorted(agg_df["probe_head"].unique()):
                sub = agg_df[(agg_df["probe_head"] == probe_head) & (agg_df["method"] == "euclid_ssl")].sort_values("fraction")
                if len(sub) == 0:
                    continue
                plt.plot(sub["fraction_percent"], sub[metric], marker="o", linewidth=2.2, label=f"Euclid SSL - {probe_head.upper()} probe")
            plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
            plt.xlabel("Label fraction")
            plt.ylabel(ylabel)
            plt.title(f"Euclid SSL: Linear vs MLP probe ({ylabel})")
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=True)
            plt.tight_layout()
            png_path = os.path.join(output_dir, f"euclid_ssl_linear_vs_mlp_{metric}.png")
            pdf_path = png_path.replace(".png", ".pdf")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.close()
            fig_paths.extend([png_path, pdf_path])
    return fig_paths


def build_paper_ready_summary(agg_df, output_dir):
    lines = ["Few-label summary (auto-generated)", ""]
    for probe_head in sorted(agg_df["probe_head"].unique()):
        lines.append(f"[{probe_head.upper()} probe]")
        for frac in sorted(agg_df["fraction"].unique().tolist()):
            sub = agg_df[(agg_df["probe_head"] == probe_head) & (agg_df["fraction"] == frac)].copy().sort_values("method")
            if len(sub) == 0:
                continue
            parts = []
            for _, row in sub.iterrows():
                parts.append(f"{row['method']}: acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}")
            lines.append(f"{fraction_to_percent_label(frac)} labels -> " + " | ".join(parts))
        lines.append("")
    txt_path = os.path.join(output_dir, "fewlabel_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


def run_fewlabel_benchmark(args, logger, writer, device):
    logger.log("===== Few-label benchmark starts =====")
    root = actual_root(args.data_root)
    ref_dataset = EuclidVISDataset(root=root, transform=SingleViewGrayTransform(args.img_size), img_size=args.img_size)
    catalog_df = load_catalog(args.catalog_path)
    records, label_col = build_labeled_records(ref_dataset.files, catalog_df, [x.strip() for x in args.probe_label_cols.split(",") if x.strip()], tuple([x.strip() for x in args.ignore_labels.split(",") if x.strip()]), logger=logger)
    methods = parse_csv_strs(args.methods)
    probe_heads = parse_csv_strs(args.probe_heads)
    fractions = sorted(parse_csv_floats(args.few_label_fractions))
    seeds = parse_csv_ints(args.seeds)
    all_rows = []
    for seed in seeds:
        set_seed(seed, deterministic=args.deterministic)
        logger.log(f"===== Seed {seed} =====")
        for frac in fractions:
            sampled = sample_fraction_per_class(records, labeled_fraction=frac, max_per_class=args.max_labeled_per_class, seed=seed, logger=logger)
            train_records, val_records, dropped_classes = stratified_split(sampled, val_frac=args.finetune_val_split, seed=seed, min_train_per_class=1, min_val_per_class=1)
            if dropped_classes:
                logger.log(f"[Split] dropped classes due to too few samples: {dropped_classes}")
            if len(train_records) == 0 or len(val_records) == 0:
                logger.log(f"[Skip] fraction={frac:.4f} seed={seed} has empty split after filtering.")
                continue
            n_classes = len(set([r["label_text"] for r in train_records] + [r["label_text"] for r in val_records]))
            logger.log(f"[Split] fraction={frac:.4f} seed={seed} train={len(train_records)} val={len(val_records)} classes={n_classes}")
            for method in methods:
                for probe_head in probe_heads:
                    exp_tag = f"{method}_{probe_head}_frac{int(round(frac * 100)):03d}_seed{seed}"
                    logger.log(f"===== Run {exp_tag} =====")
                    backbone, dataset, input_channels, init_note = build_backbone_and_dataset_for_method(args=args, method=method, root=root, device=device, logger=logger)
                    best_summary = run_one_finetune(args=args, dataset=dataset, backbone=backbone, input_channels=input_channels,
                                                    train_records=train_records, val_records=val_records, method=method,
                                                    seed=seed, probe_head=probe_head, device=device, logger=logger,
                                                    writer=writer, exp_tag=exp_tag)
                    row = {
                        "probe_head": probe_head,
                        "method": method,
                        "fraction": float(frac),
                        "fraction_percent": int(round(frac * 100)),
                        "seed": int(seed),
                        "label_col": label_col,
                        "n_train": int(best_summary["n_train"]),
                        "n_val": int(best_summary["n_val"]),
                        "best_epoch": int(best_summary["best_epoch"]),
                        "best_score_metric": best_summary["best_score_metric"],
                        "best_score": float(best_summary["best_score"]),
                        "best_accuracy": float(best_summary["last_val_metrics"]["accuracy"]),
                        "best_balanced_accuracy": float(best_summary["last_val_metrics"]["balanced_accuracy"]),
                        "best_macro_f1": float(best_summary["last_val_metrics"]["macro_f1"]),
                        "best_macro_ovr_auc": None if best_summary["last_val_metrics"]["macro_ovr_auc"] is None else float(best_summary["last_val_metrics"]["macro_ovr_auc"]),
                        "embedding_dim": int(best_summary["embedding_dim"]),
                        "input_channels": int(best_summary["input_channels"]),
                        "init_note": init_note,
                    }
                    health_before = best_summary.get("health_before")
                    if isinstance(health_before, dict):
                        for k, v in health_before.items():
                            if isinstance(v, (int, float)):
                                row[f"health_before_{k}"] = v
                    all_rows.append(row)
                    dump_json(row, os.path.join(args.output_dir, "runs", exp_tag, "best_result_row.json"))
    if len(all_rows) == 0:
        raise RuntimeError("No valid runs were completed. Please check dataset size, split, and checkpoints.")
    raw_df, agg_df = aggregate_results(all_rows)
    table_paths = save_result_tables(raw_df, agg_df, args.output_dir, metric_for_table=args.table_metric)
    fig_paths = plot_fewlabel_curves(agg_df, args.output_dir, title_suffix=args.figure_title_suffix)
    fig_paths.extend(_plot_embedding_health_summary(raw_df, args.output_dir))
    summary_txt = build_paper_ready_summary(agg_df, args.output_dir)
    summary = {
        "label_col": label_col,
        "methods": methods,
        "probe_heads": probe_heads,
        "fractions": fractions,
        "seeds": seeds,
        "table_paths": table_paths,
        "figure_paths": fig_paths,
        "summary_txt": summary_txt,
        "model_select_metric": args.model_select_metric,
        "table_metric": args.table_metric,
        "note": "MLP probing retained; full diagnostics for backbone unfreezing and embedding collapse included; finetune_epochs=20.",
    }
    dump_json(summary, os.path.join(args.output_dir, "fewlabel_benchmark_summary.json"))
    logger.log(f"Benchmark done. Summary saved to {os.path.join(args.output_dir, 'fewlabel_benchmark_summary.json')}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\datasets\gz_morphology_catalogue\VIS")
    parser.add_argument("--catalog-path", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_ssl_finetune\fewlabel_benchmark_official_dinov2_diag_mlp")
    parser.add_argument("--model-type", type=str, default="s", choices=["s", "b"])
    parser.add_argument("--patch-size", type=int, default=16, choices=[8, 16])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--methods", type=str, default="random,imagenet_dinov2,euclid_ssl")
    parser.add_argument("--probe-heads", type=str, default="linear,mlp")
    parser.add_argument("--few-label-fractions", type=str, default="0.01,0.05,0.10,1.0")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--probe-label-cols", type=str, default="morphology_label,morphology_main")
    parser.add_argument("--ignore-labels", type=str, default="uncertain")
    parser.add_argument("--max-labeled-per-class", type=int, default=None)
    parser.add_argument("--finetune-val-split", type=float, default=0.2)
    parser.add_argument("--euclid-init", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_ssl_finetune\ssl_best.pth")
    parser.add_argument("--imagenet-init", type=str, default=r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_official_dinov2\dinov2_vits14_official_backbone.pth")
    parser.add_argument("--imagenet-model-name", type=str, default="dinov2_vits14")
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-batch-size", type=int, default=64)
    parser.add_argument("--finetune-lr-head", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-backbone", type=float, default=1e-5)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--finetune-dropout", type=float, default=0.2)
    parser.add_argument("--finetune-strategy", type=str, default="linear_then_full", choices=["linear", "full", "linear_then_full"])
    parser.add_argument("--head-warmup-epochs", type=int, default=5)
    parser.add_argument("--model-select-metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy", "balanced_accuracy"])
    parser.add_argument("--mlp-hidden-dim", type=int, default=0)
    parser.add_argument("--normalize-embeddings", type=str2bool, default=True)
    parser.add_argument("--l2-normalize-embeddings", type=str2bool, default=False)
    parser.add_argument("--embedding-health-batches", type=int, default=4)
    parser.add_argument("--backbone-grad-warn-threshold", type=float, default=1e-8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-metric", type=str, default="macro_f1_mean", choices=["macro_f1_mean", "accuracy_mean", "balanced_accuracy_mean"])
    parser.add_argument("--figure-title-suffix", type=str, default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TeeLogger(os.path.join(args.output_dir, "logs", "fewlabel_benchmark.log"))
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb")) if SummaryWriter is not None else None
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"device={device}")
    logger.log(f"output_dir={args.output_dir}")
    logger.log(f"python={sys.version}")
    logger.log(f"platform={platform.platform()}")
    logger.log(f"torch={torch.__version__}")
    logger.log(f"finetune_epochs={args.finetune_epochs}")
    logger.log(f"probe_heads={args.probe_heads}")
    logger.log(f"normalize_embeddings={args.normalize_embeddings}")
    logger.log(f"l2_normalize_embeddings={args.l2_normalize_embeddings}")
    dump_json({
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "args": vars(args),
        "note": "MLP probing retained + full diagnostics + full visualization output. finetune_epochs=20.",
    }, os.path.join(args.output_dir, "config.json"))
    try:
        run_fewlabel_benchmark(args, logger, writer, device)
    finally:
        if writer is not None:
            writer.close()
    logger.log("All done.")


if __name__ == "__main__":
    main()
