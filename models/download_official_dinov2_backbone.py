# -*- coding: utf-8 -*-
"""
Author: Jinhui Xie
Email: xiejinhui22@mails.ucas.ac.cn
"""

import json
import os
import platform
import sys
from datetime import datetime

import torch


DATA_ROOT = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\datasets\gz_morphology_catalogue"
CATALOG_PATH = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\catalogs\gz_euclid_q1\morphology_catalogue_with_labels.csv"
OUTPUT_DIR = r"C:\Users\97549\Documents\trae_projects\Euclid_Joint_pytorch\Euclid_DINOv2_VIT\gz_outputs_official_dinov2"

MODEL_NAME = "dinov2_vits14"
USE_REGISTERS = False
FORCE_RELOAD = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    ensure_dir(OUTPUT_DIR)
    logs_dir = os.path.join(OUTPUT_DIR, "logs")
    ensure_dir(logs_dir)

    log_path = os.path.join(logs_dir, "download_official_dinov2.log")

    def log(msg: str):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    model_name = MODEL_NAME
    if USE_REGISTERS and not model_name.endswith("_reg"):
        model_name = model_name + "_reg"

    log(f"python={sys.version}")
    log(f"platform={platform.platform()}")
    log(f"torch={torch.__version__}")
    log(f"output_dir={OUTPUT_DIR}")
    log(f"data_root(ref only)={DATA_ROOT}")
    log(f"catalog_path(ref only)={CATALOG_PATH}")
    log(f"requested_model={model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    log("Start loading official DINOv2 model from torch.hub ...")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=True,
        force_reload=FORCE_RELOAD,
    )
    model = model.to(device)
    model.eval()
    log("Official DINOv2 model loaded successfully.")

    raw_state_dict = model.state_dict()

    ckpt = {
        "state_dict": raw_state_dict,
        "source": "official_torch_hub_facebookresearch_dinov2",
        "model_name": model_name,
        "download_time": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "note": (
            "This is an official DINOv2 backbone checkpoint exported from torch.hub. "
            "It is a clean general-pretrained backbone source, but it may not be directly "
            "compatible with custom backbones that use different patch size / input channels / "
            "implementation details."
        ),
        "reference_paths": {
            "data_root": DATA_ROOT,
            "catalog_path": CATALOG_PATH,
            "output_dir": OUTPUT_DIR,
        },
    }

    out_pth = os.path.join(OUTPUT_DIR, f"{model_name}_official_backbone.pth")
    torch.save(ckpt, out_pth)
    log(f"Saved official backbone checkpoint to: {out_pth}")

    out_state_only = os.path.join(OUTPUT_DIR, f"{model_name}_official_state_dict_only.pth")
    torch.save(raw_state_dict, out_state_only)
    log(f"Saved raw state_dict only to: {out_state_only}")

    sample_keys = list(raw_state_dict.keys())[:30]
    meta = {
        "source": "official_torch_hub_facebookresearch_dinov2",
        "model_name": model_name,
        "n_state_keys": len(raw_state_dict),
        "sample_keys": sample_keys,
        "download_time": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "important_warning": [
            "This checkpoint is the real official DINOv2 backbone.",
            "It may not directly load into your current custom DINOModel if architectures differ.",
            "Typical mismatch reasons: patch size 14 vs 16, RGB 3 channels vs grayscale 1 channel, implementation differences."
        ],
        "reference_paths": {
            "data_root": DATA_ROOT,
            "catalog_path": CATALOG_PATH,
            "output_dir": OUTPUT_DIR,
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, f"{model_name}_official_metadata.json")
    save_json(meta, meta_path)
    log(f"Saved metadata to: {meta_path}")

    try:
        dummy = torch.randn(1, 3, 518, 518, device=device)
        with torch.inference_mode():
            out = model(dummy)
        shape_info = None
        if hasattr(out, "shape"):
            shape_info = tuple(out.shape)
        elif isinstance(out, (list, tuple)) and len(out) > 0 and hasattr(out[0], "shape"):
            shape_info = [tuple(x.shape) if hasattr(x, "shape") else str(type(x)) for x in out]
        else:
            shape_info = str(type(out))
        log(f"Sanity forward passed. Output shape/info: {shape_info}")
    except Exception as e:
        log(f"Sanity forward skipped/failed: {repr(e)}")

    log("Done.")

    print("\n================ 使用提醒 ================\n")
    print("1) 你现在已经拿到了真正的官方 DINOv2 backbone。")
    print("2) 但它不一定能直接塞进你当前的自定义 DINOModel。")
    print("3) 如果 few-label benchmark 要做严格公平比较，推荐两种方案：")
    print("   A. benchmark 改成直接使用官方 DINOv2 backbone；")
    print("   B. 或者换一个与当前 DINOModel 架构完全兼容的通用预训练权重。")
    print("4) 不要再把 finetune_best.pth 当成“ImageNet DINOv2 baseline”。")
    print("\n=========================================\n")


if __name__ == "__main__":
    main()
