#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CrisisMMD v1.0 -> subset.csv
- Read official TSVs under annotations/
- Filter informative samples with confidence thresholds
- Map Type, Danger, Action
- Balance per type, split train/val/test
- Output columns: image_path,type,danger,action,split,image_id
"""

import os, re, argparse, hashlib, random
from typing import Optional, List, Dict, Tuple
import pandas as pd
from pathlib import Path

# -----------------------------
# Configurable dictionaries
# -----------------------------

TYPE_KEYWORDS = {
    "flood": [
        "flood", "flooded", "flooding", "inundat", "water level", "overflow",
        "storm surge", "harvey", "hurricane", "typhoon", "cyclone"
    ],
    "wildfire": [
        "wildfire", "forest fire", "bushfire", "smoke plume", "blaze", "wild fire"
    ],
    "earthquake": ["earthquake", "quake", "aftershock", "seismic"],
    "building collapse": ["collapse", "collapsed building", "rubble", "debris"],
    "landslide": ["landslide", "mudslide", "rockslide", "slope failure"],
    "storm": ["storm", "hurricane", "typhoon", "tornado", "cyclone", "strong winds"],
}

EVENT_TYPE_HINTS = {
    # common 2017 events (可按需要补充/修改)
    "harvey": "flood",
    "irma": "storm",
    "maria": "storm",
    "mexico": "earthquake",
    "earthquake": "earthquake",
    "wildfire": "wildfire",
    "california": "wildfire",
    "flood": "flood",
    "hurricane": "storm",
}

DANGER_KEYWORDS = {
    "high": ["severe", "major", "fatalities", "casualties", "trapped", "collapsed",
             "evacuate now", "urgent rescue", "need rescue", "critical"],
    "medium": ["damaged", "blocked", "flooded street", "power outage", "heavy smoke",
               "bridge closed", "traffic halted"],
    "low": ["minor", "under control", "monitor", "contained", "light damage"],
}
DANGER_PRIORITY = ["high", "medium", "low"]

ACTION_TEMPLATES = {
    ("flood", "High"): "Immediate boat rescue and evacuation.",
    ("flood", "Medium"): "Deploy pumps and set up evacuation shelters.",
    ("flood", "Low"): "Monitor water level and secure sandbags.",

    ("wildfire", "High"): "Immediate evacuation and aerial firefighting support.",
    ("wildfire", "Medium"): "Establish firebreaks and prepare evacuation.",
    ("wildfire", "Low"): "Monitor hotspots and restrict access.",

    ("earthquake", "High"): "Rapid search-and-rescue and medical triage.",
    ("earthquake", "Medium"): "Assess structural integrity and cordon hazardous zones.",
    ("earthquake", "Low"): "Inspect buildings and prepare relief supplies.",

    ("building collapse", "High"): "USAR team deployment and immediate rescue.",
    ("building collapse", "Medium"): "Secure perimeter and assess structural risks.",
    ("building collapse", "Low"): "Monitor area and conduct engineering inspections.",

    ("landslide", "High"): "Immediate evacuation and search for survivors.",
    ("landslide", "Medium"): "Stabilize slope and clear blocked roads.",
    ("landslide", "Low"): "Monitor terrain and warn nearby residents.",

    ("storm", "High"): "Immediate evacuation from vulnerable zones and rescue operations.",
    ("storm", "Medium"): "Distribute relief supplies and restore critical services.",
    ("storm", "Low"): "Monitor weather and prepare shelters.",
}
GENERIC_ACTION = {
    "High": "Immediate rescue and evacuation.",
    "Medium": "Mitigate risks and prepare evacuation.",
    "Low": "Monitoring and preventative measures.",
}

# -----------------------------
# Helpers
# -----------------------------

def safe_lower(x: Optional[str]) -> str:
    return "" if x is None else str(x).lower()

def hash_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12].upper()

def map_danger_from_image_damage(label: str) -> Optional[str]:
    if not label or pd.isna(label):
        return None
    l = safe_lower(label)
    if "severe" in l:
        return "High"
    if "mild" in l:
        return "Medium"
    if "little" in l or "no" in l:
        return "Low"
    if "dont" in l or "unknown" in l or "can't" in l:
        return None
    return None

def infer_danger_from_text(blob: str) -> str:
    b = safe_lower(blob)
    for lvl in DANGER_PRIORITY:
        for kw in DANGER_KEYWORDS[lvl]:
            if kw in b:
                return lvl.capitalize()
    return "Medium"

def guess_type(event_name: str, image_rel_path: str, tweet_text: str) -> Optional[str]:
    """
    综合三路信号：
    1) 事件名（来自 TSV 文件名或路径，例如 'hurricane_harvey'）
    2) image_path 顶层目录名（data_image/<event>/...）
    3) tweet_text 关键词
    """
    candidates = []
    # 1) event name hints
    ev = safe_lower(event_name)
    for k, t in EVENT_TYPE_HINTS.items():
        if k in ev:
            candidates.append(t)
    # 2) top folder name
    parts = [p for p in Path(image_rel_path).parts if p not in (".",)]
    if parts:
        top = safe_lower(parts[0])
        for k, t in EVENT_TYPE_HINTS.items():
            if k in top:
                candidates.append(t)
    # 3) keyword scan in text
    txt = safe_lower(tweet_text)
    for t, kws in TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                candidates.append(t)
                break

    if not candidates:
        return None
    # 选择出现频次最高的候选
    return max(set(candidates), key=candidates.count)

def choose_action(dtype: str, danger: str) -> str:
    return ACTION_TEMPLATES.get((dtype, danger), GENERIC_ACTION.get(danger, "Monitoring."))

def informative_ok(row, img_thresh: float, txt_thresh: float) -> bool:
    """
    行是否 informative：
    - 优先 image_info == 'informative' 且 image_info_conf >= img_thresh
    - 或 text_info == 'informative' 且 text_info_conf >= txt_thresh
    """
    ii = safe_lower(row.get("image_info", ""))
    ti = safe_lower(row.get("text_info", ""))
    iic = float(row.get("image_info_conf", 0) or 0)
    tic = float(row.get("text_info_conf", 0) or 0)
    if ii == "informative" and iic >= img_thresh:
        return True
    if ti == "informative" and tic >= txt_thresh:
        return True
    return False

# -----------------------------
# Main
# -----------------------------

def main(args):
    random.seed(args.seed)
    root = Path(args.root).resolve()
    ann_dir = root / "annotations"
    img_root = root / "data_image"
    assert ann_dir.exists(), f"annotations/ not found under {root}"
    assert img_root.exists(), f"data_image/ not found under {root}"

    # 读取所有 TSV 并标注 event 名（来自文件名）
    frames = []
    for tsv in ann_dir.glob("*.tsv"):
        ev_name = tsv.stem  # 文件名作为事件名
        df = pd.read_csv(tsv, sep="\t", dtype=str, na_filter=False)
        df["__event__"] = ev_name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No TSV files found in annotations/")

    df = pd.concat(frames, ignore_index=True)

    # 标准化重要列
    need_cols = [
        "tweet_id","image_id","tweet_text","image_url","image_path",
        "text_info","text_info_conf","image_info","image_info_conf",
        "text_human","image_human","image_damage","image_damage_conf",
        "__event__"
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = ""

    # 仅保留 informative
    mask_inf = df.apply(lambda r: informative_ok(r, args.img_info_thresh, args.txt_info_thresh), axis=1)
    df = df[mask_inf].copy()
    if df.empty:
        raise RuntimeError("No informative rows after filtering. Loosen thresholds or check TSVs.")

    # 确保 image_path 可用（优先用 data_image 下路径；若缺失，用 image_url 仅保留相对路径名）
    kept = []
    for _, r in df.iterrows():
        rel_path = r["image_path"].strip()
        # 部分数据 image_path 为相对路径（如 'hurricane_harvey/images_day1/xxx.jpg'）
        # 确保文件存在；若不存在，仍保留相对路径（训练前自行检查/补齐文件）
        # 只要相对路径非空就先使用
        if rel_path:
            final_rel = rel_path
        else:
            # 用 URL 兜底：提取文件名，挂到 event 文件夹下（你也可以按需要做更复杂的映射）
            url = r["image_url"].strip()
            if not url:
                continue
            fname = re.sub(r"[^a-zA-Z0-9._-]", "_", url.split("/")[-1] or "unknown.jpg")
            final_rel = f"{r['__event__']}/{fname}"

        # 生成稳定的 image_id（若原表有 image_id 则复用）
        image_id = r["image_id"].strip() or f"{r['tweet_id'].strip()}_{hash_id(final_rel)}"

        # Danger：优先 image_damage
        danger = map_danger_from_image_damage(r["image_damage"])
        if danger is None:
            # 退化到关键词 + human 标签
            blob = " ".join([r.get("tweet_text",""), r.get("text_human",""), r.get("image_human","")])
            danger = infer_danger_from_text(blob)

        # Type：事件名/路径/文本 综合推断
        dtype = guess_type(r["__event__"], final_rel, r["tweet_text"])
        if dtype is None:
            # 严格起见，跳过未知类型；若想保留可改为 "other"
            continue

        action = choose_action(dtype, danger)

        kept.append({
            "image_path": final_rel,     # 相对 data_image 的路径
            "type": dtype,
            "danger": danger,
            "action": action,
            "tweet_text": r["tweet_text"],
            "image_id": image_id,
            "__event__": r["__event__"],
        })

    if not kept:
        raise RuntimeError("No rows left after type/danger mapping. Check mappings or thresholds.")

    clean = pd.DataFrame(kept)

    # 去重（按 image_id / image_path）
    before = len(clean)
    clean = clean.drop_duplicates(subset=["image_id"]).copy()
    clean = clean.drop_duplicates(subset=["image_path"]).copy()
    after = len(clean)
    print(f"Dedup: {before} -> {after}")

    # 可选：每类最大样本数，便于均衡
    if args.max_per_type > 0:
        parts = []
        for t, sub in clean.groupby("type"):
            if len(sub) > args.max_per_type:
                parts.append(sub.sample(args.max_per_type, random_state=args.seed))
            else:
                parts.append(sub)
        clean = pd.concat(parts, ignore_index=True)

    # 生成 test 集（固定数量，尽量覆盖所有 type）
    types = list(clean["type"].unique())
    pool = {t: clean[clean["type"]==t].sample(frac=1.0, random_state=args.seed) for t in types}

    # 优先每类至少取 1
    picked_ids = set()
    test_rows = []
    for t in types:
        sub = pool[t]
        if not sub.empty:
            row = sub.iloc[0]
            test_rows.append(row)
            picked_ids.add(row["image_id"])

    # 填满 test_count
    if len(test_rows) < args.test_count:
        flat = clean.sample(frac=1.0, random_state=args.seed)
        for _, r in flat.iterrows():
            if len(test_rows) >= args.test_count:
                break
            if r["image_id"] in picked_ids:
                continue
            test_rows.append(r)
            picked_ids.add(r["image_id"])

    test_ids = {r["image_id"] for r in test_rows}
    remaining = clean[~clean["image_id"].isin(test_ids)].copy()

    # 按 type 分层切分 val
    def stratified_split(rem: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        val_parts, train_parts = [], []
        for t, sub in rem.groupby("type"):
            k = max(1, int(len(sub) * ratio))
            val_part = sub.sample(k, random_state=seed) if len(sub) > k else sub
            val_parts.append(val_part)
            train_parts.append(sub.drop(val_part.index))
        return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)

    train_df, val_df = stratified_split(remaining, args.val_ratio, args.seed)

    # 只保留训练必需列，路径保持相对于 data_image（训练脚本里会拼 root）
    def to_out(df_in: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = df_in.copy()
        out["split"] = split_name
        return out[["image_path","type","danger","action","split","image_id"]]

    out_df = pd.concat([
        to_out(train_df, "train"),
        to_out(val_df,   "val"),
        to_out(pd.DataFrame(test_rows), "test")
    ], ignore_index=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "subset.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved -> {out_csv}")
    print(out_df["split"].value_counts())
    print("Types:", sorted(out_df['type'].unique()))
    print("Danger dist:\n", out_df['danger'].value_counts())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,  default=r"D:\code\final_asmt")
    ap.add_argument("--outdir", type=str, default=r"D:\code\final_asmt")
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--test-count", type=int, default=20)
    ap.add_argument("--max-per-type", type=int, default=0, help="Cap per type after cleaning (0 = no cap)")
    ap.add_argument("--img-info-thresh", type=float, default=0.6)
    ap.add_argument("--txt-info-thresh", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
