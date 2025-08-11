#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import pandas as pd

REQUIRED = ["image_path","type","danger","action"]
OPTIONAL = ["caption"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="path to subset.csv")
    ap.add_argument("--images_root", default="", help="prefix for relative image_path")
    ap.add_argument("--val_ratio", type=float, default=0.12)
    ap.add_argument("--test_count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.subset)
    for c in REQUIRED:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] missing column: {c}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""

    # danger 归一化
    df["danger"] = df["danger"].astype(str).str.strip().str.capitalize()
    df.loc[~df["danger"].isin(["High","Medium","Low"]), "danger"] = "Medium"

    # 解析图片路径
    root = Path(args.images_root) if args.images_root else Path(".")
    abs_paths, missing = [], []
    for p in df["image_path"].astype(str):
        pp = Path(p)
        if not pp.is_absolute():
            pp = root / p
        pp = pp.resolve()
        abs_paths.append(str(pp))
        if not pp.exists():
            missing.append(str(pp))
    df["image_path"] = abs_paths
    if missing:
        print(f"[WARN] {len(missing)} images missing (first 10):")
        for m in missing[:10]: print(" -", m)

    # 生成训练目标文本
    df["type"]   = df["type"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip().str.rstrip(".")
    df["target"] = df.apply(lambda r: f"Type: {r['type']} | Danger: {r['danger']} | Action: {r['action']}.", axis=1)

    # 按 type 分层取 test20
    random.seed(args.seed)
    groups = {t: g.sample(frac=1.0, random_state=args.seed) for t,g in df.groupby("type")}
    types  = list(groups.keys())
    test_rows = []
    if len(df) >= args.test_count and types:
        i = 0
        while len(test_rows) < args.test_count and sum(len(g) for g in groups.values()) > 0:
            t = types[i % len(types)]
            if len(groups[t]) > 0:
                test_rows.append(groups[t].iloc[0])
                groups[t] = groups[t].iloc[1:]
            i += 1
    else:
        test_rows = df.sample(n=min(args.test_count, len(df)), random_state=args.seed).to_dict("records")
    test_df = pd.DataFrame(test_rows)

    remain_parts = [g for g in groups.values() if len(g) > 0]
    remain_df = pd.concat(remain_parts) if remain_parts else df.drop(test_df.index)

    val_size = max(1, int(len(remain_df) * args.val_ratio))
    val_df = remain_df.sample(n=val_size, random_state=args.seed)
    train_df = remain_df.drop(val_df.index)

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df[["image_path"]].assign(image_id=range(1, len(test_df)+1)).to_csv("test20.csv", index=False)

    print(f"[OK] Train:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}")
    if missing: print("[NOTE] Some images missing. Please fix paths above.")

if __name__ == "__main__":
    main()
