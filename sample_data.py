#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def show_dist(df, name):
    if df.empty:
        print(f"[{name}] EMPTY")
        return
    print(f"[{name}] size = {len(df)}")
    print(pd.crosstab(df['type'], df['danger'], normalize='all').round(3))
    print()

def cap_group(df, by_cols, cap):
    parts = []
    for keys, g in df.groupby(by_cols):
        n = min(cap, len(g))
        parts.append(g.sample(n=n, random_state=42))
    return pd.concat(parts).reset_index(drop=True) if parts else df.head(0)

def frac_group(df, by_cols, frac):
    parts = []
    for keys, g in df.groupby(by_cols):
        k = max(1, int(round(len(g)*frac)))
        parts.append(g.sample(n=k, random_state=42))
    return pd.concat(parts).reset_index(drop=True) if parts else df.head(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", default="train.csv")
    ap.add_argument("--val_in",   default="val.csv")
    ap.add_argument("--train_out", default="train_small.csv")
    ap.add_argument("--val_out",   default="val_small.csv")
    ap.add_argument("--cap", type=int, default=None, help="每个(type,danger)最多保留多少条")
    ap.add_argument("--frac", type=float, default=None, help="每个(type,danger)保留比例，例如0.25")
    args = ap.parse_args()

    assert (args.cap is None) ^ (args.frac is None), "cap 与 frac 二选一"
    train = pd.read_csv(args.train_in)
    val   = pd.read_csv(args.val_in)

    by = ["type","danger"]
    print("=== BEFORE ===")
    show_dist(train, "train"); show_dist(val, "val")

    if args.cap is not None:
        train_s = cap_group(train, by, args.cap)
        val_s   = cap_group(val, by, max(1, args.cap//5))  # 验证集少一些
    else:
        train_s = frac_group(train, by, args.frac)
        val_s   = frac_group(val, by, min(0.5, args.frac)) # 验证集比例再小一点，避免太大

    # 保留原列
    keep_cols = [c for c in train.columns]
    train_s = train_s[keep_cols]
    val_s   = val_s[keep_cols]

    train_s.to_csv(args.train_out, index=False)
    val_s.to_csv(args.val_out, index=False)

    print("=== AFTER ===")
    show_dist(train_s, f"{args.train_out}")
    show_dist(val_s,   f"{args.val_out}")
    print(f"[OK] wrote {args.train_out} ({len(train_s)} rows), {args.val_out} ({len(val_s)} rows)")

if __name__ == "__main__":
    main()
