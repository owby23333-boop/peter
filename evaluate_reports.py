#!/usr/bin/env python3
import argparse, re, pandas as pd, numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def normalize_text(s: str):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s)  # 简单去标点
    return s

def extract_danger(report: str):
    m = re.search(r"danger\s*:\s*(high|medium|low)", report, flags=re.I)
    return m.group(1).capitalize() if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="generated_reports.csv")
    ap.add_argument("--ref",  default="refs.csv", help="包含 Image_ID 与 caption（可加 danger 列）")
    ap.add_argument("--out",  default="metrics_results.csv")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    ref  = pd.read_csv(args.ref)

    # 对齐
    key = "Image_ID" if "Image_ID" in pred.columns and "Image_ID" in ref.columns else "image_id"
    df = pd.merge(pred, ref, left_on=key, right_on=key, how="inner", suffixes=("_pred","_ref"))
    if df.empty:
        raise SystemExit("对齐为空：请确认两侧都有 Image_ID（或 image_id）并一致。")

    # 文本归一 & 构造多参考
    refs = [[normalize_text(x)] for x in df["caption"].fillna("").astype(str)]
    hyps = [normalize_text(x) for x in df["Generated_Report"].fillna("").astype(str)]

    # BLEU-4（平滑）
    smoothie = SmoothingFunction().method7
    bleu4 = corpus_bleu(list(map(lambda x:[x], [r[0].split() for r in refs])),
                        [h.split() for h in hyps], smoothing_function=smoothie)

    # ROUGE（1/2/L）
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for r, h in zip(refs, hyps):
        sc = scorer.score(r[0], h)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)
    rouge1, rouge2, rougeL = np.mean(r1), np.mean(r2), np.mean(rL)

    # METEOR（逐样本平均）
    mets = []
    for r, h in zip(refs, hyps):
        mets.append(meteor_score([r[0]], h))
    meteor = float(np.mean(mets))

    # 结构化指标：危险等级准确率/宏F1（可选，若参考有 danger 列）
    acc = macro_f1 = None
    if "danger" in df.columns:
        ref_d = df["danger"].astype(str).str.strip().str.capitalize().replace({"Severe":"High","Mild":"Medium"})
        pred_d = df["Generated_Report"].apply(extract_danger).fillna("Medium")
        labels = ["High","Medium","Low"]
        cm = pd.crosstab(ref_d, pred_d, dropna=False).reindex(index=labels, columns=labels, fill_value=0)
        acc = np.trace(cm.values) / cm.values.sum()
        # 宏F1
        f1s = []
        for i,l in enumerate(labels):
            tp = cm.iloc[i,i]
            fp = cm.iloc[:,i].sum() - tp
            fn = cm.iloc[i,:].sum() - tp
            prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))

    # 导出
    out = pd.DataFrame([{
        "BLEU-4": round(bleu4, 4),
        "ROUGE-1": round(rouge1, 4),
        "ROUGE-2": round(rouge2, 4),
        "ROUGE-L": round(rougeL, 4),
        "METEOR": round(meteor, 4),
        "Danger_Accuracy": (round(acc,4) if acc is not None else "N/A"),
        "Danger_MacroF1": (round(macro_f1,4) if macro_f1 is not None else "N/A"),
        "Notes": "Lowercased, punctuation-normalized; corpus-level BLEU with smoothing."
    }])
    out.to_csv(args.out, index=False)
    print(out)

if __name__ == "__main__":
    main()
