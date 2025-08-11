#!/usr/bin/env python3
import argparse, re, pandas as pd, torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

PROMPT = "Generate a disaster danger report in the format: Type: <TYPE> | Danger: <High/Medium/Low> | Action: <ACTION>"

def normalize(report:str):
    m = re.search(r"Type:\s*([^|]+)\|\s*Danger:\s*([^|]+)\|\s*Action:\s*(.+)$", report, re.I)
    if not m:
        return "Type: Unknown | Danger: Medium | Action: Monitoring."
    t, d, a = [x.strip() for x in m.groups()]
    d = d.capitalize()
    if d not in ["High","Medium","Low"]: d = "Medium"
    return f"Type: {t} | Danger: {d} | Action: {a}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./blip2_crisismmd_qformer")
    ap.add_argument("--test_csv", default="test20.csv")
    ap.add_argument("--out_csv", default="reports.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(args.ckpt)
    model = Blip2ForConditionalGeneration.from_pretrained(args.ckpt).to(device).eval()

    df = pd.read_csv(args.test_csv)
    rows = []
    for i, r in df.iterrows():
        img_path = r["image_path"]
        image_id = r.get("image_id", i+1)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device, torch.float16 if torch.cuda.is_available() else torch.float32)
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_new_tokens=64, length_penalty=0.1, no_repeat_ngram_size=3)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        rep = normalize(text)
        danger = re.search(r"Danger:\s*(High|Medium|Low)", rep).group(1)
        rows.append([image_id, rep, danger])

    pd.DataFrame(rows, columns=["Image_ID","Generated_Report","Danger_Level"]).to_csv(args.out_csv, index=False, encoding="utf-8")
    print("[OK] wrote", args.out_csv)

if __name__ == "__main__":
    main()
