#!/usr/bin/env python3
import argparse, torch
from datasets import load_dataset
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# PROMPT = "Generate a disaster danger report in the format: Type: <TYPE> | Danger: <High/Medium/Low> | Action: <ACTION>"
from datasets import load_dataset, set_caching_enabled
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许截断JPEG
set_caching_enabled(False)              # 避免缓存脏样本带来迷惑




PROMPT = "Generate a disaster danger report in the format: Type: <TYPE> | Danger: <High/Medium/Low> | Action: <ACTION>"


def build_collate(processor):
    from torch.nn.utils.rnn import pad_sequence
    def collate(batch):
        pad_id = processor.tokenizer.pad_token_id or 0

        # pixel_values
        pixel_values = []
        for item in batch:
            pv = item["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.as_tensor(pv)
            pixel_values.append(pv)
        pixel_values = torch.stack(pixel_values)

        def to_long_list(key):
            out = []
            for item in batch:
                x = item[key]
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x, dtype=torch.long)
                else:
                    x = x.to(torch.long)
                out.append(x)
            return out

        input_ids      = pad_sequence(to_long_list("input_ids"), batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(to_long_list("attention_mask"), batch_first=True, padding_value=0)
        labels         = pad_sequence(to_long_list("labels"), batch_first=True, padding_value=pad_id)
        labels[labels == pad_id] = -100

        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}
    return collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--ckpt_dir", default="./blip2_crisismmd_qformer")
    args = ap.parse_args()

    from datasets import load_dataset, set_caching_enabled
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许截断 JPEG
    set_caching_enabled(False)  # 关缓存，避免脏样本残留


    processor = Blip2Processor.from_pretrained(args.model)
    dataset = load_dataset("csv", data_files={"train": "train_small.csv", "validation": "val_small.csv"})

    # 先单进程过滤“打不开的图片”，把真实坏样本剔掉
    def can_open(ex):
        try:
            with Image.open(ex["image_path"]) as im:
                im.verify()  # 只校验，不解码
            return True
        except Exception as e:
            # 可选：打印几条定位坏图
            # print("[BAD IMG]", ex["image_path"], e)
            return False

    dataset["train"] = dataset["train"].filter(can_open, num_proc=1, desc="Filtering broken train images")
    dataset["validation"] = dataset["validation"].filter(can_open, num_proc=1, desc="Filtering broken val images")

    PROMPT = "Generate a disaster danger report in the format: Type: <TYPE> | Danger: <High/Medium/Low> | Action: <ACTION>"

    def preprocess(ex):
        img = Image.open(ex["image_path"])
        if img.mode != "RGB":  # 调色板/带透明 -> 转 RGB
            img = img.convert("RGB")
        inputs = processor(images=img, text=PROMPT, return_tensors="pt")
        labels = processor.tokenizer(ex["target"], return_tensors="pt").input_ids
        ex["pixel_values"] = inputs["pixel_values"][0]
        ex["input_ids"] = inputs["input_ids"][0]
        ex["attention_mask"] = inputs["attention_mask"][0]
        ex["labels"] = labels[0]
        return ex

    cols = ["image_path", "type", "danger", "action", "caption", "target", "split", "image_id"]
    dataset = dataset.map(preprocess, remove_columns=cols, num_proc=1, desc="Preprocessing (single process)")


    model = Blip2ForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False

    # 只训练 Q-Former + language_projection
    for _, p in model.named_parameters(): p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith("qformer") or "language_projection" in n:
            p.requires_grad = True

    # 可选：LoRA 仅打到 Q-Former 的 Linear
    if args.use_lora:
        qformer_linear = [n for n,m in model.named_modules()
                          if n.startswith("qformer") and isinstance(m, torch.nn.Linear)]
        if not qformer_linear:  # 兜底
            qformer_linear = ["query","key","value","dense"]
        peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=qformer_linear)
        model = get_peft_model(model, peft_cfg)

    # 防呆垫片：去掉 Trainer 可能注入的 inputs_embeds
    from functools import wraps
    orig_forward = model.forward
    @wraps(orig_forward)
    def forward_patched(*a, **kw):
        kw.pop("inputs_embeds", None)
        kw.pop("labels_embeds", None)
        return orig_forward(*a, **kw)
    model.forward = forward_patched

    # 精度自动：优先 BF16（支持才开），否则 FP16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args_tr = TrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.wd,
        # ★ 关键：只跑 50 步，且第 50 步就保存
        max_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        eval_strategy="no",  # 短训不做 eval，节省时间/显存
        report_to=["none"],
        fp16=(not use_bf16), bf16=use_bf16,
        remove_unused_columns=False,
    )
    print(f"[INFO] bf16={args_tr.bf16} fp16={args_tr.fp16}")

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=build_collate(processor),
    )
    from pathlib import Path
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    try:
        trainer.train()
    finally:
        trainer.save_model(args.ckpt_dir)  # 写入 config + 权重
        processor.save_pretrained(args.ckpt_dir)  # 写入 tokenizer + image_processor
        print("[OK] saved to", args.ckpt_dir)
    # trainer.save_model(args.ckpt_dir)
    # processor.save_pretrained(args.ckpt_dir)

    import pandas as pd
    pd.DataFrame(trainer.state.log_history).to_csv("training_log.csv", index=False)
    print("[OK] saved to", args.ckpt_dir)

if __name__ == "__main__":
    main()
