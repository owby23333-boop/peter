#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable BLIP-2 trainer for Windows + RTX 4050
- 默认基座: Salesforce/blip2-opt-2.7b（更省磁盘/显存）
- 仅 Q-Former + language_projection 参与训练；ViT/LLM 在 CPU
- 支持 LoRA（仅注入 Q-Former 的 Linear）
- 过滤坏图，单进程 map（Windows 稳定）
- 健壮 collate（list→tensor，labels pad→-100）
- 训练中断也会强制保存 model + processor
"""

import os
from pathlib import Path
import argparse
import torch
from datasets import load_dataset, set_caching_enabled
from PIL import Image, ImageFile
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------- CLI ----------
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Salesforce/blip2-opt-2.7b",
                    help="Hugging Face repo id 或本地目录（推荐先下载到本地再传绝对路径）")
    ap.add_argument("--train_file", default="train.csv")
    ap.add_argument("--val_file", default="val.csv")
    ap.add_argument("--ckpt_dir", default="./blip2_crisismmd_qformer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bsz", type=int, default=1, help="单卡单步 batch（4050 建议从 1 起步）")
    ap.add_argument("--grad_accum", type=int, default=16, help="梯度累积步数，等效全局 batch = bsz*grad_accum")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--max_steps", type=int, default=80, help="短训快速出权重；设为 0 关闭按步上限")
    ap.add_argument("--save_every", type=int, default=40, help="按步保存间隔，用于断点续训/避免白跑")
    ap.add_argument("--no_eval", action="store_true", help="训练期间不做验证，进一步降低峰值")
    ap.add_argument("--offload_dir", default="./offload_blip2", help="CPU卸载目录（确保在大盘）")
    return ap.parse_args()

# ---------- Data pipeline ----------
def prepare_dataset(train_file, val_file, processor):
    set_caching_enabled(False)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ds = load_dataset("csv", data_files={"train": train_file, "validation": val_file})

    def can_open(ex):
        try:
            with Image.open(ex["image_path"]) as im:
                im.verify()
            return True
        except Exception:
            return False

    ds["train"] = ds["train"].filter(can_open, num_proc=1, desc="Filtering broken train images")
    ds["validation"] = ds["validation"].filter(can_open, num_proc=1, desc="Filtering broken val images")

    PROMPT = "Generate a disaster danger report in the format: Type: <TYPE> | Danger: <High/Medium/Low> | Action: <ACTION>"

    def preprocess(ex):
        img = Image.open(ex["image_path"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = processor(images=img, text=PROMPT, return_tensors="pt")
        labels = processor.tokenizer(ex["target"], return_tensors="pt").input_ids
        ex["pixel_values"]   = inputs["pixel_values"][0]
        ex["input_ids"]      = inputs["input_ids"][0]
        ex["attention_mask"] = inputs["attention_mask"][0]
        ex["labels"]         = labels[0]
        return ex

    cols = ["image_path","type","danger","action","caption","target","split","image_id"]
    ds = ds.map(preprocess, remove_columns=cols, num_proc=1, desc="Preprocessing (single process)")
    return ds

def build_collate(processor):
    from torch.nn.utils.rnn import pad_sequence
    def collate(batch):
        pad_id = processor.tokenizer.pad_token_id or 0

        # pixel_values: list/np -> tensor
        pvs = []
        for it in batch:
            pv = it["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.as_tensor(pv)
            pvs.append(pv)
        pixel_values = torch.stack(pvs)

        def to_long_list(key):
            out = []
            for it in batch:
                x = it[key]
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

        return {"pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}
    return collate

# ---------- Model loading (CPU/GPU split + LoRA optional) ----------
def load_model_and_processor(model_id, offload_dir, use_lora):
    is_local = Path(model_id).exists()
    processor = Blip2Processor.from_pretrained(model_id, local_files_only=is_local,use_fast=False)

    # 设备映射：仅让 Q-Former + language_projection 上 GPU
    os.makedirs(offload_dir, exist_ok=True)
    device_map = {"qformer": "cuda",
                  "language_projection": "cuda",
                  "vision_model": "cpu",   # 有些版本命名为 vision_tower，下面 try/except 兜底
                  "language_model": "cpu"}
    max_mem = {0: "5GiB", "cpu": "32GiB"}  # 按你内存/显存酌情调整

    try:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=is_local,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_mem,
            offload_folder=offload_dir,
            offload_state_dict=True
        )
    except Exception:
        # 兼容 vision_tower 命名
        device_map.pop("vision_model", None)
        device_map["vision_tower"] = "cpu"
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=is_local,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_mem,
            offload_folder=offload_dir,
            offload_state_dict=True
        )

    # 冻结除 Q-Former + language_projection 之外的参数
    for _, p in model.named_parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith("qformer") or "language_projection" in n:
            p.requires_grad = True

    # LoRA（仅注入 Q-Former 的 Linear）
    if use_lora:
        qformer_linear = [n for n, m in model.named_modules()
                          if n.startswith("qformer") and isinstance(m, torch.nn.Linear)]
        if not qformer_linear:
            qformer_linear = ["query", "key", "value", "dense"]  # 子串兜底
        peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                              r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=qformer_linear)
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # 训练稳定性
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # 去掉 Trainer 可能注入的不兼容键
    from functools import wraps
    orig_forward = model.forward
    @wraps(orig_forward)
    def forward_patched(*a, **kw):
        kw.pop("inputs_embeds", None)
        kw.pop("labels_embeds", None)
        return orig_forward(*a, **kw)
    model.forward = forward_patched

    return processor, model

# ---------- Main ----------
def main():
    args = build_args()

    # 环境自检
    print("[CUDA] available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[CUDA] device:", torch.cuda.get_device_name(0))

    # 模型 & 处理器
    processor, model = load_model_and_processor(args.model, args.offload_dir, args.use_lora)

    # 数据
    ds = prepare_dataset(args.train_file, args.val_file, processor)
    collate = build_collate(processor)

    # 精度：优先 FP16（4050 OK）
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_flag = not use_bf16

    # 训练参数（短训强制落盘，避免中途断电/黑屏白跑）
    save_strategy = "steps" if args.max_steps and args.save_every else "epoch"
    eval_strategy = "no" if args.no_eval else ("steps" if save_strategy == "steps" else "epoch")

    tr_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.wd,
        num_train_epochs=args.epochs if not args.max_steps else 1,
        max_steps=args.max_steps if args.max_steps else -1,
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=(args.save_every if save_strategy == "steps" else None),
        save_total_limit=2,
        eval_strategy=eval_strategy,
        report_to=["none"],
        fp16=fp16_flag, bf16=use_bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    print(f"[PRECISION] bf16={tr_args.bf16} fp16={tr_args.fp16}")

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds["train"],
        eval_dataset=None if args.no_eval else ds["validation"],
        data_collator=collate,
    )

    # 训练 & 总是保存
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    try:
        trainer.train(resume_from_checkpoint=True)
    finally:
        trainer.save_model(args.ckpt_dir)
        processor.save_pretrained(args.ckpt_dir)
        print("[OK] saved to", args.ckpt_dir)

    # 写训练日志
    import pandas as pd
    pd.DataFrame(trainer.state.log_history).to_csv("training_log.csv", index=False)
    print("[OK] wrote training_log.csv")

if __name__ == "__main__":
    main()
