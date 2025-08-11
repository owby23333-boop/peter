
'''

''''''
python prepare.py --subset subset.csv --images_root D:\code\final_asmt --val_ratio 0.12 --test_count 20 --seed 42
python train_b2.py --model Salesforce/blip2-flan-t5-xl --epochs 4 --bsz 2 --grad_accum 8 --lr 2e-5 --ckpt_dir blip2_crisismmd_qformer --use_lora
python info_b2.py --ckpt blip2_crisismmd_qformer --test_csv test20.csv --out_csv reports.csv
python info_b2.py --ckpt D:\code\final_asmt\blip2_crisismmd_qformer --test_csv test20.csv

# 1) 用更省的 OPT-2.7b（推荐先把模型预下载到本地目录再传绝对路径）
python train2.py --model Salesforce/blip2-opt-2.7b
  --train_file train_small.csv --val_file val_small.csv
  --ckpt_dir D:\code\final_asmt\blip2_crisismmd_qformer
  --epochs 3 --bsz 1 --grad_accum 16 --lr 2e-5 --use_lora

# 2) 如果曾因黑屏/断电中断，支持断点续训（默认 resume_from_checkpoint=True）
#    也可以用短训快速出权重（比如只跑 80 步，并每 40 步落盘）
python train2.py --max_steps 80 --save_every 40 --no_eval
Salesforce/blip2-opt-2.7b
python D:\code\final_asmt\train2.py  --model D:\hf_models\blip2-opt-2.7b  --train_file D:\code\final_asmt\train_small.csv  --val_file   D:\code\final_asmt\val_small.csv  --ckpt_dir   D:\code\final_asmt\blip2_crisismmd_qformer  --max_steps 80 --save_every 40 --no_eval --bsz 1 --grad_accum 16 --lr 2e-5 --use_lora

python D:\code\final_asmt\train2.py  --model D:\hf_models\blip2-opt-2.7b  --train_file D:\code\final_asmt\train_small.csv  --val_file   D:\code\final_asmt\val_small.csv  --ckpt_dir   D:\code\final_asmt\blip2_crisismmd_qformer  --epochs 3 --bsz 1 --grad_accum 16 --lr 2e-5 --use_lora









'''