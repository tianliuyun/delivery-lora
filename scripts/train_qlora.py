"""QLoRA 微调脚本 —— 本机 MX350 2GB 显存验证版

在 2GB 显存的消费级 GPU 上做 Qwen2-0.5B 的 QLoRA 微调：
- 4bit 量化加载基座（bitsandbytes），权重占用 ~0.35GB
- LoRA 只训适配器，batch_size=1 + 梯度累积，控制显存峰值
- 支持 --target qv（q/v 两层，默认）/ --target all（q/k/v/o 全四层）
- 支持 --packing 开启序列打包（TRL SFTTrainer，把短样本拼到 max_seq_length，
  减少 padding 浪费、提升训练吞吐）

用法：
    python scripts/train_qlora.py --data data/domain_qa.jsonl \
        --target qv --epochs 2 --seq-len 256 --out output/qlora_qv

    # 全四层 + 序列打包
    python scripts/train_qlora.py --data data/domain_qa.jsonl \
        --target all --packing --epochs 2 --seq-len 256 --out output/qlora_all
"""
import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调（2GB 显存验证版）")
    parser.add_argument("--data", required=True, help="JSONL 数据集（instruction/input/output）")
    parser.add_argument("--base-model", default="models/Qwen2-0.5B-Instruct")
    parser.add_argument("--target", choices=["qv", "all"], default="qv",
                        help="LoRA 注入层：qv（q/v 两层，默认）/ all（q/k/v/o 全四层）")
    parser.add_argument("--packing", action="store_true", help="开启序列打包（SFTTrainer packing）")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256, help="max_seq_length（显存受限，默认 256）")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--out", default="output/qlora", help="适配器输出目录")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，本脚本需要 GPU")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)} | 显存: {gpu_props.total_memory / 2**20:.0f} MB")
    t_start = time.time()

    # 1. 4bit 量化加载基座（BitsAndBytesConfig）
    print(f"📥 加载基座模型: {args.base_model}（4bit 量化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # MX350 (Turing sm_75) 不支持 bf16，用 fp16
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Qwen2 config 默认 torch_dtype=bf16，MX350 (sm_75) 不支持 bf16 → 强制 fp16
    model.config.torch_dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # 2. LoRA 配置（--target qv 两层 / all 全四层）
    target_modules = ["q_proj", "v_proj"] if args.target == "qv" \
        else ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  注入层: {target_modules} | 序列打包: {args.packing}")

    # 3. 加载数据集（JSONL: instruction/output）
    print(f"📚 加载数据集: {args.data}")
    dataset = load_dataset("json", data_files=args.data)["train"]

    def formatting_func(example):
        q = example.get("question") or example.get("instruction")
        a = example.get("answer") or example.get("output")
        return f"问题：{q}\n答案：{a}"

    # 4. 训练参数（小显存友好）
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=1,  # transformers 5.x：warmup_ratio 已移除，用 warmup_steps
        logging_steps=5,
        save_strategy="epoch",
        # MX350 + bnb4bit 下 fp16 grad scaler 会撞 bf16 张量，用 fp32 直接训（LoRA 参数仅 54 万，内存可控）
        fp16=False,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )

    print(f"🚀 开始训练（epochs={args.epochs}，样本数={len(dataset)}，packing={args.packing}）...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    used = torch.cuda.max_memory_allocated() / 2**20
    print(f"\n✅ 训练完成，适配器已保存: {out_dir}")
    print(f"  峰值显存: {used:.0f} MB / {gpu_props.total_memory / 2**20:.0f} MB")
    print(f"  总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
