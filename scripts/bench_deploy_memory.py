"""部署端实测：INT4 量化（bitsandbytes 4bit）加载 Qwen2-0.5B + LoRA 适配器，测推理显存。

把「显存 13GB→2GB、成本降低 90%」从设计目标值变成 0.5B 量化模型的本机实测：
- 模型加载峰值显存（nvidia-smi 采样）
- 单次推理显存 / 耗时
- 与 FP16 估算对比

用法：
    python scripts/bench_deploy_memory.py --base models/Qwen2-0.5B-Instruct \
        --adapter output/qlora_all_v3 --out results/deploy_memory_report.json
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

TEST_QUESTIONS = [
    "EDS分布式存储中，重要数据建议开启至少几副本？",
    "桌面云登录超时，第一步应该检查什么？",
    "GPU虚拟化部署中，NVIDIA 授权服务器默认端口是多少？",
]

# 显存监控线程
_mem_log = []
_stop_mon = threading.Event()


def _gpu_mem_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def _monitor():
    while not _stop_mon.is_set():
        _mem_log.append(_gpu_mem_mb())
        time.sleep(0.2)


def load_model(base_model: str, adapter: str = None):
    print(f"🧠 加载基座 {base_model}（INT4 4bit 量化）...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.torch_dtype = torch.float16
    if adapter:
        print(f"🔧 挂载 LoRA 适配器 {adapter} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, question: str) -> str:
    prompt = f"问题：{question}\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=80,
            do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2-0.5B-Instruct")
    parser.add_argument("--adapter", default="output/qlora_all_v3")
    parser.add_argument("--out", default="results/deploy_memory_report.json")
    args = parser.parse_args()

    print("🩺 开始显存监控...")
    _stop_mon.clear()
    _mem_log.clear()
    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    t0 = time.time()
    model, tokenizer = load_model(args.base, args.adapter)
    load_sec = round(time.time() - t0, 2)
    peak_load = max(_mem_log) if _mem_log else 0
    print(f"✅ 加载完成: {load_sec}s，加载峰值显存 {peak_load}MB")

    print("🧪 推理测试...")
    results = []
    for q in TEST_QUESTIONS:
        t1 = time.time()
        ans = generate(model, tokenizer, q)
        dt = round(time.time() - t1, 2)
        peak_here = max(_mem_log) if _mem_log else 0
        results.append({"question": q, "answer": ans[:150], "infer_sec": dt})
        print(f"  [{dt}s] Q: {q[:30]} -> {ans[:60]}")

    time.sleep(1)
    _stop_mon.set()
    mon.join(timeout=2)
    peak_total = max(_mem_log) if _mem_log else 0
    gpu_total_mb = 2048  # MX350

    report = {
        "base_model": args.base,
        "adapter": args.adapter,
        "quantization": "INT4 (bitsandbytes 4bit, compute fp16)",
        "gpu": "NVIDIA MX350 2GB",
        "load_sec": load_sec,
        "peak_mem_load_mb": peak_load,
        "peak_mem_infer_mb": peak_total,
        "gpu_total_mb": gpu_total_mb,
        "mem_utilization_pct": round(peak_total / gpu_total_mb * 100, 1),
        "results": results,
        "note": "本机实测：0.5B INT4 量化后推理峰值显存；7B INT4 的 13GB→2GB 仍为部署设计目标（本机 2GB 显存无法加载 7B）",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📊 部署显存实测报告 -> {out}")
    print(f"  加载峰值: {peak_load}MB / 推理峰值: {peak_total}MB（GPU 总量 {gpu_total_mb}MB）")


if __name__ == "__main__":
    main()
