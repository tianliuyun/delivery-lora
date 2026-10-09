"""微调前后对比评估：同一组领域问题，base vs LoRA 适配器。

用法：
    python scripts/infer_compare.py --base models/Qwen2-0.5B-Instruct \
        --adapter output/qlora_qv
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

TEST_QUESTIONS = [
    "桌面云接入速度慢怎么排查？",
    "EDS分布式存储故障怎么处理？",
    "GPU虚拟化云桌面部署有哪些步骤？",
]


def load_model(base_model: str, adapter: str = None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.config.torch_dtype = torch.float16
    if adapter:
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
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2-0.5B-Instruct")
    parser.add_argument("--adapter", default="output/qlora_qv")
    args = parser.parse_args()

    print("🧪 加载 base 模型（4bit）...")
    base_model, tok = load_model(args.base)
    print("🧪 加载 LoRA 适配器...")
    lora_model, _ = load_model(args.base, args.adapter)

    for q in TEST_QUESTIONS:
        print("=" * 70)
        print(f"❓ {q}")
        t0 = time.time()
        base_ans = generate(base_model, tok, q)
        print(f"  [微调前] ({time.time()-t0:.1f}s) {base_ans[:180]}")
        t0 = time.time()
        lora_ans = generate(lora_model, tok, q)
        print(f"  [微调后] ({time.time()-t0:.1f}s) {lora_ans[:180]}")


if __name__ == "__main__":
    main()
