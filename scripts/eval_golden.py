"""黄金评测集评测：base vs QLoRA 适配器，LLM-as-Judge 判分。

用法：
    python scripts/eval_golden.py --base models/Qwen2-0.5B-Instruct \
        --adapter output/qlora_all_v2 --eval data/domain_qa_eval.jsonl

输出：
- 每个模型的准确率（LLM-as-Judge 判对错）
- 分类别（来源文档）准确率
- 生成答案样本落盘 /tmp/eval_golden_samples.json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# DeepSeek judge
ENV_PATH = Path.home() / ".hermes/workspace/tasks/llm-course-interview/01-基础学习/week15graph和llm/.env"
API_KEY = None
BASE_URL = "https://api.deepseek.com"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY"):
            API_KEY = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("DEEPSEEK_BASE_URL"):
            BASE_URL = line.split("=", 1)[1].strip().strip('"')

JUDGE_PROMPT = """你是云计算交付领域的资深工程师，请判断模型回答是否正确。

【问题】{question}
【标准答案】{reference}
【模型回答】{prediction}

判断标准：模型回答是否答对了问题的关键点（允许措辞不同，但核心事实/数值/步骤必须正确）。
请只输出一个 JSON：{{"correct": true/false, "reason": "一句话理由"}}"""


def judge_llm(question, reference, prediction, retries=1):
    if not API_KEY:
        return None
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction[:500])
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    for attempt in range(retries):
        try:
            # 硬超时：30s 无响应直接放弃该题 judge（返回 None，不算对错，不阻塞）
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                return json.loads(m.group(0))
            return {"correct": "true" in content.lower(), "reason": content[:100]}
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ⚠️ judge 超时/失败: {type(e).__name__}")
                return None
            time.sleep(1)


def load_model(base_model, adapter=None):
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


def generate(model, tokenizer, question, max_new=120):
    prompt = f"问题：{question}\n答案："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2-0.5B-Instruct")
    parser.add_argument("--adapter", default="output/qlora_all_v2")
    parser.add_argument("--eval", default="data/domain_qa_eval.jsonl")
    args = parser.parse_args()

    eval_data = [json.loads(l) for l in Path(args.eval).read_text().splitlines() if l.strip()]
    print(f"📋 黄金评测集 {len(eval_data)} 条")

    print("🧪 加载 base + LoRA 模型...")
    base_model, tok = load_model(args.base)
    lora_model, _ = load_model(args.base, args.adapter)

    results = {"base": [], "lora": []}
    for i, item in enumerate(eval_data):
        q, ref = item["question"], item["answer"]
        base_ans = generate(base_model, tok, q)
        lora_ans = generate(lora_model, tok, q)
        base_judge = judge_llm(q, ref, base_ans)
        lora_judge = judge_llm(q, ref, lora_ans)
        results["base"].append({"q": q, "ref": ref, "pred": base_ans, "judge": base_judge})
        results["lora"].append({"q": q, "ref": ref, "pred": lora_ans, "judge": lora_judge})
        if (i + 1) % 5 == 0 or i == len(eval_data) - 1:
            print(f"  [{i+1}/{len(eval_data)}]")
        time.sleep(0.3)

    # 汇总
    print("\n" + "=" * 60)
    print("评估报告（LLM-as-Judge）")
    print("=" * 60)
    summary = {}
    for name in ["base", "lora"]:
        judged = [r for r in results[name] if r["judge"]]
        correct = sum(1 for r in judged if r["judge"].get("correct"))
        acc = correct / len(judged) if judged else 0
        summary[name] = {"accuracy": round(acc, 4), "judged": len(judged), "total": len(results[name])}
        print(f"  {name}: 准确率 {acc:.1%} ({correct}/{len(judged)} judged of {len(results[name])})")

    if summary["base"]["judged"] and summary["lora"]["judged"]:
        diff = summary["lora"]["accuracy"] - summary["base"]["accuracy"]
        print(f"\n  差值（LoRA - base）: {diff:+.1%}")

    with open("/tmp/eval_golden_samples.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✅ 样本已存 /tmp/eval_golden_samples.json")


if __name__ == "__main__":
    main()
