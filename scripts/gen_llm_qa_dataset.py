"""用 DeepSeek 从交付文档生成高质量领域 QA 数据（训练集 + 独立黄金评测集）。

数据源：delivery-rag/data/sample_docs 的 3 篇交付文档
- 训练集 data/domain_qa_train.jsonl  ：覆盖各章节的通用 QA（~200 条）
- 黄金评测集 data/domain_qa_eval.jsonl：聚焦故障码/型号/配置参数类硬问题（~100 条，与训练集分离）

设计：
- LLM-as-Generator：每篇文档按章节拆成小段，分别让 DeepSeek 生成 QA，保证覆盖全文档
- 硬问题专门生成：评测集 prompt 强制要"具体数值/型号/步骤"，避免泛泛而谈
- 可溯源：每条带 source 字段指向文档 + 章节
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---- DeepSeek 配置 ----
ENV_PATH = Path.home() / ".hermes/workspace/tasks/llm-course-interview/01-基础学习/week15graph和llm/.env"


def load_key():
    key = base = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("DEEPSEEK_BASE_URL"):
                base = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")
    return key, base or "https://api.deepseek.com"


API_KEY, BASE_URL = load_key()

# 语料来源：同工作区的 delivery-rag 项目（data/sample_docs），可用环境变量 RAG_DOCS_DIR 覆盖
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", str(Path(__file__).resolve().parents[2] / "delivery-rag" / "data" / "sample_docs")))

# 文档 → 章节拆分（按 markdown 标题）
def split_sections(text):
    sections = []
    cur_title = "概述"
    cur = []
    for line in text.splitlines():
        if line.startswith("#"):
            if cur:
                sections.append((cur_title, "\n".join(cur).strip()))
            cur_title = line.lstrip("#").strip()
            cur = []
        else:
            cur.append(line)
    if cur:
        sections.append((cur_title, "\n".join(cur).strip()))
    return [s for s in sections if s[1]]


def call_deepseek(prompt, max_tokens=1500, temperature=0.7, retries=3):
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def gen_train_qa(doc_name, section_title, content):
    """生成覆盖章节的通用 QA（训练集用）"""
    prompt = f"""你是资深云计算交付工程师，正在为领域大模型微调构造训练数据。
请基于下面文档片段，生成 5 个高质量问答对（中文）。

要求：
1. 问题要多样：现象类（怎么排查/什么原因）、操作类（怎么做/步骤）、知识类（是什么/有什么用）
2. 答案必须严格来自文档片段内容，不编造文档里没有的信息
3. 每个答案 30-80 字，简洁准确
4. 输出 JSON 数组：[{{"question": "...", "answer": "..."}}]

文档标题：{doc_name}
章节：{section_title}
文档片段：
{content}"""
    resp = call_deepseek(prompt)
    try:
        m = re.search(r"\[.*\]", resp, re.S)
        items = json.loads(m.group(0))
        for it in items:
            it["source"] = f"{doc_name} > {section_title}"
        return items
    except Exception as e:
        print(f"  ⚠️ 解析失败: {e}\n  {resp[:200]}")
        return []


def gen_eval_qa(doc_name, content):
    """生成黄金评测集：硬问题（故障码/型号/配置参数/具体数值），与训练集分离"""
    prompt = f"""你是资深云计算交付工程师，正在构造"黄金评测集"来检验微调模型对领域硬知识的掌握。
请基于下面文档，生成 10 个【硬核问题+标准答案】。

硬核问题的标准：
1. 必须能给出唯一正确答案，可客观判分（涉及具体数值、型号、端口、阈值、步骤顺序）
2. 例如：某指标的建议值是多少？某个故障的正确排查顺序？某个组件的推荐型号？
3. 不要开放式问题（"你怎么看"）或主观问题
4. 答案必须严格来自文档，20-60 字

输出 JSON 数组：[{{"question": "...", "answer": "..."}}]

文档标题：{doc_name}
文档全文：
{content}"""
    resp = call_deepseek(prompt)
    try:
        m = re.search(r"\[.*\]", resp, re.S)
        items = json.loads(m.group(0))
        for it in items:
            it["source"] = doc_name
            it["category"] = "硬知识"
        return items
    except Exception as e:
        print(f"  ⚠️ 解析失败: {e}\n  {resp[:200]}")
        return []


def main():
    docs = sorted(DOCS_DIR.glob("*.md"))
    print(f"加载文档 {len(docs)} 篇")

    train_all, eval_all = [], []
    for doc in docs:
        text = doc.read_text()
        sections = split_sections(text)
        print(f"\n📄 {doc.name}: {len(sections)} 个章节")
        # 训练集：逐章节生成
        for title, content in sections:
            if len(content) < 20:
                continue
            items = gen_train_qa(doc.name, title, content)
            train_all.extend(items)
            print(f"  [{title}] +{len(items)} 训练条")
            time.sleep(0.5)
        # 评测集：整篇生成硬问题
        evals = gen_eval_qa(doc.name, text)
        eval_all.extend(evals)
        print(f"  🎯 评测集 +{len(evals)} 条")
        time.sleep(0.5)

    out_dir = Path(__file__).resolve().parents[1] / "data"
    train_path = out_dir / "domain_qa_train.jsonl"
    eval_path = out_dir / "domain_qa_eval.jsonl"

    with open(train_path, "w") as f:
        for it in train_all:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for it in eval_all:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n✅ 训练集 {len(train_all)} 条 → {train_path}")
    print(f"✅ 评测集 {len(eval_all)} 条 → {eval_path}")
    # 样例展示
    print("\n--- 训练集样例 ---")
    for it in train_all[:3]:
        print(f"Q: {it['question']}\nA: {it['answer'][:60]}")
    print("\n--- 评测集样例（硬问题）---")
    for it in eval_all[:5]:
        print(f"Q: {it['question']}\nA: {it['answer'][:60]}")


if __name__ == "__main__":
    main()
