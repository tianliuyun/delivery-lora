"""从交付文档生成领域 QA 数据集（JSONL，供 QLoRA 微调）。

数据源：delivery-rag/data/sample_docs（GPU虚拟化云桌面部署指南/桌面云接入优化指南/EDS分布式存储故障排查）
用途：交付领域 LoRA 微调的领域数据；程序化模板生成，标注来源可溯源。
"""
import json
import os
import sys
from pathlib import Path

# 语料来源：同工作区的 delivery-rag 项目（data/sample_docs）。若不存在，可先单独生成或调整路径。
_PROJECTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS / "delivery-lora" / "src"))
sys.path.insert(0, str(_PROJECTS / "delivery-rag" / "src"))

from delivery_lora.data.dataset_builder import DatasetBuilder  # noqa: E402
from rag.document.loader import DocumentLoader  # noqa: E402
from rag.document.chunker import TextChunker  # noqa: E402


def main():
    docs_dir = _PROJECTS / "delivery-rag" / "data" / "sample_docs"
    out_path = Path(__file__).resolve().parents[1] / "data" / "domain_qa.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs = DocumentLoader.load(str(docs_dir))
    print(f"加载文档 {len(docs)} 篇")

    # 先分块（fixed 512 字符），再按块生成 QA，扩大数据量
    chunker = TextChunker(strategy="fixed", child_size=512, overlap=50)
    chunks = []
    for d in docs:
        for c in chunker.chunk(d.content):
            chunks.append({"content": c.text, "category": "交付知识", "source": str(d.source)})
    print(f"分块后共 {len(chunks)} 块")

    builder = DatasetBuilder(seed=42)
    qa_pairs = builder.build_from_documents(chunks, questions_per_doc=15, use_llm=False)
    print(f"生成 QA 对 {len(qa_pairs)} 条")

    builder.to_jsonl(qa_pairs, str(out_path))
    print(f"已保存: {out_path}")

    # 统计
    cats = {}
    for p in qa_pairs:
        cats[p.source] = cats.get(p.source, 0) + 1
    for src, cnt in cats.items():
        print(f"  {Path(src).name}: {cnt} 条")


if __name__ == "__main__":
    main()
