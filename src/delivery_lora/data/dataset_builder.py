"""数据集构造器"""
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class QAPair:
    """QA 对数据结构"""
    question: str
    answer: str
    category: str = ""
    source: str = ""
    quality_score: float = 0.0


class DatasetBuilder:
    """
    LoRA 微调数据集构造器。

    流程：原始文档 → 分块 → 生成QA对 → 质量过滤 → 最终数据集
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def build_from_documents(
        self,
        documents: List[Dict],
        questions_per_doc: int = 5,
        use_llm: bool = False,
    ) -> List[QAPair]:
        """
        从文档列表构建 QA 数据集。

        Args:
            documents: 文档列表，每个包含 {content, category, source}
            questions_per_doc: 每篇文档生成多少问题
            use_llm: 是否使用 LLM 辅助生成（False 则用模板生成）

        Returns:
            QA 对列表
        """
        qa_pairs = []
        for doc in documents:
            pairs = self._generate_qa_pairs(
                doc["content"],
                category=doc.get("category", ""),
                source=doc.get("source", ""),
                n=questions_per_doc,
                use_llm=use_llm,
            )
            qa_pairs.extend(pairs)
        return qa_pairs

    def _generate_qa_pairs(
        self,
        content: str,
        category: str,
        source: str,
        n: int,
        use_llm: bool,
    ) -> List[QAPair]:
        """从单篇文档生成 QA 对"""
        if use_llm:
            return self._generate_with_llm(content, category, source, n)
        else:
            return self._generate_template(content, category, source, n)

    def _generate_template(
        self, content: str, category: str, source: str, n: int
    ) -> List[QAPair]:
        """基于模板生成 QA 对（mock 模式）"""
        # 简单的问题模板
        question_templates = [
            "什么是{topic}？",
            "{topic}怎么操作？",
            "{topic}的步骤是什么？",
            "如何解决{topic}问题？",
            "{topic}有哪些注意事项？",
            "{topic}的常见问题有哪些？",
            "怎么排查{topic}故障？",
        ]

        # 从内容中提取关键词作为 topic（简化版）
        topics = self._extract_topics(content, n)

        pairs = []
        for i, topic in enumerate(topics[:n]):
            template = random.choice(question_templates)
            question = template.format(topic=topic)
            answer = self._extract_answer(content, topic)
            pairs.append(QAPair(
                question=question,
                answer=answer,
                category=category,
                source=source,
                quality_score=0.7 + random.random() * 0.2,
            ))
        return pairs

    def _generate_with_llm(
        self, content: str, category: str, source: str, n: int
    ) -> List[QAPair]:
        """使用 LLM 生成 QA 对（预留接口）"""
        raise NotImplementedError(
            "LLM 生成需要配置大模型 API。"
            "请接入 OpenAI / 通义千问 /  Claude 等 API。"
        )

    def _extract_topics(self, content: str, n: int) -> List[str]:
        """从内容中提取主题关键词"""
        # 简化版：按句号分句，取前 n 句的核心词
        sentences = [s.strip() for s in content.split("。") if s.strip()]
        topics = []
        for sent in sentences[:n]:
            # 取句子的前几个字作为主题
            topic = sent[:10] + "..." if len(sent) > 10 else sent
            topics.append(topic)
        if not topics:
            topics = ["相关内容"]
        return topics

    def _extract_answer(self, content: str, topic: str) -> str:
        """从内容中提取答案"""
        # 简化版：返回内容的前 200 字
        return content[:200] + "..." if len(content) > 200 else content

    def train_val_test_split(
        self,
        qa_pairs: List[QAPair],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[List[QAPair], List[QAPair], List[QAPair]]:
        """划分训练/验证/测试集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        shuffled = qa_pairs.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        return train, val, test

    def to_json(self, qa_pairs: List[QAPair], path: str):
        """保存为 JSON 格式"""
        data = [
            {
                "question": p.question,
                "answer": p.answer,
                "category": p.category,
                "source": p.source,
                "quality_score": p.quality_score,
            }
            for p in qa_pairs
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def to_jsonl(self, qa_pairs: List[QAPair], path: str):
        """保存为 JSONL 格式（微调常用）"""
        with open(path, "w", encoding="utf-8") as f:
            for p in qa_pairs:
                item = {
                    "instruction": p.question,
                    "input": "",
                    "output": p.answer,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def from_json(cls, path: str) -> List[QAPair]:
        """从 JSON 加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            QAPair(
                question=d["question"],
                answer=d["answer"],
                category=d.get("category", ""),
                source=d.get("source", ""),
                quality_score=d.get("quality_score", 0.0),
            )
            for d in data
        ]
