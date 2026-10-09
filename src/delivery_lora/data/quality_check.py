"""数据质量检查器"""
import re
from typing import List, Dict, Tuple
from collections import Counter


class QualityChecker:
    """
    QA 数据集质量检查器。

    检查维度：
    - 去重：完全重复 / 高度相似
    - 长度分布：问题长度 / 答案长度
    - 类别分布：各类别样本数
    - 质量评分分布
    - 一致性：相同问题不同答案
    """

    def __init__(self):
        pass

    def check_all(self, qa_pairs: List) -> Dict:
        """执行全部检查"""
        return {
            "total_count": len(qa_pairs),
            "duplicates": self.check_duplicates(qa_pairs),
            "length_distribution": self.check_length_distribution(qa_pairs),
            "category_distribution": self.check_category_distribution(qa_pairs),
            "quality_distribution": self.check_quality_distribution(qa_pairs),
            "consistency": self.check_consistency(qa_pairs),
        }

    def check_duplicates(self, qa_pairs: List) -> Dict:
        """检查重复"""
        questions = [p.question for p in qa_pairs]
        answers = [p.answer for p in qa_pairs]

        q_dup = len(questions) - len(set(questions))
        a_dup = len(answers) - len(set(answers))

        return {
            "question_duplicates": q_dup,
            "answer_duplicates": a_dup,
            "question_duplicate_rate": round(q_dup / max(len(qa_pairs), 1), 4),
        }

    def check_length_distribution(self, qa_pairs: List) -> Dict:
        """检查长度分布"""
        q_lengths = [len(p.question) for p in qa_pairs]
        a_lengths = [len(p.answer) for p in qa_pairs]

        def stats(lengths):
            if not lengths:
                return {"min": 0, "max": 0, "avg": 0, "median": 0}
            sorted_l = sorted(lengths)
            return {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths), 1),
                "median": sorted_l[len(sorted_l) // 2],
            }

        return {
            "question": stats(q_lengths),
            "answer": stats(a_lengths),
        }

    def check_category_distribution(self, qa_pairs: List) -> Dict:
        """检查类别分布"""
        categories = [p.category or "未分类" for p in qa_pairs]
        dist = dict(Counter(categories))
        total = len(qa_pairs)
        return {
            "distribution": dist,
            "num_categories": len(dist),
            "max_category": max(dist, key=dist.get) if dist else "",
            "min_category": min(dist, key=dist.get) if dist else "",
            "imbalance_ratio": round(
                max(dist.values()) / max(min(dist.values()), 1), 2
            ) if dist else 0,
        }

    def check_quality_distribution(self, qa_pairs: List) -> Dict:
        """检查质量评分分布"""
        scores = [p.quality_score for p in qa_pairs if p.quality_score > 0]
        if not scores:
            return {"has_scores": False}

        return {
            "has_scores": True,
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": min(scores),
            "max_score": max(scores),
            "above_0.8": sum(1 for s in scores if s >= 0.8),
            "above_0.9": sum(1 for s in scores if s >= 0.9),
        }

    def check_consistency(self, qa_pairs: List) -> Dict:
        """
        检查一致性：相同/相似问题是否有不同答案。

        简化版：检查完全相同的问题是否有不同答案。
        """
        q_to_a = {}
        conflicts = []
        for p in qa_pairs:
            if p.question in q_to_a:
                if q_to_a[p.question] != p.answer:
                    conflicts.append({
                        "question": p.question,
                        "answer1": q_to_a[p.question],
                        "answer2": p.answer,
                    })
            else:
                q_to_a[p.question] = p.answer

        return {
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:10],  # 只返回前10个
        }

    def filter_low_quality(self, qa_pairs: List, min_score: float = 0.7) -> List:
        """过滤低质量样本"""
        return [p for p in qa_pairs if p.quality_score >= min_score]

    def deduplicate(self, qa_pairs: List) -> List:
        """去重（按问题去重）"""
        seen = set()
        result = []
        for p in qa_pairs:
            if p.question not in seen:
                seen.add(p.question)
                result.append(p)
        return result

    def print_report(self, qa_pairs: List):
        """打印质量报告"""
        report = self.check_all(qa_pairs)

        print(f"{'='*60}")
        print(f"数据集质量报告")
        print(f"{'='*60}")
        print(f"总样本数: {report['total_count']}")
        print(f"{'-'*60}")
        print(f"【重复检查】")
        d = report["duplicates"]
        print(f"  问题重复: {d['question_duplicates']} 条（{d['question_duplicate_rate']:.2%}）")
        print(f"  答案重复: {d['answer_duplicates']} 条")
        print(f"{'-'*60}")
        print(f"【长度分布】")
        lens = report["length_distribution"]
        print(f"  问题长度: min={lens['question']['min']}, max={lens['question']['max']}, avg={lens['question']['avg']}")
        print(f"  答案长度: min={lens['answer']['min']}, max={lens['answer']['max']}, avg={lens['answer']['avg']}")
        print(f"{'-'*60}")
        print(f"【类别分布】")
        cat = report["category_distribution"]
        print(f"  类别数: {cat['num_categories']}")
        print(f"  不平衡比: {cat['imbalance_ratio']}")
        for c, n in sorted(cat["distribution"].items(), key=lambda x: -x[1]):
            print(f"    {c}: {n}")
        print(f"{'-'*60}")
        print(f"【一致性】")
        print(f"  冲突数: {report['consistency']['conflict_count']}")
        print(f"{'='*60}")
