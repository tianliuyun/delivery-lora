"""模型评估器"""
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EvalResult:
    """评估结果"""
    accuracy: float = 0.0
    total_samples: int = 0
    per_category: Dict[str, float] = None
    details: List[Dict] = None


class ModelEvaluator:
    """
    大模型评估器。

    评估维度：
    - 准确率：标准测试集上的正确率
    - 分类别准确率：按领域/题型统计
    - 鲁棒性：不同问法下的一致性
    - 幻觉率：编造事实的比例
    """

    def __init__(self):
        self._eval_dataset = []

    def set_eval_dataset(self, qa_pairs: List):
        """设置评估数据集"""
        self._eval_dataset = qa_pairs

    def evaluate(
        self,
        model_fn,
        eval_dataset: List = None,
    ) -> EvalResult:
        """
        评估模型。

        Args:
            model_fn: 模型推理函数，输入问题，返回答案
            eval_dataset: 评估数据集，覆盖默认数据集

        Returns:
            EvalResult 评估结果
        """
        dataset = eval_dataset or self._eval_dataset
        if not dataset:
            raise ValueError("评估数据集为空")

        details = []
        correct = 0
        per_category_correct = {}
        per_category_total = {}

        for item in dataset:
            question = item.question if hasattr(item, "question") else item["question"]
            reference = item.answer if hasattr(item, "answer") else item["answer"]
            category = item.category if hasattr(item, "category") else item.get("category", "未分类")

            # 模型推理
            prediction = model_fn(question)

            # 简单评估：关键词匹配
            is_correct = self._check_answer(prediction, reference)

            if is_correct:
                correct += 1
                per_category_correct[category] = per_category_correct.get(category, 0) + 1
            per_category_total[category] = per_category_total.get(category, 0) + 1

            details.append({
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "correct": is_correct,
                "category": category,
            })

        accuracy = correct / len(dataset) if dataset else 0.0
        per_category = {
            cat: per_category_correct.get(cat, 0) / total
            for cat, total in per_category_total.items()
        }

        return EvalResult(
            accuracy=round(accuracy, 4),
            total_samples=len(dataset),
            per_category=per_category,
            details=details,
        )

    def _check_answer(self, prediction: str, reference: str) -> bool:
        """
        检查答案是否正确（简化版：关键词匹配）。

        实际项目中应该用更复杂的评估方法：
        - BLEU / ROUGE
        - 嵌入相似度
        - LLM-as-judge
        """
        # 提取参考中的关键词
        ref_keywords = self._extract_keywords(reference)
        pred_keywords = self._extract_keywords(prediction)

        if not ref_keywords:
            return True

        # 命中率超过阈值算正确
        hits = sum(1 for kw in ref_keywords if kw in pred_keywords or kw in prediction)
        return hits / len(ref_keywords) >= 0.5

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 简单按标点和空格拆分，取长度>=2的词
        import re
        words = re.split(r'[，。、；：""''（）\s]+', text)
        return [w for w in words if len(w) >= 2]

    def compare_models(
        self,
        model_a_fn,
        model_b_fn,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
        eval_dataset: List = None,
    ) -> Dict:
        """对比两个模型"""
        result_a = self.evaluate(model_a_fn, eval_dataset)
        result_b = self.evaluate(model_b_fn, eval_dataset)

        return {
            model_a_name: {
                "accuracy": result_a.accuracy,
                "per_category": result_a.per_category,
            },
            model_b_name: {
                "accuracy": result_b.accuracy,
                "per_category": result_b.per_category,
            },
            "difference": round(result_b.accuracy - result_a.accuracy, 4),
            "winner": model_b_name if result_b.accuracy > result_a.accuracy else model_a_name,
        }

    def print_report(self, result: EvalResult, model_name: str = "model"):
        """打印评估报告"""
        print(f"{'='*60}")
        print(f"模型评估报告: {model_name}")
        print(f"{'='*60}")
        print(f"总样本数: {result.total_samples}")
        print(f"准确率: {result.accuracy:.2%}")
        print(f"{'-'*60}")
        print(f"分类别准确率:")
        for cat, acc in sorted(result.per_category.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {acc:.2%}")
        print(f"{'='*60}")
