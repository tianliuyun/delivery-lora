"""数据模块测试"""
import pytest
import tempfile
import os
from delivery_lora.data import DatasetBuilder, QualityChecker
from delivery_lora.data.dataset_builder import QAPair


class TestDatasetBuilder:

    def setup_method(self):
        self.builder = DatasetBuilder(seed=42)

    def test_build_from_documents(self):
        docs = [
            {"content": "桌面云登录失败排查步骤：检查网络连接状态。检查客户端版本是否最新。检查用户授权是否有效。检查代理设置是否正确。", "category": "云桌面", "source": "FAQ"},
            {"content": "超融合扩容操作步骤：添加新节点到集群。数据自动均衡。验证集群状态。注意业务低峰期操作。", "category": "超融合", "source": "手册"},
        ]
        pairs = self.builder.build_from_documents(docs, questions_per_doc=3)
        assert len(pairs) == 6
        assert all(isinstance(p, QAPair) for p in pairs)
        assert all(p.question and p.answer for p in pairs)

    def test_build_with_category(self):
        docs = [
            {"content": "测试内容", "category": "测试", "source": "test"},
        ]
        pairs = self.builder.build_from_documents(docs, questions_per_doc=2)
        assert pairs[0].category == "测试"
        assert pairs[0].source == "test"

    def test_train_val_test_split(self):
        pairs = [QAPair(f"q{i}", f"a{i}") for i in range(100)]
        train, val, test = self.builder.train_val_test_split(pairs, 0.7, 0.15, 0.15)
        assert len(train) + len(val) + len(test) == 100
        assert 68 <= len(train) <= 72
        assert 13 <= len(val) <= 17
        assert 13 <= len(test) <= 17

    def test_to_json_and_from_json(self):
        pairs = [
            QAPair("问题1", "答案1", "类别1", "来源1", 0.85),
            QAPair("问题2", "答案2", "类别2", "来源2", 0.92),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            self.builder.to_json(pairs, path)
            loaded = DatasetBuilder.from_json(path)
            assert len(loaded) == 2
            assert loaded[0].question == "问题1"
            assert loaded[0].answer == "答案1"
            assert loaded[0].category == "类别1"
            assert loaded[0].quality_score == 0.85
        finally:
            os.unlink(path)

    def test_to_jsonl(self):
        pairs = [QAPair("q1", "a1"), QAPair("q2", "a2")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            self.builder.to_jsonl(pairs, path)
            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
            import json
            data = json.loads(lines[0])
            assert "instruction" in data
            assert "output" in data
        finally:
            os.unlink(path)


class TestQualityChecker:

    def setup_method(self):
        self.checker = QualityChecker()
        self.pairs = [
            QAPair("q1", "a1", "cat1", "s1", 0.85),
            QAPair("q2", "a2", "cat1", "s1", 0.90),
            QAPair("q3", "a3", "cat2", "s2", 0.75),
            QAPair("q4", "a4", "cat2", "s2", 0.95),
            QAPair("q1", "a1_diff", "cat1", "s1", 0.80),  # 重复问题，答案不同
        ]

    def test_check_all(self):
        report = self.checker.check_all(self.pairs)
        assert report["total_count"] == 5
        assert "duplicates" in report
        assert "length_distribution" in report
        assert "category_distribution" in report
        assert "quality_distribution" in report
        assert "consistency" in report

    def test_check_duplicates(self):
        result = self.checker.check_duplicates(self.pairs)
        assert result["question_duplicates"] == 1  # q1 重复了

    def test_check_category_distribution(self):
        result = self.checker.check_category_distribution(self.pairs)
        assert result["num_categories"] == 2
        assert result["distribution"]["cat1"] == 3
        assert result["distribution"]["cat2"] == 2

    def test_check_quality_distribution(self):
        result = self.checker.check_quality_distribution(self.pairs)
        assert result["has_scores"] is True
        assert result["above_0.8"] >= 3  # 0.85, 0.90, 0.95, 0.80 都是 >=0.8
        assert result["above_0.9"] >= 2  # 0.90, 0.95

    def test_check_consistency(self):
        result = self.checker.check_consistency(self.pairs)
        assert result["conflict_count"] >= 1  # q1 有两个不同答案

    def test_filter_low_quality(self):
        filtered = self.checker.filter_low_quality(self.pairs, min_score=0.85)
        assert len(filtered) == 3  # 0.85, 0.90, 0.95

    def test_deduplicate(self):
        deduped = self.checker.deduplicate(self.pairs)
        assert len(deduped) == 4  # q1 去重后剩1条
