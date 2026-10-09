"""评估模块测试"""
import pytest
from delivery_lora.evaluation import ModelEvaluator, Benchmarker
from delivery_lora.data.dataset_builder import QAPair


class TestModelEvaluator:

    def setup_method(self):
        self.evaluator = ModelEvaluator()
        self.test_data = [
            QAPair("桌面云登录失败怎么办？", "请检查网络连接和客户端授权，排查步骤如下...", "云桌面"),
            QAPair("超融合怎么扩容？", "超融合扩容步骤：添加节点，数据均衡，验证。", "超融合"),
            QAPair("私有云迁移方案", "V2V迁移和P2V迁移两种方案，根据业务选择。", "私有云"),
            QAPair("云桌面打印问题", "检查打印重定向和驱动配置。", "云桌面"),
        ]

    def test_evaluate(self):
        def mock_model(q):
            if "桌面云" in q:
                return "检查网络连接和客户端授权"
            elif "超融合" in q:
                return "添加节点和数据均衡"
            else:
                return "迁移方案有V2V和P2V"

        result = self.evaluator.evaluate(mock_model, self.test_data)
        assert result.total_samples == 4
        assert 0 <= result.accuracy <= 1
        assert result.per_category is not None
        assert "云桌面" in result.per_category

    def test_compare_models(self):
        def model_a(q):
            return "简单回答"

        def model_b(q):
            return "详细的检查网络连接和客户端授权配置"

        result = self.evaluator.compare_models(
            model_a, model_b,
            "model_a", "model_b",
            self.test_data
        )
        assert "model_a" in result
        assert "model_b" in result
        assert "difference" in result
        assert "winner" in result

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError):
            self.evaluator.evaluate(lambda q: "", [])


class TestBenchmarker:

    def test_run_benchmark(self):
        benchmarker = Benchmarker()

        def mock_generate(prompts):
            import time
            time.sleep(0.001)  # 模拟耗时
            return ["这是回答" * 10 for _ in prompts]

        prompts = ["测试问题" for _ in range(10)]
        result = benchmarker.run_benchmark(
            mock_generate, prompts,
            model_name="test_model",
            batch_size=2,
            warmup=0,
        )
        assert result.model_name == "test_model"
        assert result.total_requests == 10
        assert result.total_time_s >= 0
        assert result.throughput_tps > 0
        assert result.avg_output_tokens > 0

    def test_compare_benchmarks(self):
        benchmarker = Benchmarker()

        def fast_model(prompts):
            return ["a" for _ in prompts]

        def slow_model(prompts):
            import time
            time.sleep(0.001)
            return ["a" * 100 for _ in prompts]

        prompts = ["q"] * 5
        r1 = benchmarker.run_benchmark(fast_model, prompts, "fast", warmup=0)
        r2 = benchmarker.run_benchmark(slow_model, prompts, "slow", warmup=0)

        comparison = benchmarker.compare_benchmarks(r1, r2, "fast", "slow")
        assert "throughput" in comparison
        assert "latency" in comparison
        assert "memory" in comparison
