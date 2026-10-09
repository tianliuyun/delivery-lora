"""性能基准测试"""
from typing import Dict, List
from dataclasses import dataclass, field
import time


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str = ""
    batch_size: int = 1
    total_requests: int = 0
    total_time_s: float = 0.0
    total_tokens: int = 0
    throughput_tps: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_output_tokens: float = 0.0
    memory_usage_gb: float = 0.0
    extra: Dict = field(default_factory=dict)


class Benchmarker:
    """
    模型性能基准测试工具。

    测试维度：
    - 吞吐量（tokens/s）
    - 延迟（平均/P50/P95/P99）
    - 显存占用
    - 不同 batch size 下的性能
    """

    def __init__(self):
        self.results = []

    def run_benchmark(
        self,
        generate_fn,
        prompts: List[str],
        model_name: str = "model",
        batch_size: int = 1,
        warmup: int = 2,
    ) -> BenchmarkResult:
        """
        运行基准测试。

        Args:
            generate_fn: 生成函数，输入 prompt 列表，返回答案列表
            prompts: 测试用 prompt 列表
            model_name: 模型名称
            batch_size: 批大小
            warmup: 预热轮次

        Returns:
            BenchmarkResult
        """
        # 预热
        for i in range(warmup):
            generate_fn(prompts[:batch_size])

        # 正式测试
        latencies = []
        all_outputs = []
        start_time = time.time()

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_start = time.time()
            outputs = generate_fn(batch)
            batch_time = time.time() - batch_start
            latencies.append(batch_time)
            all_outputs.extend(outputs)

        total_time = time.time() - start_time

        # 计算指标
        total_tokens = sum(len(o) for o in all_outputs)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        sorted_latencies = sorted(latencies)

        def percentile(lst, p):
            if not lst:
                return 0
            idx = int(len(lst) * p / 100)
            idx = min(idx, len(lst) - 1)
            return lst[idx]

        result = BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            total_requests=len(prompts),
            total_time_s=round(total_time, 4),
            total_tokens=total_tokens,
            throughput_tps=round(total_tokens / total_time, 2) if total_time > 0 else 0,
            avg_latency_ms=round(avg_latency * 1000, 2),
            p50_latency_ms=round(percentile(sorted_latencies, 50) * 1000, 2),
            p95_latency_ms=round(percentile(sorted_latencies, 95) * 1000, 2),
            p99_latency_ms=round(percentile(sorted_latencies, 99) * 1000, 2),
            avg_output_tokens=round(total_tokens / len(prompts), 1) if prompts else 0,
        )

        self.results.append(result)
        return result

    def compare_benchmarks(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        name_a: str = "model_a",
        name_b: str = "model_b",
    ) -> Dict:
        """对比两个基准测试结果"""
        return {
            "throughput": {
                name_a: result_a.throughput_tps,
                name_b: result_b.throughput_tps,
                "improvement": f"{(result_b.throughput_tps / result_a.throughput_tps - 1) * 100:.1f}%"
                if result_a.throughput_tps > 0 else "N/A",
            },
            "latency": {
                name_a: f"{result_a.avg_latency_ms}ms",
                name_b: f"{result_b.avg_latency_ms}ms",
                "improvement": f"{(1 - result_b.avg_latency_ms / result_a.avg_latency_ms) * 100:.1f}% faster"
                if result_a.avg_latency_ms > 0 else "N/A",
            },
            "memory": {
                name_a: f"{result_a.memory_usage_gb}GB",
                name_b: f"{result_b.memory_usage_gb}GB",
            },
        }

    def print_report(self, result: BenchmarkResult):
        """打印基准测试报告"""
        print(f"{'='*60}")
        print(f"性能基准测试: {result.model_name}")
        print(f"{'='*60}")
        print(f"总请求数: {result.total_requests}")
        print(f"批大小: {result.batch_size}")
        print(f"总耗时: {result.total_time_s}s")
        print(f"{'-'*60}")
        print(f"吞吐量: {result.throughput_tps} tokens/s")
        print(f"平均输出长度: {result.avg_output_tokens} tokens")
        print(f"{'-'*60}")
        print(f"延迟统计:")
        print(f"  平均: {result.avg_latency_ms}ms")
        print(f"  P50:  {result.p50_latency_ms}ms")
        print(f"  P95:  {result.p95_latency_ms}ms")
        print(f"  P99:  {result.p99_latency_ms}ms")
        if result.memory_usage_gb:
            print(f"{'-'*60}")
            print(f"显存占用: {result.memory_usage_gb} GB")
        print(f"{'='*60}")
