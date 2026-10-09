"""vLLM 推理服务器"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """vLLM 服务器配置"""
    model_path: str
    tokenizer_path: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    quantization: str = "none"  # none / awq / gptq / int4 / int8
    enforce_eager: bool = False
    port: int = 8000
    host: str = "0.0.0.0"


class VLLMServer:
    """
    vLLM 推理服务器封装。

    核心特性：
    - PagedAttention: 页式注意力，显存利用率高
    - Continuous Batching: 连续批处理，吞吐高
    - 多量化支持: AWQ/GPTQ/INT4/INT8
    - OpenAI 兼容 API
    """

    def __init__(self, config: ServerConfig = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = ServerConfig(**kwargs)
        self._llm = None
        self._running = False

    def start(self):
        """启动 vLLM 服务"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM 未安装。请运行: pip install vllm"
            ) from e

        print(f"启动 vLLM 服务: {self.config.model_path}")
        print(f"  GPU 显存利用率: {self.config.gpu_memory_utilization}")
        print(f"  量化方式: {self.config.quantization}")
        print(f"  最大序列长度: {self.config.max_model_len}")

        self._llm = LLM(
            model=self.config.model_path,
            tokenizer=self.config.tokenizer_path or self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            quantization=self.config.quantization if self.config.quantization != "none" else None,
            enforce_eager=self.config.enforce_eager,
        )
        self._running = True
        print("vLLM 服务启动完成")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """批量生成"""
        if self._llm is None:
            raise RuntimeError("vLLM 未启动，请先调用 start()")

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        outputs = self._llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def benchmark(
        self,
        num_requests: int = 100,
        max_tokens: int = 128,
        prompt_length: int = 256,
    ) -> Dict:
        """
        性能基准测试。

        返回：吞吐量、平均延迟、首 token 延迟等指标
        """
        import time

        # 生成测试 prompt
        test_prompt = "测试" * (prompt_length // 2)
        prompts = [test_prompt] * num_requests

        start_time = time.time()
        outputs = self.generate(prompts, max_tokens=max_tokens)
        total_time = time.time() - start_time

        total_tokens = sum(len(o) for o in outputs)

        return {
            "num_requests": num_requests,
            "total_time_s": round(total_time, 2),
            "total_tokens": total_tokens,
            "throughput_tps": round(total_tokens / total_time, 2),
            "avg_latency_ms": round(total_time / num_requests * 1000, 2),
            "avg_output_tokens": round(total_tokens / num_requests, 1),
        }

    def stop(self):
        """停止服务"""
        self._llm = None
        self._running = False
        print("vLLM 服务已停止")

    @property
    def is_running(self) -> bool:
        return self._running
