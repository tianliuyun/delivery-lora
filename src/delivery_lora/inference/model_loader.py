"""模型加载器"""
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """模型信息"""
    model_path: str
    model_type: str = "causal_lm"
    param_count: str = ""
    memory_usage: str = ""
    quantization: str = "none"


class ModelLoader:
    """
    模型加载器。

    支持多种加载方式：
    - HuggingFace Transformers
    - vLLM
    - GGUF / llama.cpp
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._info = None

    def load_transformers(self, **kwargs):
        """使用 Transformers 加载"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("需要 transformers") from e

        print(f"[Transformers] 加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if self.device == "cuda" else None,
            **kwargs
        )
        return self.model, self.tokenizer

    def load_vllm(self, **kwargs):
        """使用 vLLM 加载"""
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("需要 vllm") from e

        print(f"[vLLM] 加载模型: {self.model_path}")
        self.model = LLM(
            model=self.model_path,
            **kwargs
        )
        return self.model

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        if self._info:
            return self._info

        # 估算参数规模
        path_lower = self.model_path.lower()
        if "0.5b" in path_lower or "500m" in path_lower:
            params = "0.5B"
            memory = "~1GB (FP16)"
        elif "1.5b" in path_lower or "1_5b" in path_lower:
            params = "1.5B"
            memory = "~3GB (FP16)"
        elif "7b" in path_lower:
            params = "7B"
            memory = "~13GB (FP16)"
        elif "13b" in path_lower:
            params = "13B"
            memory = "~26GB (FP16)"
        else:
            params = "未知"
            memory = "未知"

        self._info = ModelInfo(
            model_path=self.model_path,
            param_count=params,
            memory_usage=memory,
        )
        return self._info

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        """生成文本（Transformers 方式）"""
        if self.model is None:
            self.load_transformers()

        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **kwargs
            )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def estimate_memory(self, quantization: str = "none") -> Dict:
        """
        估算显存占用。

        公式：参数量 × 字节数（FP16=2, INT8=1, INT4=0.5）+ KV缓存 + 框架开销
        """
        info = self.get_model_info()
        param_gb = self._params_to_gb(info.param_count)

        if quantization == "none" or quantization == "fp16":
            model_mem = param_gb * 2
        elif quantization == "int8":
            model_mem = param_gb * 1
        elif quantization == "int4":
            model_mem = param_gb * 0.5
        else:
            model_mem = param_gb * 2

        # KV 缓存和框架开销（估算 1-2GB）
        overhead = 1.5
        total = model_mem + overhead

        return {
            "model_memory_gb": round(model_mem, 2),
            "overhead_gb": overhead,
            "total_gb": round(total, 2),
            "quantization": quantization,
            "param_count": info.param_count,
        }

    def _params_to_gb(self, param_str: str) -> float:
        """参数字符串转 GB（FP32下1个参数4字节）"""
        param_str = param_str.upper().strip()
        if "B" in param_str:
            num = float(param_str.replace("B", ""))
            return num * 4  # 1B 参数 FP32 = 4GB
        elif "M" in param_str:
            num = float(param_str.replace("M", ""))
            return num * 4 / 1000  # 1M 参数 FP32 = 4MB
        return 0.0
