"""推理模块测试"""
import pytest
from delivery_lora.inference import ModelLoader


class TestModelLoader:

    def test_init(self):
        loader = ModelLoader("Qwen/Qwen2-0.5B-Instruct")
        assert loader.model_path == "Qwen/Qwen2-0.5B-Instruct"
        assert loader.model is None

    def test_get_model_info_05b(self):
        loader = ModelLoader("Qwen/Qwen2-0.5B-Instruct")
        info = loader.get_model_info()
        assert info.param_count == "0.5B"
        assert "1GB" in info.memory_usage

    def test_get_model_info_7b(self):
        loader = ModelLoader("Qwen/Qwen2-7B-Instruct")
        info = loader.get_model_info()
        assert info.param_count == "7B"
        assert "13GB" in info.memory_usage

    def test_get_model_info_15b(self):
        loader = ModelLoader("Qwen/Qwen2-1.5B")
        info = loader.get_model_info()
        assert info.param_count == "1.5B"
        assert "3GB" in info.memory_usage

    def test_estimate_memory_fp16(self):
        loader = ModelLoader("Qwen/Qwen2-0.5B")
        mem = loader.estimate_memory(quantization="none")
        assert mem["model_memory_gb"] > 0
        assert mem["total_gb"] > mem["model_memory_gb"]

    def test_estimate_memory_int4(self):
        loader = ModelLoader("Qwen/Qwen2-0.5B")
        mem_int4 = loader.estimate_memory(quantization="int4")
        mem_fp16 = loader.estimate_memory(quantization="none")
        # INT4 显存应该约为 FP16 的 1/4
        assert mem_int4["model_memory_gb"] < mem_fp16["model_memory_gb"]

    def test_estimate_memory_int8(self):
        loader = ModelLoader("Qwen/Qwen2-7B")
        mem = loader.estimate_memory(quantization="int8")
        assert mem["quantization"] == "int8"
        assert mem["param_count"] == "7B"
