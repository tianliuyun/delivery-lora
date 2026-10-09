"""LoRA 微调训练器"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
import os


@dataclass
class LoRAConfig:
    """LoRA 训练配置"""
    # 模型配置
    base_model: str = "Qwen/Qwen2-0.5B-Instruct"
    model_type: str = "qwen2"

    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    # 注入层：默认 q/v 两层（LoRA 常见实践，参数少、效果稳）；
    # 传 "all" 展开为 q/k/v/o 全四层；也可显式指定 ["q_proj","k_proj","v_proj","o_proj"]
    target_modules: Union[List[str], str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    task_type: str = "CAUSAL_LM"

    # 训练参数
    num_train_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    # 序列打包（Sequence Packing）：把多个短样本拼接到 max_seq_length，
    # 减少 padding 浪费、提升训练吞吐（PEFT 侧通过 packing=True 启用）
    packing: bool = False

    # 数据参数
    max_seq_length: int = 512
    dataset_path: str = ""

    # 输出
    output_dir: str = "./output"
    save_steps: int = 100
    logging_steps: int = 10

    # 设备
    device: str = "cuda"
    fp16: bool = True

    # 全四层注入（Qwen2 注意力矩阵）
    ALL_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def resolve_target_modules(self) -> List[str]:
        """解析注入层配置：'all' → q/k/v/o 全四层；列表原样返回。"""
        if self.target_modules == "all":
            return list(self.ALL_TARGET_MODULES)
        return list(self.target_modules)

    def trainable_params_ratio(self) -> float:
        """可训练参数占比（估算，随注入层数线性变化）

        基准来自本机 QLoRA 实测（Qwen2-0.5B，r=8）：
        q/v 两层注入 trainable≈0.109%（实测 print_trainable_parameters），
        单层约 0.055%，全四层（q/k/v/o）约 0.22%。
        真实占比以训练时 ``model.print_trainable_parameters()`` 输出为准。
        """
        n_modules = len(self.resolve_target_modules())
        return round(0.00055 * n_modules, 4)


class LoRATrainer:
    """
    LoRA 微调训练器。

    封装 PEFT + Transformers + SFTTrainer 的训练流程。
    """

    def __init__(self, config: LoRAConfig = None):
        self.config = config or LoRAConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._training_stats = {}

    def train(self, dataset_path: str = None) -> Dict:
        """
        执行 LoRA 微调训练。

        Args:
            dataset_path: 数据集路径（JSON/JSONL 格式）

        Returns:
            训练结果统计
        """
        dataset_path = dataset_path or self.config.dataset_path

        # 检查依赖
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "训练需要 torch, transformers, peft。"
                "请运行: pip install delivery-lora[train]"
            ) from e

        # 1. 加载基座模型和 tokenizer
        print(f"加载基座模型: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
        )

        # 2. 配置 LoRA（target_modules 支持 "all" → 全四层展开）
        resolved_modules = self.config.resolve_target_modules()
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=resolved_modules,
            task_type=self.config.task_type,
        )

        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        self.model.print_trainable_parameters()

        # 3. 加载数据（此处为框架，完整实现需用 datasets 库）
        print(f"加载数据集: {dataset_path}")

        # 4. 训练循环（此处为框架，完整实现需 SFTTrainer 或 Trainer）
        #    序列打包（config.packing=True）时：SFTTrainer(..., packing=True) 或
        #    DataCollatorForLanguageModeling + 自定义 packing collator，
        #    把多个短样本拼到 max_seq_length，减少 padding、提升吞吐。
        self._training_stats = {
            "status": "framework_only",
            "note": "完整训练逻辑请使用 train_lora.py 脚本，此处为接口框架",
            "config": {
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": resolved_modules,
                "packing": self.config.packing,
                "epochs": self.config.num_train_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "trainable_params_ratio": self.config.trainable_params_ratio(),
            },
        }

        return self._training_stats

    def save(self, output_dir: str = None):
        """保存 LoRA 适配器"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"LoRA 适配器已保存到: {output_dir}")
        else:
            print("模型未加载，无法保存")

    def merge_and_save(self, output_dir: str):
        """合并 LoRA 权重到基座模型并保存"""
        if self.model is None:
            raise ValueError("模型未加载")

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"合并后的模型已保存到: {output_dir}")

    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        return self._training_stats
