# delivery-lora

> 交付领域 LoRA 微调小模型
> 通用大模型 vs 领域 LoRA 小模型 四维对比实验

## 项目简介

针对云计算交付领域专业术语多、通用大模型理解不准、API 调用成本高、数据安全顾虑等问题，通过 LoRA 低秩微调技术训练领域专属小模型，在垂直领域实现「效果反超大模型 + 成本降 90% + 全本地部署」。

## 核心特性

- 🎯 **LoRA 低秩微调**：只训 0.22% 参数，显存占用低，训练速度快
- 📊 **四维对比实验**：准确率 / 推理速度 / 成本 / 显存，全面对比通用大模型 vs 领域 LoRA
- 🔧 **数据构造流水线**：从产品手册/FAQ/故障记录 → LLM辅助生成 → 人工抽检 → 高质量 QA 对
- 🚀 **vLLM 推理部署**：PagedAttention + continuous batching + INT4 量化，吞吐提升 10×
- 📈 **完整评估体系**：模型层 / 效果层 / 业务层 三层评估
- 🔌 **可插拔架构**：基座模型 / 训练框架 / 推理引擎均可替换

## 技术架构

```
原始文档 → 数据构造 → 数据集 → LoRA微调 → 评估 → vLLM部署 → 推理服务
                ↓                    ↑                    ↑
              LLM辅助              PEFT               PagedAttention
              人工抽检           Transformers         INT4量化
```

### 核心技术栈

| 模块 | 技术选型 |
|------|---------|
| 基座模型 | Qwen2-0.5B / 1.5B / 7B |
| 微调框架 | PEFT + Transformers + SFTTrainer |
| 微调方法 | LoRA (r=8, alpha=16) |
| 推理部署 | vLLM + PagedAttention |
| 量化 | GPTQ / AWQ / INT4 |
| 评估 | 自定义评估集 + 人工抽检 |

## 项目结构

```
delivery-lora/
├── src/
│   └── delivery_lora/
│       ├── __init__.py
│       ├── data/               # 数据构造与处理
│       │   ├── dataset_builder.py
│       │   ├── qa_generator.py
│       │   └── quality_check.py
│       ├── training/           # LoRA 微调
│       │   ├── lora_trainer.py
│       │   └── config.py
│       ├── inference/          # 推理部署
│       │   ├── vllm_server.py
│       │   └── model_loader.py
│       └── evaluation/         # 评估体系
│           ├── evaluator.py
│           └── benchmark.py
├── tests/                      # 单元测试
├── examples/                   # 使用示例
├── data/                       # 样本数据
├── README.md
├── requirements.txt
└── setup.py
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 快速体验（Mock 模式）

```python
from delivery_lora import DeliveryLoRA

# 初始化（mock 模式，无需下载模型）
lora = DeliveryLoRA(mode="mock")

# 领域问答
answer = lora.ask("桌面云客户端登录失败怎么办？")
print(answer)

# 对比通用大模型 vs LoRA 小模型
result = lora.compare("超融合集群存储池扩容怎么操作？")
print(f"通用模型: {result['general_answer']}")
print(f"LoRA小模型: {result['lora_answer']}")
print(f"准确率对比: {result['accuracy_general']} vs {result['accuracy_lora']}")
```

### 训练 LoRA 模型

```python
from delivery_lora.training import LoRATrainer, LoRAConfig

config = LoRAConfig(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    num_train_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
)

trainer = LoRATrainer(config)
trainer.train("data/delivery_qa_dataset.json")
trainer.save("outputs/delivery-lora-v1")
```

### QLoRA 实战脚本（消费级 GPU 实测）

本机 NVIDIA MX350（2GB 显存）实测可跑 Qwen2-0.5B QLoRA 微调（峰值显存 1140MB/1994MB）：

```bash
# q/v 两层注入（默认，LoRA 常见实践）
python scripts/train_qlora.py --data data/domain_qa.jsonl --target qv --epochs 2 --out output/qlora_qv

# 全四层注入（q/k/v/o）
python scripts/train_qlora.py --data data/domain_qa.jsonl --target all --epochs 2 --out output/qlora_all

# 微调前后对比评估
python scripts/infer_compare.py --base models/Qwen2-0.5B-Instruct --adapter output/qlora_qv
```

- 4bit 量化加载（bitsandbytes nf4）+ LoRA 只训适配器，显存占用低
- `--target all` 展开为 q/k/v/o 全四层；序列打包（packing）配置见 `LoRAConfig.packing`
- 数据构建：`python scripts/build_domain_dataset.py`（交付文档 → 领域 QA 对）

### vLLM 推理部署

```python
from delivery_lora.inference import VLLMServer

server = VLLMServer(
    model_path="outputs/delivery-lora-v1",
    quantize="int4",
    gpu_memory_utilization=0.8,
)
server.start()
```

## 实验结果

### 四维对比

| 维度 | 通用大模型（7B） | LoRA 小模型（0.5B） | 提升/下降 |
|------|----------------|-------------------|-----------|
| 领域问答准确率 | ~65%（目标值） | ~75%（目标值） | **+10pp** ↑（目标值） |
| 推理速度 | ~2s（目标值） | ~200ms（目标值） | **10× 更快** ↑（目标值） |
| 推理成本 | 高（~0.1元/次） | 极低（本地部署） | **降低 90%+** ↓ |
| 显存占用（训练） | 13GB+（FP16 7B） | **<2GB 实测（QLoRA 峰值 1140MB）** | **减少 85%+** ↓ |
| 显存占用（推理） | 13GB+（FP16 7B） | **547MB 实测（0.5B INT4）** | 目标：13GB→2GB ↓ |
| 数据安全 | 数据外传 | 全本地 | ✅ 安全 |
| **黄金评测集准确率** | **3.3%（实测）** | **16.7%（实测）** | **+13.3pp**（30 条硬问题，LLM-as-Judge） |

> 数字口径：领域准确率/推理速度为**目标值**（小模型实测见黄金评测集行）；训练显存为**本机实测**（MX350 2GB，QLoRA 4bit）；推理显存为**本机实测**（`scripts/bench_deploy_memory.py`：0.5B INT4 + LoRA 加载峰值 535MB / 推理峰值 547MB，MX350 2GB）；7B 的 13GB→2GB 为部署设计目标（本机 2GB 无法加载 7B）；黄金评测集准确率为**本机实测**（30 条硬问题 × LLM-as-Judge，`scripts/eval_golden.py` 可复现）。

### LoRA 配置

- 基座：Qwen2-0.5B-Instruct
- r=8, alpha=16, dropout=0.05
- 可训练参数占比（实测）：q/v 两层 0.109% / 全四层 q/k/v/o ~0.22%（本机 QLoRA 实测）
- 训练数据：目标 2000 条领域 QA 对（当前验证集 10 条）
- 训练轮次：3 epochs（目标）

## 数据构造流程

```
产品手册/FAQ/故障排查记录
    ↓
文档分块 + 清洗
    ↓
LLM 辅助生成 QA 对（5-10 题/段）
    ↓
质量过滤（去重 / 去低质 / 一致性检查）
    ↓
人工抽检（抽检率 20%，通过率 > 90% 才进入训练集）
    ↓
最终训练集（2000 条高质量领域 QA）
```

## 评估体系

### 模型层
- Loss 曲线（训练集/验证集）
- 困惑度（Perplexity）
- 过拟合检测

### 效果层
- 领域问答准确率（基于标准测试集）
- 人工评测（抽样打分，1-5 分）
- 鲁棒性测试（不同问法、边缘 case）

### 业务层
- 推理速度（tokens/s）
- 显存占用
- 成本对比
- 吞吐量（QPS）

## 路线图

- [x] 项目骨架 + Mock 模式 + 评估体系
- [x] 数据构造流水线框架
- [x] LoRA 训练配置与接口
- [x] vLLM 推理部署接口
- [ ] 真实领域数据训练
- [ ] 完整对比实验数据
- [ ] Web Demo 界面
- [ ] 多基座模型对比（Qwen / Llama / Mistral）

## License

MIT
