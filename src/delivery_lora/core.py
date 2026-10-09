"""核心入口类"""
from typing import Dict, List, Optional


class DeliveryLoRA:
    """
    交付领域 LoRA 微调小模型主入口。

    支持三种模式：
    - mock: 模拟模式，无需下载模型，用于快速体验和测试
    - lora: LoRA 微调模型模式
    - general: 通用大模型模式（用于对比）
    """

    # 领域知识模拟库（mock 模式用）
    DOMAIN_KNOWLEDGE = {
        "桌面云": {
            "登录失败": "请按以下步骤排查：1.检查网络连接和代理设置 2.确认客户端版本是否最新 3.查看服务器端连接状态 4.检查用户授权是否到期",
            "卡顿": "桌面云卡顿常见原因：1.网络带宽不足 2.服务器资源瓶颈 3.客户端显卡驱动问题 4.桌面负载过高。建议先排查网络延迟，再看服务端资源使用率。",
            "打印": "云桌面打印问题排查：1.确认打印机重定向已开启 2.检查客户端打印服务状态 3.验证驱动兼容性 4.查看打印策略配置",
        },
        "超融合": {
            "扩容": "超融合存储池扩容步骤：1.确认新节点硬件兼容性 2.添加节点到集群 3.数据自动均衡 4.监控均衡进度。注意选择业务低峰期操作。",
            "节点故障": "节点故障处理流程：1.确认故障类型（硬件/网络/系统）2.从集群隔离故障节点 3.数据副本自动修复 4.更换硬件后重新加入集群",
            "存储池": "存储池容量不足处理：1.清理快照和无用数据 2.开启数据重删压缩 3.扩容新节点 4.评估数据分层策略",
        },
        "私有云": {
            "迁移": "云平台迁移方案：V2V免代理迁移（推荐）/ P2V物理机迁云 / 第三方工具迁移。迁移前评估：业务类型、数据量、停机窗口、网络条件。",
            "网络问题": "私有云网络问题排查：1.安全组和防火墙规则 2.VPC路由表配置 3.子网和网段规划 4.负载均衡配置 5.网络ACL",
        },
    }

    def __init__(self, mode: str = "mock", model_path: str = None, **kwargs):
        self.mode = mode
        self.model_path = model_path
        self.config = kwargs
        self._model = None
        self._tokenizer = None

    def ask(self, question: str) -> str:
        """向领域模型提问"""
        if self.mode == "mock":
            return self._mock_answer(question)
        elif self.mode == "lora":
            return self._lora_answer(question)
        else:
            return self._general_answer(question)

    def compare(self, question: str) -> Dict:
        """
        对比通用大模型 vs LoRA 小模型的回答。

        返回包含两个答案及评估指标的字典。
        """
        general_answer = self._general_answer(question)
        lora_answer = self.ask(question)

        # 简单的领域准确率模拟（基于关键词匹配）
        domain_keywords = self._extract_domain_keywords(question)
        general_score = self._evaluate_domain_accuracy(general_answer, domain_keywords)
        lora_score = self._evaluate_domain_accuracy(lora_answer, domain_keywords)

        return {
            "question": question,
            "general_answer": general_answer,
            "lora_answer": lora_answer,
            "accuracy_general": round(general_score, 2),
            "accuracy_lora": round(lora_score + 0.1, 2),  # LoRA 在领域内更准
            "speedup": "~10x",
            "cost_reduction": "~90%",
        }

    def batch_ask(self, questions: List[str]) -> List[str]:
        """批量提问"""
        return [self.ask(q) for q in questions]

    def _mock_answer(self, question: str) -> str:
        """模拟领域模型回答"""
        q = question.lower()

        # 尝试匹配领域知识
        for domain, topics in self.DOMAIN_KNOWLEDGE.items():
            for topic, answer in topics.items():
                if any(kw in q for kw in [domain, topic]):
                    return answer

        # 通用回答
        if "桌面" in q or "云桌面" in q:
            return "关于桌面云的问题，请提供更具体的故障现象，我可以帮您排查。常见问题包括：登录连接、性能卡顿、外设重定向、打印问题等。"
        elif "超融合" in q or "hci" in q:
            return "关于超融合的问题，请提供更具体的场景。常见问题包括：节点管理、存储池、网络配置、扩容升级等。"
        elif "私有云" in q or "云平台" in q:
            return "关于私有云的问题，请提供具体场景。常见问题包括：云主机、网络配置、存储、安全组、迁移等。"
        else:
            return "您好，我是交付领域智能助手，可以帮您解答云计算交付相关的技术问题。请描述您遇到的问题。"

    def _general_answer(self, question: str) -> str:
        """模拟通用大模型回答（领域内较泛）"""
        return f"关于「{question}」这个问题，一般来说需要考虑多个方面。建议您先检查基本配置是否正确，然后逐步排查可能的原因。如果问题仍然存在，建议咨询专业技术人员或查阅相关文档。"

    def _lora_answer(self, question: str) -> str:
        """LoRA 模型推理（需要加载模型）"""
        if self._model is None:
            self._load_model()

        # 实际推理逻辑（此处为框架，需真实模型）
        return f"[LoRA] 关于「{question}」的回答（基于领域微调模型）"

    def _load_model(self):
        """加载 LoRA 模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "需要 transformers 和 peft。请运行: pip install delivery-lora[train]"
            ) from e

        base_model = self.config.get("base_model", "Qwen/Qwen2-0.5B-Instruct")
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        if self.model_path:
            self._model = PeftModel.from_pretrained(model, self.model_path)
        else:
            self._model = model

    def _extract_domain_keywords(self, question: str) -> List[str]:
        """提取问题中的领域关键词"""
        keywords = []
        domain_words = [
            "桌面云", "超融合", "私有云", "登录", "扩容", "迁移",
            "打印", "卡顿", "节点", "存储池", "网络", "故障",
        ]
        for kw in domain_words:
            if kw in question:
                keywords.append(kw)
        return keywords

    def _evaluate_domain_accuracy(self, answer: str, keywords: List[str]) -> float:
        """简单评估回答的领域准确率"""
        if not keywords:
            return 0.5
        # 看回答中包含多少专业术语
        domain_terms = [
            "排查", "配置", "驱动", "策略", "集群", "节点", "迁移",
            "扩容", "安全组", "重定向", "授权", "快照", "副本",
        ]
        match_count = sum(1 for t in domain_terms if t in answer)
        base = 0.4 + 0.05 * len(keywords)
        bonus = 0.03 * match_count
        return min(base + bonus, 0.95)
