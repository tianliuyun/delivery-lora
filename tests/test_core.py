"""核心模块测试"""
import pytest
from delivery_lora import DeliveryLoRA


class TestDeliveryLoRA:

    def setup_method(self):
        self.lora = DeliveryLoRA(mode="mock")

    def test_ask_returns_string(self):
        answer = self.lora.ask("桌面云登录失败怎么办？")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_云桌面_domain(self):
        answer = self.lora.ask("桌面云客户端登录失败")
        # mock 模式应该返回领域相关内容
        assert "桌面云" in answer or "排查" in answer or "登录" in answer

    def test_ask_超融合_domain(self):
        answer = self.lora.ask("超融合存储池扩容")
        assert "扩容" in answer or "超融合" in answer or "存储池" in answer

    def test_compare_returns_dict(self):
        result = self.lora.compare("桌面云卡顿怎么办？")
        assert "question" in result
        assert "general_answer" in result
        assert "lora_answer" in result
        assert "accuracy_general" in result
        assert "accuracy_lora" in result
        assert "speedup" in result
        assert "cost_reduction" in result

    def test_compare_lora_better_in_domain(self):
        result = self.lora.compare("超融合节点故障处理")
        # LoRA 在领域内准确率应该更高
        assert result["accuracy_lora"] > result["accuracy_general"]

    def test_batch_ask(self):
        questions = ["桌面云登录失败", "超融合扩容", "私有云迁移"]
        answers = self.lora.batch_ask(questions)
        assert len(answers) == 3
        assert all(isinstance(a, str) for a in answers)

    def test_general_mode(self):
        lora_general = DeliveryLoRA(mode="general")
        answer = lora_general.ask("桌面云登录失败")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_unknown_question(self):
        answer = self.lora.ask("今天吃什么？")
        # 非领域问题也应该有回复
        assert isinstance(answer, str)
        assert len(answer) > 0
