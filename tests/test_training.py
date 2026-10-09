"""训练模块测试"""
import pytest
from delivery_lora.training import LoRATrainer, LoRAConfig


class TestLoRAConfig:

    def test_default_config(self):
        config = LoRAConfig()
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.num_train_epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 2e-4

    def test_custom_config(self):
        config = LoRAConfig(
            lora_r=16,
            lora_alpha=32,
            num_train_epochs=5,
            batch_size=4,
        )
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_train_epochs == 5
        assert config.batch_size == 4

    def test_trainable_params_ratio(self):
        config = LoRAConfig()
        ratio = config.trainable_params_ratio()
        assert 0 < ratio < 1
        assert ratio < 0.01  # 应该小于1%

    def test_default_target_modules_qv(self):
        """默认注入 q/v 两层（LoRA 常见实践）。"""
        config = LoRAConfig()
        assert config.resolve_target_modules() == ["q_proj", "v_proj"]

    def test_target_modules_all_expands_four_layers(self):
        """'all' 快捷方式展开为 q/k/v/o 全四层。"""
        config = LoRAConfig(target_modules="all")
        assert config.resolve_target_modules() == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_explicit_four_layer_target_modules(self):
        config = LoRAConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        assert config.resolve_target_modules() == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_ratio_scales_with_module_count(self):
        """全四层注入的参数量约等于两层注入的两倍（0.22% → 0.44%）。"""
        qv = LoRAConfig().trainable_params_ratio()
        all4 = LoRAConfig(target_modules="all").trainable_params_ratio()
        assert all4 == pytest.approx(qv * 2, rel=0.01)

    def test_packing_flag_default_off(self):
        config = LoRAConfig()
        assert config.packing is False

    def test_packing_flag_can_enable(self):
        config = LoRAConfig(packing=True)
        assert config.packing is True


class TestLoRATrainer:

    def test_init(self):
        config = LoRAConfig()
        trainer = LoRATrainer(config)
        assert trainer.config is not None
        assert trainer.model is None

    def test_init_without_config(self):
        trainer = LoRATrainer()
        assert trainer.config is not None
        assert isinstance(trainer.config, LoRAConfig)

    def test_train_framework_mode(self):
        """测试框架模式（不实际训练）"""
        config = LoRAConfig()
        trainer = LoRATrainer(config)
        # 不调用 train（需要真实依赖），只验证初始化
        stats = trainer.get_training_stats()
        assert stats == {}
