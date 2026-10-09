"""命令行接口"""
import json
import click
from .core import DeliveryLoRA


@click.group()
def cli():
    """delivery-lora 交付领域 LoRA 微调 CLI"""
    pass


@cli.command()
@click.argument("question")
@click.option("--mode", default="mock", help="运行模式: mock/lora/general")
def ask(question, mode):
    """向领域模型提问"""
    lora = DeliveryLoRA(mode=mode)
    answer = lora.ask(question)
    click.echo(f"Q: {question}")
    click.echo(f"A: {answer}")


@cli.command()
@click.argument("question")
def compare(question):
    """对比通用大模型 vs LoRA 小模型"""
    lora = DeliveryLoRA(mode="mock")
    result = lora.compare(question)
    click.echo(f"Q: {question}")
    click.echo(f"\n通用大模型:")
    click.echo(f"  答案: {result['general_answer']}")
    click.echo(f"  领域准确率: {result['accuracy_general']:.0%}")
    click.echo(f"\nLoRA 小模型:")
    click.echo(f"  答案: {result['lora_answer']}")
    click.echo(f"  领域准确率: {result['accuracy_lora']:.0%}")
    click.echo(f"\n对比:")
    click.echo(f"  速度提升: {result['speedup']}")
    click.echo(f"  成本降低: {result['cost_reduction']}")


@cli.command()
def demo():
    """运行演示"""
    from .data import DatasetBuilder, QualityChecker

    click.echo("=== delivery-lora 演示 ===")
    click.echo()

    # 模拟数据构造
    click.echo("1. 数据构造演示")
    docs = [
        {"content": "桌面云登录失败请检查网络连接、客户端版本和用户授权。常见原因包括代理设置错误、客户端版本过低、授权过期等。", "category": "云桌面", "source": "FAQ"},
        {"content": "超融合存储池扩容步骤：添加新节点到集群，数据自动均衡，注意选择业务低峰期操作。", "category": "超融合", "source": "产品手册"},
    ]
    builder = DatasetBuilder()
    qa_pairs = builder.build_from_documents(docs, questions_per_doc=3)
    click.echo(f"   生成了 {len(qa_pairs)} 条 QA 对")

    # 质量检查
    click.echo()
    click.echo("2. 数据质量检查")
    checker = QualityChecker()
    report = checker.check_all(qa_pairs)
    click.echo(f"   样本数: {report['total_count']}")
    click.echo(f"   问题重复: {report['duplicates']['question_duplicates']}")
    click.echo(f"   类别数: {report['category_distribution']['num_categories']}")

    # 模型对比
    click.echo()
    click.echo("3. 模型对比演示")
    lora = DeliveryLoRA(mode="mock")
    test_q = "桌面云登录失败怎么办？"
    result = lora.compare(test_q)
    click.echo(f"   问题: {test_q}")
    click.echo(f"   通用模型准确率: {result['accuracy_general']:.0%}")
    click.echo(f"   LoRA 准确率: {result['accuracy_lora']:.0%}")
    click.echo(f"   速度提升: {result['speedup']}")
    click.echo(f"   成本降低: {result['cost_reduction']}")

    click.echo()
    click.echo("演示完成！")


if __name__ == "__main__":
    cli()
