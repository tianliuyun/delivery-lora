from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="delivery-lora",
    version="0.1.0",
    description="交付领域 LoRA 微调小模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="tianliuyun",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
    ],
    extras_require={
        "train": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
        ],
        "infer": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "vllm>=0.2.0",
        ],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "delivery-lora=delivery_lora.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
