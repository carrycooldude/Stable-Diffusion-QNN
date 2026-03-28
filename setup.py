from setuptools import setup, find_packages

setup(
    name="stable-diffusion-qnn",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "onnxruntime-qnn",
        "numpy",
        "pillow",
        "tokenizers"
    ],
    author="carrycooldude",
    description="Stable Diffusion v2.1 optimized for Qualcomm Snapdragon NPU",
    url="https://github.com/carrycooldude/Stable-Diffusion-QNN",
    python_requires=">=3.8",
)
