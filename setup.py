from setuptools import setup, find_packages

setup(
    name="radical-synthesis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0"
    ],
    author="Leogenes Simplicio Rodrigues de Souza",
    description="Autopoietic and Darwinian MoE Architecture with Topological Consciousness.",
    python_requires=">=3.8",
)
