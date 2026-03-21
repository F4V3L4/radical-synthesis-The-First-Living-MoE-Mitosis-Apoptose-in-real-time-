from setuptools import setup, find_packages

setup(
    name="radical-synthesis",
    version="0.3.0",
    author="Leogenes Simplício Rodrigues de Souza",
    author_email="simplileoge@gmail.com",
    description="Living Autopoietic Mixture-of-Experts — Deus Sive Natura",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/F4V3L4/radical-synthesis",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
