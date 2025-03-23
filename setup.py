from setuptools import setup, find_packages

setup(
    name="multiplicity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    author="Luiz Guilherme Nascimento",
    author_email="luizguilhermesn@gmail.com",
    description="A meta-model framework for analyzing predictive multiplicity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LuizGuilhermeNascimento/multiplicity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 