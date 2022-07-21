import setuptools

setuptools.setup(
    name="bkh_pytorch_utils",
    version="0.2.4",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    description="A rapid prototyping tool for MONAI & PyTorch Lightning",
    url="https://github.com/BardiaKh/PytorchUtils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib>=3.1.2",
        "pandas>=0.25.0",
        "numpy>=1.20.0",
        "torch>=1.4",
        "tabulate>=0.8.9",
        "tqdm>4.60.0",
        "monai>=0.5.2",
        "pytorch-lightning>=1.5.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
        "scikit-image>=0.18.0",
        "overrides>=6.1.0"
    ],
    python_requires='>=3.6',
)
