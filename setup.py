import setuptools

setuptools.setup(
    name="bkh_pytorch_utils",
    version="0.0.1",
    author="Bardia Khosravi",
    author_email="bardiakhosravi95@gmail.com",
    #description="<Template Setup.py package>",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/BardiaKh/PytorchUtils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=1.17.2","torch>=1.4","tabulate>=0.8.9","tqdm>4.60.0","monai>=0.5.2","pytorch-lightning>=1.3.0"],
    python_requires='>=3.6',
)
