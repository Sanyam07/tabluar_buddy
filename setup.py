import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tabular_buddy",
    version="0.0.29",
    author="鲲(China)",
    author_email="972775099@qq.com",
    description="machine learning toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NickYi1990/tabular_buddy",
    install_requires=["pandas-summary>=0.0.5"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
