import re
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split()

with open("pytorch_tools/__init__.py") as f:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    version = re.search(VSRE, f.read()).group(1)

setup(
    name="pytorch_tools",
    version=version,
    author="Emil Zakirov",
    author_email="bonlimezak@gmail.com",
    packages=find_packages(exclude=["test", "docs", "examples"]),
    url="https://github.com/bonlime/pytorch-tools",
    description="Tool box for PyTorch",
    long_description=readme,
    classifiers=["Programming Language :: Python :: 3.6",],
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",
    install_requires=requirements,
    license="MIT License",
)
