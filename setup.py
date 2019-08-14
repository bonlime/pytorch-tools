from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
   requirements = f.read().split()

setup(name="pytorch_tools",
      version="0.0.1",
      author="bonlime",
      author_email="bonlimezak@gmail.com",
      packages=find_packages(exclude=["test", "docs", "examples"]),
      url="https://github.com/bonlime/pytorch-tools",
      description="Tool box for PyTorch",
      long_description=readme,
      classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
      ],
      # requirements
      setup_requires=["setuptools_scm"],
      python_requires=">=3, <4",
      install_requires=requirements,

      #license="Apache License 2.0",
      #
      )
