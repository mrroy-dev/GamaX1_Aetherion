from setuptools import setup, find_packages

setup(
    name="gamax1",
    version="2.0.0",
    description="GamaX1 -- first working version of the Aetherion architecture as a real NLP language model.",
    author="MrRoy",
    packages=find_packages(include=["gamax1", "gamax1.*"]),
    install_requires=["torch>=2.0.0"],
    python_requires=">=3.9",
)
