from setuptools import setup

setup(
    name="sdkwrapper",
    version="0.1",
    description="Wrapper around Cerebras App SDK and singularity SDK to simplify usage",
    author="Bryce Allen",
    author_email="ballen@anl.gov",
    license="BSD-3-Clause",
    url="https://git.cels.anl.gov/mc-on-ai-accelerators/mc-on-cerebras",
    packages=["sdkwrapper"],
    keywords=["XSBench"],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
