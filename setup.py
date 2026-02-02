"""
Web-CogReasoner Setup Script
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="web-cogreasoner",
    version="0.1.0",
    author="Web-CogReasoner Team",
    author_email="",
    description="Knowledge-Induced Cognitive Reasoning for Web Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gnonymous/Web-CogReasoner",
    project_urls={
        "Bug Tracker": "https://github.com/Gnonymous/Web-CogReasoner/issues",
        "Documentation": "https://github.com/Gnonymous/Web-CogReasoner#readme",
        "Paper": "https://arxiv.org/abs/2508.01858",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "training": [
            "deepspeed>=0.13.0",
            "llamafactory>=0.6.0",
            "wandb>=0.16.0",
        ],
        "web": [
            "selenium>=4.18.0",
            "webdriver-manager>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI entry points here if needed
        ],
    },
)
