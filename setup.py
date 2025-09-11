"""
Setup script for HIIT DNA Methylation Analysis package.
"""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hiit-methylation-analysis",
    version="0.1.0",
    author="HIIT Methylation Analysis Team",
    author_email="hiit-analysis@example.com",
    description="Machine learning analysis of HIIT intervention effects on DNA methylation patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrewSSS/hiit-dna-methylation-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="bioinformatics, DNA methylation, HIIT, machine learning, epigenetics",
    project_urls={
        "Bug Reports": "https://github.com/BrewSSS/hiit-dna-methylation-analysis/issues",
        "Source": "https://github.com/BrewSSS/hiit-dna-methylation-analysis",
        "Documentation": "https://github.com/BrewSSS/hiit-dna-methylation-analysis/wiki",
    },
)