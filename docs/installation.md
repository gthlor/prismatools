# Installation

## Install using PyPI

**prismatools** is available on [PyPI](https://pypi.org/). To install **prismatools**, run this command in your terminal:

```bash
pip install prismatools
```

This is the preferred method to install prismatools, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Install from conda-forge

**prismatools** is also available on [conda-forge](https://anaconda.org/conda-forge/prismatools). If you have [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer, you can install **prismatools** using the following command:

```bash
conda install -c conda-forge prismatools
```

Alternatively, you can create a new conda environment and install prismatools in the new environment. This is a good practice because it avoids potential conflicts with other packages installed in your base environment.

```bash
conda install -n base mamba -c conda-forge
conda create -n prisma python=3.10
conda activate prisma
mamba install -c conda-forge prismatools
```

## Install from GitHub

To install the development version from GitHub using Git, run the following command in your terminal:

```bash
pip install git+https://github.com/gthlor/prismatools
```
