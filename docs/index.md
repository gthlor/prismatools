# Welcome to prismatools


[![image](https://img.shields.io/pypi/v/prismatools.svg)](https://pypi.python.org/pypi/prismatools)
[![image](https://img.shields.io/conda/vn/conda-forge/prismatools.svg)](https://anaconda.org/conda-forge/prismatools)

[![logo](https://raw.githubusercontent.com/gthlor/prismatools/main/docs/assets/logo150_nobg.png)](https://github.com/gthlor/prismatools/blob/main/docs/assets/logo150_nobg.png)


**prismatools** is an open-source Python package for reading, analyzing, and visualizing hyperspectral imagery from the PRISMA mission.

-   Documentation: <https://gthlor.github.io/prismatools>
-   PyPI: <https://pypi.org/project/prismatools/>
-   Conda-forge: https://anaconda.org/conda-forge/prismatools
-   Free software: [MIT License](https://opensource.org/licenses/MIT)

## Introduction

Hyperspectral remote sensing provides detailed information on the Earth's surface by capturing hundreds of contiguous, narrow spectral bands in the VNIR and SWIR ranges. This spectral richness allows scientists to analyze vegetation, water quality, soil properties, and many other environmental variables with unprecedented accuracy.

Launched in 2019 by the Italian Space Agency ([ASI](https://www.asi.it/)), **PR**ecursore **I**per**S**pettrale della **M**issione **A**pplicativa ([PRISMA](http://www.prisma-i.it/index.php/en))  is a spaceborne mission that delivers medium-resolution high-quality hyperspectral imagery (30m) combined with a panchromatic channel (5m). PRISMA products are distributed in multiple levels (L2B, L2C, L2D) and have become a key dataset for applications in agriculture, forestry, inland and coastal waters, geology, and climate studies.

With **prismatools**, researchers can explore PRISMA imagery interactively, extract meaningful biophysical information, and integrate the data into broader geospatial workflows.

## Features

- 📂 Read and Write PRISMA hyperspectral and panchromatic L2 products (L2D, L2B, L2C)
- 🎨 Visualize PRISMA data interactively (2D maps, RGB composites, custom band combinations)
- 📊 Analyze hyperspectral imagery (band math, vegetation indices, PCA)
- 🌍 Extract and plot spectral signatures from any pixel or region of interest
- 💾 Export spectral signatures as CSV files
- 🧩 Integrate seamlessly with popular Python libraries (xarray, rasterio, geopandas, matplotlib)

## Citations

If you find **prismatools** useful in your research, please consider citing the following paper: [*DOI will be provided upon publication*]. Thank you!
