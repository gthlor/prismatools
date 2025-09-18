"""The spatial module provides functions to perform spatial analysis on PRISMA data products"""

import numpy as np
import xarray as xr
import rioxarray
from typing import Union, Tuple, Dict


def band_math(
    dataset: xr.Dataset,
    expressions: dict,
    wavelength_dict: dict,
    constants: dict = None,
    inplace: bool = False,
) -> xr.Dataset:
    """
    Compute one or more spectral indices from a hyperspectral dataset using
    user-defined expressions.

    Args:
        dataset (xarray.Dataset): Dataset containing the spectral variable
            (e.g., 'reflectance') with dimension 'wavelength'.
        expressions (dict): Dictionary mapping variable names to expressions,
            e.g. {'NDVI': '(nir - red) / (nir + red)', 'GNDVI': '(nir - green)/(nir + green)'}.
        wavelength_dict (dict): Dictionary mapping band names to wavelengths,
            e.g. {'red': 660, 'green': 560, 'nir': 840}.
        constants (dict, optional): Dictionary of constants to include in
            all expressions, e.g. {'c': 0.01}. Defaults to None.
        inplace (bool, optional):
            - If True, add the computed indices to the input dataset.
            - If False, return a new dataset containing only the indices.
            Defaults to False.

    Returns:
        xarray.Dataset: Dataset containing the computed index/indices.
            If inplace=True, also keeps the original variables.

    Example:
        >>> wavelength_dict = {'red': 660, 'green': 560, 'nir': 840}
        >>> expressions = {
        ...     'NDVI': '(nir - red) / (nir + red)',
        ...     'GNDVI': '(nir - green) / (nir + green)'
        ... }
        >>> result = band_math(ds, expressions, wavelength_dict, inplace=True)
    """
    da_dict = {
        band: dataset["reflectance"].sel(wavelength=w, method="nearest")
        for band, w in wavelength_dict.items()
    }

    if constants:
        da_dict.update(constants)

    results = {}
    for varname, expr in expressions.items():
        results[varname] = eval(expr, {"__builtins__": None}, da_dict)

    if inplace:
        for varname, da in results.items():
            dataset[varname] = da
        return dataset
    else:
        return xr.Dataset(results)


def pca_image(
    input_data: Union[xr.Dataset, str], out_tif: str, n_components: int = 3
) -> Tuple[xr.Dataset, Dict[str, str]]:
    """
    Perform Principal Component Analysis (PCA) on a hyperspectral or multispectral image
    and save the resulting principal components as a GeoTIFF and an xarray.Dataset.

    This function supports both:
      - `xarray.Dataset` input containing a 'reflectance' variable with dimensions
        (y, x, wavelength)
      - Filepath to a GeoTIFF with shape (bands, rows, cols)

    The function reshapes the image to (pixels, bands), applies PCA, and reshapes the
    principal components back to image dimensions. It also preserves geospatial metadata
    (CRS and transform) when saving the output GeoTIFF.

    Args:
        input_data (xarray.Dataset or str):
            - xarray.Dataset: Must contain 'reflectance' variable with shape (y, x, wavelength)
            - str: Filepath to a GeoTIFF
        out_tif (str): Output path to save the PCA GeoTIFF.
        n_components (int, optional): Number of principal components to compute. Default is 3.

    Returns:
        ds_pca (xarray.Dataset): Dataset containing the principal components with dimensions
            ("pc", "y", "x") and coordinates:
              - pc: principal component index (1, 2, ..., n_components)
              - y: spatial y-coordinate
              - x: spatial x-coordinate
        expl_var (dict): Explained variance ratio for each principal component, formatted as
            percentages with two decimals, e.g. {"PC1": "78.32", "PC2": "12.45", ...}

    Example:
        >>> # Using xarray Dataset
        >>> ds_pca, expl_var = pca_image(ds, "output/pca_image.tif", n_components=3)
        >>> print(expl_var)
        {'PC1': '78.32', 'PC2': '12.45', 'PC3': '5.67'}

        >>> # Using GeoTIFF
        >>> ds_pca, expl_var = pca_image("input_image.tif", "output/pca_image.tif", n_components=5)

    Notes:
        - For xarray input, the function expects the CRS and transform to be set with `rioxarray`.
        - The PCA output GeoTIFF will have `n_components` bands corresponding to the principal
          components in order.
        - The explained variance ratio provides insight into how much of the data variability
          is captured by each component.
    """
    from sklearn.decomposition import PCA
    import rasterio

    # read data
    if isinstance(input_data, xr.Dataset):
        ds = input_data
        array = ds["reflectance"].values  # rows, cols, bands
        array = np.transpose(array, (2, 0, 1))  # bands, rows, cols
        bands, rows, cols = array.shape
        x_coords = ds["x"].values
        y_coords = ds["y"].values
        crs = ds.rio.crs
        geotransform = ds.rio.transform()

    elif isinstance(input_data, str):
        with rasterio.open(input_data) as src:
            array = src.read()  # shape: bands, rows, cols
            bands, rows, cols = array.shape
            crs = src.crs
            geotransform = src.transform
            x_coords = np.array([(geotransform * (col, 0))[0] for col in range(cols)])
            y_coords = np.array([(geotransform * (0, row))[1] for row in range(rows)])

    else:
        raise ValueError("input_data must be xr.Dataset or a GeoTIFF filepath")

    # PCA
    image_reshaped = array.reshape(bands, rows * cols).T  # n_samples, n_bands
    model = PCA(n_components=n_components)
    principal_components = model.fit_transform(image_reshaped)
    pca_image = principal_components.T.reshape(n_components, rows, cols)

    expl_var = {
        f"PC{i+1}": f"{v*100:.2f}"
        for i, v in enumerate(model.explained_variance_ratio_)
    }

    # write output
    ds_pca = xr.Dataset(
        data_vars=dict(
            pca=(["pc", "y", "x"], pca_image),
        ),
        coords=dict(
            pc=(["pc"], np.arange(1, n_components + 1)),
            y=(["y"], y_coords),
            x=(["x"], x_coords),
        ),
    )
    ds_pca.rio.write_crs(crs, inplace=True)
    ds_pca.rio.write_transform(geotransform, inplace=True)

    metadata = dict(
        driver="GTiff",
        height=pca_image.shape[1],
        width=pca_image.shape[2],
        count=pca_image.shape[0],
        dtype=pca_image.dtype,
        crs=crs,
        transform=geotransform,
    )

    with rasterio.open(out_tif, "w", **metadata) as dst:
        dst.write(pca_image)
        print(f"pca image saved at: {out_tif}")

    return ds_pca, expl_var
