"""The common module contains common functions and classes used by the other modules."""

# TODO: split the module into IO (read_prismaL2D, read_prismaL2D_pan, write_prismaL2D) and utils (check_valid_file, get_transform, array_to_image)

import os
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rio
import h5py

from typing import List, Tuple, Union, Optional, Any
from affine import Affine


def convert_coords(
    coords: List[Tuple[float, float]], from_epsg: str, to_epsg: str
) -> List[Tuple[float, float]]:
    """
    Convert a list of coordinates from one EPSG to another.

    Args:
        coords: List of tuples containing coordinates in the format (latitude, longitude).
        from_epsg: Source EPSG code (default is "epsg:4326").
        to_epsg: Target EPSG code (default is "epsg:32615").

    Returns:
        List of tuples containing converted coordinates in the format (x, y).
    """
    import pyproj

    # Define the coordinate transformation
    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)

    # Convert each coordinate
    converted_coords = [transformer.transform(lon, lat) for lat, lon in coords]

    return converted_coords


def extract_spectral(
    ds: xr.Dataset, lat: float, lon: float, name: str = "data"
) -> xr.DataArray:
    """
    Extracts spectral signature from a given xarray Dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the spectral data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted data.
    """

    crs = ds.rio.crs

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")[name].values

    da = xr.DataArray(values, dims=["band"], coords={"band": ds.coords["band"]})

    return da


def check_valid_file(file_path: str, type: str = "PRS_L2D") -> bool:
    """
    Checks if the given file path points to a valid file.

    Args:
        file_path (str): Path to the file.
        type (str, optional): Expected file type ('PRS_L2B', 'PRS_L2C', 'PRS_L2D'). Defaults to 'PRS_L2D'.

    Returns:
        bool: True if file_path points to the correct file, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the type is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    valid_types = {"PRS_L2B", "PRS_L2C", "PRS_L2D"}
    if type not in valid_types:
        raise ValueError(
            f"Unsupported file type: {type}. Supported types are {valid_types}."
        )

    basename = os.path.basename(file_path)
    return basename.startswith(type) and basename.endswith(".he5")


def get_transform(ul_easting: float, ul_northing: float, res: int = 30) -> Affine:
    """
    Returns an affine transformation for a given upper-left corner and resolution.

    Args:
        ul_easting (float): Easting coordinate of the upper-left corner.
        ul_northing (float): Northing coordinate of the upper-left corner.
        res (int, optional): Pixel resolution. Defaults to 30.

    Returns:
        Affine: Affine transformation object representing the spatial transform.
    """
    return Affine.translation(ul_easting, ul_northing) * Affine.scale(res, -res)


def array_to_image(
    array: np.ndarray,
    output: str,
    dtype: Optional[np.dtype] = None,
    compress: str = "lzw",
    transpose: bool = True,
    crs: Optional[str] = None,
    transform: Optional[tuple] = None,
    driver: str = "GTiff",
    **kwargs,
) -> str:
    """
    Save a NumPy array as a georeferenced raster (GeoTIFF by default).

    Args:
        array (np.ndarray): Array to save. Shape can be (rows, cols) or (bands, rows, cols).
        output (str): Path to the output file.
        dtype (np.dtype, optional): Data type for output. Auto-inferred if None.
        compress (str, optional): Compression for GTiff/COG. Defaults to "lzw".
        transpose (bool, optional): If True, expects (bands, rows, cols) and transposes.
        crs (str, optional): CRS of the output raster.
        transform (tuple, optional): Affine transform of the raster.
        driver (str, optional): GDAL driver. Defaults to "GTiff".
        **kwargs: Extra options for rasterio.open().

    Returns:
        str: Path to the saved file.
    """
    # ensure correct shape
    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))

    # ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # get driver from extension
    ext = os.path.splitext(output)[-1].lower()
    driver_map = {"": "COG", ".tif": "GTiff", ".tiff": "GTiff", ".dat": "ENVI"}
    driver = driver_map.get(ext, "COG")
    if ext == "":
        output += ".tif"

    # infer dtype if not given
    if dtype is None:
        min_val, max_val = np.nanmin(array), np.nanmax(array)
        if min_val >= 0 and max_val <= 1:
            dtype = np.float32
        elif min_val >= 0 and max_val <= 255:
            dtype = np.uint8
        elif min_val >= -128 and max_val <= 127:
            dtype = np.int8
        elif min_val >= 0 and max_val <= 65535:
            dtype = np.uint16
        elif min_val >= -32768 and max_val <= 32767:
            dtype = np.int16
        else:
            dtype = np.float64
    array = array.astype(dtype)

    # set metadata
    count = 1 if array.ndim == 2 else array.shape[2]
    metadata = dict(
        driver=driver,
        height=array.shape[0],
        width=array.shape[1],
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    )
    if compress and driver in ["GTiff", "COG"]:
        metadata["compress"] = compress
    metadata.update(**kwargs)

    # write raster
    with rasterio.open(output, "w", **metadata) as dst:
        if array.ndim == 2:  # panchromatic
            dst.write(array, 1)
            dst.set_band_description(
                1, kwargs.get("band_description", "Panchromatic band")
            )
        else:  # hyperspectral
            for i in range(array.shape[2]):
                dst.write(array[:, :, i], i + 1)
                if "wavelengths" in kwargs:
                    wl = kwargs["wavelengths"][i]
                    dst.set_band_description(i + 1, f"Band {i+1} ({wl:.1f} nm)")

    return output


def read_prismaL2D(
    file_path: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    panchromatic: bool = False,
) -> xr.Dataset:
    """
    Reads PRISMA Level-2D .he5 data (hyperspectral or panchromatic)
    and returns an xarray.Dataset with reflectance values and geospatial metadata.

    Args:
        file_path (str): Path to the PRISMA L2D .he5 file.
        wavelengths (Optional[List[float]]): List of wavelengths (in nm) to extract
            (only for hyperspectral cube).
            - If None, all valid wavelengths are used.
            - If provided, can select by exact match or nearest available wavelength.
        method (str, default "nearest"): Method for wavelength selection ("nearest" or "exact").
        panchromatic (bool, default False): If True, read the panchromatic cube.
                                            If False, read the hyperspectral cube.

    Returns:
        xr.Dataset: An xarray.Dataset containing reflectance data with coordinates.
    """
    # check if file is valid
    if not check_valid_file(file_path, type="PRS_L2D"):
        raise ValueError(
            f"The file {file_path} is not a valid PRS_L2D file or does not exist."
        )

    try:
        with h5py.File(file_path, "r") as f:
            epsg_code = f.attrs["Epsg_Code"][()]
            ul_easting = f.attrs["Product_ULcorner_easting"][()]
            ul_northing = f.attrs["Product_ULcorner_northing"][()]

            if panchromatic:
                # --- PANCHROMATIC ---
                pancube_path = "HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube"
                pancube_data = f[pancube_path][()]
                l2_scale_pan_min = f.attrs["L2ScalePanMin"][()]
                l2_scale_pan_max = f.attrs["L2ScalePanMax"][()]
                fill_value = -9999
                max_data_value = 65535

                pancube_data = l2_scale_pan_min + (
                    pancube_data.astype(np.float32) / max_data_value
                ) * (l2_scale_pan_max - l2_scale_pan_min)
                pancube_data[pancube_data == fill_value] = np.nan

                rows, cols = pancube_data.shape
                transform = get_transform(ul_easting, ul_northing, res=5)
                x_coords = np.array([transform * (i, 0) for i in range(cols)])[:, 0]
                y_coords = np.array([transform * (0, j) for j in range(rows)])[:, 1]

                ds = xr.Dataset(
                    data_vars=dict(
                        reflectance=(
                            ["y", "x"],
                            pancube_data,
                            dict(
                                units="unitless",
                                _FillValue=np.nan,
                                standard_name="reflectance",
                                long_name="Panchromatic reflectance",
                            ),
                        ),
                    ),
                    coords=dict(
                        y=(["y"], y_coords, dict(units="m")),
                        x=(["x"], x_coords, dict(units="m")),
                    ),
                )

            else:
                # --- HYPERSPECTRAL CUBE ---
                swir_cube = f["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"][()]
                vnir_cube = f["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"][()]
                vnir_wavelengths = f.attrs["List_Cw_Vnir"][()]
                swir_wavelengths = f.attrs["List_Cw_Swir"][()]
                l2_scale_vnir_min = f.attrs["L2ScaleVnirMin"][()]
                l2_scale_vnir_max = f.attrs["L2ScaleVnirMax"][()]
                l2_scale_swir_min = f.attrs["L2ScaleSwirMin"][()]
                l2_scale_swir_max = f.attrs["L2ScaleSwirMax"][()]
                fill_value = -9999
                max_data_value = 65535

                vnir_cube = l2_scale_vnir_min + (
                    vnir_cube.astype(np.float32) / max_data_value
                ) * (l2_scale_vnir_max - l2_scale_vnir_min)
                swir_cube = l2_scale_swir_min + (
                    swir_cube.astype(np.float32) / max_data_value
                ) * (l2_scale_swir_max - l2_scale_swir_min)

                vnir_cube[vnir_cube == fill_value] = np.nan
                swir_cube[swir_cube == fill_value] = np.nan

                full_cube = np.concatenate((vnir_cube, swir_cube), axis=1)
                full_wavelengths = np.concatenate((vnir_wavelengths, swir_wavelengths))

                # filter and sort wavelengths
                valid_idx = full_wavelengths > 0
                full_wavelengths = full_wavelengths[valid_idx]
                full_cube = full_cube[:, valid_idx, :]
                sort_idx = np.argsort(full_wavelengths)
                full_wavelengths = full_wavelengths[sort_idx]
                full_cube = full_cube[:, sort_idx, :]

                # select requested wavelengths
                if wavelengths is not None:
                    requested = np.array(wavelengths)
                    if method == "exact":
                        idx = np.where(np.isin(full_wavelengths, requested))[0]
                        if len(idx) == 0:
                            raise ValueError(
                                "No requested wavelengths found (exact match)."
                            )
                    else:
                        idx = np.array(
                            [np.abs(full_wavelengths - w).argmin() for w in requested]
                        )
                    full_wavelengths = full_wavelengths[idx]
                    full_cube = full_cube[:, idx, :]

                rows, cols = full_cube.shape[0], full_cube.shape[2]
                transform = get_transform(ul_easting, ul_northing, res=30)
                x_coords = np.array([transform * (i, 0) for i in range(cols)])[:, 0]
                y_coords = np.array([transform * (0, j) for j in range(rows)])[:, 1]

                ds = xr.Dataset(
                    data_vars=dict(
                        reflectance=(
                            ["y", "wavelength", "x"],
                            full_cube,
                            dict(
                                units="unitless",
                                _FillValue=np.nan,
                                standard_name="reflectance",
                                long_name="Combined atmospherically corrected surface reflectance",
                            ),
                        ),
                    ),
                    coords=dict(
                        wavelength=(
                            ["wavelength"],
                            full_wavelengths,
                            dict(long_name="center wavelength", units="nm"),
                        ),
                        y=(["y"], y_coords, dict(units="m")),
                        x=(["x"], x_coords, dict(units="m")),
                    ),
                )
                ds["reflectance"] = ds.reflectance.transpose("y", "x", "wavelength")

    except Exception as e:
        raise RuntimeError(f"Error reading the file {file_path}: {e}")

    # write CRS and transform
    crs = f"EPSG:{epsg_code}"
    ds.rio.write_crs(crs, inplace=True)
    ds.rio.write_transform(transform, inplace=True)

    # global attributes
    ds.attrs.update(
        dict(
            units="unitless",
            _FillValue=-9999,
            grid_mapping="crs",
            standard_name="reflectance",
            Conventions="CF-1.6",
            crs=ds.rio.crs.to_string(),
        )
    )

    return ds


def write_prismaL2D(
    dataset: Union[xr.Dataset, str],
    output: str,
    panchromatic: bool = False,
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> Optional[str]:
    """
    Converts a PRISMA hyperspectral dataset to a georeferenced image.

    Args:
        dataset (Union[xr.Dataset, str]): The PRISMA dataset or the path to the
            dataset file (.he5).
        output (str): File path to save the output raster.
        panchromatic (bool, optional): If True, treat array as single-band pancromatic. Defaults to False.
        wavelengths (np.ndarray, optional): Wavelengths to select from the dataset.
            If None, all wavelengths are included. Defaults to None.
        method (str, optional): Method to use for wavelength selection (e.g. "nearest").
        **kwargs (Any): Additional arguments passed to 'array_to_image()' and to 'rasterio.open()'.

    Returns:
        str: Output file path, or None if all values are NaN.
    """
    # load dataset if it's a path to .he5
    if isinstance(dataset, str):
        dataset = read_prismaL2D(dataset, panchromatic=panchromatic)

    # get np.array
    array = dataset["reflectance"].values
    if not np.any(np.isfinite(array)):
        print("Warning: All reflectance values are NaN. Output image will be blank.")
        return None

    # get band names (wavelength) and, eventually, select specific bands
    if array.ndim == 2:  # panchromatic
        kwargs["band_description"] = "Panchromatic band"
    else:  # cube
        if wavelengths is not None:
            dataset = dataset.sel(wavelength=wavelengths, method=method)
            array = dataset["reflectance"].values
        kwargs["wavelengths"] = dataset["wavelength"].values

    return array_to_image(
        array,
        output=output,
        transpose=False,
        crs=dataset.rio.crs,
        transform=dataset.rio.transform(),
        **kwargs,
    )


def extract_prisma(
    dataset: xr.Dataset,
    lat: float,
    lon: float,
    offset: float = 15.0,
) -> xr.DataArray:
    """
    Extracts an averaged reflectance spectrum from a PRISMA hyperspectral dataset.

    A square spatial window is centered at the specified latitude and longitude,
    and the reflectance values within that window are averaged across the spatial
    dimensions to produce a single spectrum.

    Args:
        dataset (xarray.Dataset): The PRISMA dataset containing reflectance data,
            with valid CRS information.
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center points.
        offset (float, optional): Half-size of the square window for extraction,
            expressed in the dataset's projected coordinate units (e.g., meters).
            Defaults to 15.0.

    Returns:
        xarray.DataArray: A 1D array containing the averaged reflectance values
        across wavelengths. If no matching pixels are found, returns NaN values.
    """
    if dataset.rio.crs is None:
        raise ValueError("Dataset CRS not set. Please provide dataset with CRS info.")

    crs = dataset.rio.crs.to_string()

    # Convert lat/lon to projected coords
    x_proj, y_proj = convert_coords([(lat, lon)], "epsg:4326", crs)[0]

    da = dataset["reflectance"]
    x_con = (da["x"] > x_proj - offset) & (da["x"] < x_proj + offset)
    y_con = (da["y"] > y_proj - offset) & (da["y"] < y_proj + offset)

    try:
        data = da.where(x_con & y_con, drop=True)
        data = data.mean(dim=["x", "y"], skipna=True)
    except ValueError:
        # No matching pixels
        data = np.full(da.sizes["wavelength"], np.nan)

    return xr.DataArray(
        data,
        dims=["wavelength"],
        coords={"wavelength": dataset.coords["wavelength"]},
    )


# debugging
# if __name__ == "__main__":
# file = r"C:/Users/loren/Desktop/PRS_L2D_STD_20240429095823_20240429095827_0001\PRS_L2D_STD_20240429095823_20240429095827_0001.he5"
# ds = read_prismaL2D(file, wavelengths=None, method="nearest")
# print(ds)

# ds_pan = read_prismaL2D(file, panchromatic=True)
# print(ds_pan)

# # case1a: Pan from path (è necessario specificare se è pan o meno)
# write_prismaL2D(file, output=r'..\out_test\imgPan_path_pan.tif', panchromatic=True)
# # case1b: Cube from path
# write_prismaL2D(file, output=r'..\out_test\imgPan_path_cube.tif')

# case2a: Pan from dataset
# write_prismaL2D(ds_pan, output=r'..\out_test\imgPan_ds.tif')
# case2b: Cube from dataset
# write_prismaL2D(ds, output=r'..\out_test\imgCube_ds.tif')

# # case3 : Cube in ENVI format
# write_prismaL2D(file, output=r'..\out_test\imgCube_ds.dat', panchromatic=False)
# write_prismaL2D(file, output=r'..\out_test\imgPanc_ds.dat', panchromatic=True)
