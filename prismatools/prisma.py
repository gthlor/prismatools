"""The prisma module provides functions to read and write PRISMA L2 data products"""

# TODO: check if "extract_prisma" works also for panchromatic data.

import os
import numpy as np
import rasterio
import xarray as xr
import rioxarray
import h5py
from typing import List, Tuple, Union, Optional, Any

from .utils import check_valid_file, convert_coords, get_transform


def read_prismaL2D(
    file_path: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    panchromatic: bool = False,
    extract_error: bool = False,
) -> xr.Dataset:
    """
    The function reads PRISMA Level-2D .he5 data (hyperspectral or panchromatic),
    applies digital number to reflectance scaling, handles fill values, optionally
    selects specific wavelengths, and builds an xarray.Dataset with spatial
    coordinates and CRS information. Optionally reads pixel-level error matrices
    when available.

    Args:
        file_path (str):
            Path to the PRISMA L2D .he5 file.
        wavelengths (Optional[List[float]], optional):
            List of wavelengths (in nm) to extract (hyperspectral only).
            - If None, all available wavelengths are used.
            - If provided, can select by exact match or nearest available wavelength
              depending on `method`.
        method (str, optional):
            Wavelength selection method (used if `wavelengths` is provided, hyperspectral only):
            - "nearest" (default): select the closest available band for each requested wavelength.
            - "exact": select only exact matches; raises ValueError if none found.
        panchromatic (bool, optional):
            If True, read the panchromatic cube instead of the hyperspectral VNIR+SWIR cubes.
            Defaults to False.
        extract_error (bool, optional):
            If True, attempt to read and attach pixel-level error matrix
            fields from the product (when present). Defaults to False.

    Returns:
        xr.Dataset:
        An xarray.Dataset containing reflectance data with dimensions:
            - Hyperspectral: ("y", "x", "wavelength")
            - Panchromatic: ("y", "x")

        The dataset may include an optional variable named `error_matrix` when
        `extract_error=True` and pixel-level error matrices are present in the
        product. The shape of `error_matrix` matches the reflectance dims.
    """
    # check if file is valid
    if not check_valid_file(file_path, type="PRS_L2D"):
        raise ValueError(f"{file_path} is not a valid PRS_L2D file or does not exist.")

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
                fill_value = 0
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
                if extract_error:
                    try:
                        err_path = (
                            "HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PIXEL_L2_ERR_MATRIX"
                        )
                        err_arr = f[err_path][()].astype(np.float32)
                        # mask error where DN fill is present
                        try:
                            err_arr[np.isnan(pancube_data)] = np.nan
                        except Exception:
                            pass
                        ds["error_matrix"] = ("y", "x"), err_arr
                    except Exception:
                        # no error matrix available; continue silently
                        pass

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
                fill_value = 0
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
                if extract_error:
                    try:
                        v_err_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX"
                        s_err_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX"
                        v_err = f[v_err_path][()].astype(np.float32)
                        s_err = f[s_err_path][()].astype(np.float32)
                        # mask raw DN fill values if raw arrays exist
                        try:
                            v_err[np.isnan(vnir_cube)] = np.nan
                        except Exception:
                            pass
                        try:
                            s_err[np.isnan(swir_cube)] = np.nan
                        except Exception:
                            pass

                        err_full = np.concatenate((v_err, s_err), axis=1)
                        # apply same wavelength filtering and sorting as reflectance
                        err_full = err_full[:, valid_idx, :]
                        err_full = err_full[:, sort_idx, :]
                        if wavelengths is not None:
                            err_full = err_full[:, idx, :]
                        ds["error_matrix"] = ("y", "wavelength", "x"), err_full
                    except Exception:
                        pass
                ds["reflectance"] = ds.reflectance.transpose("y", "x", "wavelength")
                # transpose error matrix as well to match reflectance dims
                try:
                    if "error_matrix" in ds:
                        ds["error_matrix"] = ds.error_matrix.transpose(
                            "y", "x", "wavelength"
                        )
                except Exception:
                    pass

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
            _FillValue=0,
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
    Write a PRISMA Level-2D dataset to a georeferenced raster file.

    This function takes a PRISMA Level-2D hyperspectral or panchromatic dataset
    (or the path to a `.he5` file), extracts reflectance data and georeferencing
    information, and saves it as a raster image (GeoTIFF by default).

    Args:
        dataset (Union[xr.Dataset, str]):
            PRISMA dataset as an `xarray.Dataset` or the path to a Level-2D `.he5` file.
        output (str):
            Path to the output raster file.
        panchromatic (bool, optional):
            If True, treat the dataset as a single-band panchromatic image.
            Defaults to False.
        wavelengths (Optional[np.ndarray], optional):
            Wavelengths (in nm) to select from the dataset (hyperspectral only).
            If None, all available wavelengths are included.
            Defaults to None.
        method (str, optional):
            Wavelength selection method (hyperspectral only):
                - `"nearest"`: select closest available bands.
                - `"exact"`: select exact matches only.
            Defaults to `"nearest"`.
        **kwargs (Any):
            Additional keyword arguments passed to:
                - `array_to_image()` for raster writing.
                - `rasterio.open()` for file creation.

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

    A square spatial window is centered at the given latitude and longitude.
    The reflectance values within that window are averaged across the spatial
    dimensions, producing a single spectrum.

    Args:
        dataset (xr.Dataset):
            PRISMA dataset containing a variable named "reflectance", with
            valid CRS information and shape (x, y, wavelength).
        lat (float):
            Latitude of the center point (in WGS84).
        lon (float):
            Longitude of the center point (in WGS84).
        offset (float, optional):
            Half-size of the square window for extraction, expressed in the
            dataset's projected coordinate units (e.g., meters).
            Defaults to `15.0`.

    Returns:
        xarray.DataArray: A 1D array containing the averaged reflectance values
        across wavelengths. If no matching pixels are found, returns NaN values.
    """
    if dataset.rio.crs is None:
        raise ValueError("Dataset CRS not set. Please provide dataset with CRS info.")

    crs = dataset.rio.crs.to_string()
    # convert lat/lon to projected coords
    x_proj, y_proj = convert_coords([(lat, lon)], "epsg:4326", crs)[0]

    da = dataset["reflectance"]
    x_con = (da["x"] > x_proj - offset) & (da["x"] < x_proj + offset)
    y_con = (da["y"] > y_proj - offset) & (da["y"] < y_proj + offset)

    data = da.where(x_con & y_con, drop=True)

    if "wavelength" in da.dims:
        data = data.mean(dim=["x", "y"], skipna=True)
        return xr.DataArray(
            data,
            dims=["wavelength"],
            coords={"wavelength": dataset.coords["wavelength"]},
        )
    else:
        # panchromatic
        data = data.mean(dim=["x", "y"], skipna=True)
        return xr.DataArray(
            [data.item()] if np.ndim(data) == 0 else data,
            dims=["value"],
        )


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
    Save a NumPy array as a georeferenced raster file (GeoTIFF by default).

    This function writes a 2D (single-band) or 3D (multi-band) NumPy array
    to a raster file using rasterio. Georeferencing can be applied by
    providing a CRS and an affine transform.

    Args:
        array (np.ndarray):
            Input array to save.
            - Shape (rows, cols): single-band raster.
            - Shape (bands, rows, cols): multi-band raster.
        output (str):
            Path to the output raster file.
        dtype (Optional[np.dtype], optional):
            Data type of the output raster. If None, inferred from the array.
            Defaults to None.
        compress (str, optional):
            Compression method for the raster (e.g., "lzw", "deflate").
            Defaults to "lzw".
        transpose (bool, optional):
            If True, assumes input shape is (bands, rows, cols) and transposes
            to (rows, cols, bands) before writing. Defaults to True.
        crs (Optional[str], optional):
            Coordinate reference system (e.g., "EPSG:4326").
            If None, no CRS is assigned. Defaults to None.
        transform (Optional[tuple], optional):
            Affine transform defining georeferencing.
            If None, raster is written without spatial reference.
            Defaults to None.
        driver (str, optional):
            GDAL driver to use for writing. Defaults to "GTiff".
        **kwargs:
            Additional keyword arguments passed to `rasterio.open()`.

    Returns:
        str: Path to the saved file.
    """
    # --- ensure correct shape ---
    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))

    # --- ensure output directory exists ---
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # --- get driver from extension ---
    ext = os.path.splitext(output)[-1].lower()
    driver_map = {"": "COG", ".tif": "GTiff", ".tiff": "GTiff", ".dat": "ENVI"}
    driver = driver_map.get(ext, "COG")
    if ext == "":
        output += ".tif"

    # --- infer dtype if not given ---
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

    # --- set metadata ---
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

    # --- write raster ---
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

    return output  # it's a file path


def read_prismaL2BC(
    file_path: str,
    product_type: str = "PRS_L2B",  # "PRS_L2B" or "PRS_L2C"
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    panchromatic: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, str], np.ndarray, np.ndarray]:
    """
    Reads PRISMA Level-2B or Level-2C .he5 data (VNIR+SWIR or Panchromatic),
    apply scaling from DN to reflectance and return the data together with
    geolocation arrays.

    Args:
        file_path (str):
            Path to the PRISMA L2B/L2C `.he5` file.
        product_type (str, optional):
            PRISMA product type, either "PRS_L2B" or "PRS_L2C". Defaults to "PRS_L2B".
        wavelengths (Optional[List[float]], optional):
            List of target wavelengths in nanometers to extract (for hyperspectral).
            If None, all available wavelengths are used. Defaults to None.
        method (str, optional):
            Wavelength selection method:
                - "nearest": select the closest available band for each requested wavelength.
                - "exact": select only exact matches; raises ValueError if none found.
            Defaults to "nearest".
        panchromatic (bool, optional):
            If True, read the panchromatic cube instead of the hyperspectral VNIR+SWIR cubes.
            Defaults to False.

    Returns:
        Tuple[np.ndarray, Union[np.ndarray, str], np.ndarray, np.ndarray]:
            A 4-element tuple containing:
            - cube_or_pancube (np.ndarray):
                * Hyperspectral: 3D array with shape (rows, cols, wavelengths) of reflectance
                  (dtype float32) after applying scaling attributes.
                * Panchromatic: 2D array with shape (rows, cols) of reflectance.
            - wavelengths (np.ndarray or str):
                * Hyperspectral: 1D array of wavelengths (in nm) corresponding to the band axis.
                * Panchromatic: the string "Panchromatic".
            - lat (np.ndarray): 2D latitude array with shape (rows, cols).
            - lon (np.ndarray): 2D longitude array with shape (rows, cols).
    """
    if not check_valid_file(file_path, type=product_type):
        raise ValueError(
            f"{file_path} is not a valid {product_type} file or does not exist."
        )

    try:
        with h5py.File(file_path, "r") as f:
            if panchromatic:
                # --- PANCHROMATIC ---
                cube_path = f"HDFEOS/SWATHS/{product_type}_PCO/Data Fields/Cube"
                pancube_data = f[cube_path][()]

                fill_value = 0
                max_data_value = 65535
                l2_scale_pan_min = f.attrs.get("L2ScalePanMin", 0.0)
                l2_scale_pan_max = f.attrs.get("L2ScalePanMax", 1.0)

                pancube_data = l2_scale_pan_min + (
                    pancube_data.astype(np.float32) / max_data_value
                ) * (l2_scale_pan_max - l2_scale_pan_min)
                pancube_data[pancube_data == fill_value] = np.nan

                # --- read lat/lon ---
                lat = f[
                    f"HDFEOS/SWATHS/{product_type}_PCO/Geolocation Fields/Latitude"
                ][()]
                lon = f[
                    f"HDFEOS/SWATHS/{product_type}_PCO/Geolocation Fields/Longitude"
                ][()]

                wavelengths = "Panchromatic"
                return pancube_data, wavelengths, lat, lon

            else:
                # --- HYPERSPECTRAL VNIR+SWIR ---
                cube_path = f"HDFEOS/SWATHS/{product_type}_HCO/Data Fields"
                vnir_cube = f[f"{cube_path}/VNIR_Cube"][()]
                swir_cube = f[f"{cube_path}/SWIR_Cube"][()]
                vnir_wavelengths = f.attrs["List_Cw_Vnir"][()]
                swir_wavelengths = f.attrs["List_Cw_Swir"][()]

                max_data_value = 65535
                fill_value = 0

                l2_scale_vnir_min = f.attrs["L2ScaleVnirMin"][()]
                l2_scale_vnir_max = f.attrs["L2ScaleVnirMax"][()]
                vnir_cube = l2_scale_vnir_min + (
                    vnir_cube.astype(np.float32) / max_data_value
                ) * (l2_scale_vnir_max - l2_scale_vnir_min)

                l2_scale_swir_min = f.attrs["L2ScaleSwirMin"][()]
                l2_scale_swir_max = f.attrs["L2ScaleSwirMax"][()]
                swir_cube = l2_scale_swir_min + (
                    swir_cube.astype(np.float32) / max_data_value
                ) * (l2_scale_swir_max - l2_scale_swir_min)

                vnir_cube[vnir_cube == fill_value] = np.nan
                swir_cube[swir_cube == fill_value] = np.nan

                # --- combine cubes ---
                full_cube = np.concatenate((vnir_cube, swir_cube), axis=1)
                full_wavelengths = np.concatenate((vnir_wavelengths, swir_wavelengths))

                # --- filter and sort wavelengths ---
                valid_idx = full_wavelengths > 0
                full_wavelengths = full_wavelengths[valid_idx]
                full_cube = full_cube[:, valid_idx, :]
                sort_idx = np.argsort(full_wavelengths)
                full_wavelengths = full_wavelengths[sort_idx]
                full_cube = full_cube[:, sort_idx, :]

                # --- select requested wavelengths ---
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

                full_cube = full_cube.transpose(0, 2, 1)
                # --- read lat/lon ---
                lat = f[
                    f"HDFEOS/SWATHS/{product_type}_HCO/Geolocation Fields/Latitude"
                ][()]
                lon = f[
                    f"HDFEOS/SWATHS/{product_type}_HCO/Geolocation Fields/Longitude"
                ][()]

                return full_cube, full_wavelengths, lat, lon

    except Exception as e:
        raise RuntimeError(f"Error reading the file {file_path}: {e}")


def write_prismaL2BC(
    filepath: str,
    output: str,
    product_type: str = "PRS_L2B",  # "PRS_L2B" or "PRS_L2C"
    panchromatic: bool = False,
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> Optional[str]:
    """
    Convert a PRISMA Level-2B/2C dataset (.he5) into a georeferenced raster image.

    This function loads PRISMA hyperspectral (VNIR+SWIR) or panchromatic data,
    applies scaling and georeferencing using latitude/longitude fields, and writes
    the result to a raster file (e.g., GeoTIFF).

    Args:
        filepath (str):
            Path to the input PRISMA L2B/L2C `.he5` file.
        output (str):
            Path where the output raster will be saved.
        product_type (str, optional):
            PRISMA product type, either `"PRS_L2B"` or `"PRS_L2C"`.
            Defaults to `"PRS_L2B"`.
        panchromatic (bool, optional):
            If True, treat the dataset as a single-band panchromatic cube.
            If False, read the hyperspectral VNIR+SWIR cubes.
            Defaults to False.
        wavelengths (Optional[np.ndarray], optional):
            List/array of wavelengths (in nm) to select (hyperspectral only).
            If None, all available wavelengths are included.
            Defaults to None.
        method (str, optional):
            Method for wavelength selection:
                - `"nearest"`: select the closest available band for each requested wavelength.
                - `"exact"`: select only exact matches.
            Defaults to `"nearest"`.
        **kwargs (Any):
            Additional keyword arguments passed to:
                - `array_to_image()` for raster writing
                - `rasterio.open()` for file creation


    Returns:
        str: Output file path, or None if all values are NaN.
    """
    from affine import Affine

    # load dataset if it's a path to .he5
    if isinstance(filepath, str):
        if wavelengths is not None:
            full_cube, full_wavelengths, lat, lon = read_prismaL2BC(
                filepath,
                product_type=product_type,
                wavelengths=wavelengths,
                method=method,
                panchromatic=panchromatic,
            )
        else:
            full_cube, full_wavelengths, lat, lon = read_prismaL2BC(
                filepath,
                product_type=product_type,
                method=method,
                panchromatic=panchromatic,
            )

    # get np.array
    array = full_cube
    if not np.any(np.isfinite(array)):
        print("Warning: All reflectance values are NaN. Output image will be blank.")
        return None

    # get band names (wavelength) and, eventually, select specific bands
    if panchromatic:  # panchromatic
        kwargs["band_description"] = full_wavelengths
    else:  # cube
        kwargs["wavelengths"] = full_wavelengths

    # define transform
    res_y = 30 / 111320
    res_x = 30 / (111320 * np.cos(np.deg2rad(lat.mean())))

    x_min, x_max = lon.min() - res_x / 2, lon.max() + res_x / 2
    y_min, y_max = lat.min() - res_y / 2, lat.max() + res_y / 2
    extent = (x_min, y_min, x_max, y_max)

    ext_width = extent[2] - extent[0]
    ext_height = extent[3] - extent[1]
    height, width = lat.shape

    xResolution = ext_width / width
    yResolution = ext_height / height

    if np.allclose(lat[1:, 0] - lat[:-1, 0], lat[1, 0] - lat[0, 0]) and np.allclose(
        lon[0, 1:] - lon[0, :-1], lon[0, 1] - lon[0, 0]
    ):

        transform = Affine(xResolution, 0, extent[0], 0, -yResolution, extent[3])
    else:
        x0, y0 = lon[0, 0], lat[0, 0]
        x1, y1 = lon[0, 1], lat[0, 1]
        x2, y2 = lon[1, 0], lat[1, 0]

        dx_col = x1 - x0
        dy_col = y1 - y0
        dx_row = x2 - x0
        dy_row = y2 - y0

        transform = Affine(dx_col, dx_row, x0, dy_col, dy_row, y0)

    # write output
    return array_to_image(
        array,
        output=output,
        transpose=False,
        crs="EPSG:4326",
        transform=transform,
        **kwargs,
    )
