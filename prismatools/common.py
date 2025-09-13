"""The common module contains common functions and classes used by the other modules."""

import os
import numpy as np
import xarray as xr
import rioxarray as rio
import h5py

from typing import List, Tuple, Union, Optional
from affine import Affine


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


def read_prismaL2D(
    file_path: str, wavelengths: Optional[List[float]] = None, method: str = "nearest"
) -> xr.Dataset:
    """
    Reads PRISMA hyperspectral Level-2D .he5 data and returns an xarray dataset with
    reflectance values, associated wavelengths, and geospatial metadata.

    Args:
        file_path (str): Path to the PRISMA L2D .he5 file.
        wavelengths (Optional[List[float]]): List of wavelengths (in nm) to extract.
            - If None, all valid wavelengths are used.
            - If provided, can select by exact match or nearest available wavelength.
        method (str, default "nearest"): Method to select wavelengths when `wavelengths` is provided. Options are:
            - "nearest": selects the closest available wavelength.
            - "exact": selects only wavelengths exactly matching those requested.

    Returns:
        xr.Dataset: An xarray.Dataset containing reflectance data with coordinates.
    """
    # check if file is valid
    if not check_valid_file(file_path, type="PRS_L2D"):
        raise ValueError(
            f"The file {file_path} is not a valid PRS_L2D file or does not exist."
        )

    # get data and metadata
    try:
        with h5py.File(file_path, "r") as f:
            swir_cube_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
            vnir_cube_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
            swir_cube_data = f[swir_cube_path][()]
            vnir_cube_data = f[vnir_cube_path][()]
            vnir_wavelengths = f.attrs["List_Cw_Vnir"][()]
            swir_wavelengths = f.attrs["List_Cw_Swir"][()]
            l2_scale_vnir_min = f.attrs["L2ScaleVnirMin"][()]
            l2_scale_vnir_max = f.attrs["L2ScaleVnirMax"][()]
            l2_scale_swir_min = f.attrs["L2ScaleSwirMin"][()]
            l2_scale_swir_max = f.attrs["L2ScaleSwirMax"][()]
            epsg_code = f.attrs["Epsg_Code"][()]
            ul_easting = f.attrs["Product_ULcorner_easting"][()]
            ul_northing = f.attrs["Product_ULcorner_northing"][()]
    except Exception as e:
        raise RuntimeError(f"Error reading the file {file_path}: {e}")

    fill_value = -9999
    max_data_value = 65535

    # scale data to reflectance and set fill value to -9999
    vnir_cube_data = l2_scale_vnir_min + (
        vnir_cube_data.astype(np.float32) / max_data_value
    ) * (l2_scale_vnir_max - l2_scale_vnir_min)
    swir_cube_data = l2_scale_swir_min + (
        swir_cube_data.astype(np.float32) / max_data_value
    ) * (l2_scale_swir_max - l2_scale_swir_min)

    vnir_cube_data[vnir_cube_data == fill_value] = np.nan
    swir_cube_data[swir_cube_data == fill_value] = np.nan

    # combine VNIR and SWIR data
    full_cube_data = np.concatenate((vnir_cube_data, swir_cube_data), axis=1)
    full_wavelengths = np.concatenate((vnir_wavelengths, swir_wavelengths))

    # filter wavelengths if specified or corrupted
    valid_indices = full_wavelengths > 0
    full_wavelengths = full_wavelengths[valid_indices]
    full_cube_data = full_cube_data[:, valid_indices, :]

    sort_indices = np.argsort(full_wavelengths)
    full_wavelengths = full_wavelengths[sort_indices]
    full_cube_data = full_cube_data[:, sort_indices, :]

    if wavelengths is not None:
        requested = np.array(wavelengths)
        available = full_wavelengths

        if method == "exact":
            idx = np.where(np.isin(available, requested))[0]
            if len(idx) == 0:
                raise ValueError(
                    "No requested wavelengths found in the data (exact match)."
                )
        else:  # "nearest"
            # find the closest available wavelengths to those requested
            idx = np.array([np.abs(available - w).argmin() for w in requested])

        full_cube_data = full_cube_data[:, idx, :]
        full_wavelengths = available[idx]

    # create coordinates and geotransform
    rows = full_cube_data.shape[0]
    cols = full_cube_data.shape[2]

    transform = get_transform(ul_easting, ul_northing, res=30)
    x_coords = np.array([transform * (i, 0) for i in range(cols)])[:, 0]
    y_coords = np.array([transform * (0, j) for j in range(rows)])[:, 1]

    crs = f"EPSG:{epsg_code}"
    if crs is None:
        raise ValueError(
            "Dataset has no CRS. Please ensure read_prisma writes CRS before returning."
        )

    # create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            reflectance=(
                ["y", "wavelength", "x"],
                full_cube_data,
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
    ds.rio.write_crs(crs, inplace=True)
    ds.rio.write_transform(transform, inplace=True)

    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = dict(
        units="unitless",
        _FillValue=-9999,
        grid_mapping="crs",
        standard_name="reflectance",
        long_name="atmospherically corrected surface reflectance",
        crs=ds.rio.crs.to_string(),
    )
    ds.attrs.update(global_atts)
    return ds


def read_prismaL2D_pan(file_path: str) -> xr.Dataset:
    """
    Reads PRISMA panchromatic Level-2D .he5 data and returns an xarray dataset with
    reflectance values and geospatial metadata.

    Args:
        file_path (str): Path to the PRISMA L2D panchromatic .he5 file.

    Returns:
        xr.Dataset: An xarray.Dataset containing reflectance data with coordinates.
    """
    # check if file is valid
    if not check_valid_file(file_path, type="PRS_L2D"):
        raise ValueError(
            f"The file {file_path} is not a valid PRS_L2D file or does not exist."
        )

    # get data and metadata
    try:
        with h5py.File(file_path, "r") as f:
            pancube_path = "HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube"
            pancube_data = f[pancube_path][()]
            l2_scale_pan_min = f.attrs["L2ScalePanMin"][()]
            l2_scale_pan_max = f.attrs["L2ScalePanMax"][()]
            epsg_code = f.attrs["Epsg_Code"][()]
            ul_easting = f.attrs["Product_ULcorner_easting"][()]
            ul_northing = f.attrs["Product_ULcorner_northing"][()]
    except Exception as e:
        raise RuntimeError(f"Error reading the file {file_path}: {e}")

    fill_value = -9999
    max_data_value = 65535

    # scale data to reflectance and set fill value to -9999
    pancube_data = l2_scale_pan_min + (
        pancube_data.astype(np.float32) / max_data_value
    ) * (l2_scale_pan_max - l2_scale_pan_min)
    pancube_data[pancube_data == fill_value] = np.nan

    # create coordinates and geotransform
    rows = pancube_data.shape[0]
    cols = pancube_data.shape[1]

    transform = get_transform(ul_easting, ul_northing, res=5)
    x_coords = np.array([transform * (i, 0) for i in range(cols)])[:, 0]
    y_coords = np.array([transform * (0, j) for j in range(rows)])[:, 1]

    crs = f"EPSG:{epsg_code}"
    if crs is None:
        raise ValueError(
            "Dataset has no CRS. Please ensure read_prisma writes CRS before returning."
        )
    # create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            reflectance=(
                ["y", "x"],
                pancube_data,
                dict(
                    units="unitless",
                    _FillValue=np.nan,
                    standard_name="reflectance",
                    long_name="Panchromatic atmospherically corrected surface reflectance",
                ),
            ),
        ),
        coords=dict(
            y=(["y"], y_coords, dict(units="m")),
            x=(["x"], x_coords, dict(units="m")),
        ),
    )
    ds.rio.write_crs(crs, inplace=True)
    ds.rio.write_transform(transform, inplace=True)
    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = dict(
        units="unitless",
        _FillValue=-9999,
        grid_mapping="crs",
        standard_name="reflectance",
        long_name="atmospherically corrected surface reflectance",
        crs=ds.rio.crs.to_string(),
    )
    ds.attrs.update(global_atts)
    return ds


# debugging
# if __name__ == "__main__":
#     file = r"C:/Users/loren/Desktop/PRS_L2D_STD_20240429095823_20240429095827_0001\PRS_L2D_STD_20240429095823_20240429095827_0001.he5"
#     ds = read_prismaL2D(file, wavelengths=None, method="nearest")
#     print(ds)

#     ds_pan = read_prismaL2D_pan(file)
#     print(ds_pan)
