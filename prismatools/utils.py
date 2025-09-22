"""The utils module provides functions used by the other modules."""

# TODO: get_transform assumes PRISMA L2D; does not work for L2B/L2C at the moment.

import os
import pyproj
from typing import List, Tuple
from affine import Affine


def check_valid_file(file_path: str, type: str = "PRS_L2D") -> bool:
    """
    Checks if the given file path points to a valid file.

    Args:
        file_path (str): Path to the file.
        type (str, optional): Expected file type ('PRS_L2B', 'PRS_L2C', 'PRS_L2D'). Defaults to 'PRS_L2D'.

    Returns:
        bool: True if file_path points to the correct file, False otherwise.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    valid_types = {"PRS_L2B", "PRS_L2C", "PRS_L2D"}
    if type not in valid_types:
        raise ValueError(
            f"Unsupported file type: {type}. Supported types are {valid_types}."
        )

    basename = os.path.basename(file_path)
    return basename.startswith(type) and basename.endswith(".he5")


def convert_coords(
    coords: List[Tuple[float, float]], from_epsg: str, to_epsg: str
) -> List[Tuple[float, float]]:
    """
    Convert a list of coordinates from one EPSG to another. Coordinates are
    expected in the format (lat, lon).

    Args:
        coords (List[Tuple[float, float]]):
            List of tuples containing coordinates in the format (latitude, longitude).
        from_epsg (str): Source EPSG code (e.g. "epsg:4326").
        to_epsg (str): Target EPSG code (e.g. "epsg:32632").

    Returns:
        List of tuple containing converted coordinates (x, y)
    """
    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    return [transformer.transform(lon, lat) for lat, lon in coords]


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
