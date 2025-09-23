# Usage

## Import library

To use prismatools in a project:

```python
from pathlib import Path
import prismatools

from prismatools.prisma import (
    read_prismaL2D,
    write_prismaL2D,
    extract_prisma,
    write_prismaL2BC,
)
```

## Read and Write PRISMA L2D products

- Wavelengths are in nanometers (nm).
- Use method="nearest" to select the closest available band.
- Use panchromatic=True to load the PAN band.

To read PRISMA Hyperspectral data:

```python
file = Path("/path/to/PRS_L2D_STD_20210701T102345_... .he5")

# Read the full hyperspectral cube
ds = read_prismaL2D(file)
print(ds)

# Read only selected wavelengths
ds_sel = read_prismaL2D(file, wavelengths=[490, 560, 670, 1300, 2200], method="nearest")
print(ds_sel)

# Read the panchromatic band
ds_pan = read_prismaL2D(file, panchromatic=True)
print(ds_pan)
```

To write PRISMA Hyperspectral data:

```python
out_dir = Path("path/to/outdir")
out_dir.mkdir(parents=True, exist_ok=True)

# Write PAN band from source file
write_prismaL2D(file, output=str(out_dir / "img_pan.tif"), panchromatic=True)

# Write hyperspectral cube from source file
write_prismaL2D(file, output=str(out_dir / "img_cube.tif"))

# Write hyperspectral cube from an xarray.Dataset
write_prismaL2D(ds, output=str(out_dir / "img_cube_ds.tif"))
```

## Extract data from PRISMA dataset

- Input is (lat, lon) pairs.

```python
points = [
    [41.4468, 15.4646],
    [41.4500, 15.4700],
    [41.4400, 15.4600]
 ]

extracted_values = []
for lat, lon in points:
    res = extract_prisma(ds, lat=lat, lon=lon)
    extracted_values.append(res.values)

extracted_values[0]
```

## Index computation (band math)

- multiple indices can be provided in the expressions dictionary.

```python
ds = read_prismaL2D(file_path=file)

wavelength_dict = {'red': 660, 'nir': 840}

expressions =  {'NDVI' :'(nir - red) / (nir + red)'}

constants = {}

result = band_math(ds, expressions, wavelength_dict, constants, inplace=True)

print(result)
result.NDVI.plot(cmap='YlGn')
```

## PCA

```python
input_file = "path/to/TIFF_image.tif"  # replace with actual path
output_file = "path/to/out_TIFF_image.tif" # replace with actual path
n_components = 10

ds_pca, expl_var = pca_image(input_file, output_file, n_components=n_components)
print(ds_pca)
print(expl_var)
```

## Interactive visualization

In Jupyter Notebooks:

- Create a map.

```python
m = prismatools.Map()
m
```

- Read the PRISMA data

```python
ds = prismatools.read_prismaL2D(file_path=file)
```

- Add the PRISMA data to the map.

```python
m.add_prisma(ds, wavelengths=[650.0, 550.0, 450.0], vmin=0, vmax=0.2)
m.add("spectral")
m
```