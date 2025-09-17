"""Main module."""

import numpy as np
import tempfile

import ipyleaflet
import leafmap

from .prisma import read_prismaL2D, write_prismaL2D
from .gui import main_toolbar, spectralWidget


class Map(leafmap.Map):
    """
    A class that extends leafmap.Map to provide additional functionality for
        prismatools.

    Attributes:
        Any attributes inherited from leafmap.Map.

    Methods:
        Any methods inherited from leafmap.Map.
    """

    def __init__(self, center=[0, 0], zoom=2, **kwargs):
        """
        Initializes a new instance of the Map class.

        Args:
            **kwargs: Arbitrary keyword arguments that are passed to the parent
                class's constructor.
        """
        # set off toolbars not needed from Leafmap
        kwargs["layers_control"] = False
        kwargs["fullscreen_control"] = True
        kwargs["search_control"] = False
        kwargs["draw_control"] = False
        kwargs["measure_control"] = False
        kwargs["scale_control"] = True
        kwargs["toolbar_control"] = False

        super().__init__(**kwargs)
        # add my simplified toolbar
        main_toolbar(self)

        self._spectral_data = {}
        self._plot_options = None

    def add(self, obj, position="topright", xlim=None, ylim=None, **kwargs):
        """Add a layer to the map.

        Args:
            obj (str or object): The name of the layer or a layer object.
            position (str, optional): The position of the layer widget. Can be
                'topright', 'topleft', 'bottomright', or 'bottomleft'. Defaults
                to 'topright'.
            xlim (tuple, optional): The x-axis limits of the plot. Defaults to None.
            ylim (tuple, optional): The y-axis limits of the plot. Defaults to None.
            **kwargs: Arbitrary keyword arguments that are passed to the parent
                class's add_layer method.
        """

        if isinstance(obj, str):
            if obj == "spectral":

                spectralWidget(self, position=position, xlim=xlim, ylim=ylim, **kwargs)
                self.set_plot_options(add_marker_cluster=True)
            else:
                super().add(obj, **kwargs)

        else:
            super().add(obj, **kwargs)

    def add_prisma(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=0.5,
        nodata=np.nan,
        attribution=None,
        layer_name="PRISMA",
        zoom_to_layer=True,
        visible=True,
        array_args=None,
        method="nearest",
        **kwargs,
    ):
        """Add a PRISMA dataset to the map.

        This function reads a PRISMA hyperspectral dataset, optionally selects
        specific wavelengths, converts the data to an image, and adds it as a
        raster layer to the map. The dataset can be provided as a file path or
        as an xarray Dataset.

        Args:
            source (str or xarray.Dataset): The path to the PRISMA file or an
                in-memory xarray Dataset containing PRISMA data.
            wavelengths (list or np.ndarray, optional): Specific wavelengths to
                select from the dataset. If None, all wavelengths are used.
                Defaults to None.
            indexes (int or list, optional): The band(s) to display. Band
                indexing starts at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See:
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is greyscale.
            vmin (float, optional): The minimum value for color mapping when
                plotting a single band. Defaults to 0.
            vmax (float, optional): The maximum value for color mapping when
                plotting a single band. Defaults to 0.5.
            nodata (float, optional): Value in the raster to interpret as
                no-data. Defaults to np.nan.
            attribution (str, optional): Attribution for the source raster.
                Defaults to None.
            layer_name (str, optional): The name to assign to the map layer.
                Defaults to "PRISMA".
            zoom_to_layer (bool, optional): Whether to zoom the map to the
                extent of the layer after adding it. Defaults to True.
            visible (bool, optional): Whether the layer should be visible when
                first added. Defaults to True.
            array_args (dict, optional): Additional keyword arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            method (str, optional): Method to use for wavelength interpolation
                when selecting bands. Options may include "nearest", "linear",
                etc. Defaults to "nearest".
            **kwargs: Additional keyword arguments passed to `add_raster`.
        """
        if array_args is None:
            array_args = {}

        if isinstance(source, str):
            xds = read_prismaL2D(source)
        else:
            xds = source

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name

        # create a temporary .tif with only selected wavelengths because leafmap (built on ipyleaflet) can add only images with 3 bands or 1 band
        write_prismaL2D(xds, wavelengths=wavelengths, method=method, output=temp_path)

        self.add_raster(  # this function will add the tiff to the map
            temp_path,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            array_args=array_args,
            **kwargs,
        )

        self.cog_layer_dict[layer_name][
            "xds"
        ] = xds  # avoid losing spectral information
        self.cog_layer_dict[layer_name]["hyper"] = "PRISMA"

    def add_raster(
        self,
        source,
        indexes=None,
        colormap=None,
        vmin=None,
        vmax=None,
        nodata=None,
        attribution=None,
        layer_name="Raster",
        layer_index=None,
        zoom_to_layer=True,
        visible=True,
        opacity=1.0,
        array_args=None,
        client_args={"cors_all": False},
        open_args=None,
        **kwargs,
    ):
        """Add a local raster dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'Raster'.
            layer_index (int, optional): The index of the layer. Defaults to None.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to
                True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            client_args (dict, optional): Additional arguments to pass to
                localtileserver.TileClient. Defaults to { "cors_all": False }.
            open_args (dict, optional): Additional arguments to pass to
                rioxarray.open_rasterio.

        """
        import rioxarray as rxr

        if array_args is None:
            array_args = {}
        if open_args is None:
            open_args = {}

        if nodata is None:
            nodata = np.nan

        super().add_raster(
            source,
            indexes=indexes,  # if source has more bands, you need to specify 3 indices
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            layer_index=layer_index,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            opacity=opacity,
            array_args=array_args,
            client_args=client_args,
            **kwargs,
        )

        if isinstance(source, str):
            da = rxr.open_rasterio(source, **open_args)
            dims = da.dims
            da = da.transpose(dims[1], dims[2], dims[0])

            xds = da.to_dataset(name="data")
            self.cog_layer_dict[layer_name]["xds"] = xds

    def spectral_to_df(self, **kwargs):
        """Converts the spectral data to a pandas DataFrame.

        Returns:
            pd.DataFrame: The spectral data as a pandas DataFrame.
        """
        import pandas as pd

        df = pd.DataFrame(self._spectral_data, **kwargs)
        return df

    def spectral_to_csv(self, filename, index=True, **kwargs):
        """Saves the spectral data to a CSV file.

        Args:
            filename (str): The output CSV file.
            index (bool, optional): Whether to write the index. Defaults to True.
        """
        df = self.spectral_to_df()
        df = df.rename_axis("band")
        df.to_csv(filename, index=index, **kwargs)

    def set_plot_options(
        self,
        add_marker_cluster=False,
        plot_type=None,
        overlay=False,
        position="bottomright",
        min_width=None,
        max_width=None,
        min_height=None,
        max_height=None,
        **kwargs,
    ):
        """Sets plotting options.

        Args:
            add_marker_cluster (bool, optional): Whether to add a marker cluster.
                Defaults to False.
            sample_scale (float, optional):  A nominal scale in meters of the
                projection to sample in . Defaults to None.
            plot_type (str, optional): The plot type can be one of "None", "bar",
                "scatter" or "hist". Defaults to None.
            overlay (bool, optional): Whether to overlay plotted lines on the
                figure. Defaults to False.
            position (str, optional): Position of the control, can be
                ‘bottomleft’, ‘bottomright’, ‘topleft’, or ‘topright’. Defaults
                to 'bottomright'.
            min_width (int, optional): Min width of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            max_width (int, optional): Max width of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            min_height (int, optional): Min height of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            max_height (int, optional): Max height of the widget (in pixels), if
                None it will respect the content size. Defaults to None.

        """
        plot_options_dict = {}
        plot_options_dict["add_marker_cluster"] = add_marker_cluster
        plot_options_dict["plot_type"] = plot_type
        plot_options_dict["overlay"] = overlay
        plot_options_dict["position"] = position
        plot_options_dict["min_width"] = min_width
        plot_options_dict["max_width"] = max_width
        plot_options_dict["min_height"] = min_height
        plot_options_dict["max_height"] = max_height

        for key in kwargs:
            plot_options_dict[key] = kwargs[key]

        self._plot_options = plot_options_dict

        if not hasattr(self, "_plot_marker_cluster"):
            self._plot_marker_cluster = ipyleaflet.MarkerCluster(name="Marker Cluster")

        if add_marker_cluster and (self._plot_marker_cluster not in self.layers):
            self.add(self._plot_marker_cluster)
