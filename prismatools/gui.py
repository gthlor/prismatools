"""This module contains the user interface for the prismatools package."""

import os
import math
import numpy as np
import xarray as xr
from typing import Optional

import ipyleaflet
import ipyevents
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from ipyleaflet import WidgetControl, LayersControl

import bqplot
from bqplot import pyplot as plt

from .prisma import extract_prisma


def main_toolbar(m):
    """Creates the main toolbar and adds it to the map.

    Args:
        m (leafmap.Map): The leafmap Map object.
    """
    # add here additional tools
    all_tools = {
        "map": {
            "name": "basemap",
            "tooltip": "Change basemap",
        },
    }

    tools = {}
    for tool in all_tools:
        if os.environ.get(all_tools[tool]["name"].upper(), "").upper() != "FALSE":
            tools[tool] = all_tools[tool]

    icons = list(tools.keys())
    tooltips = [item["tooltip"] for item in list(tools.values())]
    toolnames = [item["name"].upper() for item in list(all_tools.values())]
    toolnames.sort()
    setattr(m, "_ENV_VARS", toolnames)

    icon_width = "32px"
    icon_height = "32px"
    n_cols = 3
    n_rows = math.ceil(len(icons) / n_cols)

    toolbar_grid = widgets.GridBox(
        children=[
            widgets.ToggleButton(
                layout=widgets.Layout(
                    width="auto", height="auto", padding="0px 0px 0px 4px"
                ),
                button_style="primary",
                icon=icons[i],
                tooltip=tooltips[i],
            )
            for i in range(len(icons))
        ],
        layout=widgets.Layout(
            width="109px",
            grid_template_columns=(icon_width + " ") * n_cols,
            grid_template_rows=(icon_height + " ") * n_rows,
            grid_gap="1px 1px",
            padding="5px",
        ),
    )
    m.toolbar = toolbar_grid

    def tool_callback(change):
        if change["new"]:
            current_tool = change["owner"]
            for tool in toolbar_grid.children:
                if tool is not current_tool:
                    tool.value = False
            tool = change["owner"]
            tool_name = tools[tool.icon]["name"]

            if tool_name == "basemap":
                change_basemap(m)
        else:
            # tool = change["owner"]
            # tool_name = tools[tool.icon]["name"]
            pass

        m.toolbar_reset()

    for tool in toolbar_grid.children:
        tool.observe(tool_callback, "value")

    toolbar_button = widgets.ToggleButton(
        value=False,
        tooltip="Toolbar",
        icon="wrench",
        layout=widgets.Layout(width="28px", height="28px", padding="0px 0px 0px 4px"),
    )
    m.toolbar_button = toolbar_button

    layers_button = widgets.ToggleButton(
        value=False,
        tooltip="Layers",
        icon="server",
        layout=widgets.Layout(height="28px", width="72px"),
    )

    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_button]
    toolbar_header = widgets.HBox()
    toolbar_header.children = [layers_button, toolbar_button]
    toolbar_footer = widgets.VBox()
    toolbar_footer.children = [toolbar_grid]

    toolbar_event = ipyevents.Event(
        source=toolbar_widget, watched_events=["mouseenter", "mouseleave"]
    )

    def handle_toolbar_event(event):
        if event["type"] == "mouseenter":
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        elif event["type"] == "mouseleave":
            if not toolbar_button.value:
                toolbar_widget.children = [toolbar_button]
                toolbar_button.value = False
                layers_button.value = False

    toolbar_event.on_dom_event(handle_toolbar_event)

    def toolbar_btn_click(change):
        if change["new"]:
            layers_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not layers_button.value:
                toolbar_widget.children = [toolbar_button]

    toolbar_button.observe(toolbar_btn_click, "value")

    def layers_btn_click(change):
        if change["new"]:
            toolbar_footer.children = layer_manager_gui(m, return_widget=True)
            toolbar_button.value = False
        else:
            toolbar_footer.children = [toolbar_grid]

    layers_button.observe(layers_btn_click, "value")

    toolbar_control = WidgetControl(widget=toolbar_widget, position="topright")

    m.add(toolbar_control)


def change_basemap(m, position: Optional[str] = "topright"):
    """Widget for changing basemaps.

    Args:
        m (object): leafmap.Map.
    """
    from leafmap.basemaps import get_xyz_dict
    from leafmap import basemaps, get_basemap

    xyz_dict = get_xyz_dict()

    value = "OpenStreetMap"

    dropdown = widgets.Dropdown(
        options=list(basemaps.keys()),
        value=value,
        layout=widgets.Layout(width="200px"),
    )

    close_btn = widgets.Button(
        icon="times",
        tooltip="Close the basemap widget",
        button_style="primary",
        layout=widgets.Layout(width="32px"),
    )

    basemap_widget = widgets.HBox([dropdown, close_btn])

    def on_click(change):
        if change["new"]:
            basemap_name = dropdown.value
            if basemap_name not in m.get_layer_names():
                m.add_basemap(basemap_name)
                if basemap_name in xyz_dict:
                    if "bounds" in xyz_dict[basemap_name]:
                        bounds = xyz_dict[basemap_name]["bounds"]
                        bounds = [
                            bounds[0][1],
                            bounds[0][0],
                            bounds[1][1],
                            bounds[1][0],
                        ]
                        m.zoom_to_bounds(bounds)

    dropdown.observe(on_click, "value")

    def close_click(change):
        m.toolbar_reset()
        if m.basemap_ctrl is not None and m.basemap_ctrl in m.controls:
            m.remove_control(m.basemap_ctrl)
        basemap_widget.close()

    close_btn.on_click(close_click)

    basemap_control = WidgetControl(widget=basemap_widget, position=position)
    m.add(basemap_control)
    m.basemap_ctrl = basemap_control


def layer_manager_gui(
    m,
    position: Optional[str] = "topright",
    opened: Optional[bool] = True,
    return_widget: Optional[bool] = False,
):
    """Creates a layer manager widget.

    Args:
        m (geemap.Map): The geemap.Map object.
        position (str, optional): The position of the widget. Defaults to "topright".
        return_widget (bool, optional): Whether to return the widget. Defaults to False.
    """

    layers_button = widgets.ToggleButton(
        value=False,
        tooltip="Layer Manager",
        icon="server",
        layout=widgets.Layout(width="28px", height="28px", padding="0px 0px 0px 4px"),
    )

    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding="0px 0px 0px 4px"),
    )

    toolbar_header = widgets.HBox()
    toolbar_header.children = [layers_button]
    toolbar_footer = widgets.VBox()
    toolbar_footer.children = []
    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_header]

    def toolbar_btn_click(change):
        if change["new"]:
            close_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not close_button.value:
                toolbar_widget.children = [layers_button]

    layers_button.observe(toolbar_btn_click, "value")

    def close_btn_click(change):
        if change["new"]:
            layers_button.value = False
            m.toolbar_reset()
            if m.layer_manager is not None and m.layer_manager in m.controls:
                m.remove_control(m.layer_manager)
                m.layer_manager = None
            toolbar_widget.close()

    close_button.observe(close_btn_click, "value")

    def layers_btn_click(change):
        if change["new"]:
            layers_hbox = []
            all_layers_chk = widgets.Checkbox(
                value=False,
                description="All layers on/off",
                indent=False,
                layout=widgets.Layout(height="18px", padding="0px 8px 25px 8px"),
            )
            all_layers_chk.layout.width = "30ex"
            layers_hbox.append(all_layers_chk)

            def all_layers_chk_changed(change):
                if change["new"]:
                    for layer in m.layers:
                        if hasattr(layer, "visible"):
                            layer.visible = True
                else:
                    for layer in m.layers:
                        if hasattr(layer, "visible"):
                            layer.visible = False

            all_layers_chk.observe(all_layers_chk_changed, "value")

            layers = [lyr for lyr in m.layers if lyr.name]

            # if the layers contain unsupported layers (e.g., GeoJSON, GeoData), adds the ipyleaflet built-in LayerControl
            if len(layers) < (len(m.layers) - 1):
                if m.layer_control is None:
                    layer_control = LayersControl(position="topright")
                    m.layer_control = layer_control
                if m.layer_control not in m.controls:
                    m.add(m.layer_control)

            # for non-TileLayer, use layer.style={'opacity':0, 'fillOpacity': 0} to turn layer off.
            for layer in layers:
                visible = True
                if hasattr(layer, "visible"):
                    visible = layer.visible
                layer_chk = widgets.Checkbox(
                    value=visible,
                    description=layer.name,
                    indent=False,
                    layout=widgets.Layout(height="18px"),
                )
                layer_chk.layout.width = "25ex"

                if layer in m.geojson_layers:
                    try:
                        opacity = max(
                            layer.style["opacity"], layer.style["fillOpacity"]
                        )
                    except KeyError:
                        opacity = 1.0
                elif hasattr(layer, "opacity"):
                    opacity = layer.opacity
                else:
                    opacity = 1.0

                layer_opacity = widgets.FloatSlider(
                    value=opacity,
                    min=0,
                    max=1,
                    step=0.01,
                    readout=False,
                    layout=widgets.Layout(width="80px"),
                )
                layer_settings = widgets.Button(
                    icon="gear",
                    tooltip=layer.name,
                    layout=widgets.Layout(
                        width="25px", height="25px", padding="0px 0px 0px 0px"
                    ),
                )

                def layer_settings_click(b):
                    if b.tooltip in m.cog_layer_dict:
                        m._add_layer_editor(
                            position="topright",
                            layer_dict=m.cog_layer_dict[b.tooltip],
                        )

                layer_settings.on_click(layer_settings_click)

                def layer_opacity_changed(change):
                    if change["new"]:
                        layer.style = {
                            "opacity": change["new"],
                            "fillOpacity": change["new"],
                        }

                if hasattr(layer, "visible"):
                    widgets.jslink((layer_chk, "value"), (layer, "visible"))

                if layer in m.geojson_layers:
                    layer_opacity.observe(layer_opacity_changed, "value")
                elif hasattr(layer, "opacity"):
                    widgets.jsdlink((layer_opacity, "value"), (layer, "opacity"))

                hbox = widgets.HBox(
                    [layer_chk, layer_settings, layer_opacity],
                    layout=widgets.Layout(padding="0px 8px 0px 8px"),
                )
                layers_hbox.append(hbox)
                m.layer_widget = layers_hbox

            toolbar_header.children = [close_button, layers_button]
            toolbar_footer.children = layers_hbox

        else:
            toolbar_header.children = [layers_button]

    layers_button.observe(layers_btn_click, "value")
    layers_button.value = opened

    if not hasattr(m, "_layer_manager_widget"):
        m._layer_manager_widget = toolbar_footer

    if return_widget:
        return m.layer_widget
    else:
        layer_control = WidgetControl(widget=toolbar_widget, position=position)

        if layer_control not in m.controls:
            m.add_control(layer_control)
            m.layer_manager = layer_control


class spectralWidget(widgets.HBox):
    """
    A widget for spectral data visualization on a map.

    Attributes:
        _host_map (Map): The map to host the widget.
        on_close (function): Function to be called when the widget is closed.
        _output_widget (widgets.Output): The output widget to display results.
        _output_control (ipyleaflet.WidgetControl): The control for the output widget.
        _on_map_interaction (function): Function to handle map interactions.
        _spectral_widget (SpectralWidget): The spectral widget itself.
        _spectral_control (ipyleaflet.WidgetControl): The control for the spectral widget.
    """

    def __init__(
        self, host_map, stack=True, position="topright", xlim=None, ylim=None, **kwargs
    ):
        """
        Initializes a new instance of the SpectralWidget class.

        Args:
            host_map (Map): The map to host the widget.
            stack (bool, optional): Whether to stack the plots. Defaults to True.
            position (str, optional): The position of the widget on the map. Defaults to "topright".
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
        """
        self._host_map = host_map

        if not hasattr(self._host_map, "_spectral_counter"):
            self._host_map._spectral_counter = 0

        self.on_close = None
        self._stack = stack
        self._show_plot = False

        fig_margin = {"top": 20, "bottom": 35, "left": 50, "right": 20}
        fig = plt.figure(
            # title=None,
            fig_margin=fig_margin,
            layout={"width": "500px", "height": "300px"},
        )

        self._fig = fig
        self._host_map._fig = fig

        layer_names = list(host_map.cog_layer_dict.keys())
        layers_widget = widgets.Dropdown(options=layer_names)
        layers_widget.layout.width = "18ex"

        close_btn = widgets.Button(
            icon="times",
            tooltip="Close the widget",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        reset_btn = widgets.Button(
            icon="trash",
            tooltip="Remove all markers",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        stack_btn = widgets.ToggleButton(
            value=stack,
            icon="area-chart",
            tooltip="Stack spectral signatures",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        def reset_btn_click(_):
            if hasattr(self._host_map, "_plot_marker_cluster"):
                self._host_map._plot_marker_cluster.markers = []
                self._host_map._plot_markers = []

            if hasattr(self._host_map, "_spectral_data"):
                self._host_map._spectral_data = {}

            if hasattr(self._host_map, "_spectral_counter"):
                self._host_map._spectral_counter = 0

            self._output_widget.clear_output()
            self._show_plot = False
            plt.clear()

        reset_btn.on_click(reset_btn_click)

        save_btn = widgets.Button(
            icon="floppy-o",
            tooltip="Save the data to a CSV",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        def chooser_callback(chooser):
            if chooser.selected:
                file_path = chooser.selected
                self._host_map.spectral_to_csv(file_path)
                if (
                    hasattr(self._host_map, "_file_chooser_control")
                    and self._host_map._file_chooser_control in self._host_map.controls
                ):
                    self._host_map.remove_control(self._host_map._file_chooser_control)
                    self._host_map._file_chooser.close()

        def save_btn_click(_):
            if not hasattr(self._host_map, "_spectral_data"):
                return

            self._output_widget.clear_output()
            file_chooser = FileChooser(
                os.getcwd(), layout=widgets.Layout(width="454px")
            )
            file_chooser.filter_pattern = "*.csv"
            file_chooser.use_dir_icons = True
            file_chooser.title = "Save spectral data to a CSV file"
            file_chooser.default_filename = "spectral_data.csv"
            file_chooser.show_hidden = False
            file_chooser.register_callback(chooser_callback)
            file_chooser_control = ipyleaflet.WidgetControl(
                widget=file_chooser, position="topright"
            )
            self._host_map.add(file_chooser_control)
            setattr(self._host_map, "_file_chooser", file_chooser)
            setattr(self._host_map, "_file_chooser_control", file_chooser_control)

        save_btn.on_click(save_btn_click)

        def close_widget(_):
            self.cleanup()

        close_btn.on_click(close_widget)

        super().__init__([layers_widget, stack_btn, reset_btn, save_btn, close_btn])

        output = widgets.Output()
        output_control = ipyleaflet.WidgetControl(widget=output, position="bottomright")
        self._output_widget = output
        self._output_control = output_control
        self._host_map.add(output_control)

        if not hasattr(self._host_map, "_spectral_data"):
            self._host_map._spectral_data = {}

        def handle_interaction(**kwargs):

            latlon = kwargs.get("coordinates")
            lat = latlon[0]
            lon = latlon[1]
            if kwargs.get("type") == "click" and self._host_map._layer_editor is None:

                self._host_map._spectral_counter += 1
                idx = self._host_map._spectral_counter
                label_txt = f"P {idx}"

                layer_name = layers_widget.value

                if not hasattr(self._host_map, "_plot_markers"):
                    self._host_map._plot_markers = []

                markers = self._host_map._plot_markers
                marker_cluster = self._host_map._plot_marker_cluster
                marker = ipyleaflet.Marker(location=latlon, draggable=False)
                marker.popup = widgets.HTML(value=label_txt)
                markers.append(marker)
                marker_cluster.markers = markers
                self._host_map._plot_marker_cluster = marker_cluster

                xlabel = "Wavelength (nm)"
                ylabel = "Reflectance"

                ds = self._host_map.cog_layer_dict[layer_name]["xds"]

                if self._host_map.cog_layer_dict[layer_name]["hyper"] == "PRISMA":
                    da = extract_prisma(ds, lat, lon)

                self._host_map._spectral_data[f"({lat:.4f} {lon:.4f})"] = da.values

                if self._host_map.cog_layer_dict[layer_name]["hyper"] != "XARRAY":
                    da[da < 0] = np.nan
                    x_axis_options = {"label_offset": "30px"}
                else:
                    x_axis_options = {
                        "label_offset": "30px",
                        "tick_format": "0d",
                        "num_ticks": da.sizes["band"],
                    }
                axes_options = {
                    "x": x_axis_options,
                    "y": {"label_offset": "35px"},
                }

                if not stack_btn.value:
                    plt.clear()
                    plt.plot(
                        da.coords[da.dims[0]].values,
                        da.values,
                        axes_options=axes_options,
                    )
                else:
                    color = np.random.rand(
                        3,
                    )
                    if "wavelength" in da.coords:
                        xlabel = "Wavelength (nm)"
                        x_values = da["wavelength"].values
                    else:
                        xlabel = "Band"
                        x_values = da.coords[da.dims[0]].values

                    line = plt.plot(
                        x_values,
                        da.values,
                        color=color,
                        axes_options=axes_options,
                    )
                    line.labels = [label_txt]

                    plt.legend()

                    try:
                        if isinstance(self._fig.axes[0], bqplot.ColorAxis):
                            self._fig.axes = self._fig.axes[1:]
                        elif isinstance(self._fig.axes[-1], bqplot.ColorAxis):
                            self._fig.axes = self._fig.axes[:-1]
                    except Exception:
                        pass

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if xlim:
                    plt.xlim(xlim[0], xlim[1])
                if ylim:
                    plt.ylim(ylim[0], ylim[1])

                if not self._show_plot:
                    with self._output_widget:
                        plt.show()
                        self._show_plot = True

                self._host_map.default_style = {"cursor": "crosshair"}

        self._host_map.on_interaction(handle_interaction)
        self._on_map_interaction = handle_interaction

        self._spectral_widget = self
        self._spectral_control = ipyleaflet.WidgetControl(
            widget=self, position=position
        )
        self._host_map.add(self._spectral_control)

    def cleanup(self):
        """Removes the widget from the map and performs cleanup."""
        if self._host_map:
            self._host_map.default_style = {"cursor": "default"}
            self._host_map.on_interaction(self._on_map_interaction, remove=True)

            if self._output_control:
                self._host_map.remove_control(self._output_control)

                if self._output_widget:
                    self._output_widget.close()
                    self._output_widget = None

            if self._spectral_control:
                self._host_map.remove_control(self._spectral_control)
                self._spectral_control = None

                if self._spectral_widget:
                    self._spectral_widget.close()
                    self._spectral_widget = None

            if hasattr(self._host_map, "_plot_marker_cluster"):
                self._host_map._plot_marker_cluster.markers = []
                self._host_map._plot_markers = []

            if hasattr(self._host_map, "_spectral_data"):
                self._host_map._spectral_data = {}

            if hasattr(self._host_map, "_spectral_counter"):
                self._host_map._spectral_counter = 0

            if hasattr(self, "_output_widget") and self._output_widget is not None:
                self._output_widget.clear_output()

        if self.on_close is not None:
            self.on_close()
