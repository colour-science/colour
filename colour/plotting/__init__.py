from colour.utilities import is_matplotlib_installed

if not is_matplotlib_installed():
    import sys
    from unittest.mock import MagicMock
    from colour.utilities import usage_warning

    try:
        is_matplotlib_installed(raise_exception=True)
    except ImportError as error:
        usage_warning(str(error))

    for module in (
        "matplotlib",
        "matplotlib.axes",
        "matplotlib.cm",
        "matplotlib.collections",
        "matplotlib.colors",
        "matplotlib.figure",
        "matplotlib.patches",
        "matplotlib.path",
        "matplotlib.pyplot",
        "matplotlib.ticker",
        "mpl_toolkits",
        "mpl_toolkits.mplot3d",
        "mpl_toolkits.mplot3d.art3d",
        "mpl_toolkits.mplot3d.axes3d",
    ):
        sys.modules[module] = MagicMock()

from .datasets import *  # noqa: F403
from . import datasets
from .common import (
    CONSTANTS_COLOUR_STYLE,
    CONSTANTS_ARROW_STYLE,
    colour_style,
    override_style,
    XYZ_to_plotting_colourspace,
    ColourSwatch,
    colour_cycle,
    artist,
    camera,
    render,
    label_rectangles,
    uniform_axes3d,
    filter_passthrough,
    filter_RGB_colourspaces,
    filter_cmfs,
    filter_illuminants,
    filter_colour_checkers,
    update_settings_collection,
    plot_single_colour_swatch,
    plot_multi_colour_swatches,
    plot_single_function,
    plot_multi_functions,
    plot_image,
)
from .blindness import plot_cvd_simulation_Machado2009
from .colorimetry import (
    plot_single_sd,
    plot_multi_sds,
    plot_single_cmfs,
    plot_multi_cmfs,
    plot_single_illuminant_sd,
    plot_multi_illuminant_sds,
    plot_visible_spectrum,
    plot_single_lightness_function,
    plot_multi_lightness_functions,
    plot_single_luminance_function,
    plot_multi_luminance_functions,
    plot_blackbody_spectral_radiance,
    plot_blackbody_colours,
)
from .characterisation import (
    plot_single_colour_checker,
    plot_multi_colour_checkers,
)
from .diagrams import (
    METHODS_CHROMATICITY_DIAGRAM,
    LABELS_CHROMATICITY_DIAGRAM_DEFAULT,
    lines_spectral_locus,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
)
from .corresponding import (
    plot_corresponding_chromaticities_prediction,
)  # noqa: RUF100
from .graph import plot_automatic_colour_conversion_graph
from .models import (
    colourspace_model_axis_reorder,
    lines_pointer_gamut,
    plot_pointer_gamut,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf,
    plot_multi_cctfs,
    plot_constant_hue_loci,
)
from .notation import (
    plot_single_munsell_value_function,
    plot_multi_munsell_value_functions,
)
from .phenomena import (
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
)
from .quality import (
    plot_single_sd_colour_rendering_index_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_multi_sds_colour_quality_scales_bars,
)
from .section import (
    plot_visible_spectrum_section,
    plot_RGB_colourspace_section,
)
from .temperature import (
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
    plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS,
)
from .tm3018 import plot_single_sd_colour_rendition_report
from .volume import (
    plot_RGB_colourspaces_gamuts,
    plot_RGB_scatter,
)  # noqa: RUF100

__all__ = []
__all__ += datasets.__all__
__all__ += [
    "CONSTANTS_COLOUR_STYLE",
    "CONSTANTS_ARROW_STYLE",
    "colour_style",
    "override_style",
    "XYZ_to_plotting_colourspace",
    "ColourSwatch",
    "colour_cycle",
    "artist",
    "camera",
    "render",
    "label_rectangles",
    "uniform_axes3d",
    "filter_passthrough",
    "filter_RGB_colourspaces",
    "filter_cmfs",
    "filter_illuminants",
    "filter_colour_checkers",
    "update_settings_collection",
    "plot_single_colour_swatch",
    "plot_multi_colour_swatches",
    "plot_single_function",
    "plot_multi_functions",
    "plot_image",
]
__all__ += [
    "plot_cvd_simulation_Machado2009",
]
__all__ += [
    "plot_single_sd",
    "plot_multi_sds",
    "plot_single_cmfs",
    "plot_multi_cmfs",
    "plot_single_illuminant_sd",
    "plot_multi_illuminant_sds",
    "plot_visible_spectrum",
    "plot_single_lightness_function",
    "plot_multi_lightness_functions",
    "plot_single_luminance_function",
    "plot_multi_luminance_functions",
    "plot_blackbody_spectral_radiance",
    "plot_blackbody_colours",
]
__all__ += [
    "plot_single_colour_checker",
    "plot_multi_colour_checkers",
]
__all__ += [
    "METHODS_CHROMATICITY_DIAGRAM",
    "LABELS_CHROMATICITY_DIAGRAM_DEFAULT",
    "lines_spectral_locus",
    "plot_chromaticity_diagram_CIE1931",
    "plot_chromaticity_diagram_CIE1960UCS",
    "plot_chromaticity_diagram_CIE1976UCS",
    "plot_sds_in_chromaticity_diagram_CIE1931",
    "plot_sds_in_chromaticity_diagram_CIE1960UCS",
    "plot_sds_in_chromaticity_diagram_CIE1976UCS",
]
__all__ += [
    "plot_corresponding_chromaticities_prediction",
]
__all__ += [
    "plot_automatic_colour_conversion_graph",
]
__all__ += [
    "colourspace_model_axis_reorder",
    "lines_pointer_gamut",
    "plot_pointer_gamut",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS",
    "plot_single_cctf",
    "plot_multi_cctfs",
    "plot_constant_hue_loci",
]
__all__ += [
    "plot_single_munsell_value_function",
    "plot_multi_munsell_value_functions",
]
__all__ += [
    "plot_single_sd_rayleigh_scattering",
    "plot_the_blue_sky",
]
__all__ += [
    "plot_single_sd_colour_rendering_index_bars",
    "plot_multi_sds_colour_rendering_indexes_bars",
    "plot_single_sd_colour_quality_scale_bars",
    "plot_multi_sds_colour_quality_scales_bars",
]
__all__ += [
    "plot_visible_spectrum_section",
    "plot_RGB_colourspace_section",
]
__all__ += [
    "plot_planckian_locus_in_chromaticity_diagram_CIE1931",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS",
]
__all__ += [
    "plot_single_sd_colour_rendition_report",
]
__all__ += [
    "plot_RGB_colourspaces_gamuts",
    "plot_RGB_scatter",
]
