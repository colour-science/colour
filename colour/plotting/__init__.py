from colour.utilities import is_matplotlib_installed

if not is_matplotlib_installed():  # pragma: no cover
    import sys
    from unittest.mock import MagicMock

    from colour.utilities import usage_warning

    try:
        is_matplotlib_installed(raise_exception=True)
    except ImportError as error:
        usage_warning(str(error))

    for module in (
        "cycler",
        "matplotlib",
        "matplotlib.axes",
        "matplotlib.cm",
        "matplotlib.collections",
        "matplotlib.colors",
        "matplotlib.figure",
        "matplotlib.font_manager",
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

from . import datasets
from .datasets import *  # noqa: F403

# isort: split

from .common import (
    CONSTANTS_ARROW_STYLE,
    CONSTANTS_COLOUR_STYLE,
    ColourSwatch,
    XYZ_to_plotting_colourspace,
    artist,
    camera,
    colour_cycle,
    colour_style,
    filter_cmfs,
    filter_colour_checkers,
    filter_illuminants,
    filter_passthrough,
    filter_RGB_colourspaces,
    font_scaling,
    label_rectangles,
    override_style,
    plot_image,
    plot_multi_colour_swatches,
    plot_multi_functions,
    plot_ray,
    plot_single_colour_swatch,
    plot_single_function,
    render,
    uniform_axes3d,
    update_settings_collection,
)

# isort: split

from .blindness import plot_cvd_simulation_Machado2009
from .characterisation import (
    plot_multi_colour_checkers,
    plot_single_colour_checker,
)
from .colorimetry import (
    plot_blackbody_colours,
    plot_blackbody_spectral_radiance,
    plot_multi_cmfs,
    plot_multi_illuminant_sds,
    plot_multi_lightness_functions,
    plot_multi_luminance_functions,
    plot_multi_sds,
    plot_single_cmfs,
    plot_single_illuminant_sd,
    plot_single_lightness_function,
    plot_single_luminance_function,
    plot_single_sd,
    plot_visible_spectrum,
)
from .diagrams import (
    LABELS_CHROMATICITY_DIAGRAM_DEFAULT,
    METHODS_CHROMATICITY_DIAGRAM,
    lines_spectral_locus,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
)

# isort: split

from .corresponding import (  # noqa: RUF100
    plot_corresponding_chromaticities_prediction,
)
from .graph import plot_automatic_colour_conversion_graph
from .models import (
    colourspace_model_axis_reorder,
    lines_pointer_gamut,
    plot_constant_hue_loci,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_multi_cctfs,
    plot_pointer_gamut,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf,
)
from .notation import (
    plot_multi_munsell_value_functions,
    plot_single_munsell_value_function,
)
from .phenomena import (
    plot_multi_layer_stack,
    plot_multi_layer_thin_film,
    plot_single_layer_thin_film,
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
    plot_thin_film_comparison,
    plot_thin_film_iridescence,
    plot_thin_film_reflectance_map,
    plot_thin_film_spectrum,
)
from .quality import (
    plot_multi_sds_colour_quality_scales_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_single_sd_colour_rendering_index_bars,
)
from .section import (
    plot_RGB_colourspace_section,
    plot_visible_spectrum_section,
)
from .temperature import (
    LABELS_PLANCKIAN_LOCUS_DEFAULT,
    lines_daylight_locus,
    lines_planckian_locus,
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
    plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS,
)
from .tm3018 import plot_single_sd_colour_rendition_report
from .volume import (  # noqa: RUF100
    plot_RGB_colourspaces_gamuts,
    plot_RGB_scatter,
)

__all__ = datasets.__all__
__all__ += [
    "CONSTANTS_ARROW_STYLE",
    "CONSTANTS_COLOUR_STYLE",
    "ColourSwatch",
    "XYZ_to_plotting_colourspace",
    "artist",
    "camera",
    "colour_cycle",
    "colour_style",
    "filter_cmfs",
    "filter_colour_checkers",
    "filter_illuminants",
    "filter_passthrough",
    "filter_RGB_colourspaces",
    "font_scaling",
    "label_rectangles",
    "override_style",
    "plot_image",
    "plot_multi_colour_swatches",
    "plot_multi_functions",
    "plot_ray",
    "plot_single_colour_swatch",
    "plot_single_function",
    "render",
    "uniform_axes3d",
    "update_settings_collection",
]
__all__ += [
    "plot_cvd_simulation_Machado2009",
]
__all__ += [
    "plot_multi_colour_checkers",
    "plot_single_colour_checker",
]
__all__ += [
    "plot_blackbody_colours",
    "plot_blackbody_spectral_radiance",
    "plot_multi_cmfs",
    "plot_multi_illuminant_sds",
    "plot_multi_lightness_functions",
    "plot_multi_luminance_functions",
    "plot_multi_sds",
    "plot_single_cmfs",
    "plot_single_illuminant_sd",
    "plot_single_lightness_function",
    "plot_single_luminance_function",
    "plot_single_sd",
    "plot_visible_spectrum",
]
__all__ += [
    "LABELS_CHROMATICITY_DIAGRAM_DEFAULT",
    "METHODS_CHROMATICITY_DIAGRAM",
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
    "plot_constant_hue_loci",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS",
    "plot_multi_cctfs",
    "plot_pointer_gamut",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS",
    "plot_single_cctf",
]
__all__ += [
    "plot_multi_munsell_value_functions",
    "plot_single_munsell_value_function",
]
__all__ += [
    "plot_single_sd_rayleigh_scattering",
    "plot_the_blue_sky",
    "plot_single_layer_thin_film",
    "plot_multi_layer_thin_film",
    "plot_thin_film_comparison",
    "plot_thin_film_spectrum",
    "plot_thin_film_iridescence",
    "plot_thin_film_reflectance_map",
    "plot_multi_layer_stack",
]
__all__ += [
    "plot_multi_sds_colour_quality_scales_bars",
    "plot_multi_sds_colour_rendering_indexes_bars",
    "plot_single_sd_colour_quality_scale_bars",
    "plot_single_sd_colour_rendering_index_bars",
]
__all__ += [
    "plot_RGB_colourspace_section",
    "plot_visible_spectrum_section",
]
__all__ += [
    "LABELS_PLANCKIAN_LOCUS_DEFAULT",
    "lines_daylight_locus",
    "lines_planckian_locus",
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
