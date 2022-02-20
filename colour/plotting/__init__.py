from colour.utilities import is_matplotlib_installed

is_matplotlib_installed(raise_exception=True)

from .datasets import *  # noqa
from . import datasets  # noqa
from .common import (  # noqa
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
from .blindness import plot_cvd_simulation_Machado2009  # noqa
from .colorimetry import (  # noqa
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
from .characterisation import (  # noqa
    plot_single_colour_checker,
    plot_multi_colour_checkers,
)
from .diagrams import (  # noqa
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
)
from .corresponding import plot_corresponding_chromaticities_prediction  # noqa
from .graph import plot_automatic_colour_conversion_graph  # noqa
from .models import (  # noqa
    colourspace_model_axis_reorder,
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
from .notation import (  # noqa
    plot_single_munsell_value_function,
    plot_multi_munsell_value_functions,
)
from .phenomena import (  # noqa
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
)
from .quality import (  # noqa
    plot_single_sd_colour_rendering_index_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_multi_sds_colour_quality_scales_bars,
)
from .section import (  # noqa
    plot_visible_spectrum_section,
    plot_RGB_colourspace_section,
)
from .temperature import (  # noqa
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
)
from .tm3018 import plot_single_sd_colour_rendition_report  # noqa
from .volume import plot_RGB_colourspaces_gamuts, plot_RGB_scatter  # noqa

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
]
__all__ += [
    "plot_single_sd_colour_rendition_report",
]
__all__ += [
    "plot_RGB_colourspaces_gamuts",
    "plot_RGB_scatter",
]
