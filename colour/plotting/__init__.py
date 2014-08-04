from __future__ import absolute_import

from .plots import normalise_RGB, figure_size, aspect, bounding_box, display
from .plots import colour_parameter, colour_parameters_plot, single_colour_plot, multi_colour_plot
from .plots import colour_checker_plot
from .plots import single_spd_plot, multi_spd_plot
from .plots import single_cmfs_plot, multi_cmfs_plot
from .plots import single_illuminant_relative_spd_plot, multi_illuminants_relative_spd_plot
from .plots import visible_spectrum_plot
from .plots import CIE_1931_chromaticity_diagram_plot
from .plots import colourspaces_CIE_1931_chromaticity_diagram_plot, planckian_locus_CIE_1931_chromaticity_diagram_plot
from .plots import CIE_1960_UCS_chromaticity_diagram_plot
from .plots import planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot
from .plots import CIE_1976_UCS_chromaticity_diagram_plot
from .plots import single_munsell_value_function_plot, multi_munsell_value_function_plot
from .plots import single_lightness_function_plot, multi_lightness_function_plot
from .plots import single_transfer_function_plot, multi_transfer_function_plot
from .plots import blackbody_spectral_radiance_plot, blackbody_colours_plot
from .plots import colour_rendering_index_bars_plot

__all__ = ["normalise_RGB", "figure_size", "aspect", "bounding_box", "display"]
__all__ += ["colour_parameter", "colour_parameters_plot", "single_colour_plot", "multi_colour_plot"]
__all__ += ["colour_checker_plot"]
__all__ += ["single_spd_plot", "multi_spd_plot"]
__all__ += ["single_cmfs_plot", "multi_cmfs_plot"]
__all__ += ["single_illuminant_relative_spd_plot", "multi_illuminants_relative_spd_plot"]
__all__ += ["visible_spectrum_plot"]
__all__ += ["CIE_1931_chromaticity_diagram_plot"]
__all__ += ["colourspaces_CIE_1931_chromaticity_diagram_plot", "planckian_locus_CIE_1931_chromaticity_diagram_plot"]
__all__ += ["CIE_1960_UCS_chromaticity_diagram_plot"]
__all__ += ["planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot"]
__all__ += ["CIE_1976_UCS_chromaticity_diagram_plot"]
__all__ += ["single_munsell_value_function_plot", "multi_munsell_value_function_plot"]
__all__ += ["single_lightness_function_plot", "multi_lightness_function_plot"]
__all__ += ["single_transfer_function_plot", "multi_transfer_function_plot"]
__all__ += ["blackbody_spectral_radiance_plot", "blackbody_colours_plot"]
__all__ += ["colour_rendering_index_bars_plot"]

