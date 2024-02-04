#!/usr/bin/env python
"""
Generate Plots
==============
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("AGG")

import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh

import colour
from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (
    MSDS_CMFS_STANDARD_OBSERVER,
    SDS_ILLUMINANTS,
    SDS_LEFS_PHOTOPIC,
    SDS_LEFS_SCOTOPIC,
    SDS_LIGHT_SOURCES,
    SpectralDistribution,
    SpectralShape,
    sd_blackbody,
    sd_mesopic_luminous_efficiency_function,
    sd_to_XYZ,
)
from colour.geometry import primitive_cube
from colour.hints import cast
from colour.io import read_image
from colour.models import (
    RGB_COLOURSPACE_sRGB,
    RGB_to_XYZ,
    XYZ_to_sRGB,
    XYZ_to_xy,
    sRGB_to_XYZ,
)
from colour.plotting import (
    ColourSwatch,
    colour_style,
    plot_automatic_colour_conversion_graph,
    plot_blackbody_colours,
    plot_blackbody_spectral_radiance,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_constant_hue_loci,
    plot_corresponding_chromaticities_prediction,
    plot_cvd_simulation_Machado2009,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_image,
    plot_multi_cctfs,
    plot_multi_cmfs,
    plot_multi_colour_checkers,
    plot_multi_colour_swatches,
    plot_multi_functions,
    plot_multi_illuminant_sds,
    plot_multi_lightness_functions,
    plot_multi_luminance_functions,
    plot_multi_munsell_value_functions,
    plot_multi_sds,
    plot_multi_sds_colour_quality_scales_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
    plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS,
    plot_pointer_gamut,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_colourspace_section,
    plot_RGB_colourspaces_gamuts,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_scatter,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf,
    plot_single_cmfs,
    plot_single_colour_checker,
    plot_single_colour_swatch,
    plot_single_function,
    plot_single_illuminant_sd,
    plot_single_lightness_function,
    plot_single_luminance_function,
    plot_single_munsell_value_function,
    plot_single_sd,
    plot_single_sd_colour_quality_scale_bars,
    plot_single_sd_colour_rendering_index_bars,
    plot_single_sd_colour_rendition_report,
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
    plot_visible_spectrum,
    plot_visible_spectrum_section,
    render,
)
from colour.plotting.diagrams import (
    plot_chromaticity_diagram,
    plot_chromaticity_diagram_colours,
    plot_sds_in_chromaticity_diagram,
    plot_spectral_locus,
)
from colour.plotting.models import (
    plot_ellipses_MacAdam1942_in_chromaticity_diagram,
    plot_RGB_chromaticities_in_chromaticity_diagram,
    plot_RGB_colourspaces_in_chromaticity_diagram,
)
from colour.plotting.quality import plot_colour_quality_bars
from colour.plotting.section import (
    plot_hull_section_colours,
    plot_hull_section_contour,
)
from colour.plotting.temperature import (
    plot_daylight_locus,
    plot_planckian_locus,
    plot_planckian_locus_in_chromaticity_diagram,
)
from colour.quality import colour_quality_scale
from colour.utilities import (  # ; noqa: RUF100
    domain_range_scale,
    filter_warnings,
)

__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "generate_documentation_plots",
]


def generate_documentation_plots(output_directory: str):
    """
    Generate documentation plots.

    Parameters
    ----------
    output_directory
        Output directory.
    """

    filter_warnings()

    colour_style()

    np.random.seed(0)

    # *************************************************************************
    # "README.rst"
    # *************************************************************************
    filename = os.path.join(
        output_directory, "Examples_Colour_Automatic_Conversion_Graph.png"
    )
    plot_automatic_colour_conversion_graph(filename)

    arguments = {
        "tight_layout": True,
        "transparent_background": True,
        "filename": os.path.join(
            output_directory, "Examples_Plotting_Visible_Spectrum.png"
        ),
    }
    plt.close(
        plot_visible_spectrum("CIE 1931 2 Degree Standard Observer", **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Illuminant_F1_SD.png"
    )
    plt.close(plot_single_illuminant_sd("FL1", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Blackbodies.png"
    )
    blackbody_sds = [
        sd_blackbody(i, SpectralShape(1, 10001, 10)) for i in range(1000, 15000, 1000)
    ]
    plt.close(
        plot_multi_sds(
            blackbody_sds,
            y_label="W / (sr m$^2$) / m",
            plot_kwargs={"use_sd_colours": True, "normalise_sd_colours": True},
            legend_location="upper right",
            bounding_box=(0, 1250, 0, 2.5e6),
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Cone_Fundamentals.png"
    )
    plt.close(
        plot_single_cmfs(
            "Stockman & Sharpe 2 Degree Cone Fundamentals",
            y_label="Sensitivity",
            bounding_box=(390, 870, 0, 1.1),
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Luminous_Efficiency.png"
    )
    plt.close(
        plot_multi_sds(
            (
                sd_mesopic_luminous_efficiency_function(0.2),
                SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"],
                SDS_LEFS_SCOTOPIC["CIE 1951 Scotopic Standard Observer"],
            ),
            y_label="Luminous Efficiency",
            legend_location="upper right",
            y_tighten=True,
            margins=(0, 0, 0, 0.1),
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_BabelColor_Average.png"
    )
    plt.close(
        plot_multi_sds(
            SDS_COLOURCHECKERS["BabelColor Average"].values(),
            plot_kwargs={"use_sd_colours": True},
            title="BabelColor Average - Spectral Distributions",
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_ColorChecker_2005.png"
    )
    plt.close(
        plot_single_colour_checker(
            "ColorChecker 2005", text_kwargs={"visible": False}, **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Chromaticities_Prediction.png"
    )
    plt.close(
        plot_corresponding_chromaticities_prediction(2, "Von Kries", **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Examples_Plotting_Chromaticities_CIE_1931_Chromaticity_Diagram.png",
    )
    RGB = np.random.random((32, 32, 3))
    plt.close(
        plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB,
            "ITU-R BT.709",
            colourspaces=["ACEScg", "S-Gamut"],
            show_pointer_gamut=True,
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(output_directory, "Examples_Plotting_CRI.png")
    plt.close(
        plot_single_sd_colour_rendering_index_bars(SDS_ILLUMINANTS["FL2"], **arguments)[
            0
        ]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Colour_Rendition_Report.png"
    )
    plt.close(
        plot_single_sd_colour_rendition_report(SDS_ILLUMINANTS["FL2"], **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Plot_Visible_Spectrum_Section.png"
    )
    plt.close(
        plot_visible_spectrum_section(
            section_colours="RGB", section_opacity=0.15, **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Examples_Plotting_Plot_RGB_Colourspace_Section.png"
    )
    plt.close(
        plot_RGB_colourspace_section(
            "sRGB", section_colours="RGB", section_opacity=0.15, **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Examples_Plotting_CCT_CIE_1960_UCS_Chromaticity_Diagram.png",
    )
    plt.close(
        plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
            ["A", "B", "C"], **arguments
        )[0]
    )

    # *************************************************************************
    # Documentation
    # *************************************************************************
    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_CVD_Simulation_Machado2009.png"
    )
    plt.close(plot_cvd_simulation_Machado2009(RGB, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Colour_Checker.png"
    )
    plt.close(plot_single_colour_checker("ColorChecker 2005", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Colour_Checkers.png"
    )
    plt.close(
        plot_multi_colour_checkers(
            ["ColorChecker 1976", "ColorChecker 2005"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_SD.png"
    )
    data = {
        500: 0.0651,
        520: 0.0705,
        540: 0.0772,
        560: 0.0870,
        580: 0.1128,
        600: 0.1360,
    }
    sd = SpectralDistribution(data, name="Custom")
    plt.close(plot_single_sd(sd, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_SDS.png"
    )
    data_1 = {
        500: 0.004900,
        510: 0.009300,
        520: 0.063270,
        530: 0.165500,
        540: 0.290400,
        550: 0.433450,
        560: 0.594500,
    }
    data_2 = {
        500: 0.323000,
        510: 0.503000,
        520: 0.710000,
        530: 0.862000,
        540: 0.954000,
        550: 0.994950,
        560: 0.995000,
    }
    spd1 = SpectralDistribution(data_1, name="Custom 1")
    spd2 = SpectralDistribution(data_2, name="Custom 2")
    plot_kwargs = [
        {"use_sd_colours": True},
        {"use_sd_colours": True, "linestyle": "dashed"},
    ]
    plt.close(plot_multi_sds([spd1, spd2], plot_kwargs=plot_kwargs, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_CMFS.png"
    )
    plt.close(plot_single_cmfs("CIE 1931 2 Degree Standard Observer", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_CMFS.png"
    )
    cmfs = [
        "CIE 1931 2 Degree Standard Observer",
        "CIE 1964 10 Degree Standard Observer",
    ]
    plt.close(plot_multi_cmfs(cmfs, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Illuminant_SD.png"
    )
    plt.close(plot_single_illuminant_sd("A", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Illuminant_SDS.png"
    )
    plt.close(plot_multi_illuminant_sds(["A", "B", "C"], **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Visible_Spectrum.png"
    )
    plt.close(plot_visible_spectrum(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Lightness_Function.png"
    )
    plt.close(plot_single_lightness_function("CIE 1976", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Lightness_Functions.png"
    )
    plt.close(
        plot_multi_lightness_functions(["CIE 1976", "Wyszecki 1963"], **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Luminance_Function.png"
    )
    plt.close(plot_single_luminance_function("CIE 1976", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Luminance_Functions.png"
    )
    plt.close(
        plot_multi_luminance_functions(["CIE 1976", "Newhall 1943"], **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Blackbody_Spectral_Radiance.png"
    )
    plt.close(
        plot_blackbody_spectral_radiance(3500, blackbody="VY Canis Major", **arguments)[
            0
        ]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Blackbody_Colours.png"
    )
    plt.close(plot_blackbody_colours(SpectralShape(150, 12500, 50), **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Colour_Swatch.png"
    )
    RGB = ColourSwatch((0.45620519, 0.03081071, 0.04091952))
    plt.close(plot_single_colour_swatch(RGB, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Colour_Swatches.png"
    )
    RGB_1 = ColourSwatch((0.45293517, 0.31732158, 0.26414773))
    RGB_2 = ColourSwatch((0.77875824, 0.57726450, 0.50453169))
    plt.close(plot_multi_colour_swatches([RGB_1, RGB_2], **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Function.png"
    )
    plt.close(plot_single_function(lambda x: x ** (1 / 2.2), **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Functions.png"
    )
    functions = {
        "Gamma 2.2": lambda x: x ** (1 / 2.2),
        "Gamma 2.4": lambda x: x ** (1 / 2.4),
        "Gamma 2.6": lambda x: x ** (1 / 2.6),
    }
    plt.close(plot_multi_functions(functions, **arguments)[0])

    arguments["filename"] = os.path.join(output_directory, "Plotting_Plot_Image.png")
    path = os.path.join(
        colour.__path__[0],  # pyright: ignore
        "examples",
        "plotting",
        "resources",
        "Ishihara_Colour_Blindness_Test_Plate_3.png",
    )
    plt.close(plot_image(read_image(str(path)), **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Corresponding_Chromaticities_Prediction.png",
    )
    plt.close(
        plot_corresponding_chromaticities_prediction(1, "Von Kries", **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Spectral_Locus.png"
    )
    plt.close(plot_spectral_locus(spectral_locus_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Chromaticity_Diagram_Colours.png"
    )
    plt.close(plot_chromaticity_diagram_colours(diagram_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Chromaticity_Diagram.png"
    )
    plt.close(plot_chromaticity_diagram(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Chromaticity_Diagram_CIE1931.png"
    )
    plt.close(plot_chromaticity_diagram_CIE1931(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Chromaticity_Diagram_CIE1960UCS.png"
    )
    plt.close(plot_chromaticity_diagram_CIE1960UCS(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Chromaticity_Diagram_CIE1976UCS.png"
    )
    plt.close(plot_chromaticity_diagram_CIE1976UCS(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_SDS_In_Chromaticity_Diagram.png"
    )
    A = SDS_ILLUMINANTS["A"]
    D65 = SDS_ILLUMINANTS["D65"]
    annotate_kwargs = [
        {"xytext": (-25, 15), "arrowprops": {"arrowstyle": "-"}},
        {},
    ]
    plot_kwargs = [
        {
            "illuminant": SDS_ILLUMINANTS["E"],
            "markersize": 15,
            "normalise_sd_colours": True,
            "use_sd_colours": True,
        },
        {"illuminant": SDS_ILLUMINANTS["E"]},
    ]
    plt.close(
        plot_sds_in_chromaticity_diagram(
            [A, D65],
            annotate_kwargs=annotate_kwargs,
            plot_kwargs=plot_kwargs,
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_SDS_In_Chromaticity_Diagram_CIE1931.png",
    )
    plt.close(plot_sds_in_chromaticity_diagram_CIE1931([A, D65], **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_SDS_In_Chromaticity_Diagram_CIE1960UCS.png",
    )
    plt.close(plot_sds_in_chromaticity_diagram_CIE1960UCS([A, D65], **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_SDS_In_Chromaticity_Diagram_CIE1976UCS.png",
    )
    plt.close(plot_sds_in_chromaticity_diagram_CIE1976UCS([A, D65], **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Pointer_Gamut.png"
    )
    plt.close(plot_pointer_gamut(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Colourspaces_In_Chromaticity_Diagram.png",
    )
    plot_kwargs = [
        {"color": "r"},
        {"linestyle": "dashed"},
        {"marker": None},
    ]
    plt.close(
        plot_RGB_colourspaces_in_chromaticity_diagram(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"],
            plot_kwargs=plot_kwargs,
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1931.png",
    )
    plt.close(
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1960UCS.png",
    )
    plt.close(
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1976UCS.png",
    )
    plt.close(
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Chromaticities_In_Chromaticity_Diagram.png",
    )
    RGB = np.random.random((128, 128, 3))
    plt.close(
        plot_RGB_chromaticities_in_chromaticity_diagram(
            RGB, "ITU-R BT.709", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1931.png",
    )
    plt.close(
        plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            RGB, "ITU-R BT.709", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1960UCS.png",
    )
    plt.close(
        plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
            RGB, "ITU-R BT.709", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1976UCS.png",
    )
    plt.close(
        plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
            RGB, "ITU-R BT.709", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram.png",
    )
    plt.close(plot_ellipses_MacAdam1942_in_chromaticity_diagram(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1931.png",
    )
    plt.close(plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1960UCS.png",
    )
    plt.close(
        plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS(**arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1976UCS.png",
    )
    plt.close(
        plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(**arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_CCTF.png"
    )
    plt.close(plot_single_cctf("ITU-R BT.709", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_CCTFs.png"
    )
    plt.close(plot_multi_cctfs(["ITU-R BT.709", "sRGB"], **arguments)[0])

    data = [
        [
            None,
            np.array([0.95010000, 1.00000000, 1.08810000]),
            np.array([0.40920000, 0.28120000, 0.30600000]),
            np.array(
                [
                    [0.02495100, 0.01908600, 0.02032900],
                    [0.10944300, 0.06235900, 0.06788100],
                    [0.27186500, 0.18418700, 0.19565300],
                    [0.48898900, 0.40749400, 0.44854600],
                ]
            ),
            None,
        ],
        [
            None,
            np.array([0.95010000, 1.00000000, 1.08810000]),
            np.array([0.30760000, 0.48280000, 0.42770000]),
            np.array(
                [
                    [0.02108000, 0.02989100, 0.02790400],
                    [0.06194700, 0.11251000, 0.09334400],
                    [0.15255800, 0.28123300, 0.23234900],
                    [0.34157700, 0.56681300, 0.47035300],
                ]
            ),
            None,
        ],
        [
            None,
            np.array([0.95010000, 1.00000000, 1.08810000]),
            np.array([0.39530000, 0.28120000, 0.18450000]),
            np.array(
                [
                    [0.02436400, 0.01908600, 0.01468800],
                    [0.10331200, 0.06235900, 0.02854600],
                    [0.26311900, 0.18418700, 0.12109700],
                    [0.43158700, 0.40749400, 0.39008600],
                ]
            ),
            None,
        ],
        [
            None,
            np.array([0.95010000, 1.00000000, 1.08810000]),
            np.array([0.20510000, 0.18420000, 0.57130000]),
            np.array(
                [
                    [0.03039800, 0.02989100, 0.06123300],
                    [0.08870000, 0.08498400, 0.21843500],
                    [0.18405800, 0.18418700, 0.40111400],
                    [0.32550100, 0.34047200, 0.50296900],
                    [0.53826100, 0.56681300, 0.80010400],
                ]
            ),
            None,
        ],
        [
            None,
            np.array([0.95010000, 1.00000000, 1.08810000]),
            np.array([0.35770000, 0.28120000, 0.11250000]),
            np.array(
                [
                    [0.03678100, 0.02989100, 0.01481100],
                    [0.17127700, 0.11251000, 0.01229900],
                    [0.30080900, 0.28123300, 0.21229800],
                    [0.52976000, 0.40749400, 0.11720000],
                ]
            ),
            None,
        ],
    ]
    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Constant_Hue_Loci.png"
    )
    plt.close(plot_constant_hue_loci(data, "IPT", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_Munsell_Value_Function.png"
    )
    plt.close(plot_single_munsell_value_function("ASTM D1535", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Multi_Munsell_Value_Functions.png"
    )
    plt.close(
        plot_multi_munsell_value_functions(["ASTM D1535", "McCamy 1987"], **arguments)[
            0
        ]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Single_SD_Rayleigh_Scattering.png"
    )
    plt.close(plot_single_sd_rayleigh_scattering(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_The_Blue_Sky.png"
    )
    plt.close(plot_the_blue_sky(**arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Colour_Quality_Bars.png"
    )
    illuminant = SDS_ILLUMINANTS["FL2"]
    light_source = SDS_LIGHT_SOURCES["Kinoton 75P"]
    light_source = light_source.copy().align(SpectralShape(360, 830, 1))
    cqs_i = colour_quality_scale(illuminant, additional_data=True)
    cqs_l = colour_quality_scale(light_source, additional_data=True)
    plt.close(
        plot_colour_quality_bars(
            [cqs_i, cqs_l],  # pyright: ignore
            **arguments,  # pyright: ignore
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Single_SD_Colour_Rendering_Index_Bars.png",
    )
    illuminant = SDS_ILLUMINANTS["FL2"]
    plt.close(plot_single_sd_colour_rendering_index_bars(illuminant, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Multi_SDS_Colour_Rendering_Indexes_Bars.png",
    )
    light_source = SDS_LIGHT_SOURCES["Kinoton 75P"]
    plt.close(
        plot_multi_sds_colour_rendering_indexes_bars(
            [illuminant, light_source], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Single_SD_Colour_Quality_Scale_Bars.png",
    )
    illuminant = SDS_ILLUMINANTS["FL2"]
    plt.close(plot_single_sd_colour_quality_scale_bars(illuminant, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Multi_SDS_Colour_Quality_Scales_Bars.png",
    )
    light_source = SDS_LIGHT_SOURCES["Kinoton 75P"]
    plt.close(
        plot_multi_sds_colour_quality_scales_bars(
            [illuminant, light_source], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Hull_Section_Colours.png"
    )
    vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
    XYZ_vertices = RGB_to_XYZ(vertices["position"] + 0.5, RGB_COLOURSPACE_sRGB)
    hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)
    plt.close(plot_hull_section_colours(hull, section_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Hull_Section_Contour.png"
    )
    plt.close(plot_hull_section_contour(hull, section_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Visible_Spectrum_Section.png"
    )
    plt.close(
        plot_visible_spectrum_section(
            section_colours="RGB", section_opacity=0.15, **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_RGB_Colourspace_Section.png"
    )
    plt.close(
        plot_RGB_colourspace_section(
            "sRGB", section_colours="RGB", section_opacity=0.15, **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Daylight_Locus.png"
    )
    plt.close(plot_daylight_locus(daylight_locus_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_Planckian_Locus.png"
    )
    plt.close(plot_planckian_locus(planckian_locus_colours="RGB", **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Planckian_Locus_In_Chromaticity_Diagram.png",
    )
    annotate_kwargs = [
        {"xytext": (-25, 15), "arrowprops": {"arrowstyle": "-"}},
        {"arrowprops": {"arrowstyle": "-["}},
        {},
    ]
    plot_kwargs = [
        {
            "markersize": 15,
        },
        {"color": "r"},
        {},
    ]
    plt.close(
        plot_planckian_locus_in_chromaticity_diagram(
            ["A", "B", "C"],
            annotate_kwargs=annotate_kwargs,
            plot_kwargs=plot_kwargs,
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1931.png",
    )
    plt.close(
        plot_planckian_locus_in_chromaticity_diagram_CIE1931(
            ["A", "B", "C"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1960UCS.png",
    )
    plt.close(
        plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
            ["A", "B", "C"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1976UCS.png",
    )
    plt.close(
        plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS(
            ["A", "B", "C"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Single_SD_Colour_Rendition_Report_Full.png",
    )
    plt.close(
        plot_single_sd_colour_rendition_report(SDS_ILLUMINANTS["FL2"], **arguments)[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Single_SD_Colour_Rendition_Report_Intermediate.png",
    )
    plt.close(
        plot_single_sd_colour_rendition_report(
            SDS_ILLUMINANTS["FL2"], "Intermediate", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory,
        "Plotting_Plot_Single_SD_Colour_Rendition_Report_Simple.png",
    )
    plt.close(
        plot_single_sd_colour_rendition_report(
            SDS_ILLUMINANTS["FL2"], "Simple", **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_RGB_Colourspaces_Gamuts.png"
    )
    plt.close(
        plot_RGB_colourspaces_gamuts(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_RGB_Colourspaces_Gamuts.png"
    )
    plt.close(
        plot_RGB_colourspaces_gamuts(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"], **arguments
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Plotting_Plot_RGB_Scatter.png"
    )
    plt.close(plot_RGB_scatter(RGB, "ITU-R BT.709", **arguments)[0])

    filename = os.path.join(
        output_directory, "Plotting_Plot_Colour_Automatic_Conversion_Graph.png"
    )
    plot_automatic_colour_conversion_graph(filename)

    # *************************************************************************
    # "tutorial.rst"
    # *************************************************************************
    arguments["filename"] = os.path.join(
        output_directory, "Tutorial_Visible_Spectrum.png"
    )
    plt.close(plot_visible_spectrum(**arguments)[0])

    arguments["filename"] = os.path.join(output_directory, "Tutorial_Sample_SD.png")
    sample_sd_data = {
        380: 0.048,
        385: 0.051,
        390: 0.055,
        395: 0.060,
        400: 0.065,
        405: 0.068,
        410: 0.068,
        415: 0.067,
        420: 0.064,
        425: 0.062,
        430: 0.059,
        435: 0.057,
        440: 0.055,
        445: 0.054,
        450: 0.053,
        455: 0.053,
        460: 0.052,
        465: 0.052,
        470: 0.052,
        475: 0.053,
        480: 0.054,
        485: 0.055,
        490: 0.057,
        495: 0.059,
        500: 0.061,
        505: 0.062,
        510: 0.065,
        515: 0.067,
        520: 0.070,
        525: 0.072,
        530: 0.074,
        535: 0.075,
        540: 0.076,
        545: 0.078,
        550: 0.079,
        555: 0.082,
        560: 0.087,
        565: 0.092,
        570: 0.100,
        575: 0.107,
        580: 0.115,
        585: 0.122,
        590: 0.129,
        595: 0.134,
        600: 0.138,
        605: 0.142,
        610: 0.146,
        615: 0.150,
        620: 0.154,
        625: 0.158,
        630: 0.163,
        635: 0.167,
        640: 0.173,
        645: 0.180,
        650: 0.188,
        655: 0.196,
        660: 0.204,
        665: 0.213,
        670: 0.222,
        675: 0.231,
        680: 0.242,
        685: 0.251,
        690: 0.261,
        695: 0.271,
        700: 0.282,
        705: 0.294,
        710: 0.305,
        715: 0.318,
        720: 0.334,
        725: 0.354,
        730: 0.372,
        735: 0.392,
        740: 0.409,
        745: 0.420,
        750: 0.436,
        755: 0.450,
        760: 0.462,
        765: 0.465,
        770: 0.448,
        775: 0.432,
        780: 0.421,
    }

    sd = SpectralDistribution(sample_sd_data, name="Sample")
    plt.close(plot_single_sd(sd, **arguments)[0])

    arguments["filename"] = os.path.join(
        output_directory, "Tutorial_SD_Interpolation.png"
    )
    sd_copy = sd.copy()
    sd_copy.interpolate(SpectralShape(400, 770, 1))
    plt.close(
        plot_multi_sds([sd, sd_copy], bounding_box=[730, 780, 0.25, 0.5], **arguments)[
            0
        ]
    )

    arguments["filename"] = os.path.join(output_directory, "Tutorial_Sample_Swatch.png")
    sd = SpectralDistribution(sample_sd_data)
    cmfs = MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"]
    illuminant = SDS_ILLUMINANTS["D65"]
    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd, cmfs, illuminant)
        RGB = XYZ_to_sRGB(XYZ)
    plt.close(
        plot_single_colour_swatch(
            ColourSwatch(RGB, "Sample"),
            text_kwargs={"size": "x-large"},
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(output_directory, "Tutorial_Neutral5.png")
    patch_name = "neutral 5 (.70 D)"
    patch_sd = SDS_COLOURCHECKERS["ColorChecker N Ohta"][patch_name]
    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(patch_sd, cmfs, illuminant)
        RGB = XYZ_to_sRGB(XYZ)
    plt.close(
        plot_single_colour_swatch(
            ColourSwatch(RGB, patch_name.title()),
            text_kwargs={"size": "x-large"},
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Tutorial_Colour_Checker.png"
    )
    plt.close(
        plot_single_colour_checker(
            colour_checker="ColorChecker 2005",
            text_kwargs={"visible": False},
            **arguments,
        )[0]
    )

    arguments["filename"] = os.path.join(
        output_directory, "Tutorial_CIE_1931_Chromaticity_Diagram.png"
    )
    xy = cast(tuple[float, float], XYZ_to_xy(XYZ))
    plot_chromaticity_diagram_CIE1931(standalone=False)
    plt.plot(xy[0], xy[1], "o-", color="white")
    # Annotating the plot.
    plt.annotate(
        patch_sd.name.title(),
        xy=xy,
        xytext=(-50, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3, rad=-0.2"},
    )
    plt.close(
        render(
            standalone=True,
            limits=(-0.1, 0.9, -0.1, 0.9),
            x_tighten=True,
            y_tighten=True,
            **arguments,
        )[0]
    )

    # *************************************************************************
    # "basics.rst"
    # *************************************************************************
    arguments["filename"] = os.path.join(
        output_directory, "Basics_Logo_Small_001_CIE_XYZ.png"
    )
    RGB = read_image(os.path.join(output_directory, "Logo_Small_001.png"))[..., 0:3]
    XYZ = sRGB_to_XYZ(RGB)
    plt.close(plot_image(XYZ, text_kwargs={"text": "sRGB to XYZ"}, **arguments)[0])


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    generate_documentation_plots(os.path.join("..", "docs", "_static"))
