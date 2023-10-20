"""
Colour Models Plotting
======================

Defines the colour models plotting objects:

-   :func:`colour.plotting.lines_pointer_gamut`
-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.plot_single_cctf`
-   :func:`colour.plotting.plot_multi_cctfs`
-   :func:`colour.plotting.plot_constant_hue_loci`

References
----------
-   :cite:`Ebner1998` : Ebner, F., & Fairchild, M. D. (1998). Finding constant
    hue surfaces in color space. In G. B. Beretta & R. Eschbach (Eds.), Proc.
    SPIE 3300, Color Imaging: Device-Independent Color, Color Hardcopy, and
    Graphic Arts III, (2 January 1998) (pp. 107-117). doi:10.1117/12.298269
-   :cite:`Hung1995` : Hung, P.-C., & Berns, R. S. (1995). Determination of
    constant Hue Loci for a CRT gamut and their predictions using color
    appearance spaces. Color Research & Application, 20(5), 285-295.
    doi:10.1002/col.5080200506
-   :cite:`Mansencal2019` : Mansencal, T. (2019). Colour - Datasets.
    doi:10.5281/zenodo.3362520
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.path import Path

from colour.adaptation import chromatic_adaptation_VonKries
from colour.algebra import normalise_maximum
from colour.colorimetry import MultiSpectralDistributions
from colour.constants import DEFAULT_FLOAT_DTYPE, EPSILON
from colour.geometry import (
    ellipse_coefficients_canonical_form,
    ellipse_fitting,
    point_at_angle_on_ellipse,
)
from colour.graph import convert
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    List,
    Literal,
    LiteralColourspaceModel,
    LiteralRGBColourspace,
    NDArrayFloat,
    Sequence,
    Tuple,
    cast,
)
from colour.models import (
    CCS_ILLUMINANT_POINTER_GAMUT,
    CCS_POINTER_GAMUT_BOUNDARY,
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    COLOURSPACE_MODELS_AXIS_LABELS,
    COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE,
    DATA_MACADAM_1942_ELLIPSES,
    DATA_POINTER_GAMUT_VOLUME,
    Lab_to_XYZ,
    LCHab_to_Lab,
    RGB_Colourspace,
    RGB_to_RGB,
    RGB_to_XYZ,
    XYZ_to_RGB,
    XYZ_to_xy,
    xy_to_XYZ,
)
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    METHODS_CHROMATICITY_DIAGRAM,
    XYZ_to_plotting_colourspace,
    artist,
    colour_cycle,
    colour_style,
    filter_cmfs,
    filter_passthrough,
    filter_RGB_colourspaces,
    override_style,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_multi_functions,
    render,
    update_settings_collection,
)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import (
    CanonicalMapping,
    as_array,
    as_float_array,
    as_int_array,
    domain_range_scale,
    first_item,
    optional,
    tsplit,
    validate_method,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "COLOURSPACE_MODELS_AXIS_ORDER",
    "colourspace_model_axis_reorder",
    "lines_pointer_gamut",
    "plot_pointer_gamut",
    "plot_RGB_colourspaces_in_chromaticity_diagram",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS",
    "plot_RGB_chromaticities_in_chromaticity_diagram",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS",
    "plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS",
    "ellipses_MacAdam1942",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS",
    "plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS",
    "plot_single_cctf",
    "plot_multi_cctfs",
    "plot_constant_hue_loci",
]

COLOURSPACE_MODELS_AXIS_ORDER: CanonicalMapping = CanonicalMapping(
    {
        "CAM02LCD": (1, 2, 0),
        "CAM02SCD": (1, 2, 0),
        "CAM02UCS": (1, 2, 0),
        "CAM16LCD": (1, 2, 0),
        "CAM16SCD": (1, 2, 0),
        "CAM16UCS": (1, 2, 0),
        "CIE XYZ": (0, 1, 2),
        "CIE xyY": (0, 1, 2),
        "CIE Lab": (1, 2, 0),
        "CIE LCHab": (1, 2, 0),
        "CIE Luv": (1, 2, 0),
        "CIE LCHuv": (1, 2, 0),
        "CIE UCS": (0, 1, 2),
        "CIE UVW": (1, 2, 0),
        "DIN99": (1, 2, 0),
        "Hunter Lab": (1, 2, 0),
        "Hunter Rdab": (1, 2, 0),
        "ICaCb": (1, 2, 0),
        "ICtCp": (1, 2, 0),
        "IPT": (1, 2, 0),
        "IPT Ragoo 2021": (1, 2, 0),
        "IgPgTg": (1, 2, 0),
        "Jzazbz": (1, 2, 0),
        "OSA UCS": (1, 2, 0),
        "Oklab": (1, 2, 0),
        "hdr-CIELAB": (1, 2, 0),
        "hdr-IPT": (1, 2, 0),
        "Yrg": (1, 2, 0),
    }
)
"""Colourspace models axis order."""


def colourspace_model_axis_reorder(
    a: ArrayLike,
    model: LiteralColourspaceModel | str,
    direction: Literal["Forward", "Inverse"] | str = "Forward",
) -> NDArrayFloat:
    """
    Reorder the axes of given colourspace model :math:`a` array according to
    the most common volume plotting axes order.

    Parameters
    ----------
    a
        Colourspace model :math:`a` array.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    direction
        Reordering direction.

    Returns
    -------
    :class:`numpy.ndarray`
        Reordered colourspace model :math:`a` array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> colourspace_model_axis_reorder(a, "CIE Lab")
    array([ 1.,  2.,  0.])
    >>> colourspace_model_axis_reorder(a, "IPT")
    array([ 1.,  2.,  0.])
    >>> colourspace_model_axis_reorder(a, "OSA UCS")
    array([ 1.,  2.,  0.])
    >>> b = np.array([1, 2, 0])
    >>> colourspace_model_axis_reorder(b, "OSA UCS", "Inverse")
    array([ 0.,  1.,  2.])
    """

    a = as_float_array(a)

    model = validate_method(
        model,
        tuple(COLOURSPACE_MODELS_AXIS_ORDER),
        '"{0}" model is invalid, it must be one of {1}!',
    )

    direction = validate_method(
        direction,
        ("Forward", "Inverse"),
        '"{0}" direction is invalid, it must be one of {1}!',
    )

    order = COLOURSPACE_MODELS_AXIS_ORDER.get(model, (0, 1, 2))

    if direction == "forward":
        indexes = (order[0], order[1], order[2])
    else:
        indexes = (order.index(0), order.index(1), order.index(2))

    return a[..., indexes]


def lines_pointer_gamut(
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931"
):
    """
    Return the *Pointer's Gamut* line vertices, i.e. positions, normals and
    colours, according to given method.

    Parameters
    ----------
    method
        *Chromaticity Diagram* method.

    Returns
    -------
    :class:`tuple`
        Tuple of *Pointer's Gamut* boundary and volume vertices.

    Examples
    --------
    >>> lines = lines_pointer_gamut()
    >>> len(lines)
    2
    >>> lines[0].dtype
    dtype([('position', '<f8', (2,)), ('normal', '<f8', (2,)), \
('colour', '<f8', (3,))])
    >>> lines[1].dtype
    dtype([('position', '<f8', (2,)), ('normal', '<f8', (2,)), \
('colour', '<f8', (3,))])
    """

    method = validate_method(
        method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS")
    )

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    XYZ_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["XYZ_to_ij"]
    ij_to_XYZ = METHODS_CHROMATICITY_DIAGRAM[method]["ij_to_XYZ"]

    XYZ = xy_to_XYZ(CCS_POINTER_GAMUT_BOUNDARY)
    XYZ = chromatic_adaptation_VonKries(
        XYZ, xy_to_XYZ(CCS_ILLUMINANT_POINTER_GAMUT), xy_to_XYZ(illuminant)
    )
    ij_b = XYZ_to_ij(XYZ)
    ij_b = np.vstack([ij_b, ij_b[0]])
    colours_b = normalise_maximum(
        XYZ_to_plotting_colourspace(ij_to_XYZ(ij_b, illuminant), illuminant),
        axis=-1,
    )

    lines_b = zeros(
        ij_b.shape[0],
        [
            ("position", DEFAULT_FLOAT_DTYPE, 2),
            ("normal", DEFAULT_FLOAT_DTYPE, 2),
            ("colour", DEFAULT_FLOAT_DTYPE, 3),
        ],  # pyright: ignore
    )

    lines_b["position"] = ij_b
    lines_b["colour"] = colours_b

    XYZ = Lab_to_XYZ(
        LCHab_to_Lab(DATA_POINTER_GAMUT_VOLUME), CCS_ILLUMINANT_POINTER_GAMUT
    )
    XYZ = chromatic_adaptation_VonKries(
        XYZ, xy_to_XYZ(CCS_ILLUMINANT_POINTER_GAMUT), xy_to_XYZ(illuminant)
    )
    ij_v = XYZ_to_ij(XYZ)

    colours_v = normalise_maximum(
        XYZ_to_plotting_colourspace(ij_to_XYZ(ij_v, illuminant), illuminant),
        axis=-1,
    )

    lines_v = zeros(
        ij_v.shape[0],
        [
            ("position", DEFAULT_FLOAT_DTYPE, 2),
            ("normal", DEFAULT_FLOAT_DTYPE, 2),
            ("colour", DEFAULT_FLOAT_DTYPE, 3),
        ],  # pyright: ignore
    )

    lines_v["position"] = ij_v
    lines_v["colour"] = colours_v

    return lines_b, lines_v


@override_style()
def plot_pointer_gamut(
    pointer_gamut_colours: ArrayLike | str | None = None,
    pointer_gamut_opacity: float = 1,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot *Pointer's Gamut* according to given method.

    Parameters
    ----------
    pointer_gamut_colours
       Colours of the *Pointer's Gamut*.
    pointer_gamut_opacity
       Opacity of the *Pointer's Gamut*.
    method
        Plotting method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_pointer_gamut(pointer_gamut_colours="RGB")  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Pointer_Gamut.png
        :align: center
        :alt: plot_pointer_gamut
    """

    method = validate_method(
        method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS")
    )

    pointer_gamut_colours = optional(
        pointer_gamut_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    use_RGB_colours = str(pointer_gamut_colours).upper() == "RGB"

    pointer_gamut_opacity = optional(
        pointer_gamut_opacity, CONSTANTS_COLOUR_STYLE.opacity.high
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    lines_b, lines_v = lines_pointer_gamut(method)

    axes.add_collection(
        LineCollection(
            np.concatenate(
                [lines_b["position"][:-1], lines_b["position"][1:]],
                axis=1,  # pyright: ignore
            ).reshape([-1, 2, 2]),
            colors=lines_b["colour"]
            if use_RGB_colours
            else pointer_gamut_colours,
            alpha=pointer_gamut_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
        )
    )

    scatter_settings = {
        "alpha": pointer_gamut_opacity / 2,
        "c": lines_v["colour"] if use_RGB_colours else pointer_gamut_colours,
        "marker": "+",
        "zorder": CONSTANTS_COLOUR_STYLE.zorder.foreground_scatter,
    }
    axes.scatter(
        lines_v["position"][..., 0],
        lines_v["position"][..., 1],
        **scatter_settings,
    )

    settings.update({"axes": axes})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram(
    colourspaces: RGB_Colourspace
    | LiteralRGBColourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str],
    cmfs: MultiSpectralDistributions
    | str
    | Sequence[
        MultiSpectralDistributions | str
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable: Callable = plot_chromaticity_diagram,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    show_whitepoints: bool = True,
    show_pointer_gamut: bool = False,
    chromatically_adapt: bool = False,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspaces in the *Chromaticity Diagram* according
    to given method.

    Parameters
    ----------
    colourspaces
        *RGB* colourspaces to plot. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.common.filter_cmfs` definition.
    chromaticity_diagram_callable
        Callable responsible for drawing the *Chromaticity Diagram*.
    method
        *Chromaticity Diagram* method.
    show_whitepoints
        Whether to display the *RGB* colourspaces whitepoints.
    show_pointer_gamut
        Whether to display the *Pointer's Gamut*.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted *RGB* colourspaces.
        ``plot_kwargs`` can be either a single dictionary applied to all the
        plotted *RGB* colourspaces with the same settings or a sequence of
        dictionaries with different settings for each plotted *RGB*
        colourspace.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.plot_pointer_gamut`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_kwargs = [
    ...     {"color": "r"},
    ...     {"linestyle": "dashed"},
    ...     {"marker": None},
    ... ]
    >>> plot_RGB_colourspaces_in_chromaticity_diagram(
    ...     ["ITU-R BT.709", "ACEScg", "S-Gamut"], plot_kwargs=plot_kwargs
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram
    """

    method = validate_method(
        method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS")
    )

    colourspaces = cast(
        List[RGB_Colourspace],
        list(filter_RGB_colourspaces(colourspaces).values()),
    )  # pyright: ignore

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    title = (
        f"{', '.join([colourspace.name for colourspace in colourspaces])}\n"
        f"{cmfs.name} - {method.upper()} Chromaticity Diagram"
    )

    settings = {"axes": axes, "title": title, "method": method}
    settings.update(kwargs)
    settings["show"] = False

    chromaticity_diagram_callable(**settings)

    if show_pointer_gamut:
        settings = {"axes": axes, "method": method}
        settings.update(kwargs)
        settings["show"] = False

        plot_pointer_gamut(**settings)

    xy_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["xy_to_ij"]

    if method == "cie 1931":
        x_limit_min, x_limit_max = [-0.1], [0.9]
        y_limit_min, y_limit_max = [-0.1], [0.9]

    elif method == "cie 1960 ucs":
        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.2], [0.6]

    elif method == "cie 1976 ucs":
        x_limit_min, x_limit_max = [-0.1], [0.7]
        y_limit_min, y_limit_max = [-0.1], [0.7]

    settings = {"colour_cycle_count": len(colourspaces)}
    settings.update(kwargs)

    cycle = colour_cycle(**settings)

    plotting_colourspace = CONSTANTS_COLOUR_STYLE.colour.colourspace

    plot_settings_collection = [
        {
            "label": f"{colourspace.name}",
            "marker": "o",
            "color": next(cycle)[:3],
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
        }
        for colourspace in colourspaces
    ]

    if plot_kwargs is not None:
        update_settings_collection(
            plot_settings_collection, plot_kwargs, len(colourspaces)
        )

    for i, colourspace in enumerate(colourspaces):
        plot_settings = plot_settings_collection[i]

        if chromatically_adapt and not np.array_equal(
            colourspace.whitepoint, plotting_colourspace.whitepoint
        ):
            colourspace = colourspace.chromatically_adapt(  # noqa: PLW2901
                plotting_colourspace.whitepoint,
                plotting_colourspace.whitepoint_name,
            )

        # RGB colourspaces such as *ACES2065-1* have primaries with
        # chromaticity coordinates set to 0 thus we prevent nan from being
        # yield by zero division in later colour transformations.
        P = np.where(
            colourspace.primaries == 0,
            EPSILON,
            colourspace.primaries,
        )
        P = xy_to_ij(P)
        W = xy_to_ij(colourspace.whitepoint)

        P_p = np.vstack([P, P[0]])
        axes.plot(P_p[..., 0], P_p[..., 1], **plot_settings)

        if show_whitepoints:
            plot_settings["marker"] = "o"
            plot_settings.pop("label")

            W_p = np.vstack([W, W])
            axes.plot(W_p[..., 0], W_p[..., 1], **plot_settings)

        x_limit_min.append(cast(float, np.amin(P[..., 0]) - 0.1))
        y_limit_min.append(cast(float, np.amin(P[..., 1]) - 0.1))
        x_limit_max.append(cast(float, np.amax(P[..., 0]) + 0.1))
        y_limit_max.append(cast(float, np.amax(P[..., 1]) + 0.1))

    bounding_box = (
        min(x_limit_min),
        max(x_limit_max),
        min(y_limit_min),
        max(y_limit_max),
    )

    settings.update(
        {
            "show": True,
            "legend": True,
            "bounding_box": bounding_box,
        }
    )
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    colourspaces: RGB_Colourspace
    | LiteralRGBColourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str],
    cmfs: MultiSpectralDistributions
    | str
    | Sequence[
        MultiSpectralDistributions | str
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_chromaticity_diagram_CIE1931
    ),
    show_whitepoints: bool = True,
    show_pointer_gamut: bool = False,
    chromatically_adapt: bool = False,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspaces in the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces
        *RGB* colourspaces to plot. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.common.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1931
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    show_whitepoints
        Whether to display the *RGB* colourspaces whitepoints.
    show_pointer_gamut
        Whether to display the *Pointer's Gamut*.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted *RGB* colourspaces.
        ``plot_kwargs`` can be either a single dictionary applied to all the
        plotted *RGB* colourspaces with the same settings or a sequence of
        dictionaries with different settings for each plotted *RGB*
        colourspace.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.plot_pointer_gamut`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    ...     ["ITU-R BT.709", "ACEScg", "S-Gamut"]
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces,
        cmfs,
        chromaticity_diagram_callable_CIE1931,
        show_whitepoints=show_whitepoints,
        show_pointer_gamut=show_pointer_gamut,
        chromatically_adapt=chromatically_adapt,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
    colourspaces: RGB_Colourspace
    | LiteralRGBColourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str],
    cmfs: MultiSpectralDistributions
    | str
    | Sequence[
        MultiSpectralDistributions | str
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS
    ),
    show_whitepoints: bool = True,
    show_pointer_gamut: bool = False,
    chromatically_adapt: bool = False,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspaces in the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces
        *RGB* colourspaces to plot. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.common.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1960UCS
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    show_whitepoints
        Whether to display the *RGB* colourspaces whitepoints.
    show_pointer_gamut
        Whether to display the *Pointer's Gamut*.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted *RGB* colourspaces.
        ``plot_kwargs`` can be either a single dictionary applied to all the
        plotted *RGB* colourspaces with the same settings or a sequence of
        dictionaries with different settings for each plotted *RGB*
        colourspace.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.plot_pointer_gamut`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(
    ...     ["ITU-R BT.709", "ACEScg", "S-Gamut"]
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces,
        cmfs,
        chromaticity_diagram_callable_CIE1960UCS,
        show_whitepoints=show_whitepoints,
        show_pointer_gamut=show_pointer_gamut,
        chromatically_adapt=chromatically_adapt,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
    colourspaces: RGB_Colourspace
    | LiteralRGBColourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str],
    cmfs: MultiSpectralDistributions
    | str
    | Sequence[
        MultiSpectralDistributions | str
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1976UCS: Callable = (
        plot_chromaticity_diagram_CIE1976UCS
    ),
    show_whitepoints: bool = True,
    show_pointer_gamut: bool = False,
    chromatically_adapt: bool = False,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspaces in the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    colourspaces
        *RGB* colourspaces to plot. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.common.filter_cmfs` definition.
    chromaticity_diagram_callable_CIE1976UCS
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    show_whitepoints
        Whether to display the *RGB* colourspaces whitepoints.
    show_pointer_gamut
        Whether to display the *Pointer's Gamut*.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted *RGB* colourspaces.
        ``plot_kwargs`` can be either a single dictionary applied to all the
        plotted *RGB* colourspaces with the same settings or a sequence of
        dictionaries with different settings for each plotted *RGB*
        colourspace.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.plot_pointer_gamut`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
    ...     ["ITU-R BT.709", "ACEScg", "S-Gamut"]
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Colourspaces_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_RGB_colourspaces_in_chromaticity_diagram(
        colourspaces,
        cmfs,
        chromaticity_diagram_callable_CIE1976UCS,
        show_whitepoints=show_whitepoints,
        show_pointer_gamut=show_pointer_gamut,
        chromatically_adapt=chromatically_adapt,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram(
    RGB: ArrayLike,
    colourspace: RGB_Colourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str] = "sRGB",
    chromaticity_diagram_callable: Callable = (
        plot_RGB_colourspaces_in_chromaticity_diagram
    ),
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    scatter_kwargs: dict | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspace array in the *Chromaticity Diagram* according
    to given method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    chromaticity_diagram_callable
        Callable responsible for drawing the *Chromaticity Diagram*.
    method
        *Chromaticity Diagram* method.
    scatter_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.scatter` definition.
        The following special keyword arguments can also be used:

        -   ``c`` : If ``c`` is set to *RGB*, the scatter will use the colours
            as given by the ``RGB`` argument.
        -   ``apply_cctf_encoding`` : If ``apply_cctf_encoding`` is set to
            *False*, the encoding colour component transfer function /
            opto-electronic transfer function is not applied when encoding the
            samples to the plotting space.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram(RGB, "ITU-R BT.709")
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram
    """

    RGB = np.reshape(as_float_array(RGB), (-1, 3))
    method = validate_method(
        method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS")
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    scatter_settings = {
        "s": 40,
        "c": "RGB",
        "marker": "o",
        "alpha": 0.85,
        "zorder": CONSTANTS_COLOUR_STYLE.zorder.midground_scatter,
        "apply_cctf_encoding": True,
    }
    if scatter_kwargs is not None:
        scatter_settings.update(scatter_kwargs)

    settings = dict(kwargs)
    settings.update({"axes": axes, "show": False})

    colourspace = cast(
        RGB_Colourspace,
        first_item(filter_RGB_colourspaces(colourspace).values()),
    )

    settings["colourspaces"] = [colourspace, *settings.get("colourspaces", [])]

    chromaticity_diagram_callable(**settings)

    use_RGB_colours = str(scatter_settings["c"]).upper() == "RGB"
    apply_cctf_encoding = scatter_settings.pop("apply_cctf_encoding")
    if use_RGB_colours:
        RGB = RGB[RGB[:, 1].argsort()]
        scatter_settings["c"] = np.clip(
            np.reshape(
                RGB_to_RGB(
                    RGB,
                    colourspace,
                    CONSTANTS_COLOUR_STYLE.colour.colourspace,
                    apply_cctf_encoding=apply_cctf_encoding,
                ),
                (-1, 3),
            ),
            0,
            1,
        )

    XYZ = RGB_to_XYZ(RGB, colourspace)

    XYZ_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["XYZ_to_ij"]

    ij = XYZ_to_ij(XYZ, colourspace.whitepoint)

    axes.scatter(ij[..., 0], ij[..., 1], **scatter_settings)

    settings.update({"show": True})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    RGB: ArrayLike,
    colourspace: RGB_Colourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str] = "sRGB",
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931
    ),
    scatter_kwargs: dict | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspace array in the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    chromaticity_diagram_callable_CIE1931
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    scatter_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.scatter` definition.
        The following special keyword arguments can also be used:

        -   ``c`` : If ``c`` is set to *RGB*, the scatter will use the colours
            as given by the ``RGB`` argument.
        -   ``apply_cctf_encoding`` : If ``apply_cctf_encoding`` is set to
            *False*, the encoding colour component transfer function /
            opto-electronic transfer function is not applied when encoding the
            samples to the plotting space.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    ...     RGB, "ITU-R BT.709"
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1931,
        scatter_kwargs=scatter_kwargs,
        **settings,
    )


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
    RGB: ArrayLike,
    colourspace: RGB_Colourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str] = "sRGB",
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS
    ),
    scatter_kwargs: dict | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspace array in the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    chromaticity_diagram_callable_CIE1960UCS
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    scatter_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.scatter` definition.
        The following special keyword arguments can also be used:

        -   ``c`` : If ``c`` is set to *RGB*, the scatter will use the colours
            as given by the ``RGB`` argument.
        -   ``apply_cctf_encoding`` : If ``apply_cctf_encoding`` is set to
            *False*, the encoding colour component transfer function /
            opto-electronic transfer function is not applied when encoding the
            samples to the plotting space.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
    ...     RGB, "ITU-R BT.709"
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1960UCS,
        scatter_kwargs=scatter_kwargs,
        **settings,
    )


@override_style()
def plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
    RGB: ArrayLike,
    colourspace: RGB_Colourspace
    | str
    | Sequence[RGB_Colourspace | LiteralRGBColourspace | str] = "sRGB",
    chromaticity_diagram_callable_CIE1976UCS: Callable = (
        plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS
    ),
    scatter_kwargs: dict | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *RGB* colourspace array in the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_RGB_colourspaces` definition.
    chromaticity_diagram_callable_CIE1976UCS
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    scatter_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.scatter` definition.
        The following special keyword arguments can also be used:

        -   ``c`` : If ``c`` is set to *RGB*, the scatter will use the colours
            as given by the ``RGB`` argument.
        -   ``apply_cctf_encoding`` : If ``apply_cctf_encoding`` is set to
            *False*, the encoding colour component transfer function /
            opto-electronic transfer function is not applied when encoding the
            samples to the plotting space.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
    ...     RGB, "ITU-R BT.709"
    ... )
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_RGB_Chromaticities_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_RGB_chromaticities_in_chromaticity_diagram(
        RGB,
        colourspace,
        chromaticity_diagram_callable_CIE1976UCS,
        scatter_kwargs=scatter_kwargs,
        **settings,
    )


def ellipses_MacAdam1942(
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931"
) -> List[NDArrayFloat]:
    """
    Return *MacAdam (1942) Ellipses (Observer PGN)* coefficients according to
    given method.

    Parameters
    ----------
    method
        Computation method.

    Returns
    -------
    :class:`list`
        *MacAdam (1942) Ellipses (Observer PGN)* coefficients.

    Examples
    --------
    >>> ellipses_MacAdam1942()[0]  # doctest: +SKIP
    array([  1.60000000e-01,   5.70000000e-02,   5.00000023e-03,
             1.56666660e-02,  -2.77000015e+01])
    """

    method = validate_method(
        method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS")
    )

    xy_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["xy_to_ij"]

    x, y, _a, _b, _theta, a, b, theta = tsplit(DATA_MACADAM_1942_ELLIPSES)

    ellipses_coefficients = []
    for i in range(len(theta)):
        xy = point_at_angle_on_ellipse(
            np.linspace(0, 360, 36),
            [x[i], y[i], a[i] / 60, b[i] / 60, theta[i]],
        )
        ij = xy_to_ij(xy)
        ellipses_coefficients.append(
            ellipse_coefficients_canonical_form(ellipse_fitting(ij))
        )

    return ellipses_coefficients


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram(
    chromaticity_diagram_callable: Callable = plot_chromaticity_diagram,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    chromaticity_diagram_clipping: bool = False,
    ellipse_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot *MacAdam (1942) Ellipses (Observer PGN)* in the
    *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    chromaticity_diagram_callable
        Callable responsible for drawing the *Chromaticity Diagram*.
    method
        *Chromaticity Diagram* method.
    chromaticity_diagram_clipping
        Whether to clip the *Chromaticity Diagram* colours with the ellipses.
    ellipse_kwargs
        Parameters for the :class:`Ellipse` class, ``ellipse_kwargs`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram
    """

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    settings = dict(kwargs)
    settings.update({"axes": axes, "show": False})

    ellipses_coefficients = ellipses_MacAdam1942(method=method)

    if chromaticity_diagram_clipping:
        diagram_clipping_path_x = []
        diagram_clipping_path_y = []
        for coefficients in ellipses_coefficients:
            coefficients = np.copy(coefficients)  # noqa: PLW2901

            coefficients[2:4] /= 2

            x, y = tsplit(
                point_at_angle_on_ellipse(
                    np.linspace(0, 360, 36),
                    coefficients,
                )
            )
            diagram_clipping_path_x.append(x)
            diagram_clipping_path_y.append(y)

        diagram_clipping_path = np.rollaxis(
            np.array([diagram_clipping_path_x, diagram_clipping_path_y]), 0, 3
        )
        diagram_clipping_path = Path.make_compound_path_from_polys(
            diagram_clipping_path
        ).vertices
        settings.update({"diagram_clipping_path": diagram_clipping_path})

    chromaticity_diagram_callable(**settings)

    ellipse_settings_collection = [
        {
            "color": CONSTANTS_COLOUR_STYLE.colour.cycle[4],
            "alpha": 0.4,
            "linewidth": colour_style()["lines.linewidth"],
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.midground_polygon,
        }
        for _ellipses_coefficient in ellipses_coefficients
    ]

    if ellipse_kwargs is not None:
        update_settings_collection(
            ellipse_settings_collection,
            ellipse_kwargs,
            len(ellipses_coefficients),
        )

    for i, coefficients in enumerate(ellipses_coefficients):
        x_c, y_c, a_a, a_b, theta_e = coefficients
        ellipse = Ellipse(
            (x_c, y_c),
            a_a,
            a_b,
            angle=theta_e,
            **ellipse_settings_collection[i],
        )
        axes.add_artist(ellipse)

    settings.update({"show": True})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931(
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_chromaticity_diagram_CIE1931
    ),
    chromaticity_diagram_clipping: bool = False,
    ellipse_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1931
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    chromaticity_diagram_clipping
        Whether to clip the *CIE 1931 Chromaticity Diagram* colours with the
        ellipses.
    ellipse_kwargs
        Parameters for the :class:`Ellipse` class, ``ellipse_kwargs`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1931,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_kwargs=ellipse_kwargs,
        **settings,
    )


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS(
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS
    ),
    chromaticity_diagram_clipping: bool = False,
    ellipse_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1960UCS
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    chromaticity_diagram_clipping
        Whether to clip the *CIE 1960 UCS Chromaticity Diagram* colours with
        the ellipses.
    ellipse_kwargs
        Parameters for the :class:`Ellipse` class, ``ellipse_kwargs`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1960UCS,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_kwargs=ellipse_kwargs,
        **settings,
    )


@override_style()
def plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(
    chromaticity_diagram_callable_CIE1976UCS: Callable = (
        plot_chromaticity_diagram_CIE1976UCS
    ),
    chromaticity_diagram_clipping: bool = False,
    ellipse_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot *MacAdam (1942) Ellipses (Observer PGN)* in the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    chromaticity_diagram_callable_CIE1976UCS
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    chromaticity_diagram_clipping
        Whether to clip the *CIE 1976 UCS Chromaticity Diagram* colours with
        the ellipses.
    ellipse_kwargs
        Parameters for the :class:`Ellipse` class, ``ellipse_kwargs`` can
        be either a single dictionary applied to all the ellipses with same
        settings or a sequence of dictionaries with different settings for each
        ellipse.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram`},
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS()
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/\
Plotting_Plot_Ellipses_MacAdam1942_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_ellipses_MacAdam1942_in_chromaticity_diagram(
        chromaticity_diagram_callable_CIE1976UCS,
        chromaticity_diagram_clipping=chromaticity_diagram_clipping,
        ellipse_kwargs=ellipse_kwargs,
        **settings,
    )


@override_style()
def plot_single_cctf(
    cctf: Callable | str, cctf_decoding: bool = False, **kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    Plot given colourspace colour component transfer function.

    Parameters
    ----------
    cctf
        Colour component transfer function to plot. ``function`` can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    cctf_decoding
        Plot the decoding colour component transfer function instead.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_cctf("ITU-R BT.709")  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Single_CCTF.png
        :align: center
        :alt: plot_single_cctf
    """

    settings: Dict[str, Any] = {
        "title": f"{cctf} - {'Decoding' if cctf_decoding else 'Encoding'} CCTF"
    }
    settings.update(kwargs)

    return plot_multi_cctfs([cctf], cctf_decoding, **settings)


@override_style()
def plot_multi_cctfs(
    cctfs: Callable | str | Sequence[Callable | str],
    cctf_decoding: bool = False,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given colour component transfer functions.

    Parameters
    ----------
    cctfs
        Colour component transfer function to plot. ``cctfs`` elements can be
        of any type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    cctf_decoding
        Plot the decoding colour component transfer function instead.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_cctfs(["ITU-R BT.709", "sRGB"])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Multi_CCTFs.png
        :align: center
        :alt: plot_multi_cctfs
    """

    cctfs_filtered = filter_passthrough(
        CCTF_DECODINGS if cctf_decoding else CCTF_ENCODINGS, cctfs
    )

    mode = "Decoding" if cctf_decoding else "Encoding"
    title = f"{', '.join(list(cctfs_filtered))} - {mode} CCTFs"

    settings: Dict[str, Any] = {
        "bounding_box": (0, 1, 0, 1),
        "legend": True,
        "title": title,
        "x_label": "Signal Value" if cctf_decoding else "Tristimulus Value",
        "y_label": "Tristimulus Value" if cctf_decoding else "Signal Value",
    }
    settings.update(kwargs)

    with domain_range_scale("1"):
        return plot_multi_functions(cctfs_filtered, **settings)


@override_style()
def plot_constant_hue_loci(
    data: ArrayLike,
    model: LiteralColourspaceModel | str = "CIE Lab",
    scatter_kwargs: dict | None = None,
    convert_kwargs: dict | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given constant hue loci colour matches data such as that from
    :cite:`Hung1995` or :cite:`Ebner1998` that are easily loaded with
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`__.

    Parameters
    ----------
    data
        Constant hue loci colour matches data expected to be an `ArrayLike` as
        follows::

            [
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
    {metadata}),
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
    {metadata}),
                ('name', XYZ_r, XYZ_cr, (XYZ_ct, XYZ_ct, XYZ_ct, ...), \
    {metadata}),
                ...
            ]

        where ``name`` is the hue angle or name, ``XYZ_r`` the *CIE XYZ*
        tristimulus values of the reference illuminant, ``XYZ_cr`` the
        *CIE XYZ* tristimulus values of the reference colour under the
        reference illuminant, ``XYZ_ct`` the *CIE XYZ* tristimulus values of
        the colour matches under the reference illuminant and ``metadata`` the
        dataset metadata.
    model
        Colourspace model, see :attr:`colour.COLOURSPACE_MODELS` attribute for
        the list of supported colourspace models.
    scatter_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.scatter` definition.
        The following special keyword arguments can also be used:

        -   ``c`` : If ``c`` is set to *RGB*, the scatter will use the colours
            as given by the ``RGB`` argument.
    convert_kwargs
        Keyword arguments for the :func:`colour.convert` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    References
    ----------
    :cite:`Ebner1998`, :cite:`Hung1995`, :cite:`Mansencal2019`

    Examples
    --------
    >>> data = [
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.40920000, 0.28120000, 0.30600000]),
    ...         np.array(
    ...             [
    ...                 [0.02495100, 0.01908600, 0.02032900],
    ...                 [0.10944300, 0.06235900, 0.06788100],
    ...                 [0.27186500, 0.18418700, 0.19565300],
    ...                 [0.48898900, 0.40749400, 0.44854600],
    ...             ]
    ...         ),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.30760000, 0.48280000, 0.42770000]),
    ...         np.array(
    ...             [
    ...                 [0.02108000, 0.02989100, 0.02790400],
    ...                 [0.06194700, 0.11251000, 0.09334400],
    ...                 [0.15255800, 0.28123300, 0.23234900],
    ...                 [0.34157700, 0.56681300, 0.47035300],
    ...             ]
    ...         ),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.39530000, 0.28120000, 0.18450000]),
    ...         np.array(
    ...             [
    ...                 [0.02436400, 0.01908600, 0.01468800],
    ...                 [0.10331200, 0.06235900, 0.02854600],
    ...                 [0.26311900, 0.18418700, 0.12109700],
    ...                 [0.43158700, 0.40749400, 0.39008600],
    ...             ]
    ...         ),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.20510000, 0.18420000, 0.57130000]),
    ...         np.array(
    ...             [
    ...                 [0.03039800, 0.02989100, 0.06123300],
    ...                 [0.08870000, 0.08498400, 0.21843500],
    ...                 [0.18405800, 0.18418700, 0.40111400],
    ...                 [0.32550100, 0.34047200, 0.50296900],
    ...                 [0.53826100, 0.56681300, 0.80010400],
    ...             ]
    ...         ),
    ...         None,
    ...     ],
    ...     [
    ...         None,
    ...         np.array([0.95010000, 1.00000000, 1.08810000]),
    ...         np.array([0.35770000, 0.28120000, 0.11250000]),
    ...         np.array(
    ...             [
    ...                 [0.03678100, 0.02989100, 0.01481100],
    ...                 [0.17127700, 0.11251000, 0.01229900],
    ...                 [0.30080900, 0.28123300, 0.21229800],
    ...                 [0.52976000, 0.40749400, 0.11720000],
    ...             ]
    ...         ),
    ...         None,
    ...     ],
    ... ]
    >>> plot_constant_hue_loci(data, "CIE Lab")  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Constant_Hue_Loci.png
        :align: center
        :alt: plot_constant_hue_loci
    """

    # TODO: Filter appropriate colour models.
    # NOTE: "dtype=object" is required for ragged array support
    # in "Numpy" 1.24.0.
    data = as_array(data, dtype=object)  # pyright: ignore

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    scatter_settings = {
        "s": 40,
        "c": "RGB",
        "marker": "o",
        "alpha": 0.85,
        "zorder": CONSTANTS_COLOUR_STYLE.zorder.foreground_scatter,
    }
    if scatter_kwargs is not None:
        scatter_settings.update(scatter_kwargs)

    convert_kwargs = optional(convert_kwargs, {})

    use_RGB_colours = str(scatter_settings["c"]).upper() == "RGB"

    colourspace = CONSTANTS_COLOUR_STYLE.colour.colourspace
    for hue_data in data:
        _name, XYZ_r, XYZ_cr, XYZ_ct, _metadata = hue_data

        xy_r = XYZ_to_xy(XYZ_r)

        convert_settings = {"illuminant": xy_r}
        convert_settings.update(convert_kwargs)

        ijk_ct = colourspace_model_axis_reorder(
            convert(XYZ_ct, "CIE XYZ", model, **convert_settings), model
        )
        ijk_cr = colourspace_model_axis_reorder(
            convert(XYZ_cr, "CIE XYZ", model, **convert_settings), model
        )

        ijk_ct *= COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model]
        ijk_cr *= COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE[model]

        def _linear_equation(
            x: NDArrayFloat, a: NDArrayFloat, b: NDArrayFloat
        ) -> NDArrayFloat:
            """Define the canonical linear equation for a line."""

            return a * x + b

        popt, _pcov = scipy.optimize.curve_fit(
            _linear_equation, ijk_ct[..., 0], ijk_ct[..., 1]
        )

        axes.plot(
            ijk_ct[..., 0],
            _linear_equation(ijk_ct[..., 0], *popt),
            c=CONSTANTS_COLOUR_STYLE.colour.average,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
        )

        if use_RGB_colours:
            RGB_ct = XYZ_to_RGB(
                XYZ_ct, colourspace, xy_r, apply_cctf_encoding=True
            )
            scatter_settings["c"] = np.clip(RGB_ct, 0, 1)
            RGB_cr = XYZ_to_RGB(
                XYZ_cr, colourspace, xy_r, apply_cctf_encoding=True
            )
            RGB_cr = np.clip(np.ravel(RGB_cr), 0, 1)
        else:
            scatter_settings["c"] = CONSTANTS_COLOUR_STYLE.colour.dark
            RGB_cr = CONSTANTS_COLOUR_STYLE.colour.dark

        axes.scatter(ijk_ct[..., 0], ijk_ct[..., 1], **scatter_settings)

        axes.plot(
            ijk_cr[..., 0],
            ijk_cr[..., 1],
            "s",
            c=RGB_cr,
            markersize=CONSTANTS_COLOUR_STYLE.geometry.short * 8,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
        )

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[model])[
        as_int_array(colourspace_model_axis_reorder([0, 1, 2], model))
    ]

    settings = {
        "axes": axes,
        "title": f"Constant Hue Loci - {model}",
        "x_label": labels[0],
        "y_label": labels[1],
    }
    settings.update(kwargs)

    return render(**settings)
