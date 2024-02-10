"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`colour.plotting.lines_daylight_locus`
-   :func:`colour.plotting.lines_planckian_locus`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS`
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from colour.algebra import normalise_maximum, normalise_vector
from colour.colorimetry import CCS_ILLUMINANTS, MSDS_CMFS
from colour.constants import DTYPE_FLOAT_DEFAULT
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    List,
    Literal,
    NDArray,
    Sequence,
    Tuple,
    cast,
)
from colour.models import (
    UCS_uv_to_xy,
    xy_to_XYZ,
)
from colour.plotting import (
    CONSTANTS_ARROW_STYLE,
    CONSTANTS_COLOUR_STYLE,
    METHODS_CHROMATICITY_DIAGRAM,
    XYZ_to_plotting_colourspace,
    artist,
    filter_passthrough,
    override_style,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    render,
    update_settings_collection,
)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.temperature import CCT_to_uv, CCT_to_xy_CIE_D, mired_to_CCT
from colour.utilities import (
    CanonicalMapping,
    as_float_array,
    as_float_scalar,
    as_int_scalar,
    full,
    optional,
    tstack,
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
    "lines_daylight_locus",
    "plot_daylight_locus",
    "LABELS_PLANCKIAN_LOCUS_DEFAULT",
    "lines_planckian_locus",
    "plot_planckian_locus",
    "plot_planckian_locus_in_chromaticity_diagram",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1931",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS",
]


def lines_daylight_locus(
    mireds: bool = False,
    method: (Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"] | str) = "CIE 1931",
) -> Tuple[NDArray]:
    """
    Return the *Daylight Locus* line vertices, i.e. positions, normals and
    colours, according to given method.

    Parameters
    ----------
    mireds
        Whether to use micro reciprocal degrees for the iso-temperature lines.
    method
        *Daylight Locus* method.

    Returns
    -------
    :class:`tuple`
        Tuple of *Spectral Locus* vertices.

    Examples
    --------
    >>> lines = lines_daylight_locus()
    >>> len(lines)
    1
    >>> lines[0].dtype
    dtype([('position', '<f8', (2,)), ('normal', '<f8', (2,)), \
('colour', '<f8', (3,))])
    """

    method = validate_method(method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"))

    xy_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["xy_to_ij"]

    def CCT_to_plotting_colourspace(CCT):
        """
        Convert given correlated colour temperature :math:`T_{cp}` to the
        default plotting colourspace.
        """

        return normalise_maximum(
            XYZ_to_plotting_colourspace(xy_to_XYZ(CCT_to_xy_CIE_D(CCT))),
            axis=-1,
        )

    start, end = (0, 1000) if mireds else (1e6 / 600, 1e6 / 10)

    CCT = np.arange(start, end + 100, 10) * 1.4388 / 1.4380
    CCT = mired_to_CCT(CCT) if mireds else CCT

    ij_sl = np.reshape(xy_to_ij(CCT_to_xy_CIE_D(CCT)), (-1, 2))
    colour_sl = np.reshape(CCT_to_plotting_colourspace(CCT), (-1, 3))

    lines_sl = zeros(
        ij_sl.shape[0],
        [
            ("position", DTYPE_FLOAT_DEFAULT, 2),
            ("normal", DTYPE_FLOAT_DEFAULT, 2),
            ("colour", DTYPE_FLOAT_DEFAULT, 3),
        ],  # pyright: ignore
    )

    lines_sl["position"] = ij_sl
    lines_sl["colour"] = colour_sl

    return (lines_sl,)


@override_style()
def plot_daylight_locus(
    daylight_locus_colours: ArrayLike | str | None = None,
    daylight_locus_opacity: float = 1,
    daylight_locus_mireds: bool = False,
    method: (Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"] | str) = "CIE 1931",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Daylight Locus* according to given method.

    Parameters
    ----------
    daylight_locus_colours
        Colours of the *Daylight Locus*, if ``daylight_locus_colours`` is set
        to *RGB*, the colours will be computed according to the corresponding
        chromaticity coordinates.
    daylight_locus_opacity
       Opacity of the *Daylight Locus*.
    daylight_locus_mireds
        Whether to use micro reciprocal degrees for the iso-temperature lines.
    method
        *Chromaticity Diagram* method.

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
    >>> plot_daylight_locus(daylight_locus_colours="RGB")
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Daylight_Locus.png
        :align: center
        :alt: plot_daylight_locus
    """

    method = validate_method(method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"))

    use_RGB_daylight_locus_colours = str(daylight_locus_colours).upper() == "RGB"

    daylight_locus_colours = optional(
        daylight_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    lines_sl, *_ = lines_daylight_locus(daylight_locus_mireds, method)

    line_collection = LineCollection(
        np.reshape(
            np.concatenate(
                [lines_sl["position"][:-1], lines_sl["position"][1:]], axis=1
            ),
            (-1, 2, 2),
        ),  # pyright: ignore
        colors=(
            lines_sl["colour"]
            if use_RGB_daylight_locus_colours
            else daylight_locus_colours
        ),
        alpha=daylight_locus_opacity,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
    )
    axes.add_collection(line_collection)

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**settings)


LABELS_PLANCKIAN_LOCUS_DEFAULT: CanonicalMapping = CanonicalMapping(
    {
        "Default": (1e6 / 600, 2000, 2500, 3000, 4000, 6000, 1e6 / 100),
        "Mireds": (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    }
)
"""*Planckian Locus* default labels."""


def lines_planckian_locus(
    labels: Sequence | None = None,
    mireds: bool = False,
    iso_temperature_lines_D_uv: float = 0.05,
    method: (Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"] | str) = "CIE 1931",
) -> Tuple[NDArray, NDArray]:
    """
    Return the *Planckian Locus* line vertices, i.e. positions, normals and
    colours, according to given method.

    Parameters
    ----------
    labels
        Array of labels used to customise which iso-temperature lines will be
        drawn along the *Planckian Locus*. Passing an empty array will result
        in no iso-temperature lines being drawn.
    mireds
        Whether to use micro reciprocal degrees for the iso-temperature lines.
    iso_temperature_lines_D_uv
        Iso-temperature lines :math:`\\Delta_{uv}` length on each side of the
        *Planckian Locus*.
    method
        *Planckian Locus* method.

    Returns
    -------
    :class:`tuple`
        Tuple of *Planckian Locus* vertices and wavelength labels vertices.

    Examples
    --------
    >>> lines = lines_planckian_locus()
    >>> len(lines)
    2
    >>> lines[0].dtype
    dtype([('position', '<f8', (2,)), ('normal', '<f8', (2,)), \
('colour', '<f8', (3,))])
    >>> lines[1].dtype
    dtype([('position', '<f8', (2,)), ('normal', '<f8', (2,)), \
('colour', '<f8', (3,))])
    """

    method = validate_method(method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"))

    labels = cast(
        tuple,
        optional(
            labels,
            LABELS_PLANCKIAN_LOCUS_DEFAULT["Mireds" if mireds else "Default"],
        ),
    )
    D_uv = iso_temperature_lines_D_uv

    uv_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["uv_to_ij"]

    def CCT_D_uv_to_plotting_colourspace(CCT_D_uv):
        """
        Convert given correlated colour temperature :math:`T_{cp}` and
        :math:`\\Delta_{uv}` to the default plotting colourspace.
        """

        return normalise_maximum(
            XYZ_to_plotting_colourspace(
                xy_to_XYZ(UCS_uv_to_xy(CCT_to_uv(CCT_D_uv, "Robertson 1968")))
            ),
            axis=-1,
        )

    # Planckian Locus
    start, end = (0, 1000) if mireds else (1e6 / 600, 1e6 / 10)

    CCT = np.arange(start, end + 100, 100)
    CCT = mired_to_CCT(CCT) if mireds else CCT
    CCT_D_uv = tstack([CCT, zeros(CCT.shape)])

    ij_pl = uv_to_ij(CCT_to_uv(CCT_D_uv, "Robertson 1968"))
    colour_pl = CCT_D_uv_to_plotting_colourspace(CCT_D_uv)

    lines_pl = zeros(
        ij_pl.shape[0],
        [
            ("position", DTYPE_FLOAT_DEFAULT, 2),
            ("normal", DTYPE_FLOAT_DEFAULT, 2),
            ("colour", DTYPE_FLOAT_DEFAULT, 3),
        ],  # pyright: ignore
    )

    lines_pl["position"] = ij_pl
    lines_pl["colour"] = colour_pl

    # Labels
    ij_itl, normal_itl, colour_itl = [], [], []
    for label in labels:
        CCT_D_uv = tstack(
            [
                full(
                    20,
                    as_float_scalar(mired_to_CCT(label)) if mireds else label,
                ),
                np.linspace(-D_uv, D_uv, 20),
            ]
        )

        ij = uv_to_ij(CCT_to_uv(CCT_D_uv, "Robertson 1968"))
        ij_itl.append(ij)
        normal_itl.append(np.tile(normalise_vector(ij[-1, ...] - ij[0, ...]), (20, 1)))
        colour_itl.append(CCT_D_uv_to_plotting_colourspace(CCT_D_uv))

    ij_l = np.reshape(as_float_array(ij_itl), (-1, 2))
    normal_l = np.reshape(as_float_array(normal_itl), (-1, 2))
    colour_l = np.reshape(as_float_array(colour_itl), (-1, 3))

    lines_l = zeros(
        ij_l.shape[0],
        [
            ("position", DTYPE_FLOAT_DEFAULT, 2),
            ("normal", DTYPE_FLOAT_DEFAULT, 2),
            ("colour", DTYPE_FLOAT_DEFAULT, 3),
        ],  # pyright: ignore
    )

    lines_l["position"] = ij_l
    lines_l["normal"] = normal_l
    lines_l["colour"] = colour_l

    return lines_pl, lines_l


@override_style()
def plot_planckian_locus(
    planckian_locus_colours: ArrayLike | str | None = None,
    planckian_locus_opacity: float = 1,
    planckian_locus_labels: Sequence | None = None,
    planckian_locus_mireds: bool = False,
    planckian_locus_iso_temperature_lines_D_uv: float = 0.05,
    method: (Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"] | str) = "CIE 1931",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Planckian Locus* according to given method.

    Parameters
    ----------
    planckian_locus_colours
        Colours of the *Planckian Locus*, if ``planckian_locus_colours`` is set
        to *RGB*, the colours will be computed according to the corresponding
        chromaticity coordinates.
    planckian_locus_opacity
       Opacity of the *Planckian Locus*.
    planckian_locus_labels
        Array of labels used to customise which iso-temperature lines will be
        drawn along the *Planckian Locus*. Passing an empty array will result
        in no iso-temperature lines being drawn.
    planckian_locus_mireds
        Whether to use micro reciprocal degrees for the iso-temperature lines.
    planckian_locus_iso_temperature_lines_D_uv
        Iso-temperature lines :math:`\\Delta_{uv}` length on each side of the
        *Planckian Locus*.
    method
        *Chromaticity Diagram* method.

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
    >>> plot_planckian_locus(planckian_locus_colours="RGB")
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus.png
        :align: center
        :alt: plot_planckian_locus
    """

    planckian_locus_colours = optional(
        planckian_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    use_RGB_planckian_locus_colours = str(planckian_locus_colours).upper() == "RGB"

    labels = cast(
        tuple,
        optional(
            planckian_locus_labels,
            LABELS_PLANCKIAN_LOCUS_DEFAULT[
                "Mireds" if planckian_locus_mireds else "Default"
            ],
        ),
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    lines_pl, lines_l = lines_planckian_locus(
        labels,
        planckian_locus_mireds,
        planckian_locus_iso_temperature_lines_D_uv,
        method,
    )

    axes.add_collection(
        LineCollection(
            np.reshape(
                np.concatenate(
                    [lines_pl["position"][:-1], lines_pl["position"][1:]], axis=1
                ),
                (-1, 2, 2),
            ),  # pyright: ignore
            colors=(
                lines_pl["colour"]
                if use_RGB_planckian_locus_colours
                else planckian_locus_colours
            ),
            alpha=planckian_locus_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
        )
    )

    lines_itl = np.reshape(lines_l["position"], (len(labels), 20, 2))
    colours_itl = np.reshape(lines_l["colour"], (len(labels), 20, 3))
    for i, label in enumerate(labels):
        axes.add_collection(
            LineCollection(
                np.reshape(
                    np.concatenate(
                        [lines_itl[i][:-1], lines_itl[i][1:]],  # pyright: ignore
                        axis=1,
                    ),
                    (-1, 2, 2),
                ),
                colors=(
                    colours_itl[i]
                    if use_RGB_planckian_locus_colours
                    else planckian_locus_colours
                ),
                alpha=planckian_locus_opacity,
                zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
            )
        )

        axes.text(
            lines_itl[i][-1, 0],
            lines_itl[i][-1, 1],
            f'{as_int_scalar(label)}{"M" if planckian_locus_mireds else "K"}',
            clip_on=True,
            ha="left",
            va="bottom",
            fontsize="x-small-colour-science",
            zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_label,
        )

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram(
    illuminants: str | Sequence[str],
    chromaticity_diagram_callable: Callable = plot_chromaticity_diagram,
    method: (Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"] | str) = "CIE 1931",
    annotate_kwargs: dict | List[dict] | None = None,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Planckian Locus* and given illuminants in the
    *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    chromaticity_diagram_callable
        Callable responsible for drawing the *Chromaticity Diagram*.
    method
        *Chromaticity Diagram* method.
    annotate_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.annotate`
        definition, used to annotate the resulting chromaticity coordinates
        with their respective spectral distribution names. ``annotate_kwargs``
        can be either a single dictionary applied to all the arrows with same
        settings or a sequence of dictionaries with different settings for each
        spectral distribution. The following special keyword arguments can also
        be used:

        -   ``annotate`` : Whether to annotate the spectral distributions.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted illuminants. ``plot_kwargs``
        can be either a single dictionary applied to all the plotted
        illuminants with the same settings or a sequence of dictionaries with
        different settings for eachplotted illuminant.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> annotate_kwargs = [
    ...     {"xytext": (-25, 15), "arrowprops": {"arrowstyle": "-"}},
    ...     {"arrowprops": {"arrowstyle": "-["}},
    ...     {},
    ... ]
    >>> plot_kwargs = [
    ...     {
    ...         "markersize": 15,
    ...     },
    ...     {"color": "r"},
    ...     {},
    ... ]
    >>> plot_planckian_locus_in_chromaticity_diagram(
    ...     ["A", "B", "C"],
    ...     annotate_kwargs=annotate_kwargs,
    ...     plot_kwargs=plot_kwargs,
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram
    """

    method = validate_method(method, ("CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"))

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    illuminants_filtered = filter_passthrough(CCS_ILLUMINANTS[cmfs.name], illuminants)

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    settings = {"axes": axes, "method": method}
    settings.update(kwargs)
    settings["show"] = False

    chromaticity_diagram_callable(**settings)

    plot_planckian_locus(**settings)

    xy_to_ij = METHODS_CHROMATICITY_DIAGRAM[method]["xy_to_ij"]

    if method == "CIE 1931":
        bounding_box = (-0.1, 0.9, -0.1, 0.9)

    elif method == "CIE 1960 UCS":
        bounding_box = (-0.1, 0.7, -0.2, 0.6)

    elif method == "CIE 1976 UCS":
        bounding_box = (-0.1, 0.7, -0.1, 0.7)

    annotate_settings_collection = [
        {
            "annotate": True,
            "xytext": (-50, 30),
            "textcoords": "offset points",
            "arrowprops": CONSTANTS_ARROW_STYLE,
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.foreground_annotation,
        }
        for _ in range(len(illuminants_filtered))
    ]

    if annotate_kwargs is not None:
        update_settings_collection(
            annotate_settings_collection,
            annotate_kwargs,
            len(illuminants_filtered),
        )

    plot_settings_collection = [
        {
            "color": CONSTANTS_COLOUR_STYLE.colour.brightest,
            "label": f"{illuminant}",
            "marker": "o",
            "markeredgecolor": CONSTANTS_COLOUR_STYLE.colour.dark,
            "markeredgewidth": CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
            "markersize": (
                CONSTANTS_COLOUR_STYLE.geometry.short * 6
                + CONSTANTS_COLOUR_STYLE.geometry.short * 0.75
            ),
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
        }
        for illuminant in illuminants_filtered
    ]

    if plot_kwargs is not None:
        update_settings_collection(
            plot_settings_collection, plot_kwargs, len(illuminants_filtered)
        )

    for i, (illuminant, xy) in enumerate(illuminants_filtered.items()):
        plot_settings = plot_settings_collection[i]

        ij = cast(tuple[float, float], xy_to_ij(xy))

        axes.plot(ij[0], ij[1], **plot_settings)

        if annotate_settings_collection[i]["annotate"]:
            annotate_settings = annotate_settings_collection[i]
            annotate_settings.pop("annotate")

            axes.annotate(illuminant, xy=ij, **annotate_settings)

    title = (
        (
            f"{', '.join(illuminants_filtered)} Illuminants - Planckian Locus\n"
            f"{method.upper()} Chromaticity Diagram - "
            "CIE 1931 2 Degree Standard Observer"
        )
        if illuminants_filtered
        else (
            f"Planckian Locus\n{method.upper()} Chromaticity Diagram - "
            f"CIE 1931 2 Degree Standard Observer"
        )
    )

    settings.update(
        {
            "axes": axes,
            "show": True,
            "bounding_box": bounding_box,
            "title": title,
        }
    )
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1931(
    illuminants: str | Sequence[str],
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_chromaticity_diagram_CIE1931
    ),
    annotate_kwargs: dict | List[dict] | None = None,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Planckian Locus* and given illuminants in
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    chromaticity_diagram_callable_CIE1931
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    annotate_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.annotate`
        definition, used to annotate the resulting chromaticity coordinates
        with their respective spectral distribution names. ``annotate_kwargs``
        can be either a single dictionary applied to all the arrows with same
        settings or a sequence of dictionaries with different settings for each
        spectral distribution. The following special keyword arguments can also
        be used:

        -   ``annotate`` : Whether to annotate the spectral distributions.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted illuminants. ``plot_kwargs``
        can be either a single dictionary applied to all the plotted
        illuminants with the same settings or a sequence of dictionaries with
        different settings for eachplotted illuminant.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1931(["A", "B", "C"])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_planckian_locus_in_chromaticity_diagram(
        illuminants,
        chromaticity_diagram_callable_CIE1931,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
    illuminants: str | Sequence[str],
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS
    ),
    annotate_kwargs: dict | List[dict] | None = None,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Planckian Locus* and given illuminants in
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    chromaticity_diagram_callable_CIE1960UCS
        Callable responsible for drawing the
        *CIE 1960 UCS Chromaticity Diagram*.
    annotate_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.annotate`
        definition, used to annotate the resulting chromaticity coordinates
        with their respective spectral distribution names. ``annotate_kwargs``
        can be either a single dictionary applied to all the arrows with same
        settings or a sequence of dictionaries with different settings for each
        spectral distribution. The following special keyword arguments can also
        be used:

        -   ``annotate`` : Whether to annotate the spectral distributions.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted illuminants. ``plot_kwargs``
        can be either a single dictionary applied to all the plotted
        illuminants with the same settings or a sequence of dictionaries with
        different settings for eachplotted illuminant.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
    ...     ["A", "C", "E"]
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_planckian_locus_in_chromaticity_diagram(
        illuminants,
        chromaticity_diagram_callable_CIE1960UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS(
    illuminants: str | Sequence[str],
    chromaticity_diagram_callable_CIE1976UCS: Callable = (
        plot_chromaticity_diagram_CIE1976UCS
    ),
    annotate_kwargs: dict | List[dict] | None = None,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the *Planckian Locus* and given illuminants in
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.
    chromaticity_diagram_callable_CIE1976UCS
        Callable responsible for drawing the
        *CIE 1976 UCS Chromaticity Diagram*.
    annotate_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.annotate`
        definition, used to annotate the resulting chromaticity coordinates
        with their respective spectral distribution names. ``annotate_kwargs``
        can be either a single dictionary applied to all the arrows with same
        settings or a sequence of dictionaries with different settings for each
        spectral distribution. The following special keyword arguments can also
        be used:

        -   ``annotate`` : Whether to annotate the spectral distributions.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted illuminants. ``plot_kwargs``
        can be either a single dictionary applied to all the plotted
        illuminants with the same settings or a sequence of dictionaries with
        different settings for eachplotted illuminant.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.temperature.plot_planckian_locus`,
        :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS(
    ...     ["A", "C", "E"]
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_planckian_locus_in_chromaticity_diagram(
        illuminants,
        chromaticity_diagram_callable_CIE1976UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )
