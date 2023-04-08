"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from colour.algebra import normalise_maximum
from colour.colorimetry import MSDS_CMFS, CCS_ILLUMINANTS
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
    List,
    Literal,
    NDArrayFloat,
    Sequence,
    Tuple,
    cast,
)
from colour.models import (
    UCS_to_uv,
    UCS_uv_to_xy,
    XYZ_to_UCS,
    xy_to_Luv_uv,
    xy_to_UCS_uv,
    xy_to_XYZ,
)
from colour.temperature import mired_to_CCT, CCT_to_uv, CCT_to_xy_CIE_D
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    CONSTANTS_ARROW_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    filter_passthrough,
    override_style,
    render,
    update_settings_collection,
)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import (
    as_int_scalar,
    as_float_scalar,
    full,
    optional,
    tstack,
    validate_method,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_daylight_locus",
    "plot_planckian_locus",
    "plot_planckian_locus_in_chromaticity_diagram",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1931",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1976UCS",
]


@override_style()
def plot_daylight_locus(
    daylight_locus_colours: ArrayLike | str | None = None,
    daylight_locus_opacity: float = 1,
    daylight_locus_use_mireds: bool = False,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
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
    daylight_locus_use_mireds
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

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    daylight_locus_colours = optional(
        daylight_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if method == "cie 1931":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

    elif method == "cie 1960 ucs":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_UCS_uv(xy)

    elif method == "cie 1976 ucs":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_Luv_uv(xy)

    def CCT_to_plotting_colourspace(CCT):
        """
        Convert given correlated colour temperature :math:`T_{cp}` to the
        default plotting colourspace.
        """

        return normalise_maximum(
            XYZ_to_plotting_colourspace(xy_to_XYZ(CCT_to_xy_CIE_D(CCT))),
            axis=-1,
        )

    start, end = (
        (0, 1000) if daylight_locus_use_mireds else (1e6 / 600, 1e6 / 10)
    )

    CCT = np.arange(start, end + 100, 10) * 1.4388 / 1.4380
    CCT = mired_to_CCT(CCT) if daylight_locus_use_mireds else CCT
    ij = xy_to_ij(CCT_to_xy_CIE_D(CCT)).reshape(-1, 1, 2)

    use_RGB_daylight_locus_colours = (
        str(daylight_locus_colours).upper() == "RGB"
    )
    if use_RGB_daylight_locus_colours:
        pl_colours = CCT_to_plotting_colourspace(CCT)
    else:
        pl_colours = daylight_locus_colours

    line_collection = LineCollection(
        np.concatenate([ij[:-1], ij[1:]], axis=1),
        colors=pl_colours,
        alpha=daylight_locus_opacity,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
    )
    axes.add_collection(line_collection)

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus(
    planckian_locus_colours: ArrayLike | str | None = None,
    planckian_locus_opacity: float = 1,
    planckian_locus_labels: Sequence | None = None,
    planckian_locus_use_mireds: bool = False,
    planckian_locus_iso_temperature_lines_D_uv: float = 0.05,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
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
    planckian_locus_use_mireds
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

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    planckian_locus_colours = optional(
        planckian_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    labels = cast(
        tuple,
        optional(
            planckian_locus_labels,
            (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
            if planckian_locus_use_mireds
            else (1e6 / 600, 2000, 2500, 3000, 4000, 6000, 1e6 / 100),
        ),
    )
    D_uv = planckian_locus_iso_temperature_lines_D_uv

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if method == "cie 1931":

        def uv_to_ij(uv: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_uv_to_xy(uv)

    elif method == "cie 1960 ucs":

        def uv_to_ij(uv: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return uv

    elif method == "cie 1976 ucs":

        def uv_to_ij(uv: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy_to_Luv_uv(UCS_uv_to_xy(uv))

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

    start, end = (
        (0, 1000) if planckian_locus_use_mireds else (1e6 / 600, 1e6 / 10)
    )

    CCT = np.arange(start, end + 100, 10)
    CCT = mired_to_CCT(CCT) if planckian_locus_use_mireds else CCT
    CCT_D_uv = np.reshape(tstack([CCT, zeros(CCT.shape)]), (-1, 1, 2))
    ij = uv_to_ij(CCT_to_uv(CCT_D_uv, "Robertson 1968"))

    use_RGB_planckian_locus_colours = (
        str(planckian_locus_colours).upper() == "RGB"
    )
    if use_RGB_planckian_locus_colours:
        pl_colours = CCT_D_uv_to_plotting_colourspace(CCT_D_uv)
    else:
        pl_colours = planckian_locus_colours

    line_collection = LineCollection(
        np.concatenate([ij[:-1], ij[1:]], axis=1),
        colors=pl_colours,
        alpha=planckian_locus_opacity,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
    )
    axes.add_collection(line_collection)

    for label in labels:
        CCT_D_uv = np.reshape(
            tstack(
                [
                    full(
                        10,
                        as_float_scalar(mired_to_CCT(label))
                        if planckian_locus_use_mireds
                        else label,
                    ),
                    np.linspace(-D_uv, D_uv, 10),
                ]
            ),
            (-1, 1, 2),
        )

        if use_RGB_planckian_locus_colours:
            itl_colours = CCT_D_uv_to_plotting_colourspace(CCT_D_uv)
        else:
            itl_colours = planckian_locus_colours

        ij = uv_to_ij(CCT_to_uv(CCT_D_uv, "Robertson 1968"))

        line_collection = LineCollection(
            np.concatenate([ij[:-1], ij[1:]], axis=1),
            colors=itl_colours,
            alpha=planckian_locus_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
        )
        axes.add_collection(line_collection)
        axes.annotate(
            f'{as_int_scalar(label)}{"M" if planckian_locus_use_mireds else "K"}',
            xy=(ij[-1, :, 0], ij[-1, :, 1]),
            xytext=(0, CONSTANTS_COLOUR_STYLE.geometry.long / 2),
            textcoords="offset points",
            size="x-small",
            zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_label,
        )

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram(
    illuminants: str | Sequence[str],
    chromaticity_diagram_callable: Callable = plot_chromaticity_diagram,
    method: Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    | str = "CIE 1931",
    annotate_kwargs: dict | List[dict] | None = None,
    plot_kwargs: dict | List[dict] | None = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
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

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    illuminants_filtered = filter_passthrough(
        CCS_ILLUMINANTS[cmfs.name], illuminants
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    settings = {"axes": axes, "method": method}
    settings.update(kwargs)
    settings["standalone"] = False

    chromaticity_diagram_callable(**settings)

    plot_planckian_locus(**settings)

    if method == "CIE 1931":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == "CIE 1960 UCS":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)

    elif method == "CIE 1976 UCS":

        def xy_to_ij(xy: NDArrayFloat) -> NDArrayFloat:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy_to_Luv_uv(xy)

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

        ij = xy_to_ij(xy)

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
            "standalone": True,
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
) -> Tuple[plt.Figure, plt.Axes]:
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
) -> Tuple[plt.Figure, plt.Axes]:
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
) -> Tuple[plt.Figure, plt.Axes]:
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
