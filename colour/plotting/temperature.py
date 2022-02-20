"""
Colour Temperature & Correlated Colour Temperature Plotting
===========================================================

Defines the colour temperature and correlated colour temperature plotting
objects:

-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS`
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
    Floating,
    List,
    Literal,
    NDArray,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from colour.models import (
    UCS_to_uv,
    UCS_uv_to_xy,
    XYZ_to_UCS,
    xy_to_Luv_uv,
    xy_to_XYZ,
)
from colour.temperature import CCT_to_uv
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    CONSTANTS_ARROW_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    filter_passthrough,
    override_style,
    render,
    update_settings_collection,
)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import (
    as_int_scalar,
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
    "plot_planckian_locus",
    "plot_planckian_locus_in_chromaticity_diagram",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1931",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS",
]


@override_style()
def plot_planckian_locus(
    planckian_locus_colours: Optional[Union[ArrayLike, str]] = None,
    planckian_locus_opacity: Floating = 1,
    planckian_locus_labels: Optional[Sequence] = None,
    method: Union[
        Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"], str
    ] = "CIE 1931",
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
    >>> plot_planckian_locus(planckian_locus_colours='RGB')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

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
        Tuple,
        optional(
            planckian_locus_labels,
            (10**6 / 600, 2000, 2500, 3000, 4000, 6000, 10**6 / 100),
        ),
    )
    D_uv = 0.05

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if method == "cie 1931":

        def uv_to_ij(uv: NDArray) -> NDArray:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_uv_to_xy(uv)

    elif method == "cie 1960 ucs":

        def uv_to_ij(uv: NDArray) -> NDArray:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return uv

    elif method == "cie 1976 ucs":

        def uv_to_ij(uv: NDArray) -> NDArray:
            """
            Convert given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy_to_Luv_uv(UCS_uv_to_xy(uv))

    def CCT_D_uv_to_plotting_colourspace(CCT_D_uv):
        """
        Convert given *uv* chromaticity coordinates to the default plotting
        colourspace.
        """

        return normalise_maximum(
            XYZ_to_plotting_colourspace(
                xy_to_XYZ(UCS_uv_to_xy(CCT_to_uv(CCT_D_uv, "Robertson 1968")))
            ),
            axis=-1,
        )

    start, end = 10**6 / 600, 10**6 / 10
    CCT = np.arange(start, end + 100, 100)
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
            tstack([full(10, label), np.linspace(-D_uv, D_uv, 10)]), (-1, 1, 2)
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
            f"{as_int_scalar(label)}K",
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
    illuminants: Union[str, Sequence[str]],
    chromaticity_diagram_callable: Callable = (
        plot_chromaticity_diagram  # type: ignore[has-type]
    ),
    method: Union[Literal["CIE 1931", "CIE 1960 UCS"], str] = "CIE 1931",
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
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
        :func:`colour.plotting.filter_passthrough` definition.
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
    ...     {'xytext': (-25, 15), 'arrowprops':{'arrowstyle':'-'}},
    ...     {'arrowprops':{'arrowstyle':'-['}},
    ...     {},
    ... ]
    >>> plot_kwargs = [
    ...     {
    ...         'markersize' : 15,
    ...     },
    ...     {   'color': 'r'},
    ...     {},
    ... ]
    >>> plot_planckian_locus_in_chromaticity_diagram(
    ...     ['A', 'B', 'C'],
    ...     annotate_kwargs=annotate_kwargs,
    ...     plot_kwargs=plot_kwargs
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Planckian_Locus_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_planckian_locus_in_chromaticity_diagram
    """

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    illuminants_filtered = filter_passthrough(
        CCS_ILLUMINANTS.get(cmfs.name), illuminants  # type: ignore[arg-type]
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

        def xy_to_ij(xy: NDArray) -> NDArray:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return xy

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == "CIE 1960 UCS":

        def xy_to_ij(xy: NDArray) -> NDArray:
            """
            Convert given *CIE xy* chromaticity coordinates to *ij*
            chromaticity coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)
    else:
        raise ValueError(
            f'Invalid method: "{method}", must be one of '
            f'["CIE 1931", "CIE 1960 UCS"]'
        )

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
    illuminants: Union[str, Sequence[str]],
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_chromaticity_diagram_CIE1931  # type: ignore[has-type]
    ),
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
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
        :func:`colour.plotting.filter_passthrough` definition.
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
    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1931(['A', 'B', 'C'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

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
    illuminants: Union[str, Sequence[str]],
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS  # type: ignore[has-type]
    ),
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
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
        :func:`colour.plotting.filter_passthrough` definition.
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
    ...     ['A', 'C', 'E'])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

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
