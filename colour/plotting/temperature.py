# -*- coding: utf-8 -*-
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

from colour.colorimetry import MSDS_CMFS, CCS_ILLUMINANTS
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Dict,
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
    UCS_uv_to_xy,
    XYZ_to_UCS,
    UCS_to_uv,
    xy_to_XYZ,
)
from colour.temperature import CCT_to_uv
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    CONSTANTS_ARROW_STYLE,
    artist,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    filter_passthrough,
    override_style,
    render,
    update_settings_collection,
)
from colour.plotting.diagrams import plot_chromaticity_diagram
from colour.utilities import optional, tstack, validate_method, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_planckian_locus",
    "plot_planckian_locus_CIE1931",
    "plot_planckian_locus_CIE1960UCS",
    "plot_planckian_locus_in_chromaticity_diagram",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1931",
    "plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS",
]


@override_style()
def plot_planckian_locus(
    planckian_locus_colours: Optional[Union[ArrayLike, str]] = None,
    method: Union[Literal["CIE 1931", "CIE 1960 UCS"], str] = "CIE 1931",
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* according to given method.

    Parameters
    ----------
    planckian_locus_colours
        *Planckian Locus* colours.
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
    >>> plot_planckian_locus()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus.png
        :align: center
        :alt: plot_planckian_locus
    """

    method = validate_method(method, ["CIE 1931", "CIE 1960 UCS"])

    planckian_locus_colours = cast(
        Union[ArrayLike, str],
        optional(planckian_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark),
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if method == "cie 1931":

        def uv_to_ij(uv: NDArray) -> NDArray:
            """
            Converts given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_uv_to_xy(uv)

        D_uv = 0.025

    elif method == "cie 1960 ucs":

        def uv_to_ij(uv: NDArray) -> NDArray:
            """
            Converts given *uv* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return uv

        D_uv = 0.025

    start, end = 1667, 100000
    CCT = np.arange(start, end + 250, 250)
    CCT_D_uv = tstack([CCT, zeros(CCT.shape)])
    ij = uv_to_ij(CCT_to_uv(CCT_D_uv, "Robertson 1968"))

    axes.plot(ij[..., 0], ij[..., 1], color=planckian_locus_colours)

    for i in (1667, 2000, 2500, 3000, 4000, 6000, 10000):
        i0, j0 = uv_to_ij(CCT_to_uv(np.array([i, -D_uv]), "Robertson 1968"))
        i1, j1 = uv_to_ij(CCT_to_uv(np.array([i, D_uv]), "Robertson 1968"))
        axes.plot((i0, i1), (j0, j1), color=planckian_locus_colours)
        axes.annotate(
            "{0}K".format(i),
            xy=(i0, j0),
            xytext=(0, -10),
            textcoords="offset points",
            size="x-small",
        )

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_planckian_locus_CIE1931(
    planckian_locus_colours: Optional[Union[ArrayLike, str]] = None, **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* according to *CIE 1931* method.

    Parameters
    ----------
    planckian_locus_colours
        *Planckian Locus* colours.

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
    >>> plot_planckian_locus_CIE1931()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus_CIE1931.png
        :align: center
        :alt: plot_planckian_locus_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_planckian_locus(planckian_locus_colours, **settings)


@override_style()
def plot_planckian_locus_CIE1960UCS(
    planckian_locus_colours: Optional[Union[ArrayLike, str]] = None, **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* according to *CIE 1960 UCS* method.

    Parameters
    ----------
    planckian_locus_colours
        *Planckian Locus* colours.

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
    >>> plot_planckian_locus_CIE1960UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Planckian_Locus_CIE1960UCS.png
        :align: center
        :alt: plot_planckian_locus_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_planckian_locus(planckian_locus_colours, **settings)


@override_style()
def plot_planckian_locus_in_chromaticity_diagram(
    illuminants: Union[str, Sequence[str]],
    chromaticity_diagram_callable: Callable = (
        plot_chromaticity_diagram  # type: ignore[has-type]
    ),
    planckian_locus_callable: Callable = plot_planckian_locus,
    method: Union[Literal["CIE 1931", "CIE 1960 UCS"], str] = "CIE 1931",
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* and given illuminants in the
    *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.
    chromaticity_diagram_callable
        Callable responsible for drawing the *Chromaticity Diagram*.
    planckian_locus_callable
        Callable responsible for drawing the *Planckian Locus*.
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

    planckian_locus_callable(**settings)

    if method == "CIE 1931":

        def xy_to_ij(xy: NDArray) -> NDArray:
            """
            Converts given *CIE xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return xy

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == "CIE 1960 UCS":

        def xy_to_ij(xy: NDArray) -> NDArray:
            """
            Converts given *CIE xy* chromaticity coordinates to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            "['CIE 1931', 'CIE 1960 UCS']".format(method)
        )

    annotate_settings_collection = [
        {
            "annotate": True,
            "xytext": (-50, 30),
            "textcoords": "offset points",
            "arrowprops": CONSTANTS_ARROW_STYLE,
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
            "label": "{0}".format(illuminant),
            "marker": "o",
            "markeredgecolor": CONSTANTS_COLOUR_STYLE.colour.dark,
            "markeredgewidth": CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
            "markersize": (
                CONSTANTS_COLOUR_STYLE.geometry.short * 6
                + CONSTANTS_COLOUR_STYLE.geometry.short * 0.75
            ),
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
            "{0} Illuminants - Planckian Locus\n"
            "{1} Chromaticity Diagram - "
            "CIE 1931 2 Degree Standard Observer"
        ).format(", ".join(illuminants_filtered), method.upper())
        if illuminants_filtered
        else (
            "Planckian Locus\n{0} Chromaticity Diagram - "
            "CIE 1931 2 Degree Standard Observer".format(method.upper())
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
    planckian_locus_callable_CIE1931: Callable = plot_planckian_locus_CIE1931,
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* and given illuminants in
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.
    chromaticity_diagram_callable_CIE1931
        Callable responsible for drawing the *CIE 1931 Chromaticity Diagram*.
    planckian_locus_callable_CIE1931
        Callable responsible for drawing the *Planckian Locus* according to
        *CIE 1931* method.
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
        planckian_locus_callable_CIE1931,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings
    )


@override_style()
def plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
    illuminants: Union[str, Sequence[str]],
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS  # type: ignore[has-type]
    ),
    planckian_locus_callable_CIE1960UCS: Callable = plot_planckian_locus_CIE1960UCS,
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the *Planckian Locus* and given illuminants in
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
    planckian_locus_callable_CIE1960UCS
        Callable responsible for drawing the *Planckian Locus* according to
        *CIE 1960 UCS* method.
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
        planckian_locus_callable_CIE1960UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings
    )
