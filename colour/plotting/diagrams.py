"""
CIE Chromaticity Diagrams Plotting
==================================

Defines the *CIE* chromaticity diagrams plotting objects:

-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.plot_chromaticity_diagram_CIE1976UCS`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1931`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1960UCS`
-   :func:`colour.plotting.plot_sds_in_chromaticity_diagram_CIE1976UCS`
"""

from __future__ import annotations

import bisect
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from colour.algebra import normalise_maximum, normalise_vector
from colour.colorimetry import (
    MultiSpectralDistributions,
    SDS_ILLUMINANTS,
    SpectralDistribution,
    sd_to_XYZ,
    sds_and_msds_to_sds,
)
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    Dict,
    Floating,
    Integer,
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
    Luv_to_uv,
    Luv_uv_to_xy,
    UCS_to_uv,
    UCS_uv_to_xy,
    XYZ_to_Luv,
    XYZ_to_UCS,
    XYZ_to_xy,
    xy_to_XYZ,
)
from colour.notation import HEX_to_RGB
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    CONSTANTS_ARROW_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    filter_cmfs,
    filter_illuminants,
    override_style,
    render,
    update_settings_collection,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    first_item,
    is_string,
    optional,
    tsplit,
    tstack,
    suppress_warnings,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_spectral_locus",
    "plot_chromaticity_diagram_colours",
    "plot_chromaticity_diagram",
    "plot_chromaticity_diagram_CIE1931",
    "plot_chromaticity_diagram_CIE1960UCS",
    "plot_chromaticity_diagram_CIE1976UCS",
    "plot_sds_in_chromaticity_diagram",
    "plot_sds_in_chromaticity_diagram_CIE1931",
    "plot_sds_in_chromaticity_diagram_CIE1960UCS",
    "plot_sds_in_chromaticity_diagram_CIE1976UCS",
]


@override_style()
def plot_spectral_locus(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    spectral_locus_colours: Optional[Union[ArrayLike, str]] = None,
    spectral_locus_opacity: Floating = 1,
    spectral_locus_labels: Optional[Sequence] = None,
    method: Union[
        Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"], str
    ] = "CIE 1931",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Spectral Locus* according to given method.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    spectral_locus_colours
        Colours of the *Spectral Locus*, if ``spectral_locus_colours`` is set
        to *RGB*, the colours will be computed according to the corresponding
        chromaticity coordinates.
    spectral_locus_opacity
        Opacity of the *Spectral Locus*.
    spectral_locus_labels
        Array of wavelength labels used to customise which labels will be drawn
        around the spectral locus. Passing an empty array will result in no
        wavelength labels being drawn.
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
    >>> plot_spectral_locus(spectral_locus_colours='RGB')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Spectral_Locus.png
        :align: center
        :alt: plot_spectral_locus
    """

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    spectral_locus_colours = optional(
        spectral_locus_colours, CONSTANTS_COLOUR_STYLE.colour.dark
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    wavelengths = list(cmfs.wavelengths)
    equal_energy = np.array([1 / 3] * 2)

    if method == "cie 1931":
        ij = XYZ_to_xy(cmfs.values, illuminant)
        labels = cast(
            Tuple,
            optional(
                spectral_locus_labels,
                (
                    390,
                    460,
                    470,
                    480,
                    490,
                    500,
                    510,
                    520,
                    540,
                    560,
                    580,
                    600,
                    620,
                    700,
                ),
            ),
        )
    elif method == "cie 1960 ucs":
        ij = UCS_to_uv(XYZ_to_UCS(cmfs.values))
        labels = cast(
            Tuple,
            optional(
                spectral_locus_labels,
                (
                    420,
                    440,
                    450,
                    460,
                    470,
                    480,
                    490,
                    500,
                    510,
                    520,
                    530,
                    540,
                    550,
                    560,
                    570,
                    580,
                    590,
                    600,
                    610,
                    620,
                    630,
                    645,
                    680,
                ),
            ),
        )
    elif method == "cie 1976 ucs":
        ij = Luv_to_uv(XYZ_to_Luv(cmfs.values, illuminant), illuminant)
        labels = cast(
            Tuple,
            optional(
                spectral_locus_labels,
                (
                    420,
                    440,
                    450,
                    460,
                    470,
                    480,
                    490,
                    500,
                    510,
                    520,
                    530,
                    540,
                    550,
                    560,
                    570,
                    580,
                    590,
                    600,
                    610,
                    620,
                    630,
                    645,
                    680,
                ),
            ),
        )

    pl_ij = np.reshape(
        tstack(
            [
                np.linspace(ij[0][0], ij[-1][0], 20),
                np.linspace(ij[0][1], ij[-1][1], 20),
            ]
        ),
        (-1, 1, 2),
    )
    sl_ij = np.copy(ij).reshape(-1, 1, 2)

    purple_line_colours: Optional[Union[ArrayLike, str]]
    if str(spectral_locus_colours).upper() == "RGB":
        spectral_locus_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(cmfs.values), axis=-1
        )

        if method == "cie 1931":
            XYZ = xy_to_XYZ(pl_ij)
        elif method == "cie 1960 ucs":
            XYZ = xy_to_XYZ(UCS_uv_to_xy(pl_ij))
        elif method == "cie 1976 ucs":
            XYZ = xy_to_XYZ(Luv_uv_to_xy(pl_ij))

        purple_line_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(np.reshape(XYZ, (-1, 3))), axis=-1
        )
    else:
        purple_line_colours = spectral_locus_colours

    for slp_ij, slp_colours in (
        (pl_ij, purple_line_colours),
        (sl_ij, spectral_locus_colours),
    ):
        line_collection = LineCollection(
            np.concatenate([slp_ij[:-1], slp_ij[1:]], axis=1),
            colors=slp_colours,
            alpha=spectral_locus_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_scatter,
        )
        axes.add_collection(line_collection)

    wl_ij = dict(zip(wavelengths, ij))
    for label in labels:
        ij_l = wl_ij.get(label)

        if ij_l is None:
            continue

        ij_l = as_float_array([ij_l])
        i, j = tsplit(ij_l)

        index = bisect.bisect(wavelengths, label)
        left = wavelengths[index - 1] if index >= 0 else wavelengths[index]
        right = (
            wavelengths[index] if index < len(wavelengths) else wavelengths[-1]
        )

        dx = wl_ij[right][0] - wl_ij[left][0]
        dy = wl_ij[right][1] - wl_ij[left][1]

        direction = np.array([-dy, dx])

        normal = (
            np.array([-dy, dx])
            if np.dot(
                normalise_vector(ij_l - equal_energy),
                normalise_vector(direction),
            )
            > 0
            else np.array([dy, -dx])
        )
        normal = normalise_vector(normal) / 30

        label_colour = (
            spectral_locus_colours
            if is_string(spectral_locus_colours)
            else spectral_locus_colours[index]  # type: ignore[index]
        )
        axes.plot(
            (i, i + normal[0] * 0.75),
            (j, j + normal[1] * 0.75),
            color=label_colour,
            alpha=spectral_locus_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_line,
        )

        axes.plot(
            i,
            j,
            "o",
            color=label_colour,
            alpha=spectral_locus_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_line,
        )

        axes.text(
            i + normal[0],
            j + normal[1],
            label,
            clip_on=True,
            ha="left" if normal[0] >= 0 else "right",
            va="center",
            fontdict={"size": "small"},
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_label,
        )

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**kwargs)


@override_style()
def plot_chromaticity_diagram_colours(
    samples: Integer = 256,
    diagram_colours: Optional[Union[ArrayLike, str]] = None,
    diagram_opacity: Floating = 1,
    diagram_clipping_path: Optional[ArrayLike] = None,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    method: Union[
        Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"], str
    ] = "CIE 1931",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Chromaticity Diagram* colours according to given method.

    Parameters
    ----------
    samples
        Samples count on one axis when computing the *Chromaticity Diagram*
        colours.
    diagram_colours
        Colours of the *Chromaticity Diagram*, if ``diagram_colours`` is set
        to *RGB*, the colours will be computed according to the corresponding
        coordinates.
    diagram_opacity
        Opacity of the *Chromaticity Diagram*.
    diagram_clipping_path
        Path of points used to clip the *Chromaticity Diagram* colours.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
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
    >>> plot_chromaticity_diagram_colours(diagram_colours='RGB')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_Colours.png
        :align: center
        :alt: plot_chromaticity_diagram_colours
    """

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    diagram_colours = cast(
        ArrayLike,
        optional(
            diagram_colours, HEX_to_RGB(CONSTANTS_COLOUR_STYLE.colour.average)
        ),
    )

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    if method == "cie 1931":
        spectral_locus = XYZ_to_xy(cmfs.values, illuminant)
    elif method == "cie 1960 ucs":
        spectral_locus = UCS_to_uv(XYZ_to_UCS(cmfs.values))
    elif method == "cie 1976 ucs":
        spectral_locus = Luv_to_uv(
            XYZ_to_Luv(cmfs.values, illuminant), illuminant
        )

    use_RGB_diagram_colours = str(diagram_colours).upper() == "RGB"
    if use_RGB_diagram_colours:
        ii, jj = np.meshgrid(
            np.linspace(0, 1, samples), np.linspace(1, 0, samples)
        )
        ij = tstack([ii, jj])

        # NOTE: Various values in the grid have potential to generate
        # zero-divisions, they could be avoided by perturbing the grid, e.g.
        # adding a small epsilon. It was decided instead to disable warnings.
        with suppress_warnings(python_warnings=True):
            if method == "cie 1931":
                XYZ = xy_to_XYZ(ij)
            elif method == "cie 1960 ucs":
                XYZ = xy_to_XYZ(UCS_uv_to_xy(ij))
            elif method == "cie 1976 ucs":
                XYZ = xy_to_XYZ(Luv_uv_to_xy(ij))

        diagram_colours = normalise_maximum(
            XYZ_to_plotting_colourspace(XYZ, illuminant), axis=-1
        )

    polygon = Polygon(
        spectral_locus
        if diagram_clipping_path is None
        else diagram_clipping_path,
        facecolor="none"
        if use_RGB_diagram_colours
        else np.hstack([diagram_colours, diagram_opacity]),
        edgecolor="none"
        if use_RGB_diagram_colours
        else np.hstack([diagram_colours, diagram_opacity]),
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )
    axes.add_patch(polygon)

    if use_RGB_diagram_colours:
        # Preventing bounding box related issues as per
        # https://github.com/matplotlib/matplotlib/issues/10529
        image = axes.imshow(
            diagram_colours,
            interpolation="bilinear",
            extent=(0, 1, 0, 1),
            clip_path=None,
            alpha=diagram_opacity,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
        )
        image.set_clip_path(polygon)

    settings = {"axes": axes}
    settings.update(kwargs)

    return render(**kwargs)


@override_style()
def plot_chromaticity_diagram(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    show_diagram_colours: Boolean = True,
    show_spectral_locus: Boolean = True,
    method: Union[
        Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"], str
    ] = "CIE 1931",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Chromaticity Diagram* according to given method.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus
        Whether to display the *Spectral Locus*.
    method
        *Chromaticity Diagram* method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_spectral_locus`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram_colours`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_chromaticity_diagram()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram.png
        :align: center
        :alt: plot_chromaticity_diagram
    """

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    if show_diagram_colours:
        settings = {"axes": axes, "method": method, "diagram_colours": "RGB"}
        settings.update(kwargs)
        settings["standalone"] = False
        settings["cmfs"] = cmfs

        plot_chromaticity_diagram_colours(**settings)

    if show_spectral_locus:
        settings = {"axes": axes, "method": method}
        settings.update(kwargs)
        settings["standalone"] = False
        settings["cmfs"] = cmfs

        plot_spectral_locus(**settings)

    if method == "cie 1931":
        x_label, y_label = "CIE x", "CIE y"
    elif method == "cie 1960 ucs":
        x_label, y_label = "CIE u", "CIE v"
    elif method == "cie 1976 ucs":
        x_label, y_label = (
            "CIE u'",
            "CIE v'",
        )

    title = f"{method.upper()} Chromaticity Diagram - {cmfs.strict_name}"

    settings.update(
        {
            "axes": axes,
            "standalone": True,
            "bounding_box": (0, 1, 0, 1),
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
    )
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_chromaticity_diagram_CIE1931(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    show_diagram_colours: Boolean = True,
    show_spectral_locus: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus
        Whether to display the *Spectral Locus*.

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
    >>> plot_chromaticity_diagram_CIE1931()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_chromaticity_diagram(
        cmfs, show_diagram_colours, show_spectral_locus, **settings
    )


@override_style()
def plot_chromaticity_diagram_CIE1960UCS(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    show_diagram_colours: Boolean = True,
    show_spectral_locus: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus
        Whether to display the *Spectral Locus*.

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
    >>> plot_chromaticity_diagram_CIE1960UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_chromaticity_diagram(
        cmfs, show_diagram_colours, show_spectral_locus, **settings
    )


@override_style()
def plot_chromaticity_diagram_CIE1976UCS(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    show_diagram_colours: Boolean = True,
    show_spectral_locus: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    show_diagram_colours
        Whether to display the *Chromaticity Diagram* background colours.
    show_spectral_locus
        Whether to display the *Spectral Locus*.

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
    >>> plot_chromaticity_diagram_CIE1976UCS()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_chromaticity_diagram(
        cmfs, show_diagram_colours, show_spectral_locus, **settings
    )


@override_style()
def plot_sds_in_chromaticity_diagram(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable: Callable = plot_chromaticity_diagram,
    method: Union[
        Literal["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"], str
    ] = "CIE 1931",
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distribution chromaticity coordinates into the
    *Chromaticity Diagram* using given method.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
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
        used to control the style of the plotted spectral distributions.
        `plot_kwargs`` can be either a single dictionary applied to all the
        plotted spectral distributions with the same settings or a sequence of
        dictionaries with different settings for each plotted spectral
        distributions. The following special keyword arguments can also be
        used:

        -   ``illuminant`` : The illuminant used to compute the spectral
            distributions colours. The default is the illuminant associated
            with the whitepoint of the default plotting colourspace.
            ``illuminant`` can be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``cmfs`` : The standard observer colour matching functions used for
            computing the spectral distributions colours. ``cmfs`` can be of
            any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``normalise_sd_colours`` : Whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   ``use_sd_colours`` : Whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the
            :func:`matplotlib.pyplot.plot` definition ``color`` argument with
            pre-computed values. The default is *True*.

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
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> annotate_kwargs = [
    ...     {'xytext': (-25, 15), 'arrowprops':{'arrowstyle':'-'}},
    ...     {}
    ... ]
    >>> plot_kwargs = [
    ...     {
    ...         'illuminant': SDS_ILLUMINANTS['E'],
    ...         'markersize' : 15,
    ...         'normalise_sd_colours': True,
    ...         'use_sd_colours': True
    ...     },
    ...     {'illuminant': SDS_ILLUMINANTS['E']},
    ... ]
    >>> plot_sds_in_chromaticity_diagram(
    ...     [A, D65], annotate_kwargs=annotate_kwargs, plot_kwargs=plot_kwargs)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_SDS_In_Chromaticity_Diagram.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram
    """

    method = validate_method(
        method, ["CIE 1931", "CIE 1960 UCS", "CIE 1976 UCS"]
    )

    sds_converted = sds_and_msds_to_sds(sds)

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    settings.update(
        {
            "axes": axes,
            "standalone": False,
            "method": method,
            "cmfs": cmfs,
        }
    )

    chromaticity_diagram_callable(**settings)

    if method == "cie 1931":

        def XYZ_to_ij(XYZ: NDArray) -> NDArray:
            """
            Convert given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return XYZ_to_xy(XYZ)

        bounding_box = (-0.1, 0.9, -0.1, 0.9)
    elif method == "cie 1960 ucs":

        def XYZ_to_ij(XYZ: NDArray) -> NDArray:
            """
            Convert given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return UCS_to_uv(XYZ_to_UCS(XYZ))

        bounding_box = (-0.1, 0.7, -0.2, 0.6)

    elif method == "cie 1976 ucs":

        def XYZ_to_ij(XYZ: NDArray) -> NDArray:
            """
            Convert given *CIE XYZ* tristimulus values to *ij* chromaticity
            coordinates.
            """

            return Luv_to_uv(XYZ_to_Luv(XYZ))

        bounding_box = (-0.1, 0.7, -0.1, 0.7)

    annotate_settings_collection = [
        {
            "annotate": True,
            "xytext": (-50, 30),
            "textcoords": "offset points",
            "arrowprops": CONSTANTS_ARROW_STYLE,
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.midground_annotation,
        }
        for _ in range(len(sds_converted))
    ]

    if annotate_kwargs is not None:
        update_settings_collection(
            annotate_settings_collection, annotate_kwargs, len(sds_converted)
        )

    plot_settings_collection = [
        {
            "color": CONSTANTS_COLOUR_STYLE.colour.brightest,
            "label": f"{sd.strict_name}",
            "marker": "o",
            "markeredgecolor": CONSTANTS_COLOUR_STYLE.colour.dark,
            "markeredgewidth": CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
            "markersize": (
                CONSTANTS_COLOUR_STYLE.geometry.short * 6
                + CONSTANTS_COLOUR_STYLE.geometry.short * 0.75
            ),
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.midground_line,
            "cmfs": cmfs,
            "illuminant": SDS_ILLUMINANTS[
                CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint_name
            ],
            "use_sd_colours": False,
            "normalise_sd_colours": False,
        }
        for sd in sds_converted
    ]

    if plot_kwargs is not None:
        update_settings_collection(
            plot_settings_collection, plot_kwargs, len(sds_converted)
        )

    for i, sd in enumerate(sds_converted):
        plot_settings = plot_settings_collection[i]

        cmfs = cast(
            MultiSpectralDistributions,
            first_item(filter_cmfs(plot_settings.pop("cmfs")).values()),
        )
        illuminant = cast(
            SpectralDistribution,
            first_item(
                filter_illuminants(plot_settings.pop("illuminant")).values()
            ),
        )
        normalise_sd_colours = plot_settings.pop("normalise_sd_colours")
        use_sd_colours = plot_settings.pop("use_sd_colours")

        with domain_range_scale("1"):
            XYZ = sd_to_XYZ(sd, cmfs, illuminant)

        if use_sd_colours:
            if normalise_sd_colours:
                XYZ /= XYZ[..., 1]

            plot_settings["color"] = np.clip(
                XYZ_to_plotting_colourspace(XYZ), 0, 1
            )

        ij = XYZ_to_ij(XYZ)

        axes.plot(ij[0], ij[1], **plot_settings)

        if sd.name is not None and annotate_settings_collection[i]["annotate"]:
            annotate_settings = annotate_settings_collection[i]
            annotate_settings.pop("annotate")

            axes.annotate(sd.name, xy=ij, **annotate_settings)

    settings.update({"standalone": True, "bounding_box": bounding_box})
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1931(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1931: Callable = (
        plot_chromaticity_diagram_CIE1931
    ),
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distribution chromaticity coordinates into the
    *CIE 1931 Chromaticity Diagram*.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
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
        used to control the style of the plotted spectral distributions.
        `plot_kwargs`` can be either a single dictionary applied to all the
        plotted spectral distributions with the same settings or a sequence of
        dictionaries with different settings for each plotted spectral
        distributions. The following special keyword arguments can also be
        used:

        -   ``illuminant`` : The illuminant used to compute the spectral
            distributions colours. The default is the illuminant associated
            with the whitepoint of the default plotting colourspace.
            ``illuminant`` can be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``cmfs`` : The standard observer colour matching functions used for
            computing the spectral distributions colours. ``cmfs`` can be of
            any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``normalise_sd_colours`` : Whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   ``use_sd_colours`` : Whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the
            :func:`matplotlib.pyplot.plot` definition ``color`` argument with
            pre-computed values. The default is *True*.

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
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1931([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1931.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1931
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1931"})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1931,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1960UCS(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1960UCS: Callable = (
        plot_chromaticity_diagram_CIE1960UCS
    ),
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distribution chromaticity coordinates into the
    *CIE 1960 UCS Chromaticity Diagram*.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
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
        used to control the style of the plotted spectral distributions.
        `plot_kwargs`` can be either a single dictionary applied to all the
        plotted spectral distributions with the same settings or a sequence of
        dictionaries with different settings for each plotted spectral
        distributions. The following special keyword arguments can also be
        used:

        -   ``illuminant`` : The illuminant used to compute the spectral
            distributions colours. The default is the illuminant associated
            with the whitepoint of the default plotting colourspace.
            ``illuminant`` can be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``cmfs`` : The standard observer colour matching functions used for
            computing the spectral distributions colours. ``cmfs`` can be of
            any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``normalise_sd_colours`` : Whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   ``use_sd_colours`` : Whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the
            :func:`matplotlib.pyplot.plot` definition ``color`` argument with
            pre-computed values. The default is *True*.

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
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1960UCS([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1960UCS.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1960UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1960 UCS"})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1960UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )


@override_style()
def plot_sds_in_chromaticity_diagram_CIE1976UCS(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromaticity_diagram_callable_CIE1976UCS: Callable = (
        plot_chromaticity_diagram_CIE1976UCS
    ),
    annotate_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distribution chromaticity coordinates into the
    *CIE 1976 UCS Chromaticity Diagram*.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single :class:`colour.MultiSpectralDistributions`
        class instance, a list of :class:`colour.MultiSpectralDistributions`
        class instances or a list of :class:`colour.SpectralDistribution` class
        instances.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
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
        used to control the style of the plotted spectral distributions.
        `plot_kwargs`` can be either a single dictionary applied to all the
        plotted spectral distributions with the same settings or a sequence of
        dictionaries with different settings for each plotted spectral
        distributions. The following special keyword arguments can also be
        used:

        -   ``illuminant`` : The illuminant used to compute the spectral
            distributions colours. The default is the illuminant associated
            with the whitepoint of the default plotting colourspace.
            ``illuminant`` can be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``cmfs`` : The standard observer colour matching functions used for
            computing the spectral distributions colours. ``cmfs`` can be of
            any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``normalise_sd_colours`` : Whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   ``use_sd_colours`` : Whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the
            :func:`matplotlib.pyplot.plot` definition ``color`` argument with
            pre-computed values. The default is *True*.

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
    >>> A = SDS_ILLUMINANTS['A']
    >>> D65 = SDS_ILLUMINANTS['D65']
    >>> plot_sds_in_chromaticity_diagram_CIE1976UCS([A, D65])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_SDS_In_Chromaticity_Diagram_CIE1976UCS.png
        :align: center
        :alt: plot_sds_in_chromaticity_diagram_CIE1976UCS
    """

    settings = dict(kwargs)
    settings.update({"method": "CIE 1976 UCS"})

    return plot_sds_in_chromaticity_diagram(
        sds,
        cmfs,
        chromaticity_diagram_callable_CIE1976UCS,
        annotate_kwargs=annotate_kwargs,
        plot_kwargs=plot_kwargs,
        **settings,
    )
