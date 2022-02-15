"""
Colour Models Volume Plotting
=============================

Defines the colour models volume and gamut plotting objects:

-   :func:`colour.plotting.plot_RGB_colourspaces_gamuts`
-   :func:`colour.plotting.plot_RGB_scatter`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from colour.colorimetry import MultiSpectralDistributions
from colour.geometry import (
    primitive_vertices_cube_mpl,
    primitive_vertices_grid_mpl,
)
from colour.graph import convert
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
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
from colour.models import RGB_Colourspace, RGB_to_XYZ
from colour.models.common import COLOURSPACE_MODELS_AXIS_LABELS
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    colourspace_model_axis_reorder,
    filter_RGB_colourspaces,
    filter_cmfs,
    override_style,
    render,
)
from colour.utilities import (
    Structure,
    as_float_array,
    as_int_scalar,
    as_int_array,
    first_item,
    full,
    is_integer,
    ones,
    optional,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "nadir_grid",
    "RGB_identity_cube",
    "plot_RGB_colourspaces_gamuts",
    "plot_RGB_scatter",
]


def nadir_grid(
    limits: Optional[ArrayLike] = None,
    segments: Integer = 10,
    labels: Optional[Sequence[str]] = None,
    axes: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Return a grid on *CIE xy* plane made of quad geometric elements and its
    associated faces and edges colours. Ticks and labels are added to the
    given axes according to the extended grid settings.

    Parameters
    ----------
    limits
        Extended grid limits.
    segments
        Edge segments count for the extended grid.
    labels
        Axis labels.
    axes
        Axes to add the grid.

    Other Parameters
    ----------------
    grid_edge_alpha
        Grid edge opacity value such as `grid_edge_alpha = 0.5`.
    grid_edge_colours
        Grid edge colours array such as
        `grid_edge_colours = (0.25, 0.25, 0.25)`.
    grid_face_alpha
        Grid face opacity value such as `grid_face_alpha = 0.1`.
    grid_face_colours
        Grid face colours array such as
        `grid_face_colours = (0.25, 0.25, 0.25)`.
    ticks_and_label_location
        Location of the *X* and *Y* axis ticks and labels such as
        `ticks_and_label_location = ('-x', '-y')`.
    x_axis_colour
        *X* axis colour array such as `x_axis_colour = (0.0, 0.0, 0.0, 1.0)`.
    x_label_colour
        *X* axis label colour array such as
        `x_label_colour = (0.0, 0.0, 0.0, 0.85)`.
    x_ticks_colour
        *X* axis ticks colour array such as
        `x_ticks_colour = (0.0, 0.0, 0.0, 0.85)`.
    y_axis_colour
        *Y* axis colour array such as `y_axis_colour = (0.0, 0.0, 0.0, 1.0)`.
    y_label_colour
        *Y* axis label colour array such as
        `y_label_colour = (0.0, 0.0, 0.0, 0.85)`.
    y_ticks_colour
        *Y* axis ticks colour array such as
        `y_ticks_colour = (0.0, 0.0, 0.0, 0.85)`.

    Returns
    -------
    :class:`tuple`
        Grid quads, faces colours, edges colours.

    Examples
    --------
    >>> nadir_grid(segments=1)
    (array([[[-1.   , -1.   ,  0.   ],
            [ 1.   , -1.   ,  0.   ],
            [ 1.   ,  1.   ,  0.   ],
            [-1.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   , -1.   ,  0.   ],
            [ 0.   , -1.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ],
            [-1.   ,  0.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ],
            [ 0.   ,  1.   ,  0.   ],
            [-1.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[ 0.   , -1.   ,  0.   ],
            [ 1.   , -1.   ,  0.   ],
            [ 1.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ]],
    <BLANKLINE>
           [[ 0.   ,  0.   ,  0.   ],
            [ 1.   ,  0.   ,  0.   ],
            [ 1.   ,  1.   ,  0.   ],
            [ 0.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   , -0.001,  0.   ],
            [ 1.   , -0.001,  0.   ],
            [ 1.   ,  0.001,  0.   ],
            [-1.   ,  0.001,  0.   ]],
    <BLANKLINE>
           [[-0.001, -1.   ,  0.   ],
            [ 0.001, -1.   ,  0.   ],
            [ 0.001,  1.   ,  0.   ],
            [-0.001,  1.   ,  0.   ]]]), array([[ 0.25,  0.25,  0.25,  0.1 ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]]), array([[ 0.5 ,  0.5 ,  0.5 ,  0.5 ],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.  ,  0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]]))
    """

    limits = as_float_array(
        cast(ArrayLike, optional(limits, np.array([[-1, 1], [-1, 1]])))
    )
    labels = cast(Sequence, optional(labels, ("x", "y")))

    extent = np.max(np.abs(limits[..., 1] - limits[..., 0]))

    settings = Structure(
        **{
            "grid_face_colours": (0.25, 0.25, 0.25),
            "grid_edge_colours": (0.50, 0.50, 0.50),
            "grid_face_alpha": 0.1,
            "grid_edge_alpha": 0.5,
            "x_axis_colour": (0.0, 0.0, 0.0, 1.0),
            "y_axis_colour": (0.0, 0.0, 0.0, 1.0),
            "x_ticks_colour": (0.0, 0.0, 0.0, 0.85),
            "y_ticks_colour": (0.0, 0.0, 0.0, 0.85),
            "x_label_colour": (0.0, 0.0, 0.0, 0.85),
            "y_label_colour": (0.0, 0.0, 0.0, 0.85),
            "ticks_and_label_location": ("-x", "-y"),
        }
    )
    settings.update(**kwargs)

    # Outer grid.
    quads_g = primitive_vertices_grid_mpl(
        origin=(-extent / 2, -extent / 2),
        width=extent,
        height=extent,
        height_segments=segments,
        width_segments=segments,
    )

    RGB_g = ones((quads_g.shape[0], quads_g.shape[-1]))
    RGB_gf = RGB_g * settings.grid_face_colours
    RGB_gf = np.hstack(
        [RGB_gf, full((RGB_gf.shape[0], 1), settings.grid_face_alpha)]
    )
    RGB_ge = RGB_g * settings.grid_edge_colours
    RGB_ge = np.hstack(
        [RGB_ge, full((RGB_ge.shape[0], 1), settings.grid_edge_alpha)]
    )

    # Inner grid.
    quads_gs = primitive_vertices_grid_mpl(
        origin=(-extent / 2, -extent / 2),
        width=extent,
        height=extent,
        height_segments=segments * 2,
        width_segments=segments * 2,
    )

    RGB_gs = ones((quads_gs.shape[0], quads_gs.shape[-1]))
    RGB_gsf = RGB_gs * 0
    RGB_gsf = np.hstack([RGB_gsf, full((RGB_gsf.shape[0], 1), 0)])
    RGB_gse = np.clip(RGB_gs * settings.grid_edge_colours * 1.5, 0, 1)
    RGB_gse = np.hstack(
        (RGB_gse, full((RGB_gse.shape[0], 1), settings.grid_edge_alpha / 2))
    )

    # Axis.
    thickness = extent / 1000
    quad_x = primitive_vertices_grid_mpl(
        origin=(limits[0, 0], -thickness / 2), width=extent, height=thickness
    )
    RGB_x = ones((quad_x.shape[0], quad_x.shape[-1] + 1))
    RGB_x = RGB_x * settings.x_axis_colour

    quad_y = primitive_vertices_grid_mpl(
        origin=(-thickness / 2, limits[1, 0]), width=thickness, height=extent
    )
    RGB_y = ones((quad_y.shape[0], quad_y.shape[-1] + 1))
    RGB_y = RGB_y * settings.y_axis_colour

    if axes is not None:
        # Ticks.
        x_s = 1 if "+x" in settings.ticks_and_label_location else -1
        y_s = 1 if "+y" in settings.ticks_and_label_location else -1
        for i, axis in enumerate("xy"):
            h_a = "center" if axis == "x" else "left" if x_s == 1 else "right"
            v_a = "center"

            ticks = list(sorted(set(quads_g[..., 0, i])))
            ticks += [ticks[-1] + ticks[-1] - ticks[-2]]
            for tick in ticks:
                x = (
                    limits[1, 1 if x_s == 1 else 0] + (x_s * extent / 25)
                    if i
                    else tick
                )
                y = (
                    tick
                    if i
                    else limits[0, 1 if y_s == 1 else 0] + (y_s * extent / 25)
                )

                tick = as_int_scalar(tick) if is_integer(tick) else tick
                c = settings[f"{axis}_ticks_colour"]

                axes.text(
                    x,
                    y,
                    0,
                    tick,
                    "x",
                    horizontalalignment=h_a,
                    verticalalignment=v_a,
                    color=c,
                    clip_on=True,
                )

        # Labels.
        for i, axis in enumerate("xy"):
            h_a = "center" if axis == "x" else "left" if x_s == 1 else "right"
            v_a = "center"

            x = (
                limits[1, 1 if x_s == 1 else 0] + (x_s * extent / 10)
                if i
                else 0
            )
            y = (
                0
                if i
                else limits[0, 1 if y_s == 1 else 0] + (y_s * extent / 10)
            )

            c = settings[f"{axis}_label_colour"]

            axes.text(
                x,
                y,
                0,
                labels[i],
                "x",
                horizontalalignment=h_a,
                verticalalignment=v_a,
                color=c,
                size=20,
                clip_on=True,
            )

    quads = np.vstack([quads_g, quads_gs, quad_x, quad_y])
    RGB_f = np.vstack([RGB_gf, RGB_gsf, RGB_x, RGB_y])
    RGB_e = np.vstack([RGB_ge, RGB_gse, RGB_x, RGB_y])

    return quads, RGB_f, RGB_e


def RGB_identity_cube(
    width_segments: Integer = 16,
    height_segments: Integer = 16,
    depth_segments: Integer = 16,
    planes: Optional[
        Literal[
            "-x",
            "+x",
            "-y",
            "+y",
            "-z",
            "+z",
            "xy",
            "xz",
            "yz",
            "yx",
            "zx",
            "zy",
        ]
    ] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Return an *RGB* identity cube made of quad geometric elements and its
    associated *RGB* colours.

    Parameters
    ----------
    width_segments
        Cube segments, quad counts along the width.
    height_segments
        Cube segments, quad counts along the height.
    depth_segments
        Cube segments, quad counts along the depth.
    planes
        Grid primitives to include in the cube construction.

    Returns
    -------
    :class:`tuple`
        Cube quads, *RGB* colours.

    Examples
    --------
    >>> vertices, RGB = RGB_identity_cube(1, 1, 1)
    >>> vertices
    array([[[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 0.,  1.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  1.],
            [ 1.,  0.,  1.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 0.,  1.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 1.,  0.,  1.]]])
    >>> RGB
    array([[ 0.5,  0.5,  0. ],
           [ 0.5,  0.5,  1. ],
           [ 0.5,  0. ,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0. ,  0.5,  0.5],
           [ 1. ,  0.5,  0.5]])
    """

    quads = primitive_vertices_cube_mpl(
        width=1,
        height=1,
        depth=1,
        width_segments=width_segments,
        height_segments=height_segments,
        depth_segments=depth_segments,
        planes=planes,
    )
    RGB = np.average(quads, axis=-2)

    return quads, RGB


@override_style()
def plot_RGB_colourspaces_gamuts(
    colourspaces: Union[
        RGB_Colourspace, str, Sequence[Union[RGB_Colourspace, str]]
    ],
    reference_colourspace: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    segments: Integer = 8,
    show_grid: Boolean = True,
    grid_segments: Integer = 10,
    show_spectral_locus: Boolean = False,
    spectral_locus_colour: Optional[Union[ArrayLike, str]] = None,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromatically_adapt: Boolean = False,
    convert_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *RGB* colourspaces gamuts in given reference colourspace.

    Parameters
    ----------
    colourspaces
        *RGB* colourspaces to plot the gamuts. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.filter_RGB_colourspaces` definition.
    reference_colourspace
        Reference colourspace model to plot the gamuts into, see
        :attr:`colour.COLOURSPACE_MODELS` attribute for the list of supported
        colourspace models.
    segments
        Edge segments count for each *RGB* colourspace cubes.
    show_grid
        Whether to show a grid at the bottom of the *RGB* colourspace cubes.
    grid_segments
        Edge segments count for the grid.
    show_spectral_locus
        Whether to show the spectral locus.
    spectral_locus_colour
        Spectral locus colour.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    convert_kwargs
        Keyword arguments for the :func:`colour.convert` definition.

    Other Parameters
    ----------------
    edge_colours
        Edge colours array such as `edge_colours = (None, (0.5, 0.5, 1.0))`.
    edge_alpha
        Edge opacity value such as `edge_alpha = (0.0, 1.0)`.
    face_alpha
        Face opacity value such as `face_alpha = (0.5, 1.0)`.
    face_colours
        Face colours array such as `face_colours = (None, (0.5, 0.5, 1.0))`.
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.volume.nadir_grid`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_gamuts(['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes3DSubplot...>)

    .. image:: ../_static/Plotting_Plot_RGB_Colourspaces_Gamuts.png
        :align: center
        :alt: plot_RGB_colourspaces_gamuts
    """

    colourspaces = cast(
        List[RGB_Colourspace],
        list(filter_RGB_colourspaces(colourspaces).values()),
    )

    convert_kwargs = optional(convert_kwargs, {})

    count_c = len(colourspaces)

    title = (
        f"{', '.join([colourspace.name for colourspace in colourspaces])} "
        f"- {reference_colourspace} Reference Colourspace"
    )

    illuminant = CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint

    convert_settings = {"illuminant": illuminant}
    convert_settings.update(convert_kwargs)

    settings = Structure(
        **{
            "face_colours": [None] * count_c,
            "edge_colours": [None] * count_c,
            "face_alpha": [1] * count_c,
            "edge_alpha": [1] * count_c,
            "title": title,
        }
    )
    settings.update(kwargs)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")

    points = zeros((4, 3))
    if show_spectral_locus:
        cmfs = cast(
            MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
        )
        XYZ = cmfs.values

        points = colourspace_model_axis_reorder(
            convert(XYZ, "CIE XYZ", reference_colourspace, **convert_settings),
            reference_colourspace,
        )

        points[np.isnan(points)] = 0

        c = (
            (0.0, 0.0, 0.0, 0.5)
            if spectral_locus_colour is None
            else spectral_locus_colour
        )

        axes.plot(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color=c,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
        )
        axes.plot(
            (points[-1][0], points[0][0]),
            (points[-1][1], points[0][1]),
            (points[-1][2], points[0][2]),
            color=c,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
        )

    plotting_colourspace = CONSTANTS_COLOUR_STYLE.colour.colourspace

    quads_c: List = []
    RGB_cf: List = []
    RGB_ce: List = []
    for i, colourspace in enumerate(colourspaces):

        if chromatically_adapt and not np.array_equal(
            colourspace.whitepoint, plotting_colourspace.whitepoint
        ):
            colourspace = colourspace.chromatically_adapt(
                plotting_colourspace.whitepoint,
                plotting_colourspace.whitepoint_name,
            )

        quads_cb, RGB = RGB_identity_cube(
            width_segments=segments,
            height_segments=segments,
            depth_segments=segments,
        )

        XYZ = RGB_to_XYZ(
            quads_cb,
            colourspace.whitepoint,
            colourspace.whitepoint,
            colourspace.matrix_RGB_to_XYZ,
        )

        convert_settings = {"illuminant": colourspace.whitepoint}
        convert_settings.update(convert_kwargs)

        quads_c.extend(
            colourspace_model_axis_reorder(
                convert(
                    XYZ, "CIE XYZ", reference_colourspace, **convert_settings
                ),
                reference_colourspace,
            )
        )

        if settings.face_colours[i] is not None:
            RGB = ones(RGB.shape) * settings.face_colours[i]

        RGB_cf.extend(
            np.hstack([RGB, full((RGB.shape[0], 1), settings.face_alpha[i])])
        )

        if settings.edge_colours[i] is not None:
            RGB = ones(RGB.shape) * settings.edge_colours[i]

        RGB_ce.extend(
            np.hstack([RGB, full((RGB.shape[0], 1), settings.edge_alpha[i])])
        )

    quads = as_float_array(quads_c)
    RGB_f = as_float_array(RGB_cf)
    RGB_e = as_float_array(RGB_ce)

    quads[np.isnan(quads)] = 0

    if quads.size != 0:
        for i, axis in enumerate("xyz"):
            min_a = np.minimum(np.min(quads[..., i]), np.min(points[..., i]))
            max_a = np.maximum(np.max(quads[..., i]), np.max(points[..., i]))
            getattr(axes, f"set_{axis}lim")((min_a, max_a))

    labels = np.array(COLOURSPACE_MODELS_AXIS_LABELS[reference_colourspace])[
        as_int_array(
            colourspace_model_axis_reorder([0, 1, 2], reference_colourspace)
        )
    ]
    for i, axis in enumerate("xyz"):
        getattr(axes, f"set_{axis}label")(labels[i])

    if show_grid:
        limits = np.array([[-1.5, 1.5], [-1.5, 1.5]])

        quads_g, RGB_gf, RGB_ge = nadir_grid(
            limits, grid_segments, labels, axes, **settings
        )
        quads = np.vstack([quads_g, quads])
        RGB_f = np.vstack([RGB_gf, RGB_f])
        RGB_e = np.vstack([RGB_ge, RGB_e])

    collection = Poly3DCollection(quads)
    collection.set_facecolors(RGB_f)
    collection.set_edgecolors(RGB_e)

    axes.add_collection3d(collection)

    settings.update(
        {"axes": axes, "axes_visible": False, "camera_aspect": "equal"}
    )
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_scatter(
    RGB: ArrayLike,
    colourspace: Union[
        RGB_Colourspace, str, Sequence[Union[RGB_Colourspace, str]]
    ],
    reference_colourspace: Union[
        Literal[
            "CAM02LCD",
            "CAM02SCD",
            "CAM02UCS",
            "CAM16LCD",
            "CAM16SCD",
            "CAM16UCS",
            "CIE XYZ",
            "CIE xyY",
            "CIE Lab",
            "CIE Luv",
            "CIE UCS",
            "CIE UVW",
            "DIN99",
            "Hunter Lab",
            "Hunter Rdab",
            "ICaCb",
            "ICtCp",
            "IPT",
            "IgPgTg",
            "Jzazbz",
            "OSA UCS",
            "Oklab",
            "hdr-CIELAB",
            "hdr-IPT",
        ],
        str,
    ] = "CIE xyY",
    colourspaces: Optional[
        Union[RGB_Colourspace, str, Sequence[Union[RGB_Colourspace, str]]]
    ] = None,
    segments: Integer = 8,
    show_grid: Boolean = True,
    grid_segments: Integer = 10,
    show_spectral_locus: Boolean = False,
    spectral_locus_colour: Optional[Union[ArrayLike, str]] = None,
    points_size: Floating = 12,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    chromatically_adapt: Boolean = False,
    convert_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *RGB* colourspace array in a scatter plot.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        *RGB* colourspace of the *RGB* array. ``colourspace`` can be of any
        type or form supported by the
        :func:`colour.plotting.filter_RGB_colourspaces` definition.
    reference_colourspace
        Reference colourspace model to plot the gamuts into, see
        :attr:`colour.COLOURSPACE_MODELS` attribute for the list of supported
        colourspace models.
    colourspaces
        *RGB* colourspaces to plot the gamuts. ``colourspaces`` elements
        can be of any type or form supported by the
        :func:`colour.plotting.filter_RGB_colourspaces` definition.
    segments
        Edge segments count for each *RGB* colourspace cubes.
    show_grid
        Whether to show a grid at the bottom of the *RGB* colourspace cubes.
    grid_segments
        Edge segments count for the grid.
    show_spectral_locus
        Whether to show the spectral locus.
    spectral_locus_colour
        Spectral locus colour.
    points_size
        Scatter points size.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral locus boundaries. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    chromatically_adapt
        Whether to chromatically adapt the *RGB* colourspaces given in
        ``colourspaces`` to the whitepoint of the default plotting colourspace.
    convert_kwargs
        Keyword arguments for the :func:`colour.convert` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_RGB_colourspaces_gamuts`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_scatter(RGB, 'ITU-R BT.709')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes3DSubplot...>)

    .. image:: ../_static/Plotting_Plot_RGB_Scatter.png
        :align: center
        :alt: plot_RGB_scatter
    """

    colourspace = cast(
        RGB_Colourspace,
        first_item(filter_RGB_colourspaces(colourspace).values()),
    )
    colourspaces = cast(List[str], optional(colourspaces, [colourspace.name]))

    convert_kwargs = optional(convert_kwargs, {})

    count_c = len(colourspaces)
    settings = Structure(
        **{
            "face_colours": [None] * count_c,
            "edge_colours": [(0.25, 0.25, 0.25)] * count_c,
            "face_alpha": [0.0] * count_c,
            "edge_alpha": [0.1] * count_c,
        }
    )
    settings.update(kwargs)
    settings["standalone"] = False

    plot_RGB_colourspaces_gamuts(
        colourspaces=colourspaces,
        reference_colourspace=reference_colourspace,
        segments=segments,
        show_grid=show_grid,
        grid_segments=grid_segments,
        show_spectral_locus=show_spectral_locus,
        spectral_locus_colour=spectral_locus_colour,
        cmfs=cmfs,
        chromatically_adapt=chromatically_adapt,
        **settings,
    )

    XYZ = RGB_to_XYZ(
        RGB,
        colourspace.whitepoint,
        colourspace.whitepoint,
        colourspace.matrix_RGB_to_XYZ,
    )

    convert_settings = {"illuminant": colourspace.whitepoint}
    convert_settings.update(convert_kwargs)

    points = colourspace_model_axis_reorder(
        convert(XYZ, "CIE XYZ", reference_colourspace, **convert_settings),
        reference_colourspace,
    )

    axes = plt.gca()
    axes.scatter(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        color=np.reshape(RGB, (-1, 3)),
        s=points_size,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_scatter,
    )

    settings.update({"axes": axes, "standalone": True})
    settings.update(kwargs)

    return render(**settings)
