"""
Automatic Colour Conversion Graph Plotting
==========================================

Define the automatic colour conversion graph plotting objects:

-   :func:`colour.plotting.plot_automatic_colour_conversion_graph`
"""

from __future__ import annotations

import os

import colour
from colour.graph import (
    CONVERSION_GRAPH_NODE_LABELS,
    describe_conversion_path,
)
from colour.hints import Literal, cast
from colour.utilities import required, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_automatic_colour_conversion_graph",
]


@required("Pydot")
@required("NetworkX")
def plot_automatic_colour_conversion_graph(
    filename: str,
    prog: Literal["circo", "dot", "fdp", "neato", "nop", "twopi"] | str = "fdp",
) -> Dot:  # pyright: ignore  # noqa: F821  # pragma: no cover
    """
    Plot *Colour* automatic colour conversion graph using
    `Graphviz <https://www.graphviz.org>`__ and
    `pyraphviz <https://pygraphviz.github.io>`__.

    Parameters
    ----------
    filename
        Filename to use to save the image.
    prog
        *Graphviz* layout method.

    Returns
    -------
    :class:`pydot.Dot`
        *Pydot* graph.

    Notes
    -----
    -   This definition does not directly plot the *Colour* automatic colour
        conversion graph but instead write it to an image.

    Examples
    --------
    >>> import tempfile
    >>> import colour
    >>> from colour import read_image
    >>> from colour.plotting import plot_image
    >>> filename = "{0}.png".format(tempfile.mkstemp()[-1])
    >>> _ = plot_automatic_colour_conversion_graph(filename, "dot")
    ... # doctest: +SKIP
    >>> plot_image(read_image(filename))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Colour_Automatic_Conversion_Graph.png
        :align: center
        :alt: plot_automatic_colour_conversion_graph
    """

    import networkx as nx

    prog = validate_method(
        prog,
        ("circo", "dot", "fdp", "neato", "nop", "twopi"),
        '"{0}" program is invalid, it must be one of {1}!',
    )

    # TODO: Investigate API to trigger the conversion graph build.
    describe_conversion_path("RGB", "RGB", print_callable=lambda x: x)

    dot = nx.drawing.nx_pydot.to_pydot(cast(nx.DiGraph, colour.graph.CONVERSION_GRAPH))

    for node in dot.get_nodes():
        label = CONVERSION_GRAPH_NODE_LABELS.get(node.get_name())

        if label is None:
            continue

        node.set_label(label)
        node.set_style("filled")
        node.set_shape("circle")
        node.set_color("#2196F3FF")
        node.set_fillcolor("#2196F370")
        node.set_fontname("Helvetica")
        node.set_fontcolor("#263238")

    for name in ("CIE XYZ", "RGB", "Spectral Distribution"):
        node = next(iter(dot.get_node(name.lower())))

        node.set_shape("doublecircle")
        node.set_color("#673AB7FF")
        node.set_fillcolor("#673AB770")
        node.set_fontsize(30)

    for name in (
        "ATD95",
        "CAM16",
        "CIECAM02",
        "Hellwig 2022",
        "Hunt",
        "Kim 2009",
        "LLAB",
        "Nayatani95",
        "RLAB",
        "ZCAM",
    ):
        node = next(iter(dot.get_node(name.lower())))

        node.set_color("#00BCD4FF")
        node.set_fillcolor("#00BCD470")

    for edge in dot.get_edges():
        edge.set_color("#26323870")

    file_format = os.path.splitext(filename)[-1][1:]
    write_method = getattr(dot, f"write_{file_format}")
    write_method(filename, prog=prog, f=file_format)

    return dot
