# -*- coding: utf-8 -*-
"""
Automatic Colour Conversion Graph Plotting
==========================================

Defines the automatic colour conversion graph plotting objects:

-   :func:`colour.plotting.plot_automatic_colour_conversion_graph`
"""

from __future__ import division

from colour.graph import CONVERSION_GRAPH, CONVERSION_GRAPH_NODE_LABELS
from colour.utilities import is_networkx_installed

if is_networkx_installed():  # pragma: no cover
    import networkx as nx

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['plot_automatic_colour_conversion_graph']


def plot_automatic_colour_conversion_graph(filename, prog='fdp', args=''):
    """
    Plots *Colour* automatic colour conversion graph using
    `Graphviz <https://www.graphviz.org/>`_ and
    `pyraphviz <https://pygraphviz.github.io>`_.

    Parameters
    ----------
    filename : unicode
        Filename to use to save the image.
    prog : unicode, optional
        {'neato', 'dot', 'twopi', 'circo', 'fdp', 'nop'},
        *Graphviz* layout method.
    args : unicode, optional
         Additional arguments for *Graphviz*.

    Returns
    -------
    AGraph
        *Pyraphviz* graph.

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
    >>> filename = '{0}.png'.format(tempfile.mkstemp()[-1])
    >>> _ = plot_automatic_colour_conversion_graph(filename, 'dot')
    ... # doctest: +SKIP
    >>> plot_image(read_image(filename))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Colour_Automatic_Conversion_Graph.png
        :align: center
        :alt: plot_automatic_colour_conversion_graph
    """

    if is_networkx_installed(raise_exception=True):  # pragma: no cover
        agraph = nx.nx_agraph.to_agraph(CONVERSION_GRAPH)

        for node in agraph.nodes():
            node.attr.update(label=CONVERSION_GRAPH_NODE_LABELS[node.name])

        agraph.node_attr.update(
            style='filled',
            shape='circle',
            color='#2196F3FF',
            fillcolor='#2196F370',
            fontname='Helvetica',
            fontcolor="#263238")
        agraph.edge_attr.update(color='#26323870')
        for node in ('CIE XYZ', 'RGB', 'Spectral Distribution'):
            agraph.get_node(node.lower()).attr.update(
                shape='doublecircle',
                color='#673AB7FF',
                fillcolor='#673AB770',
                fontsize=30)
        for node in ('ATD95', 'CAM16', 'CIECAM02', 'Hunt', 'LLAB',
                     'Nayatani95', 'RLAB'):
            agraph.get_node(node.lower()).attr.update(
                color='#00BCD4FF', fillcolor='#00BCD470')

        agraph.draw(filename, prog=prog, args=args)

        return agraph
