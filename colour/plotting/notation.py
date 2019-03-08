# -*- coding: utf-8 -*-
"""
Colour Notation Systems Plotting
================================

Defines the colour notation systems plotting objects:

-   :func:`colour.plotting.plot_single_munsell_value_function`
-   :func:`colour.plotting.plot_multi_munsell_value_functions`
"""

from __future__ import division

import numpy as np

from colour.notation import MUNSELL_VALUE_METHODS
from colour.plotting import (filter_passthrough, plot_multi_functions,
                             override_style)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'plot_single_munsell_value_function', 'plot_multi_munsell_value_functions'
]


@override_style()
def plot_single_munsell_value_function(function='ASTM D1535-08', **kwargs):
    """
    Plots given *Lightness* function.

    Parameters
    ----------
    function : unicode, optional
        *Munsell* value function to plot.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_single_munsell_value_function('ASTM D1535-08')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Single_Munsell_Value_Function.png
        :align: center
        :alt: plot_single_munsell_value_function
    """

    settings = {'title': '{0} - Munsell Value Function'.format(function)}
    settings.update(kwargs)

    return plot_multi_munsell_value_functions((function, ), **settings)


@override_style()
def plot_multi_munsell_value_functions(functions=None, **kwargs):
    """
    Plots given *Munsell* value functions.

    Parameters
    ----------
    functions : array_like, optional
        *Munsell* value functions to plot.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_munsell_value_functions(['ASTM D1535-08', 'McCamy 1987'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Multi_Munsell_Value_Functions.png
        :align: center
        :alt: plot_multi_munsell_value_functions
    """

    if functions is None:
        functions = ('ASTM D1535-08', 'McCamy 1987')

    functions = filter_passthrough(MUNSELL_VALUE_METHODS, functions)

    settings = {
        'bounding_box': (0, 100, 0, 10),
        'legend': True,
        'title': '{0} - Munsell Functions'.format(', '.join(functions)),
        'x_label': 'Luminance Y',
        'y_label': 'Munsell Value V',
    }
    settings.update(kwargs)

    return plot_multi_functions(
        functions, samples=np.linspace(0, 100, 1000), **settings)
