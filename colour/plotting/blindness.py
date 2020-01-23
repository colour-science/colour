# -*- coding: utf-8 -*-
"""
Colour Blindness Plotting
=========================

Defines the colour blindness plotting objects:

-   :func:`plot_cvd_simulation_Machado2009`
"""

from __future__ import division

from colour.blindness import cvd_matrix_Machado2009
from colour.plotting import COLOUR_STYLE_CONSTANTS, plot_image, override_style
from colour.utilities import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['plot_cvd_simulation_Machado2009']


@override_style()
def plot_cvd_simulation_Machado2009(RGB,
                                    deficiency='Protanomaly',
                                    severity=0.5,
                                    M_a=None,
                                    **kwargs):
    """
    Performs colour vision deficiency simulation on given *RGB* colourspace
    array using *Machado et al. (2009)* model.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    deficiency : unicode, optional
        {'Protanomaly', 'Deuteranomaly', 'Tritanomaly'}
        Colour blindness / vision deficiency type.
    severity : numeric, optional
        Severity of the colour vision deficiency in domain [0, 1].
    M_a : array_like, optional
        Anomalous trichromacy matrix to use instead of Machado (2010)
        pre-computed matrix.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.plot_image`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Notes
    -----
    -  Input *RGB* array is expected to be linearly encoded.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.random.rand(32, 32, 3)
    >>> plot_cvd_simulation_Machado2009(RGB)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.AxesSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_CVD_Simulation_Machado2009.png
        :align: center
        :alt: plot_cvd_simulation_Machado2009
    """

    if M_a is None:
        M_a = cvd_matrix_Machado2009(deficiency, severity)

    text = 'Deficiency: {0} - Severity: {1}'.format(deficiency, severity)

    settings = {'text_parameters': {'text': None if M_a is None else text}}
    settings.update(kwargs)

    return plot_image(
        COLOUR_STYLE_CONSTANTS.colour.colourspace.cctf_encoding(
            dot_vector(M_a, RGB)), **settings)
