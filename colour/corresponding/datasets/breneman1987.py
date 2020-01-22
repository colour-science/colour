# -*- coding: utf-8 -*-
"""
Breneman Corresponding Chromaticities Dataset
=============================================

Defines *Breneman (1987)* results for corresponding chromaticities experiments.

See Also
--------
`Corresponding Chromaticities Prediction Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/corresponding/prediction.ipynb>`_

References
----------
-   :cite:`Breneman1987b` : Breneman, E. J. (1987). Corresponding
    chromaticities for different states of adaptation to complex visual fields.
    Journal of the Optical Society of America A, 4(6), 1115.
    doi:10.1364/JOSAA.4.001115
"""

from __future__ import division, unicode_literals

import numpy as np

from collections import namedtuple

from colour.utilities.documentation import DocstringDict

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'BrenemanExperimentResult', 'PrimariesChromaticityCoordinates',
    'BRENEMAN_EXPERIMENT_1_RESULTS', 'BRENEMAN_EXPERIMENT_2_RESULTS',
    'BRENEMAN_EXPERIMENT_3_RESULTS', 'BRENEMAN_EXPERIMENT_4_RESULTS',
    'BRENEMAN_EXPERIMENT_5_RESULTS', 'BRENEMAN_EXPERIMENT_6_RESULTS',
    'BRENEMAN_EXPERIMENT_7_RESULTS', 'BRENEMAN_EXPERIMENT_10_RESULTS',
    'BRENEMAN_EXPERIMENT_8_RESULTS', 'BRENEMAN_EXPERIMENT_9_RESULTS',
    'BRENEMAN_EXPERIMENT_11_RESULTS', 'BRENEMAN_EXPERIMENT_12_RESULTS',
    'BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES', 'BRENEMAN_EXPERIMENTS'
]


class BrenemanExperimentResult(
        namedtuple('BrenemanExperimentResult',
                   ('name', 'uv_t', 'uv_m', 's_uv', 'd_uv_i', 'd_uv_g'))):
    """
    Experiment result.

    Parameters
    ----------
    name : unicode
        Test colour name.
    uv_t : numeric
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    uv_m : array_like, (2,)
        Chromaticity coordinates :math:`uv_m^p` of matching colour.
    s_uv : array_like, (2,), optional
        Interobserver variation (:math:`x10^3`) :math:`\\sigma_uv^p`.
    d_uv_i : array_like, (2,), optional
        Deviation of individual linear transformation (:math:`x10^3`)
        :math:`\\delta_uv_i^p`.
    d_uv_g : array_like, (2,), optional
        Deviation of individual linear transformation (:math:`x10^3`)
        :math:`\\delta_uv_g^p`.
    """

    def __new__(cls, name, uv_t, uv_m, s_uv=None, d_uv_i=None, d_uv_g=None):
        """
        Returns a new instance of the
        :class:`colour.corresponding.datasets.corresponding_chromaticities.\
BrenemanExperimentResult` class.
        """

        return super(BrenemanExperimentResult, cls).__new__(
            cls, name, np.array(uv_t), np.array(uv_m), np.array(s_uv),
            np.array(d_uv_i), np.array(d_uv_g))


class PrimariesChromaticityCoordinates(
        namedtuple(
            'PrimariesChromaticityCoordinates',
            ('experiment', 'illuminants', 'Y', 'P_uvp', 'D_uvp', 'T_uvp'))):
    """
    Chromaticity coordinates of primaries.

    Parameters
    ----------
    experiment : integer
        Experiment.
    illuminants : array_like, (2,)
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    Y : numeric
        White luminance :math:`Y` in :math:`cd/m^2`.
    P_uvp : numeric
        Chromaticity coordinates :math:`uv^p` of primary :math:`P`.
    D_uvp : numeric
        Chromaticity coordinates :math:`uv^p` of primary :math:`D`.
    T_uvp : numeric
        Chromaticity coordinates :math:`uv^p` of primary :math:`T`.
    """

    def __new__(cls,
                experiment,
                illuminants,
                Y,
                P_uvp=None,
                D_uvp=None,
                T_uvp=None):
        """
        Returns a new instance of the
        :class:`colour.corresponding.datasets.corresponding_chromaticities.\
PrimariesChromaticityCoordinates` class.
        """

        return super(PrimariesChromaticityCoordinates, cls).__new__(
            cls, experiment, np.array(illuminants), np.array(Y),
            np.array(P_uvp), np.array(D_uvp), np.array(T_uvp))


# yapf: disable
BRENEMAN_EXPERIMENT_1_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.259, 0.526), (0.200, 0.475)),
    BrenemanExperimentResult(
        'Gray',
        (0.259, 0.524), (0.199, 0.487), (4, 4), (2, 3), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.459, 0.522), (0.420, 0.509), (19, 4), (-10, -7), (-19, -3)),
    BrenemanExperimentResult(
        'Skin',
        (0.307, 0.526), (0.249, 0.497), (7, 4), (-1, 1), (-6, -1)),
    BrenemanExperimentResult(
        'Orange',
        (0.360, 0.544), (0.302, 0.548), (12, 1), (1, -2), (-7, -6)),
    BrenemanExperimentResult(
        'Brown',
        (0.350, 0.541), (0.290, 0.537), (11, 4), (3, 0), (-5, -3)),
    BrenemanExperimentResult(
        'Yellow',
        (0.318, 0.550), (0.257, 0.554), (8, 2), (0, 2), (-5, -5)),
    BrenemanExperimentResult(
        'Foliage',
        (0.258, 0.542), (0.192, 0.529), (4, 6), (3, 2), (3, -6)),
    BrenemanExperimentResult(
        'Green',
        (0.193, 0.542), (0.129, 0.521), (7, 5), (3, 2), (9, -7)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.180, 0.516), (0.133, 0.469), (4, 6), (-3, -2), (2, -5)),
    BrenemanExperimentResult(
        'Blue',
        (0.186, 0.445), (0.158, 0.340), (13, 33), (2, 7), (1, 13)),
    BrenemanExperimentResult(
        'Sky',
        (0.226, 0.491), (0.178, 0.426), (3, 14), (1, -3), (0, -1)),
    BrenemanExperimentResult(
        'Purple',
        (0.278, 0.456), (0.231, 0.365), (4, 25), (0, 2), (-5, 7)))
# yapf: enable
"""
*Breneman (1987)* experiment 1 results.

BRENEMAN_EXPERIMENT_1_RESULTS : tuple

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 1500 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_2_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.222, 0.521), (0.204, 0.479)),
    BrenemanExperimentResult(
        'Gray',
        (0.227, 0.517), (0.207, 0.486), (2, 5), (-1, 0), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.464, 0.520), (0.449, 0.511), (22, 3), (-8, -8), (-7, -2)),
    BrenemanExperimentResult(
        'Skin',
        (0.286, 0.526), (0.263, 0.505), (7, 2), (0, -1), (0, -1)),
    BrenemanExperimentResult(
        'Orange',
        (0.348, 0.546), (0.322, 0.545), (13, 3), (3, -1), (3, -2)),
    BrenemanExperimentResult(
        'Brown',
        (0.340, 0.543), (0.316, 0.537), (11, 3), (1, 1), (0, 0)),
    BrenemanExperimentResult(
        'Yellow',
        (0.288, 0.554), (0.265, 0.553), (5, 2), (-2, 2), (-1, -2)),
    BrenemanExperimentResult(
        'Foliage',
        (0.244, 0.547), (0.221, 0.538), (4, 3), (-2, 1), (0, -3)),
    BrenemanExperimentResult(
        'Green',
        (0.156, 0.548), (0.135, 0.532), (4, 3), (-1, 3), (3, -4)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.159, 0.511), (0.145, 0.472), (9, 7), (-1, 2), (2, 1)),
    BrenemanExperimentResult(
        'Blue',
        (0.160, 0.406), (0.163, 0.331), (23, 31), (2, -3), (-1, 3)),
    BrenemanExperimentResult(
        'Sky',
        (0.190, 0.481), (0.176, 0.431), (5, 24), (2, -2), (2, 0)),
    BrenemanExperimentResult(
        'Purple',
        (0.258, 0.431), (0.244, 0.349), (4, 19), (-3, 13), (-4, 19)))
# yapf: enable
"""
*Breneman (1987)* experiment 2 results.

BRENEMAN_EXPERIMENT_2_RESULTS : tuple

Notes
-----
-   Illuminants : *Projector*, *D55*
-   White Luminance : 1500 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_3_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.223, 0.521), (0.206, 0.478)),
    BrenemanExperimentResult(
        'Gray',
        (0.228, 0.517), (0.211, 0.494), (1, 3), (0, 2), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.462, 0.519), (0.448, 0.505), (11, 4), (-3, 6), (-4, 6)),
    BrenemanExperimentResult(
        'Skin',
        (0.285, 0.524), (0.267, 0.507), (6, 3), (-1, 1), (-2, 1)),
    BrenemanExperimentResult(
        'Orange',
        (0.346, 0.546), (0.325, 0.541), (11, 3), (1, -2), (2, 3)),
    BrenemanExperimentResult(
        'Brown',
        (0.338, 0.543), (0.321, 0.532), (9, 6), (-3, 2), (-3, 7)),
    BrenemanExperimentResult(
        'Yellow',
        (0.287, 0.554), (0.267, 0.548), (4, 5), (1, -2), (0, 5)),
    BrenemanExperimentResult(
        'Foliage',
        (0.244, 0.547), (0.226, 0.531), (3, 6), (-1, 3), (-2, 8)),
    BrenemanExperimentResult(
        'Green',
        (0.157, 0.548), (0.141, 0.528), (9, 6), (2, 2), (0, 6)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.160, 0.510), (0.151, 0.486), (8, 5), (-2, -1), (-2, -5)),
    BrenemanExperimentResult(
        'Blue',
        (0.162, 0.407), (0.158, 0.375), (6, 7), (1, -6), (4, -23)),
    BrenemanExperimentResult(
        'Sky',
        (0.191, 0.482), (0.179, 0.452), (4, 5), (0, 1), (1, -7)),
    BrenemanExperimentResult(
        'Purple',
        (0.258, 0.432), (0.238, 0.396), (4, 8), (5, 3), (4, -11)))
# yapf: enable
"""
*Breneman (1987)* experiment 3 results.

BRENEMAN_EXPERIMENT_3_RESULTS : tuple

Notes
-----
-   Illuminants : *Projector*, *D55*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_4_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.258, 0.523), (0.199, 0.467)),
    BrenemanExperimentResult(
        'Gray',
        (0.257, 0.524), (0.205, 0.495), (2, 2), (0, 4), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.460, 0.521), (0.416, 0.501), (11, 6), (-6, 4), (-6, 9)),
    BrenemanExperimentResult(
        'Skin',
        (0.308, 0.526), (0.253, 0.503), (7, 3), (-1, 1), (-1, 0)),
    BrenemanExperimentResult(
        'Orange',
        (0.360, 0.544), (0.303, 0.541), (14, 5), (1, -4), (1, 2)),
    BrenemanExperimentResult(
        'Brown',
        (0.350, 0.541), (0.296, 0.527), (11, 7), (-2, 4), (-3, 9)),
    BrenemanExperimentResult(
        'Yellow',
        (0.317, 0.550), (0.260, 0.547), (9, 5), (1, -3), (0, 3)),
    BrenemanExperimentResult(
        'Foliage',
        (0.258, 0.543), (0.203, 0.520), (4, 6), (0, 8), (0, 9)),
    BrenemanExperimentResult(
        'Green',
        (0.193, 0.543), (0.142, 0.516), (6, 9), (3, 8), (2, 6)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.180, 0.516), (0.140, 0.484), (9, 5), (-2, -1), (-1, -9)),
    BrenemanExperimentResult(
        'Blue',
        (0.185, 0.445), (0.151, 0.394), (8, 10), (2, -8), (8, -24)),
    BrenemanExperimentResult(
        'Sky',
        (0.225, 0.490), (0.180, 0.448), (4, 8), (1, -1), (3, -11)),
    BrenemanExperimentResult(
        'Purple',
        (0.278, 0.455), (0.229, 0.388), (6, 14), (1, 12), (3, 0)))
# yapf: enable
"""
*Breneman (1987)* experiment 4 results.

BRENEMAN_EXPERIMENT_4_RESULTS : tuple

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_5_RESULTS = (
    BrenemanExperimentResult(
        'Gray',
        (0.028, 0.480), (0.212, 0.491), (2, 2)),
    BrenemanExperimentResult(
        'Red',
        (0.449, 0.512), (0.408, 0.514), (11, 5)),
    BrenemanExperimentResult(
        'Skin',
        (0.269, 0.505), (0.262, 0.511), (4, 2)),
    BrenemanExperimentResult(
        'Orange',
        (0.331, 0.548), (0.303, 0.545), (4, 3)),
    BrenemanExperimentResult(
        'Brown',
        (0.322, 0.541), (0.303, 0.538), (4, 4)),
    BrenemanExperimentResult(
        'Yellow',
        (0.268, 0.555), (0.264, 0.550), (3, 2)),
    BrenemanExperimentResult(
        'Foliage',
        (0.224, 0.538), (0.227, 0.535), (3, 3)),
    BrenemanExperimentResult(
        'Green',
        (0.134, 0.531), (0.159, 0.530), (9, 3)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.145, 0.474), (0.165, 0.490), (8, 3)),
    BrenemanExperimentResult(
        'Blue',
        (0.163, 0.329), (0.173, 0.378), (7, 12)),
    BrenemanExperimentResult(
        'Sky',
        (0.179, 0.438), (0.189, 0.462), (5, 4)),
    BrenemanExperimentResult(
        'Purple',
        (0.245, 0.364), (0.239, 0.401), (4, 16)))
# yapf: enable
"""
*Breneman (1987)* experiment 5 results.

BRENEMAN_EXPERIMENT_5_RESULTS : tuple

Notes
-----
-   Effective White Levels : 130 and 2120 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_6_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.257, 0.525), (0.201, 0.482)),
    BrenemanExperimentResult(
        'Gray',
        (0.267, 0.521), (0.207, 0.485), (5, 3), (-1, 0), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.457, 0.521), (0.398, 0.516), (9, 4), (-2, -5), (1, -9)),
    BrenemanExperimentResult(
        'Skin',
        (0.316, 0.526), (0.253, 0.503), (5, 3), (-3, -2), (-1, -3)),
    BrenemanExperimentResult(
        'Orange',
        (0.358, 0.545), (0.287, 0.550), (7, 3), (3, 0), (7, -6)),
    BrenemanExperimentResult(
        'Brown',
        (0.350, 0.541), (0.282, 0.540), (6, 3), (-1, 0), (2, -5)),
    BrenemanExperimentResult(
        'Yellow',
        (0.318, 0.551), (0.249, 0.556), (7, 2), (-1, 1), (2, -5)),
    BrenemanExperimentResult(
        'Foliage',
        (0.256, 0.547), (0.188, 0.537), (5, 4), (3, 1), (4, -2)),
    BrenemanExperimentResult(
        'Green',
        (0.193, 0.542), (0.133, 0.520), (13, 3), (5, -2), (5, -4)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.180, 0.516), (0.137, 0.466), (12, 10), (0, 0), (-2, 2)),
    BrenemanExperimentResult(
        'Blue',
        (0.186, 0.445), (0.156, 0.353), (12, 45), (6, 1), (2, 6)),
    BrenemanExperimentResult(
        'Sky',
        (0.225, 0.492), (0.178, 0.428), (6, 14), (1, -1), (-1, 3)),
    BrenemanExperimentResult(
        'Purple',
        (0.276, 0.456), (0.227, 0.369), (6, 27), (-2, 4), (-3, 9)))
# yapf: enable
"""
*Breneman (1987)* experiment 6 results.

BRENEMAN_EXPERIMENT_6_RESULTS : tuple

Notes
-----
-   Illuminants : *A*, *D55*
-   White Luminance : 11100 :math:`cd/m^2`
-   Observers Count : 8
"""

# yapf: disable
BRENEMAN_EXPERIMENT_7_RESULTS = (
    BrenemanExperimentResult(
        'Gray',
        (0.208, 0.481), (0.211, 0.486), (2, 3)),
    BrenemanExperimentResult(
        'Red',
        (0.448, 0.512), (0.409, 0.516), (9, 2)),
    BrenemanExperimentResult(
        'Skin',
        (0.269, 0.505), (0.256, 0.506), (4, 3)),
    BrenemanExperimentResult(
        'Orange',
        (0.331, 0.549), (0.305, 0.547), (5, 4)),
    BrenemanExperimentResult(
        'Brown',
        (0.322, 0.541), (0.301, 0.539), (5, 2)),
    BrenemanExperimentResult(
        'Yellow',
        (0.268, 0.555), (0.257, 0.552), (3, 4)),
    BrenemanExperimentResult(
        'Foliage',
        (0.225, 0.538), (0.222, 0.536), (3, 2)),
    BrenemanExperimentResult(
        'Green',
        (0.135, 0.531), (0.153, 0.529), (8, 2)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.145, 0.475), (0.160, 0.484), (3, 5)),
    BrenemanExperimentResult(
        'Blue',
        (0.163, 0.331), (0.171, 0.379), (4, 11)),
    BrenemanExperimentResult(
        'Sky',
        (0.179, 0.438), (0.187, 0.452), (4, 7)),
    BrenemanExperimentResult(
        'Purple',
        (0.245, 0.365), (0.240, 0.398), (4, 10)))
# yapf: enable
"""
*Breneman (1987)* experiment 7 results.

BRENEMAN_EXPERIMENT_7_RESULTS : tuple

Notes
-----
-   Effective White Levels : 850 and 11100 :math:`cd/m^2`
-   Observers Count : 8
"""

# yapf: disable
BRENEMAN_EXPERIMENT_8_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.258, 0.524), (0.195, 0.469)),
    BrenemanExperimentResult(
        'Gray',
        (0.257, 0.525), (0.200, 0.494), (2, 3), (1, 2), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.458, 0.522), (0.410, 0.508), (12, 4), (-3, 5), (-7, 2)),
    BrenemanExperimentResult(
        'Skin',
        (0.308, 0.526), (0.249, 0.502), (6, 2), (-1, 1), (-3, -1)),
    BrenemanExperimentResult(
        'Orange',
        (0.359, 0.545), (0.299, 0.545), (12, 4), (0, -2), (-3, 0)),
    BrenemanExperimentResult(
        'Brown',
        (0.349, 0.540), (0.289, 0.532), (10, 4), (0, 1), (-2, 2)),
    BrenemanExperimentResult(
        'Yellow',
        (0.317, 0.550), (0.256, 0.549), (9, 5), (0, -3), (-3, 1)),
    BrenemanExperimentResult(
        'Foliage',
        (0.260, 0.545), (0.198, 0.529), (5, 5), (3, 1), (0, 3)),
    BrenemanExperimentResult(
        'Green',
        (0.193, 0.543), (0.137, 0.520), (9, 5), (3, 0), (2, 1)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.182, 0.516), (0.139, 0.477), (9, 4), (-3, 0), (-2, -4)),
    BrenemanExperimentResult(
        'Blue',
        (0.184, 0.444), (0.150, 0.387), (5, 11), (3, -10), (6, -22)),
    BrenemanExperimentResult(
        'Sky',
        (0.224, 0.489), (0.177, 0.439), (5, 6), (1, 1), (1, -7)),
    BrenemanExperimentResult(
        'Purple',
        (0.277, 0.454), (0.226, 0.389), (4, 10), (1, 4), (1, -8)))
# yapf: enable
"""
*Breneman (1987)* experiment 8 results.

BRENEMAN_EXPERIMENT_8_RESULTS : tuple

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 350 :math:`cd/m^2`
-   Observers Count : 8
"""

# yapf: disable
BRENEMAN_EXPERIMENT_9_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.254, 0.525), (0.195, 0.465)),
    BrenemanExperimentResult(
        'Gray',
        (0.256, 0.524), (0.207, 0.496), (4, 6), (3, 2), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.459, 0.521), (0.415, 0.489), (20, 14), (2, 12), (-2, 21)),
    BrenemanExperimentResult(
        'Skin',
        (0.307, 0.525), (0.261, 0.500), (7, 7), (0, 1), (-5, 2)),
    BrenemanExperimentResult(
        'Orange',
        (0.359, 0.545), (0.313, 0.532), (7, 5), (-2, -3), (-6, 13)),
    BrenemanExperimentResult(
        'Brown',
        (0.349, 0.540), (0.302, 0.510), (11, 15), (0, 12), (-5, 24)),
    BrenemanExperimentResult(
        'Yellow',
        (0.317, 0.550), (0.268, 0.538), (7, 10), (1, -4), (-4, 12)),
    BrenemanExperimentResult(
        'Foliage',
        (0.259, 0.544), (0.212, 0.510), (10, 11), (0, 14), (-4, 22)),
    BrenemanExperimentResult(
        'Green',
        (0.193, 0.542), (0.150, 0.506), (6, 10), (-1, 13), (-2, 15)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.181, 0.517), (0.144, 0.487), (9, 6), (-3, 0), (-1, -9)),
    BrenemanExperimentResult(
        'Blue',
        (0.184, 0.444), (0.155, 0.407), (4, 11), (-2, -6), (6, -36)),
    BrenemanExperimentResult(
        'Sky',
        (0.225, 0.490), (0.183, 0.458), (5, 8), (1, -3), (2, -19)),
    BrenemanExperimentResult(
        'Purple',
        (0.276, 0.454), (0.233, 0.404), (7, 12), (2, 9), (0, -16)),
    BrenemanExperimentResult(
        '(Gray)h',
        (0.256, 0.525), (0.208, 0.498)),
    BrenemanExperimentResult(
        '(Red)h',
        (0.456, 0.521), (0.416, 0.501), (15, 7), None, (-6, -9)),
    BrenemanExperimentResult(
        '(Brown)h',
        (0.349, 0.539), (0.306, 0.526), (11, 8), None, (-8, 7)),
    BrenemanExperimentResult(
        '(Foliage)h',
        (0.260, 0.545), (0.213, 0.528), (7, 9), None, (-4, 5)),
    BrenemanExperimentResult(
        '(Green)h',
        (0.193, 0.543), (0.149, 0.525), (10, 8), None, (-1, -1)),
    BrenemanExperimentResult(
        '(Blue)h',
        (0.184, 0.444), (0.156, 0.419), (7, 8), None, (4, -45)),
    BrenemanExperimentResult(
        '(Purple)h',
        (0.277, 0.456), (0.236, 0.422), (6, 11), None, (-2, -29)))
# yapf: enable
"""
*Breneman (1987)* experiment 9 results.

BRENEMAN_EXPERIMENT_9_RESULTS : tuple

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 15 :math:`cd/m^2`
-   Observers Count : 8
-   The colors indicated by (.)h are the darker colors presented at the higher
    luminescence level of the lighter colors.
"""

# yapf: disable
BRENEMAN_EXPERIMENT_10_RESULTS = (
    BrenemanExperimentResult(
        'Gray',
        (0.208, 0.482), (0.213, 0.494), (3, 3)),
    BrenemanExperimentResult(
        'Red',
        (0.447, 0.512), (0.411, 0.506), (15, 7)),
    BrenemanExperimentResult(
        'Skin',
        (0.269, 0.505), (0.269, 0.511), (4, 3)),
    BrenemanExperimentResult(
        'Orange',
        (0.331, 0.549), (0.315, 0.536), (7, 8)),
    BrenemanExperimentResult(
        'Brown',
        (0.323, 0.542), (0.310, 0.526), (6, 8)),
    BrenemanExperimentResult(
        'Yellow',
        (0.268, 0.556), (0.268, 0.541), (3, 6)),
    BrenemanExperimentResult(
        'Foliage',
        (0.226, 0.538), (0.230, 0.525), (4, 8)),
    BrenemanExperimentResult(
        'Green',
        (0.135, 0.531), (0.158, 0.524), (6, 3)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.145, 0.476), (0.161, 0.491), (4, 4)),
    BrenemanExperimentResult(
        'Blue',
        (0.163, 0.330), (0.171, 0.377), (6, 19)),
    BrenemanExperimentResult(
        'Sky',
        (0.179, 0.439), (0.187, 0.465), (5, 5)),
    BrenemanExperimentResult(
        'Purple',
        (0.245, 0.366), (0.240, 0.402), (3, 12)))
# yapf: enable
"""
*Breneman (1987)* experiment 10 results.

BRENEMAN_EXPERIMENT_10_RESULTS : tuple

Notes
-----
-   Effective White Levels : 15 and 270 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_11_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.208, 0.482), (0.174, 0.520)),
    BrenemanExperimentResult(
        'Gray',
        (0.209, 0.483), (0.176, 0.513), (3, 4), (2, 2), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.450, 0.512), (0.419, 0.524), (10, 2), (3, 2), (8, -1)),
    BrenemanExperimentResult(
        'Skin',
        (0.268, 0.506), (0.240, 0.528), (6, 2), (-4, 0), (-3, 0)),
    BrenemanExperimentResult(
        'Orange',
        (0.331, 0.547), (0.293, 0.553), (6, 2), (3, -1), (5, 1)),
    BrenemanExperimentResult(
        'Brown',
        (0.323, 0.542), (0.290, 0.552), (5, 2), (-1, -3), (0, -1)),
    BrenemanExperimentResult(
        'Yellow',
        (0.266, 0.549), (0.236, 0.557), (4, 2), (-3, -2), (-4, 2)),
    BrenemanExperimentResult(
        'Foliage',
        (0.227, 0.538), (0.194, 0.552), (4, 2), (2, -3), (-1, 1)),
    BrenemanExperimentResult(
        'Green',
        (0.146, 0.534), (0.118, 0.551), (8, 3), (4, -2), (-6, 3)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.160, 0.475), (0.130, 0.513), (9, 4), (1, -1), (-4, -3)),
    BrenemanExperimentResult(
        'Blue',
        (0.177, 0.340), (0.133, 0.427), (6, 14), (4, -17), (11, -29)),
    BrenemanExperimentResult(
        'Sky',
        (0.179, 0.438), (0.146, 0.482), (6, 10), (1, 4), (0, -1)),
    BrenemanExperimentResult(
        'Purple',
        (0.245, 0.366), (0.216, 0.419), (4, 13), (-3, 8), (4, -2)))
# yapf: enable
"""
*Breneman (1987)* experiment 1 results.

BRENEMAN_EXPERIMENT_11_RESULTS : tuple

Notes
-----
-   Illuminants : *green*, *D65*
-   White Luminance : 1560 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENT_12_RESULTS = (
    BrenemanExperimentResult(
        'Illuminant',
        (0.205, 0.482), (0.174, 0.519)),
    BrenemanExperimentResult(
        'Gray',
        (0.208, 0.482), (0.181, 0.507), (4, 3), (0, 1), (0, 0)),
    BrenemanExperimentResult(
        'Red',
        (0.451, 0.512), (0.422, 0.526), (20, 3), (0, -5), (10, -5)),
    BrenemanExperimentResult(
        'Skin',
        (0.268, 0.506), (0.244, 0.525), (5, 2), (-6, 0), (-2, -1)),
    BrenemanExperimentResult(
        'Orange',
        (0.331, 0.548), (0.292, 0.553), (10, 2), (5, 2), (11, 1)),
    BrenemanExperimentResult(
        'Brown',
        (0.324, 0.542), (0.286, 0.554), (8, 1), (5, -3), (10, -4)),
    BrenemanExperimentResult(
        'Yellow',
        (0.266, 0.548), (0.238, 0.558), (6, 2), (-3, -1), (-1, -2)),
    BrenemanExperimentResult(
        'Foliage',
        (0.227, 0.538), (0.196, 0.555), (6, 3), (3, -4), (2, -5)),
    BrenemanExperimentResult(
        'Green',
        (0.145, 0.534), (0.124, 0.551), (8, 6), (1, -1), (-8, -1)),
    BrenemanExperimentResult(
        'Blue-green',
        (0.160, 0.474), (0.135, 0.505), (5, 2), (1, -1), (-4, -3)),
    BrenemanExperimentResult(
        'Blue',
        (0.178, 0.339), (0.149, 0.392), (4, 20), (-1, -5), (3, -7)),
    BrenemanExperimentResult(
        'Sky',
        (0.179, 0.440), (0.150, 0.473), (4, 8), (3, 2), (2, 0)),
    BrenemanExperimentResult(
        'Purple',
        (0.246, 0.366), (0.222, 0.404), (5, 15), (-4, 2), (4, 2)))
# yapf: enable
"""
*Breneman (1987)* experiment 12 results.

BRENEMAN_EXPERIMENT_12_RESULTS : tuple

Notes
-----
-   Illuminants : *D55*, *green*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""

# yapf: disable
BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES = DocstringDict({
    1: PrimariesChromaticityCoordinates(
        1, ('A', 'D65'), 1500,
        (0.671, 0.519), (-0.586, 0.627), (0.253, 0.016)),
    2: PrimariesChromaticityCoordinates(
        2, ('Projector', 'D55'), 1500,
        (0.675, 0.523), (-0.466, 0.617), (0.255, 0.018)),
    3: PrimariesChromaticityCoordinates(
        3, ('Projector', 'D55'), 75,
        (0.664, 0.510), (-0.256, 0.729), (0.244, 0.003)),
    4: PrimariesChromaticityCoordinates(
        4, ('A', 'D65'), 75,
        (0.674, 0.524), (-0.172, 0.628), (0.218, -0.026)),
    6: PrimariesChromaticityCoordinates(
        6, ('A', 'D55'), 11100,
        (0.659, 0.506), (-0.141, 0.615), (0.249, 0.009)),
    8: PrimariesChromaticityCoordinates(
        8, ('A', 'D65'), 350,
        (0.659, 0.505), (-0.246, 0.672), (0.235, -0.006)),
    9: PrimariesChromaticityCoordinates(
        9, ('A', 'D65'), 15,
        (0.693, 0.546), (-0.446, 0.773), (0.221, -0.023)),
    11: PrimariesChromaticityCoordinates(
        11, ('D55', 'green'), 1560,
        (0.680, 0.529), (0.018, 0.576), (0.307, 0.080)),
    12: PrimariesChromaticityCoordinates(
        12, ('D55', 'green'), 75,
        (0.661, 0.505), (0.039, 0.598), (0.345, 0.127))})
# yapf: enable
BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES.__doc__ = """
*Breneman (1987)* experiments primaries chromaticities.

References
----------
:cite:`Breneman1987b`

BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES : dict
"""

BRENEMAN_EXPERIMENTS = DocstringDict({
    1: BRENEMAN_EXPERIMENT_1_RESULTS,
    2: BRENEMAN_EXPERIMENT_2_RESULTS,
    3: BRENEMAN_EXPERIMENT_3_RESULTS,
    4: BRENEMAN_EXPERIMENT_4_RESULTS,
    5: BRENEMAN_EXPERIMENT_5_RESULTS,
    6: BRENEMAN_EXPERIMENT_6_RESULTS,
    7: BRENEMAN_EXPERIMENT_7_RESULTS,
    8: BRENEMAN_EXPERIMENT_8_RESULTS,
    9: BRENEMAN_EXPERIMENT_9_RESULTS,
    10: BRENEMAN_EXPERIMENT_10_RESULTS,
    11: BRENEMAN_EXPERIMENT_11_RESULTS,
    12: BRENEMAN_EXPERIMENT_12_RESULTS
})
BRENEMAN_EXPERIMENTS.__doc__ = """
*Breneman (1987)* experiments.

References
----------
:cite:`Breneman1987b`

BRENEMAN_EXPERIMENTS : dict
"""
