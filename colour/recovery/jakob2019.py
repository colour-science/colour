# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery using *Jakob and Hanika (2019)*
method:

-   :func:`colour.recovery.XYZ_to_sd_Jakob2019`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147–155. doi:10.1111/cgf.13626
"""

from __future__ import division, unicode_literals

import numpy as np
import struct
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from colour import ILLUMINANT_SDS
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.volume import is_within_visible_spectrum
from colour.colorimetry import (STANDARD_OBSERVER_CMFS, SpectralDistribution,
                                SpectralShape, sd_to_XYZ,
                                multi_sds_to_XYZ_integration)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_xy, XYZ_to_Lab
from colour.utilities import as_float_array, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DEFAULT_SPECTRAL_SHAPE_JAKOB_2019', 'spectral_model', 'spectral_values',
    'solve_Jakob2019', 'XYZ_to_sd_Jakob2019', 'Jakob2019Interpolator'
]

DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 = SpectralShape(360, 780, 5)
"""
DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 : SpectralShape
"""


def spectral_model(wl, coefficients):
    """
    Spectral model given by *Jakob and Hanika (2019)*.
    """

    c_1, c_2, c_3 = coefficients
    x = c_1 * wl ** 2 + c_2 * wl + c_3

    return 1 / 2 + x / (2 * np.sqrt(1 + x ** 2))


def spectral_values(coefficients,
                    shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019,
                    initialised=True):
    """
    Create a SpectralDistribution using given coefficients
    """

    wl = shape.range()
    wl_p = (wl - shape.start) / (shape.end - shape.start)
    wl = wl_p if initialised else wl

    return spectral_model(wl, coefficients)


# TODO: This code is *very* slow and needs a rework.
def solve_Jakob2019(XYZ,
                    cmfs,
                    illuminant,
                    coefficients_0=(0, 0, 0),
                    try_hard=True,
                    verbose=False):
    """
    Computes the coefficients for *Jakob and Hanika (2019)* reflectance
    spectral model for a given XYZ color and optional starting point.
    """

    XYZ = as_float_array(XYZ)

    if XYZ[1] > 1:
        raise ValueError(
            'Non physically-realisable tristimulus reflectance values with '
            '"Luminance Y"={0}!'.format(XYZ[1]))

    if np.allclose(XYZ, [0, 0, 0]):
        raise ValueError(
            'Almost null tristimulus reflectance values are not recoverable!'
            .format(XYZ[1]))

    # TODO: Code below assumes we can always get a near-zero delta E and will
    #        fail if it's not possible.
    if not is_within_visible_spectrum(XYZ, cmfs, illuminant):
        raise ValueError(
            'Non physically-realisable tristimulus reflectance values outside '
            'the visible spectrum!')

    xy_w = XYZ_to_xy(sd_to_XYZ(illuminant))

    def objective_function(ccp, target):
        """
        Computes :math:`\\Delta E_{76}` between the target colour and the
        colour defined by given spectral model parameters and illuminant.
        """

        sd_v = spectral_values(ccp)
        Lab = XYZ_to_Lab(
            multi_sds_to_XYZ_integration(
                sd_v,
                cmfs=cmfs,
                illuminant=illuminant,
                shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019) / 100, xy_w)

        return delta_E_CIE1976(target, Lab)

    def optimize(XYZ, coefficients_0):
        """
        Performs the actual minimization. This function will be called multiple
        times if minimization diverges and intermediate solutions are required.
        """

        Lab = XYZ_to_Lab(XYZ, xy_w)

        result = minimize(
            objective_function,
            coefficients_0, (Lab, ),
            method='Nelder-Mead',
            options={'disp': verbose})

        if verbose:
            print(result)

        return result

    if verbose:
        print('Trying the target directly, XYZ={0}'.format(XYZ))

    result = optimize(XYZ, coefficients_0)

    if result.fun < 0.1 or not try_hard:
        return result.x, result.fun

    # The coefficients below are good only in case of D65, but this should be
    # good enough.
    good_XYZ = (1 / 3, 1 / 3, 1 / 3)
    good_ccp = (2.1276356, -1.07293026, -0.29583292)

    divisions = 3
    while divisions < 10:
        if verbose:
            print('Trying with {0} divisions'.format(divisions))

        keep_divisions = False
        XYZ_ref = good_XYZ
        ccp_ref = good_ccp

        coefficients_0 = ccp_ref
        for i in range(1, divisions):
            intermediate_XYZ = XYZ_ref + (XYZ - XYZ_ref) * i / (divisions - 1)
            if verbose:
                print(
                    'Intermediate step {0}/{1}, XYZ={2} with ccp0={3}'.format(
                        i + 1, divisions, intermediate_XYZ, coefficients_0))

            result = optimize(intermediate_XYZ, coefficients_0)
            if result.fun > 0.1:
                if verbose:
                    print('WARNING: intermediate optimization failed')
                break
            else:
                good_XYZ = intermediate_XYZ
                good_ccp = result.x
                keep_divisions = True

            coefficients_0 = result.x
        else:
            return result.x, result.fun

        if not keep_divisions:
            divisions += 2

    raise RuntimeError('Optimization failed for XYZ={0}, ccp0={1}'.format(
        XYZ, coefficients_0))


def XYZ_to_sd_Jakob2019(
        XYZ,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        optimisation_kwargs=None,
        verbose=False):
    """
    Recovers the spectral distribution of given *CIE XYZ* tristimulus values
    using *Jakob and Hanika (2019)* method.

    TODO: documentation
    """

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    coefficients, _ = solve_Jakob2019(XYZ, cmfs, illuminant, verbose=verbose)

    return SpectralDistribution(
        spectral_values(coefficients),
        cmfs.shape.range(),
        name='Jakob (2019) - {0}'.format(XYZ))


class Jakob2019Interpolator:
    def __init__(self):
        pass

    def from_file(self, path):
        with open(path, 'rb') as fd:
            if fd.read(4).decode('ISO-8859-1') != 'SPEC':
                raise ValueError(
                    'Bad magic number, this likely is not the right file type!'
                )

            self.res = struct.unpack('i', fd.read(4))[0]
            self.scale = np.fromfile(fd, count=self.res, dtype=np.float32)
            coeffs = np.fromfile(
                fd, count=3 * self.res ** 3 * 3, dtype=np.float32)
            coeffs = coeffs.reshape(3, self.res, self.res, self.res, 3)

        samples = np.linspace(0, 1, self.res)
        axes = ([0, 1, 2], self.scale, samples, samples)
        self.cubes = RegularGridInterpolator(
            axes, coeffs[:, :, :, :, :], bounds_error=False)

    def coefficients(self, RGB):
        RGB = np.asarray(RGB, dtype=DEFAULT_FLOAT_DTYPE)
        vmax = np.max(RGB, axis=-1)
        imax = np.argmax(RGB, axis=-1)
        chroma = RGB / (np.expand_dims(vmax, -1) + 1e-10
                        )  # Avoid division by zero
        vmax = np.max(RGB, axis=-1)
        v2 = np.take_along_axis(
            chroma, np.expand_dims((imax + 2) % 3, axis=-1),
            axis=-1).squeeze(axis=-1)
        v3 = np.take_along_axis(
            chroma, np.expand_dims((imax + 1) % 3, axis=-1),
            axis=-1).squeeze(axis=-1)
        coords = np.stack([imax, vmax, v2, v3], axis=-1)
        return self.cubes(coords).squeeze()

    def __call__(self, RGB):
        return SpectralDistribution(
            spectral_values(self.coefficients(RGB), initialised=False),
            name='Jakob (2019) - {0} (RGB)'.format(RGB))
