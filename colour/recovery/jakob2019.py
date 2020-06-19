# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
=========================================

Defines objects for reflectance recovery using *Jakob and Hanika (2019)*
method:

-   :func:`colour.recovery.XYZ_to_sd_Jakob2019`

References
----------
-   :cite:`Jakob2019Spectral` : Wenzel Jakob and Johannes Hanika. 2019.
    A Low-Dimensional Function Space for Efficient Spectral Upsampling. In
    Computer Graphics Forum (Proceedings of Eurographics) 38(2).    
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
                                SpectralShape, sd_ones, sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_xy, XYZ_to_Lab
from colour.utilities import runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['DEFAULT_SPECTRAL_SHAPE_MENG_2015', 'XYZ_to_sd_Meng2015']

DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 = SpectralShape(360, 780, 5)
"""
DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 : SpectralShape
"""



def model(wvlp, ccp):
    """
    This is the model of spectral reflectivity described in the article.
    """
    yy = ccp[0] * wvlp**2 + ccp[1] * wvlp + ccp[2]
    return 1 / 2 + yy / (2 * np.sqrt(1 + yy ** 2))

def model_sd(ccp, name='Jakob and Hanika (2019)', primed=True):
    """
    Create a SpectralDistribution using given coefficients
    """
    # FIXME: don't hardcode the wavelength grid; there should be a way
    #        of creating a SpectralDistribution from the function alone
    wvl = np.arange(360, 830, 5)
    wvlp = (wvl - 360) / (830 - 360)
    grid = wvlp if primed else wvl
    return SpectralDistribution(model(grid, ccp), wvl, name=name)



# FIXME: This code is *very* slow and needs a rework
def solve_Jakob2019(
        XYZ, cmfs, illuminant, ccp0=(0, 0, 0),
        try_hard=True, verbose=False):
    """
    Computes the coefficients for Jakob and Hanika (2019) reflecitvity model
    for a given XYZ color and an optional starting point.
    """

    if XYZ[1] > 1:
        raise ValueError('Lightness Y = {0} is greater than one; the result would have to be unphysical'
                         .format(XYZ[1]))

    # FIXME: Code below assumes we can always get a near-zero delta E and will
    #        fail if it's not possible.
    if not is_within_visible_spectrum(XYZ, cmfs, illuminant):
        raise ValueError('The current implementation of Jakob (2019) cannot handle XYZ values corresponding to unphysical spectra')

    # A special case that's hard to solve numerically
    if np.allclose(XYZ, [0, 0, 0]):
        return np.array([0, 0, -1e+9], dtype=DEFAULT_FLOAT_DTYPE), 0.0

    ill_xy = XYZ_to_xy(sd_to_XYZ(illuminant))



    def objective_function(ccp, target):
        """
        Computes the CIE 1976 delta E between the target color and the color defined
        by given model parameters and illuminant.
        """
        sd = model_sd(ccp)
        Lab = XYZ_to_Lab(sd_to_XYZ(sd, illuminant=illuminant) / 100, ill_xy)
        return delta_E_CIE1976(target, Lab)

    def optimize(XYZ, ccp0):
        """
        Performs the actual minimization. This function will be called multiple
        times if minimization diverges and intermediate solutions are required.
        """
        Lab = XYZ_to_Lab(XYZ, ill_xy)
        opt = minimize(
            objective_function, ccp0, (Lab,),
            method='Nelder-Mead', options={"disp": verbose}
        )
        if verbose:
            print(opt)
        return opt



    if verbose:
        print('Trying the target directly, XYZ={0}'.format(XYZ))
    opt = optimize(XYZ, ccp0)
    if opt.fun < 0.1 or not try_hard:
        return opt.x, opt.fun



    # The coefficients below are good only in case of D65, but this should be
    # good enough.
    good_XYZ = (1/3, 1/3, 1/3)
    good_ccp = (2.1276356, -1.07293026, -0.29583292)

    divisions = 3
    while divisions < 10:
        if verbose:
            print('Trying with {0} divisions'.format(divisions))

        keep_divisions = False
        ref_XYZ = good_XYZ
        ref_ccp = good_ccp

        ccp0 = ref_ccp
        for i in range(1, divisions):
            intermediate_XYZ = ref_XYZ + (XYZ - ref_XYZ) * i / (divisions - 1)
            if verbose:
                print('Intermediate step {0}/{1}, XYZ={2} with ccp0={3}'.format(
                    i + 1, divisions, intermediate_XYZ, ccp0))

            opt = optimize(intermediate_XYZ, ccp0)
            if opt.fun > 0.1:
                if verbose:
                    print('WARNING: intermediate optimization failed')
                break
            else:
                good_XYZ = intermediate_XYZ
                good_ccp = opt.x
                keep_divisions = True

            ccp0 = opt.x
        else:
            return opt.x, opt.fun

        if not keep_divisions:
            divisions += 2

    raise RuntimeError(
        'Optimization failed for XYZ={0}, ccp0={1}'.format(XYZ, ccp0))



def XYZ_to_sd_Jakob2019(
        XYZ,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'],
        optimisation_parameters=None,
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

    ccp, _ = solve_Jakob2019(XYZ, cmfs, illuminant, verbose=verbose)
    return model_sd(ccp, name='Jakob (2019) - {0}'.format(XYZ))



class Jakob2019Interpolator:
    def __init__(self):
        pass

    def read_file(self, path):
        with open(path, 'rb') as fd:
            if fd.read(4).decode('ISO-8859-1') != 'SPEC':
                raise ValueError('Bad magic number, this likely is not the right file')

            self.res = struct.unpack('i', fd.read(4))[0]
            self.scale = np.fromfile(fd, count=self.res, dtype=np.float32)
            coeffs = np.fromfile(fd, count=3*self.res**3*3, dtype=np.float32)
            coeffs = coeffs.reshape(3, self.res, self.res, self.res, 3)

        t = np.linspace(0, 1, self.res)
        axes = ([0, 1, 2], self.scale, t, t)
        self.cubes = RegularGridInterpolator(axes, coeffs[:, :, :, :, :],
                                             bounds_error=False)

    def __call__(self, RGB):
        RGB = np.asarray(RGB, dtype=DEFAULT_FLOAT_DTYPE)
        vmax = np.max(RGB, axis=-1)
        imax = np.argmax(RGB, axis=-1)
        chroma = RGB / (np.expand_dims(vmax, -1) + 1e-10) # Avoid division by zero
        vmax = np.max(RGB, axis=-1)
        v2 = np.take_along_axis(chroma, np.expand_dims((imax + 2) % 3, axis=-1),
                                axis=-1).squeeze(axis=-1)
        v3 = np.take_along_axis(chroma, np.expand_dims((imax + 1) % 3, axis=-1),
                                axis=-1).squeeze(axis=-1)
        coords = np.stack([imax, vmax, v2, v3], axis=-1)
        ccp = self.cubes(coords).squeeze()
        return model_sd(ccp, primed=False)
