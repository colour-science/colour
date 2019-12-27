# -*- coding: utf-8 -*-
"""
Common Gamut Boundary Descriptor (GDB) Utilities
================================================

Defines various *Gamut Boundary Descriptor (GDB)* common utilities.

-   :func:`colour.`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate
import scipy.ndimage

from colour.algebra import cartesian_to_polar, cartesian_to_spherical, \
    polar_to_cartesian, spherical_to_cartesian
from colour.utilities import (as_int_array, as_float_array,
                              is_trimesh_installed, orient, tsplit, warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ij_to_polar', 'polar_to_ij', 'Jab_to_spherical', 'spherical_to_Jab',
    'close_gamut_boundary_descriptor', 'fill_gamut_boundary_descriptor',
    'sample_volume_boundary_descriptor',
    'tessellate_volume_boundary_descriptor'
]


def ij_to_polar(ij):
    return cartesian_to_polar(np.roll(ij, 1, -1))


def polar_to_ij(rho_phi):
    return np.roll(polar_to_cartesian(rho_phi), 1, -1)


def Jab_to_spherical(Jab):
    return cartesian_to_spherical(np.roll(Jab, 2, -1))


def spherical_to_Jab(rho_phi_theta):
    return np.roll(spherical_to_cartesian(rho_phi_theta), 1, -1)


def gamut_boundary_descriptor_to_cartesian(GBD):
    GBD = as_float_array(GBD)

    if GBD.shape[-1] == 3:
        a = spherical_to_Jab(GBD)
    else:
        a = polar_to_ij(GBD)

    return a


def cartesian_to_gamut_boundary_descriptor(a):
    a = as_float_array(a)

    if a.shape[-1] == 3:
        GBD = Jab_to_spherical(a)
    else:
        GBD = ij_to_polar(a)

    return GBD


def close_gamut_boundary_descriptor(GBD):
    GBD = gamut_boundary_descriptor_to_cartesian(GBD)

    # Index of first row with values.
    r_t_i = np.argmin(np.all(np.any(np.isnan(GBD), axis=-1), axis=1))

    # First row with values.
    GBD_t = GBD[r_t_i, ...]
    GBD_t = GBD_t[~np.any(np.isnan(GBD_t), axis=-1)]
    if not np.allclose(GBD_t, GBD_t[0]) or GBD_t.shape[0] == 1:
        Jab_t = np.nanmean(
            GBD_t[GBD_t[..., 0] == np.max(GBD_t[..., 0])], axis=0)

        if r_t_i != 0:
            r_t_i -= 1

        warning('Closing top of GBD at row {0} with value {1}'.format(
            r_t_i, Jab_t))

        GBD[r_t_i] = np.tile(Jab_t, [1, GBD.shape[1], 1])

    # Index of last row with values.
    r_b_i = np.argmin(np.flip(np.all(np.any(np.isnan(GBD), axis=-1), axis=1)))
    r_b_i = GBD.shape[0] - 1 - r_b_i

    # Last row with values.
    GBD_b = GBD[r_b_i, ...]
    GBD_b = GBD_b[~np.any(np.isnan(GBD_b), axis=-1)]
    if not np.allclose(GBD_b, GBD_b[0]) or GBD_b.shape[0] == 1:
        Jab_b = np.nanmean(
            GBD_b[GBD_b[..., 0] == np.min(GBD_b[..., 0])], axis=0)

        if r_b_i != GBD.shape[0] - 1:
            r_b_i += 1

        warning('Closing bottom of GBD at row {0} with value {1}'.format(
            r_b_i, Jab_b))

        GBD[r_b_i] = np.tile(Jab_b, [1, GBD.shape[1], 1])

    return cartesian_to_gamut_boundary_descriptor(GBD)


def fill_gamut_boundary_descriptor(GBD):
    GBD = gamut_boundary_descriptor_to_cartesian(GBD)

    GBD_i = np.copy(GBD)

    shape_r, shape_c = GBD.shape[0], GBD.shape[1]
    r_slice = np.s_[0:shape_r]
    c_slice = np.s_[0:shape_c]

    # If bounding columns have NaN, :math:`GBD_m` matrix is tiled
    # horizontally so that right values interpolate with left values and
    # vice-versa.
    if np.any(np.isnan(GBD_i[..., 0])) or np.any(np.isnan(GBD_i[..., -1])):
        warning(
            'Gamut boundary descriptor matrix bounding columns contains NaN '
            'and will be horizontally tiled for filling!')
        c_slice = np.s_[shape_c:shape_c * 2]
        GBD_i = np.hstack([GBD_i] * 3)

    # If bounding rows have NaN, :math:`GBD_m` matrix is reflected vertically
    # so that top and bottom values are replicated via interpolation, i.e.
    # equivalent to nearest-neighbour interpolation.
    if np.any(np.isnan(GBD_i[0, ...])) or np.any(np.isnan(GBD_i[-1, ...])):
        warning('Gamut boundary descriptor matrix bounding rows contains NaN '
                'and will be vertically reflected for filling!')
        r_slice = np.s_[shape_r:shape_r * 2]
        GBD_f = orient(GBD_i, 'Flop')
        GBD_i = np.vstack([GBD_f, GBD_i, GBD_f])

    shape_r, shape_c = GBD_i.shape[0], GBD_i.shape[1]
    mask = np.any(~np.isnan(GBD_i), axis=-1)
    for i in range(GBD.shape[-1]):
        x = np.linspace(0, 1, shape_r)
        y = np.linspace(0, 1, shape_c)
        x_g, y_g = np.meshgrid(x, y, indexing='ij')
        values = GBD_i[mask]

        GBD_i[..., i] = scipy.interpolate.griddata(
            (x_g[mask], y_g[mask]),
            values[..., i], (x_g, y_g),
            method='linear')

    return cartesian_to_gamut_boundary_descriptor(GBD_i[r_slice, c_slice])


def sample_volume_boundary_descriptor(GBD, theta_alpha):
    GBD = spherical_to_Jab(GBD)
    theta, alpha = tsplit(theta_alpha)

    GBD_s = np.zeros(list(theta_alpha.shape[:2]) + [3])

    x = np.linspace(0, 1, GBD.shape[0])
    y = np.linspace(0, 1, GBD.shape[1])
    x_g, y_g = np.meshgrid(x, y, indexing='ij')

    theta_i = np.radians(theta) / np.pi
    alpha_i = np.radians(alpha) / (2 * np.pi)

    for i in range(GBD.shape[-1]):
        GBD_s[..., i] = scipy.interpolate.griddata(
            (np.ravel(x_g), np.ravel(y_g)),
            np.ravel(GBD[..., i]), (theta_i, alpha_i),
            method='linear')

    GBD_s = Jab_to_spherical(GBD_s)

    return GBD_s


def tessellate_volume_boundary_descriptor(GBD):
    if is_trimesh_installed():
        import trimesh

        vertices = spherical_to_Jab(GBD)

        # Wrapping :math:`GBD_m` to create faces between the outer columns.
        vertices = np.insert(
            vertices, vertices.shape[1], vertices[:, 0, :], axis=1)

        shape_r, shape_c = vertices.shape[0], vertices.shape[1]

        faces = []
        for i in np.arange(shape_r - 1):
            for j in np.arange(shape_c - 1):
                a = [i, j]
                b = [i, j + 1]
                c = [i + 1, j]
                d = [i + 1, j + 1]

                # Avoiding overlapping triangles when tessellating the bottom.
                if not i == 0:
                    faces.append([a, c, b])

                # Avoiding overlapping triangles when tessellating the top.
                if not i == shape_r - 2:
                    faces.append([c, d, b])

        indices = np.ravel_multi_index(
            np.transpose(as_int_array(faces)), [shape_r, shape_c])

        GBD_t = trimesh.Trimesh(
            vertices=vertices.reshape([-1, 3]),
            faces=np.transpose(indices),
            validate=True)

        if not GBD_t.is_watertight:
            warning('Tessellated mesh has holes!')

        return GBD_t
