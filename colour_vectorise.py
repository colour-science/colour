# # Colour - Vectorise

# ## Data


from __future__ import division, with_statement

import numpy as np

from pprint import pprint

DATA_HD1 = np.random.rand(1920 * 1080, 3)
DATA_HD2 = np.random.rand(1920 * 1080, 3)
DATA_HD3 = np.random.rand(1920 * 1080, 3)

DATA_PAL1 = np.random.rand(640 * 480, 3)
DATA_PAL2 = np.random.rand(640 * 480, 3)
DATA_PAL3 = np.random.rand(640 * 480, 3)

DATA1, DATA2, DATA3 = DATA_HD1, DATA_HD2, DATA_HD3


# ## Utilities
from colour.utilities import (
    CaseInsensitiveMapping,
    as_array,
    as_numeric,
    as_shape,
    as_stack,
    handle_numpy_errors,
    ignore_numpy_errors,
    is_iterable,
    is_numeric,
    message_box,
    print_numpy_errors,
    raise_numpy_errors,
    row_as_diagonal,
    warn_numpy_errors,
    warning)


# #############################################################################
# #############################################################################
# ## colour.adaptation.cat
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.chromatic_adaptation_matrix_VonKries
# #############################################################################
from colour.adaptation import *
from colour.adaptation.vonkries import *

message_box('chromatic_adaptation_matrix_VonKries')


def chromatic_adaptation_matrix_VonKries_2d(data1, data2):
    for i in range(len(data1)):
        chromatic_adaptation_matrix_VonKries(data1[i], data2[i])


# get_ipython().magic(u'timeit chromatic_adaptation_matrix_VonKries_2d(DATA1, DATA2)')


def chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w,
                                                   XYZ_wr,
                                                   transform='CAT02'):
    M = CHROMATIC_ADAPTATION_TRANSFORMS.get(transform)

    if M is None:
        raise KeyError(
            '"{0}" chromatic adaptation transform is not defined! Supported '
            'methods: "{1}".'.format(transform,
                                     CHROMATIC_ADAPTATION_TRANSFORMS.keys()))
    M = as_array(M, (3, 3))

    XYZ_w = as_array(XYZ_w, (-1, 3))
    XYZ_wr = as_array(XYZ_wr, (-1, 3))

    rgb_w = np.dot(XYZ_w, np.transpose(M))
    rgb_wr = np.dot(XYZ_wr, np.transpose(M))

    D = rgb_wr / rgb_w

    D = row_as_diagonal(D)

    cat = np.einsum('...ij,...jk->...ik', np.linalg.inv(M), D)
    cat = np.einsum('...ij,...jk->...ik', cat, M)

    cat = np.squeeze(cat)

    return cat


print('Reference:')
XYZ_w = np.array([1.09846607, 1., 0.3558228])
XYZ_wr = np.array([0.95042855, 1., 1.08890037])
print(chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr))

print('\n')

print('1d array:')
print(chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr))

print('\n')

print('2d array:')
XYZ_w = np.tile(XYZ_w, (5, 1))
XYZ_wr = np.tile(XYZ_wr, (5, 1))
print(chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr))

# get_ipython().magic(u'timeit chromatic_adaptation_matrix_VonKries_vectorise(DATA1, DATA2)')

print('\n')

# #############################################################################
# ### colour.chromatic_adaptation_VonKries
# #############################################################################
message_box('chromatic_adaptation_VonKries')


def chromatic_adaptation_VonKries_2d(data1, data2, data3):
    for i in range(len(data1)):
        chromatic_adaptation_VonKries(data1[i], data2[i], data3[i])


# get_ipython().magic(u'timeit chromatic_adaptation_VonKries_2d(DATA1, DATA2, DATA3)')


def chromatic_adaptation_VonKries_vectorise(XYZ, XYZ_w,
                                            XYZ_wr,
                                            transform='CAT02'):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))

    cat = chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr,
                                                         transform)
    XYZ_a = as_stack(np.einsum('...i,...ji', XYZ, cat), direction='I',
                     shape=shape)

    return XYZ_a


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
XYZ_w = np.array([1.09846607, 1., 0.3558228])
XYZ_wr = np.array([0.95042855, 1., 1.08890037])
print(chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr))

print('\n')

print('1d array:')
print(chromatic_adaptation_VonKries_vectorise(XYZ, XYZ_w, XYZ_wr))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
XYZ_w = np.tile(XYZ_w, (5, 1))
XYZ_wr = np.tile(XYZ_wr, (5, 1))
print(chromatic_adaptation_VonKries_vectorise(XYZ, XYZ_w, XYZ_wr))

# get_ipython().magic(u'timeit chromatic_adaptation_VonKries_vectorise(DATA1, DATA2, DATA3)')

print('\n')

# #############################################################################
# #############################################################################
# ## colour.algebra.coordinates.transformations
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.cartesian_to_spherical
# #############################################################################
from colour.algebra.coordinates.transformations import *

message_box('cartesian_to_spherical')


def cartesian_to_spherical_2d(vectors):
    for vector in vectors:
        cartesian_to_spherical(vector)


# get_ipython().magic(u'timeit cartesian_to_spherical_2d(DATA1)')


def cartesian_to_spherical_vectorise(vector):
    shape = as_shape(vector)
    vector = as_array(vector, (-1, 3))

    r = np.linalg.norm(vector, axis=1)
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]

    theta = np.arctan2(z, np.linalg.norm(
        np.dstack((x, y)).reshape((-1, 2)), axis=1))
    phi = np.arctan2(y, x)

    return as_stack((r, theta, phi), shape=shape)


print('Reference:')
vector = np.array([3, 1, 6])
print(cartesian_to_spherical(vector))

print('\n')

print('1d array:')
print(cartesian_to_spherical_vectorise(vector))

print('\n')

print('2d array:')
vector = np.tile(vector, (5, 1))
print(cartesian_to_spherical_vectorise(vector))

# get_ipython().magic(u'timeit cartesian_to_spherical_vectorise(DATA1)')

print('\n')

# #############################################################################
# ### colour.spherical_to_cartesian
# #############################################################################
message_box('spherical_to_cartesian')


def spherical_to_cartesian_2d(vectors):
    for vector in vectors:
        spherical_to_cartesian(vector)


# get_ipython().magic(u'timeit spherical_to_cartesian_2d(DATA1)')


def spherical_to_cartesian_vectorise(vector):
    shape = as_shape(vector)
    vector = as_array(vector, (-1, 3))

    r, theta, phi = vector[:, 0], vector[:, 1], vector[:, 2]

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    return as_stack((x, y, z), shape=shape)


print('Reference:')
vector = np.array([6.78232998, 1.08574654, 0.32175055])
print(spherical_to_cartesian(vector))

print('\n')

print('1d array:')
print(spherical_to_cartesian_vectorise(vector))

print('\n')

print('2d array:')
vector = np.tile(vector, (5, 1))
print(spherical_to_cartesian_vectorise(vector))

# get_ipython().magic(u'timeit spherical_to_cartesian_vectorise(DATA1)')

print('\n')

# #############################################################################
# ### colour.cartesian_to_cylindrical
# #############################################################################
message_box('cartesian_to_cylindrical')


def cartesian_to_cylindrical_2d(vectors):
    for vector in vectors:
        cartesian_to_cylindrical(vector)


# get_ipython().magic(u'timeit cartesian_to_cylindrical_2d(DATA1)')


def cartesian_to_cylindrical_vectorise(vector):
    shape = as_shape(vector)
    vector = as_array(vector, (-1, 3))

    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]

    theta = np.arctan2(y, x)
    rho = np.linalg.norm(
        np.dstack((x, y)).reshape((-1, 2)), axis=1)

    return as_stack((z, theta, rho), shape=shape)


print('Reference:')
vector = np.array([3, 1, 6])
print(cartesian_to_cylindrical(vector))

print('\n')

print('1d array:')
print(cartesian_to_cylindrical_vectorise(vector))

print('\n')

print('2d array:')
vector = np.tile(vector, (5, 1))
print(cartesian_to_cylindrical_vectorise(vector))

# get_ipython().magic(u'timeit cartesian_to_cylindrical_vectorise(DATA1)')

print('\n')


# #############################################################################
# ### colour.cylindrical_to_cartesian
# #############################################################################
message_box('cylindrical_to_cartesian')


def cylindrical_to_cartesian_2d(vectors):
    for vector in vectors:
        cylindrical_to_cartesian(vector)


# get_ipython().magic(u'timeit cylindrical_to_cartesian_2d(DATA1)')


def cylindrical_to_cartesian_vectorise(vector):
    shape = as_shape(vector)
    vector = as_array(vector, (-1, 3))

    z, theta, rho = vector[:, 0], vector[:, 1], vector[:, 2]

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return as_stack((x, y, z), shape=shape)


print('Reference:')
vector = np.array([6, 0.32175055, 3.16227766])
print(cylindrical_to_cartesian(vector))

print('\n')

print('1d array:')
print(cylindrical_to_cartesian_vectorise(vector))

print('\n')

print('2d array:')
vector = np.tile(vector, (5, 1))
print(cylindrical_to_cartesian_vectorise(vector))

# get_ipython().magic(u'timeit cylindrical_to_cartesian_vectorise(DATA1)')

print('\n')

# #############################################################################
# #############################################################################
# ## colour.colorimetry.blackbody
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.planck_law
# #############################################################################
from colour.colorimetry.blackbody import *

message_box('planck_law')


def planck_law_2d(wavelengths):
    for wavelength in wavelengths:
        planck_law(wavelength, 5500)


WAVELENGTHS = np.linspace(1, 15000, 100000) * 1e-9


# get_ipython().magic(u'timeit planck_law_2d(WAVELENGTHS)')


@handle_numpy_errors(over='ignore')
def planck_law_vectorise(wavelength, temperature, c1=C1, c2=C2, n=N):
    t = temperature
    l = as_array(wavelength)

    p = (((c1 * n ** -2 * l ** -5) / np.pi) *
         (np.exp(c2 / (n * l * t)) - 1) ** -1)

    return as_numeric(p)


print('Reference:')
print(planck_law(500 * 1e-9, 5500))

print('\n')

print('1d array:')
print(planck_law_vectorise(500 * 1e-9, 5500))

print('\n')

print('2d array:')
print(planck_law_vectorise([500 * 1e-9] * 5, 5500))

# get_ipython().magic(u'timeit planck_law_vectorise(WAVELENGTHS, 5500)')

print('\n')

from colour.colorimetry.spectrum import *


def blackbody_spd_vectorise(temperature,
                            shape=DEFAULT_SPECTRAL_SHAPE,
                            c1=C1,
                            c2=C2,
                            n=N):
    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name='{0}K Blackbody'.format(temperature),
        data=dict(
            zip(wavelengths, planck_law_vectorise(
                wavelengths * 1e-9, temperature, c1, c2, n))))


print(blackbody_spd_vectorise(5000).values)


# #############################################################################
# #############################################################################
# ## colour.colorimetry.lightness
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.lightness_Glasser1958
# #############################################################################
from colour.colorimetry.lightness import *

message_box('lightness_Glasser1958')


def lightness_Glasser1958_2d(Y):
    for Y_ in Y:
        lightness_Glasser1958(Y_)


Y = np.linspace(0, 100, 1000000)

# get_ipython().magic(u'timeit lightness_Glasser1958_2d(Y)')


def lightness_Glasser1958_vectorise(Y, **kwargs):
    Y = as_array(Y)

    L = 25.29 * (Y ** (1 / 3)) - 18.38

    return as_numeric(L)


print('Reference:')
print(lightness_Glasser1958(10.08))

print('\n')

print('1d array:')
print(lightness_Glasser1958_vectorise(10.08))

print('\n')

print('2d array:')
print(lightness_Glasser1958_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit lightness_Glasser1958_vectorise(Y)')

print('\n')

# #############################################################################
# ### colour.lightness_Wyszecki1963
# ##############################################################################
message_box('lightness_Wyszecki1963')


def lightness_Wyszecki1963_2d(Y):
    for Y_ in Y:
        lightness_Wyszecki1963(Y_)


# get_ipython().magic(u'timeit lightness_Wyszecki1963_2d(Y)')


def lightness_Wyszecki1963_vectorise(Y, **kwargs):
    Y = as_array(Y)

    if np.any(Y < 1) or np.any(Y > 98):
        warning(('"W*" Lightness computation is only applicable for '
                 '1% < "Y" < 98%, unpredictable results may occur!'))

    W = 25 * (Y ** (1 / 3)) - 17

    return as_numeric(W)


print('Reference:')
print(lightness_Wyszecki1963(10.08))

print('\n')

print('1d array:')
print(lightness_Wyszecki1963_vectorise(10.08))

print('\n')

print('2d array:')
print(lightness_Wyszecki1963_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit lightness_Wyszecki1963_vectorise(Y)')

print('\n')

# ##############################################################################
# ### colour.lightness_1976
###############################################################################
message_box('lightness_1976')


def lightness_1976_2d(Y):
    for Y_ in Y:
        lightness_1976(Y_)

# get_ipython().magic(u'timeit lightness_1976_2d(Y)')

from colour.constants import CIE_E, CIE_K


def lightness_1976_vectorise(Y, Y_n=100):
    Y = as_array(Y)
    Y_n = np.resize(as_array(Y_n), Y.shape)

    Lstar = Y / Y_n

    Lstar = np.where(Lstar <= CIE_E,
                     CIE_K * Lstar,
                     116 * Lstar ** (1 / 3) - 16)

    return as_numeric(Lstar)


print('Reference:')
print(lightness_1976(10.08, 100))

print('\n')

print('1d array:')
print(lightness_1976_vectorise(10.08, 100))

print('\n')

print('2d array:')
print(lightness_1976_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit lightness_1976_vectorise(Y)')

print('\n')

###############################################################################
###############################################################################
### colour.colorimetry.luminance
###############################################################################
###############################################################################

###############################################################################
#### colour.luminance_Newhall1943
###############################################################################
from colour.colorimetry.luminance import *

message_box('luminance_Newhall1943')


def luminance_Newhall1943_2d(L):
    for L_ in L:
        luminance_Newhall1943(L_)


L = np.linspace(0, 100, 1000000)

# get_ipython().magic(u'timeit luminance_Newhall1943_2d(L)')


def luminance_Newhall1943_vectorise(V, **kwargs):
    V = as_array(V)

    R_Y = (1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) - 0.021009 *
           (V ** 4) + 0.0008404 * (V ** 5))

    return as_numeric(R_Y)


print('Reference:')
print(luminance_Newhall1943(3.74629715382))

print('\n')

print('1d array:')
print(luminance_Newhall1943_vectorise(3.74629715382))

print('\n')

print('2d array:')
print(luminance_Newhall1943_vectorise(
    [3.74629715382, 3.74629715382, 3.74629715382, 3.74629715382,
     3.74629715382]))

# get_ipython().magic(u'timeit luminance_Newhall1943_vectorise(L)')

print('\n')

###############################################################################
#### colour.luminance_ASTMD153508
###############################################################################
message_box('luminance_ASTMD153508')


def luminance_ASTMD153508_2d(L):
    for L_ in L:
        luminance_ASTMD153508(L_)


# get_ipython().magic(u'timeit luminance_ASTMD153508_2d(L)')


def luminance_ASTMD153508_vectorise(V, **kwargs):
    V = as_array(V)

    Y = (1.1914 * V - 0.22533 * (V ** 2) + 0.23352 * (V ** 3) - 0.020484 *
         (V ** 4) + 0.00081939 * (V ** 5))

    return as_numeric(Y)


print('Reference:')
print(luminance_ASTMD153508(3.74629715382))

print('\n')

print('1d array:')
print(luminance_ASTMD153508_vectorise(3.74629715382))

print('\n')

print('2d array:')
print(luminance_ASTMD153508_vectorise(
    [3.74629715382, 3.74629715382, 3.74629715382, 3.74629715382,
     3.74629715382]))

# get_ipython().magic(u'timeit luminance_ASTMD153508_vectorise(L)')

print('\n')

###############################################################################
#### colour.luminance_1976
###############################################################################
message_box('luminance_1976')


def luminance_1976_2d(L):
    for L_ in L:
        luminance_1976(L_)


# get_ipython().magic(u'timeit luminance_1976_2d(L)')


def luminance_1976_vectorise(Lstar, Y_n=100):
    Lstar = as_array(Lstar)
    Y_n = np.resize(as_array(Y_n), Lstar.shape)

    Y = np.where(Lstar > CIE_K * CIE_E,
                 Y_n * ((Lstar + 16) / 116) ** 3,
                 Y_n * (Lstar / CIE_K))

    return as_numeric(Y)


print('Reference:')
print(luminance_1976(37.9856290977))

print('\n')

print('1d array:')
print(luminance_1976_vectorise(37.9856290977))

print('\n')

print('2d array:')
print(luminance_1976_vectorise(
    [37.9856290977, 37.9856290977, 37.9856290977, 37.9856290977,
     37.9856290977]))

# get_ipython().magic(u'timeit luminance_1976_vectorise(L)')

print('\n')

###############################################################################
###############################################################################
### colour.colorimetry.spectrum
###############################################################################
###############################################################################

###############################################################################
#### colour.SpectralShape
###############################################################################
from colour.colorimetry.spectrum import *

message_box('SpectralShape')


def SpectralShape__contains__(self, wavelength):
    wavelengths = self.range()
    return all(wavelength in wavelengths
               for wavelength in as_array(wavelength))


SpectralShape.__contains__ = SpectralShape__contains__

print(380 in SpectralShape(360, 830, 1))

print((380, 480) in SpectralShape(360, 830, 1))

print((380, 480.5) in SpectralShape(360, 830, 1))

###############################################################################
#### colour.SpectralPowerDistribution
###############################################################################
message_box('SpectralPowerDistribution')


def SpectralPowerDistribution__getitem__(self, wavelength):
    if is_numeric(wavelength):
        # self.data ===> self.__data
        return self.data.__getitem__(wavelength)
    elif is_iterable(wavelength):
        # self.data ===> self.__data
        return np.array([self.data.__getitem__(x) for x in wavelength])
    elif type(wavelength) is slice:
        return self.values[wavelength]
    else:
        raise NotImplementedError(
            '"{0}" type is not supported for indexing!'.format(
                type(wavelength)))


SpectralPowerDistribution.__getitem__ = SpectralPowerDistribution__getitem__

data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
spd = SpectralPowerDistribution('Spd', data)

print(spd[510])

print('\n')

print(spd[(510, 520)])

print('\n')

print(spd[0:-1])


def SpectralPowerDistribution__setitem__(self, wavelength, value):
    if is_numeric(wavelength) or is_iterable(wavelength):
        wavelengths = as_array(wavelength)
    elif type(wavelength) is slice:
        wavelengths = self.wavelengths[wavelength]
    else:
        raise NotImplementedError(
            '"{0}" type is not supported for indexing!'.format(
                type(wavelength)))

    values = np.resize(as_array(value), wavelengths.shape)
    for i in range(len(wavelengths)):
        # self.data ===> self.__data
        self.data.__setitem__(wavelengths[i], values[i])


SpectralPowerDistribution.__setitem__ = SpectralPowerDistribution__setitem__

spd = SpectralPowerDistribution('Spd', {})

spd[510] = 49.67
pprint(list(spd.items))

print('\n')

spd[(520, 530)] = (69.59, 81.73)
pprint(list(spd.items))

print('\n')

spd[(540, 550)] = 88.19
pprint(list(spd.items))

print('\n')

spd[:] = 49.67
pprint(list(spd.items))

print('\n')

spd[0:3] = 69.59
pprint(list(spd.items))


def SpectralPowerDistribution_get(self, wavelength, default=None):
    wavelength = as_array(wavelength)

    return as_numeric(np.array([self.data.get(x, default)
                                for x in wavelength]))


SpectralPowerDistribution.get = SpectralPowerDistribution_get

data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
spd = SpectralPowerDistribution('Spd', data)

print(spd.get(510))

print('\n')

print(spd.get((510, 520)))

print('\n')

print(spd.get((510, 520, 521)))


def SpectralPowerDistribution__contains__(self, wavelength):
    wavelengths = self.wavelengths
    return all(wavelength in wavelengths
               for wavelength in as_array(wavelength))


SpectralPowerDistribution.__contains__ = SpectralPowerDistribution__contains__

data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
spd = SpectralPowerDistribution('Spd', data)

print(510 in spd)

print('\n')

print((510, 520) in spd)

print('\n')

print((510, 520, 521) in spd)

###############################################################################
#### colour.TriSpectralPowerDistribution
###############################################################################
message_box('TriSpectralPowerDistribution')


def TriSpectralPowerDistribution__getitem__(self, wavelength):
    return np.swapaxes(np.array((self.x[wavelength],
                                 self.y[wavelength],
                                 self.z[wavelength])), 0, 1)


TriSpectralPowerDistribution.__getitem__ = TriSpectralPowerDistribution__getitem__

x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

print(tri_spd[510])

print('\n')

print(tri_spd[(510, 520)])

print('\n')

print(tri_spd[0:-1])


def TriSpectralPowerDistribution__setitem__(self, wavelength, value):
    if is_numeric(wavelength) or is_iterable(wavelength):
        wavelengths = as_array(wavelength)
    elif type(wavelength) is slice:
        wavelengths = self.wavelengths[wavelength]
    else:
        raise NotImplementedError(
            '"{0}" type is not supported for indexing!'.format(
                type(wavelength)))

    value = np.resize(as_array(value), (wavelengths.shape[0], 3))
    x, y, z = value[:, 0], value[:, 1], value[:, 2]

    self.x.__setitem__(wavelengths, x)
    self.y.__setitem__(wavelengths, y)
    self.z.__setitem__(wavelengths, z)


TriSpectralPowerDistribution.__setitem__ = TriSpectralPowerDistribution__setitem__

data = {'x_bar': {}, 'y_bar': {}, 'z_bar': {}}
mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

tri_spd[510] = 49.67
pprint(list(tri_spd.items))

print('\n')

tri_spd[(520, 530)] = (69.59, 81.73)
pprint(list(tri_spd.items))

print('\n')

tri_spd[(540, 550)] = ((49.67, 69.59, 81.73), (81.73, 69.59, 49.67))
pprint(list(tri_spd.items))

print('\n')

tri_spd[:] = 49.67
pprint(list(tri_spd.items))

print('\n')

tri_spd[0:3] = ((81.73, 69.59, 49.67), (49.67, 69.59, 81.73))
pprint(list(tri_spd.items))


def TriSpectralPowerDistribution_get(self, wavelength, default=None):
    wavelength = as_array(wavelength)

    return np.squeeze(np.array([(self.x.get(x, default),
                                 self.y.get(x, default),
                                 self.z.get(x, default)) for x in wavelength]))


TriSpectralPowerDistribution.get = TriSpectralPowerDistribution_get

x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

print(tri_spd.get(510))

print('\n')

print(tri_spd.get((510, 520)))

print('\n')

print(tri_spd.get((510, 520, 521)))

###############################################################################
###############################################################################
### colour.colorimetry.transformations
###############################################################################
###############################################################################

###############################################################################
#### colour.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
###############################################################################
from colour import PHOTOPIC_LEFS, RGB_CMFS
from colour.colorimetry.transformations import *


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wavelength):
    wavelength = as_array(wavelength)

    cmfs = RGB_CMFS.get('Wright & Guild 1931 2 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    rgb = rgb_bar / np.sum(rgb_bar)

    M1 = np.array([[0.49000, 0.31000, 0.20000],
                   [0.17697, 0.81240, 0.01063],
                   [0.00000, 0.01000, 0.99000]])

    M2 = np.array([[0.66697, 1.13240, 1.20063],
                   [0.66697, 1.13240, 1.20063],
                   [0.66697, 1.13240, 1.20063]])

    xyz = np.einsum('...i,...ji', rgb, M1)
    xyz /= np.einsum('...i,...ji', rgb, M2)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    V = PHOTOPIC_LEFS.get('CIE 1924 Photopic Standard Observer').clone()
    V.align(cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return as_stack((x_bar, y_bar, z_bar))


print('Reference:')
print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

print('\n')

print('1d array:')
print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(700))

print('\n')

print('2d array:')
print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(
    [700, 700, 700, 700, 700]))

###############################################################################
#### colour.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
###############################################################################
message_box('RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs')


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wavelength):
    wavelength = as_array(wavelength)

    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[0.341080, 0.189145, 0.387529],
                  [0.139058, 0.837460, 0.073316],
                  [0.000000, 0.039553, 2.026200]])

    xyz_bar = np.squeeze(np.einsum('...i,...ji', rgb_bar, M))

    return xyz_bar


print('Reference:')
print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

print('\n')

print('1d array:')
print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(700))

print('\n')

print('2d array:')
print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(
    [700, 700, 700, 700, 700]))


###############################################################################
#### colour.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
###############################################################################
message_box('RGB_10_degree_cmfs_to_LMS_10_degree_cmfs')


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(wavelength):
    wavelength = as_array(wavelength)

    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[0.1923252690, 0.749548882, 0.0675726702],
                  [0.0192290085, 0.940908496, 0.113830196],
                  [0.0000000000, 0.0105107859, 0.991427669]])

    lms_bar = np.einsum('...i,...ji', rgb_bar, M)

    lms_bar[wavelength > 505, 2] = 0

    lms_bar = np.squeeze(lms_bar)

    return lms_bar


print('Reference:')
print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700))

print('\n')

print('1d array:')
print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(700))

print('\n')

print('2d array:')
print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(
    [700, 700, 700, 700, 700]))


###############################################################################
#### colour.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
###############################################################################
from colour import LMS_CMFS

message_box('LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs')


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wavelength):
    wavelength = as_array(wavelength)

    cmfs = LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals')

    try:
        lms_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[1.94735469, -1.41445123, 0.36476327],
                  [0.68990272, 0.34832189, 0.00000000],
                  [0.00000000, 0.00000000, 1.93485343]])

    xyz_bar = np.squeeze(np.einsum('...i,...ji', lms_bar, M))

    return xyz_bar


print('Reference:')
print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

print('\n')

print('1d array:')
print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(700))

print('\n')

print('2d array:')
print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(
    [700, 700, 700, 700, 700]))

###############################################################################
#### colour.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
###############################################################################
message_box('LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs')


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wavelength):
    wavelength = as_array(wavelength)

    cmfs = LMS_CMFS.get('Stockman & Sharpe 10 Degree Cone Fundamentals')

    try:
        lms_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[1.93986443, -1.34664359, 0.43044935],
                  [0.69283932, 0.34967567, 0.00000000],
                  [0.00000000, 0.00000000, 2.14687945]])

    xyz_bar = np.squeeze(np.einsum('...i,...ji', lms_bar, M))

    return xyz_bar


print('Reference:')
print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

print('\n')

print('1d array:')
print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(700))

print('\n')

print('2d array:')
print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(
    [700, 700, 700, 700, 700]))

###############################################################################
###############################################################################
### colour.colorimetry.tristimulus
###############################################################################
###############################################################################

###############################################################################
#### colour.wavelength_to_XYZ
###############################################################################
from colour.colorimetry.tristimulus import *

message_box('wavelength_to_XYZ')


def wavelength_to_XYZ_2d(wavelengths):
    for wavelength in wavelengths:
        wavelength_to_XYZ(wavelength)


WAVELENGTHS = np.linspace(400, 700, 1000)

# get_ipython().magic(u'timeit wavelength_to_XYZ_2d(WAVELENGTHS)')

from colour import (
    STANDARD_OBSERVERS_CMFS,
    SpragueInterpolator,
    SplineInterpolator)


def wavelength_to_XYZ_vectorise(wavelength,
                                cmfs=STANDARD_OBSERVERS_CMFS.get(
                                    'CIE 1931 2 Degree Standard Observer')):
    wavelength = as_array(wavelength)

    shape = cmfs.shape
    if np.min(wavelength) < shape.start or np.max(wavelength) > shape.end:
        raise ValueError(
            '"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, shape.start, shape.end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values
        interpolator = (SpragueInterpolator
                        if cmfs.is_uniform() else
                        SplineInterpolator)

        interpolators = [interpolator(wavelengths, values[:, i])
                         for i in range(values.shape[-1])]

        return np.squeeze(np.swapaxes(np.array([interpolator(wavelength)
                                                for interpolator in
                                                interpolators]), 0, 1))
    else:
        return np.array(cmfs.get(wavelength))


print('Reference:')
print(wavelength_to_XYZ(480))

print('\n')

print('1d array:')
print(wavelength_to_XYZ_vectorise(480))

print('\n')

print('2d array:')
print(wavelength_to_XYZ_vectorise([480, 480, 480, 480, 480]))

print('\n')

print('2d array:')
print(wavelength_to_XYZ_vectorise([480, 480, 480, 480, 480.5]))

# get_ipython().magic(u'timeit wavelength_to_XYZ_vectorise(WAVELENGTHS)')

print('\n')

###############################################################################
###############################################################################
### colour.colorimetry.whiteness
###############################################################################
###############################################################################

###############################################################################
#### colour.whiteness_Berger1959
###############################################################################
from colour.colorimetry.whiteness import *

message_box('whiteness_Berger1959')


def whiteness_Berger1959_2d(XYZ, XYZ_0):
    for i in range(len(XYZ)):
        whiteness_Berger1959(XYZ[i], XYZ_0[i])


# get_ipython().magic(u'timeit whiteness_Berger1959_2d(DATA1, DATA2)')


def whiteness_Berger1959_vectorise(XYZ, XYZ_0):
    XYZ = as_array(XYZ, (-1, 3))
    XYZ_0 = np.resize(as_array(XYZ_0), XYZ.shape)

    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    X_0, Y_0, Z_0 = XYZ_0[:, 0], XYZ_0[:, 1], XYZ_0[:, 2]

    WI = np.squeeze(0.333 * Y + 125 * (Z / Z_0) - 125 * (X / X_0))

    return WI


print('Reference:')
XYZ = np.array([95., 100., 105.])
XYZ_0 = np.array([94.80966767, 100., 107.30513595])
print(whiteness_Berger1959(XYZ, XYZ_0))

print('\n')

print('1d array:')
print(whiteness_Berger1959_vectorise(XYZ, XYZ_0))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
XYZ_0 = np.tile(XYZ_0, (5, 1))
print(whiteness_Berger1959_vectorise(XYZ, XYZ_0))

# get_ipython().magic(u'timeit whiteness_Berger1959_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
#### colour.whiteness_Taube1960
###############################################################################
message_box('whiteness_Taube1960')


def whiteness_Taube1960_2d(XYZ, XYZ_0):
    for i in range(len(XYZ)):
        whiteness_Taube1960(XYZ[i], XYZ_0[i])


# get_ipython().magic(u'timeit whiteness_Taube1960_2d(DATA1, DATA2)')


def whiteness_Taube1960_vectorise(XYZ, XYZ_0):
    XYZ = as_array(XYZ, (-1, 3))
    XYZ_0 = np.resize(as_array(XYZ_0), XYZ.shape)
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    X_0, Y_0, Z_0 = XYZ_0[:, 0], XYZ_0[:, 1], XYZ_0[:, 2]

    WI = np.squeeze(400 * (Z / Z_0) - 3 * Y)

    return WI


print('Reference:')
XYZ = np.array([95., 100., 105.])
XYZ_0 = np.array([94.80966767, 100., 107.30513595])
print(whiteness_Taube1960(XYZ, XYZ_0))

print('\n')

print('1d array:')
print(whiteness_Taube1960_vectorise(XYZ, XYZ_0))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
XYZ_0 = np.tile(XYZ_0, (5, 1))
print(whiteness_Taube1960_vectorise(XYZ, XYZ_0))

# get_ipython().magic(u'timeit whiteness_Taube1960_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
#### colour.whiteness_Stensby1968
###############################################################################
message_box('whiteness_Stensby1968')


def whiteness_Stensby1968_2d(Lab):
    for i in range(len(Lab)):
        whiteness_Stensby1968(Lab[i])


# get_ipython().magic(u'timeit whiteness_Stensby1968_2d(DATA1)')


def whiteness_Stensby1968_vectorise(Lab):
    Lab = as_array(Lab, (-1, 3))
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]

    WI = np.squeeze(L - 3 * b + 3 * a)

    return WI


print('Reference:')
Lab = np.array([100., -2.46875131, -16.72486654])
print(whiteness_Stensby1968(Lab))

print('\n')

print('1d array:')
print(whiteness_Stensby1968_vectorise(Lab))

print('\n')

print('2d array:')
Lab = np.tile(Lab, (5, 1))
print(whiteness_Stensby1968_vectorise(Lab))

# get_ipython().magic(u'timeit whiteness_Stensby1968_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.whiteness_ASTM313
###############################################################################
message_box('whiteness_ASTM313')


def whiteness_ASTM313_2d(XYZ):
    for i in range(len(XYZ)):
        whiteness_ASTM313(XYZ[i])


# get_ipython().magic(u'timeit whiteness_ASTM313_2d(DATA1)')


def whiteness_ASTM313_vectorise(XYZ):
    XYZ = as_array(XYZ, (-1, 3))
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    WI = np.squeeze(3.388 * Z - 3 * Y)

    return WI


print('Reference:')
XYZ = np.array([95., 100., 105.])
print(whiteness_ASTM313(XYZ))

print('\n')

print('1d array:')
print(whiteness_ASTM313_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(whiteness_ASTM313_vectorise(XYZ))

# get_ipython().magic(u'timeit whiteness_ASTM313_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.whiteness_Ganz1979
###############################################################################
message_box('whiteness_Ganz1979')


def whiteness_Ganz1979_2d(xy, Y):
    for i in range(len(xy)):
        whiteness_Ganz1979(xy[i], Y[i])


# get_ipython().magic(u'timeit whiteness_Ganz1979_2d(DATA1[:,0:2], DATA2[:,0])')


def whiteness_Ganz1979_vectorise(xy, Y):
    xy = as_array(xy, (-1, 2))
    x, y = xy[:, 0], xy[:, 1]

    Y = np.resize(as_array(Y), x.shape)

    W = Y - 1868.322 * x - 3695.690 * y + 1809.441
    T = -1001.223 * x + 748.366 * y + 68.261

    WT = as_stack((W, T))

    return WT


print('Reference:')
xy = (0.3167, 0.3334)
Y = 100.
print(whiteness_Ganz1979(xy, Y))

print('\n')

print('1d array:')
print(whiteness_Ganz1979_vectorise(xy, Y))

print('\n')

print('2d array:')
xy = np.tile(xy, (5, 1))
Y = np.tile(Y, (5, 1))
print(whiteness_Ganz1979_vectorise(xy, Y))

# get_ipython().magic(u'timeit whiteness_Ganz1979_vectorise(DATA1[:,0:2], DATA2[:,0])')

print('\n')

###############################################################################
#### colour.whiteness_CIE2004
###############################################################################
message_box('whiteness_CIE2004')


def whiteness_CIE2004_2d(xy, Y, xy_n):
    for i in range(len(xy)):
        whiteness_CIE2004(xy[i], Y[i], xy_n[i])


# get_ipython().magic(u'timeit whiteness_CIE2004_2d(DATA1[:,0:2], DATA2[:,0], DATA1[:,0:2])')


def whiteness_CIE2004_vectorise(xy,
                                Y,
                                xy_n,
                                observer='CIE 1931 2 Degree Standard Observer'):
    xy = as_array(xy, (-1, 2))
    x, y = xy[:, 0], xy[:, 1]

    Y = np.resize(as_array(Y), x.shape)

    xy_n = as_array(xy_n, (-1, 2))
    xy_n = np.resize(as_array(xy_n), xy.shape)
    x_n, y_n = xy_n[:, 0], xy_n[:, 1]

    W = Y + 800 * (x_n - x) + 1700 * (y_n - y)
    T = (1000 if '1931' in observer else 900) * (x_n - x) - 650 * (y_n - y)

    WT = as_stack((W, T))

    return WT


print('Reference:')
xy = (0.3167, 0.3334)
Y = 100.
xy_n = (0.3139, 0.3311)
print(whiteness_CIE2004(xy, Y, xy_n))

print('\n')

print('1d array:')
print(whiteness_CIE2004_vectorise(xy, Y, xy_n))

print('\n')

print('2d array:')
xy = np.tile(xy, (5, 1))
Y = np.tile(Y, (5, 1))
xy_n = np.tile(xy_n, (5, 1))
print(whiteness_CIE2004_vectorise(xy, Y, xy_n))

# get_ipython().magic(u'timeit whiteness_CIE2004_vectorise(DATA1[:,0:2], DATA2[:,0], DATA1[:,0:2])')

print('\n')

###############################################################################
###############################################################################
### colour.difference.delta_e
###############################################################################
###############################################################################

################################################################################
### colour.delta_E_CIE1976
###############################################################################
from colour.difference.delta_e import *

message_box('delta_E_CIE1976')


def delta_E_CIE1976_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CIE1976(Lab1[i], Lab2[i])


# get_ipython().magic(u'timeit delta_E_CIE1976_2d(DATA1, DATA2)')


def delta_E_CIE1976_vectorise(Lab1, Lab2, **kwargs):
    Lab1 = as_array(Lab1, (-1, 3))
    Lab2 = as_array(Lab2, (-1, 3))

    return as_numeric(np.linalg.norm(np.array(Lab1) - np.array(Lab2), axis=1))


print('Reference:')
Lab1 = np.array([100, 21.57210357, 272.2281935])
Lab2 = np.array([100, 426.67945353, 72.39590835])
print(delta_E_CIE1976(Lab1, Lab2))

print('\n')

print('1d array:')
print(delta_E_CIE1976_vectorise(Lab1, Lab2))

print('\n')

print('2d array:')
Lab1 = np.tile(Lab1, (5, 1))
Lab2 = np.tile(Lab2, (5, 1))
print(delta_E_CIE1976_vectorise(Lab1, Lab2))

# get_ipython().magic(u'timeit delta_E_CIE1976_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
#### colour.delta_E_CIE1994
###############################################################################
message_box('delta_E_CIE1994')


def delta_E_CIE1994_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CIE1994(Lab1[i], Lab2[i])


# get_ipython().magic(u'timeit delta_E_CIE1994_2d(DATA1, DATA2)')


def delta_E_CIE1994_vectorise(Lab1, Lab2, textiles=True, **kwargs):
    k1 = 0.048 if textiles else 0.045
    k2 = 0.014 if textiles else 0.015
    kL = 2 if textiles else 1
    kC = 1
    kH = 1

    Lab1 = as_array(Lab1, (-1, 3))
    Lab2 = as_array(Lab2, (-1, 3))

    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)

    sL = 1
    sC = 1 + k1 * C1
    sH = 1 + k2 * C1

    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_A = a1 - a2
    delta_B = b1 - b2

    delta_H = np.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)

    L = (delta_L / (kL * sL)) ** 2
    C = (delta_C / (kC * sC)) ** 2
    H = (delta_H / (kH * sH)) ** 2

    return as_numeric(np.sqrt(L + C + H))


print('Reference:')
Lab1 = np.array([100, 21.57210357, 272.2281935])
Lab2 = np.array([100, 426.67945353, 72.39590835])
print(delta_E_CIE1994(Lab1, Lab2))

print('\n')

print('1d array:')
print(delta_E_CIE1994_vectorise(Lab1, Lab2))

print('\n')

print('2d array:')
Lab1 = np.tile(Lab1, (5, 1))
Lab2 = np.tile(Lab2, (5, 1))
print(delta_E_CIE1994_vectorise(Lab1, Lab2))

# get_ipython().magic(u'timeit delta_E_CIE1994_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
#### colour.delta_E_CIE2000
###############################################################################
message_box('delta_E_CIE2000')


def delta_E_CIE2000_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CIE2000(Lab1[i], Lab2[i])


# get_ipython().magic(u'timeit delta_E_CIE2000_2d(DATA1, DATA2)')


def delta_E_CIE2000_vectorise(Lab1, Lab2, **kwargs):
    kL = 1
    kC = 1
    kH = 1

    Lab1 = as_array(Lab1, (-1, 3))
    Lab2 = as_array(Lab2, (-1, 3))

    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = np.rad2deg(np.arctan2(b1, a1_prime))
    h1_prime[h1_prime < 0.0] += 360

    h2_prime = np.rad2deg(np.arctan2(b2, a2_prime))
    h2_prime[h2_prime < 0.0] += 360

    h_bar_prime = np.where(np.fabs(h1_prime - h2_prime) <= 180,
                           0.5 * (h1_prime + h2_prime),
                           (0.5 * (h1_prime + h2_prime + 360)))

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h2_prime - h1_prime
    delta_h_prime = np.where(h2_prime <= h1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * np.sqrt(c1_prime * c2_prime) *
                     np.sin(np.deg2rad(0.5 * delta_h_prime)))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * np.sin(np.deg2rad(2 * delta_theta))

    return as_numeric(np.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT))


print('Reference:')
Lab1 = np.array([100, 21.57210357, 272.2281935])
Lab2 = np.array([100, 426.67945353, 72.39590835])
print(delta_E_CIE2000(Lab1, Lab2))

print('\n')

print('1d array:')
print(delta_E_CIE2000_vectorise(Lab1, Lab2))

print('\n')

print('2d array:')
Lab1 = np.tile(Lab1, (5, 1))
Lab2 = np.tile(Lab2, (5, 1))
print(delta_E_CIE2000_vectorise(Lab1, Lab2))

# get_ipython().magic(u'timeit delta_E_CIE2000_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
#### colour.delta_E_CMC
###############################################################################
message_box('delta_E_CMC')


def delta_E_CMC_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CMC(Lab1[i], Lab2[i])


# get_ipython().magic(u'timeit delta_E_CMC_2d(DATA1, DATA2)')


def delta_E_CMC_vectorise(Lab1, Lab2, l=2, c=1):
    Lab1 = as_array(Lab1, (-1, 3))
    Lab2 = as_array(Lab2, (-1, 3))

    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    sl = np.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1))
    sc = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    h1 = np.where(c1 < 0.000001, 0, np.rad2deg(np.arctan2(b1, a1)))

    while np.any(h1 < 0):
        h1[h1 < 0] += 360

    while np.any(h1 >= 360):
        h1[h1 >= 360] -= 360

    t = np.where(np.logical_and(h1 >= 164, h1 <= 345),
                 0.56 + np.fabs(0.2 * np.cos(np.deg2rad(h1 + 168))),
                 0.36 + np.fabs(0.4 * np.cos(np.deg2rad(h1 + 35))))

    c4 = c1 * c1 * c1 * c1
    f = np.sqrt(c4 / (c4 + 1900))
    sh = sc * (f * t + 1 - f)

    delta_L = L1 - L2
    delta_C = c1 - c2
    delta_A = a1 - a2
    delta_B = b1 - b2
    delta_H2 = delta_A * delta_A + delta_B * delta_B - delta_C * delta_C

    v1 = delta_L / (l * sl)
    v2 = delta_C / (c * sc)
    v3 = sh

    return as_numeric(np.sqrt(v1 * v1 + v2 * v2 + (delta_H2 / (v3 * v3))))


print('Reference:')
Lab1 = np.array([100, 21.57210357, 272.2281935])
Lab2 = np.array([100, 426.67945353, 72.39590835])
print(delta_E_CMC(Lab1, Lab2))

print('\n')

print('1d array:')
print(delta_E_CMC_vectorise(Lab1, Lab2))

print('\n')

print('2d array:')
Lab1 = np.tile(Lab1, (5, 1))
Lab2 = np.tile(Lab2, (5, 1))
print(delta_E_CMC_vectorise(Lab1, Lab2))

# get_ipython().magic(u'timeit delta_E_CMC_vectorise(DATA1, DATA2)')

print('\n')

###############################################################################
###############################################################################
### colour.models.cie_xyy
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_xyY
###############################################################################
from colour.models.cie_xyy import *

message_box('XYZ_to_xyY')


def XYZ_to_xyY_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_xyY(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_xyY_2d(DATA1)')

from colour import ILLUMINANTS


@handle_numpy_errors(divide='ignore', invalid='ignore')
def XYZ_to_xyY_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))
    illuminant = np.resize(as_array(illuminant, (-1, 2)), XYZ.shape)

    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    xyY = as_stack(
        np.where(XYZ == 0,
                 np.dstack((illuminant[:, 0], illuminant[:, 1], Y)),
                 np.dstack((X / (X + Y + Z), Y / (X + Y + Z), Y))),
        shape=shape)

    return xyY


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_xyY(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_xyY_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_xyY_vectorise(XYZ))

XYZ = np.tile((0, 0, 0), (5, 1))
print(XYZ_to_xyY_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_xyY_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.xyY_to_XYZ
###############################################################################
message_box('xyY_to_XYZ')


def xyY_to_XYZ_2d(xyY):
    for i in range(len(xyY)):
        xyY_to_XYZ(xyY[i])


# get_ipython().magic(u'timeit xyY_to_XYZ_2d(DATA1)')


@handle_numpy_errors(divide='ignore')
def xyY_to_XYZ_vectorise(xyY):
    shape = as_shape(xyY)
    xyY = as_array(xyY, (-1, 3))

    x, y, Y = xyY[:, 0], xyY[:, 1], xyY[:, 2]

    XYZ = as_stack(
        np.where((y == 0)[:, np.newaxis],
                 np.dstack((y, y, y)),
                 np.dstack((x * Y / y, Y, (1 - x - y) * Y / y))),
        shape=shape)

    return XYZ


print('Reference:')
xyY = np.array([0.26414772, 0.37770001, 0.1008])
print(xyY_to_XYZ(xyY))

print('\n')

print('1d array:')
print(xyY_to_XYZ_vectorise(xyY))

print('\n')

print('2d array:')
xyY = np.tile(xyY, (5, 1))
print(xyY_to_XYZ_vectorise(xyY))

# get_ipython().magic(u'timeit xyY_to_XYZ_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.xy_to_XYZ
###############################################################################
message_box('xy_to_XYZ')


def xy_to_XYZ_2d(xy):
    for i in range(len(xy)):
        xy_to_XYZ(xy[i])


# get_ipython().magic(u'timeit xy_to_XYZ_2d(DATA1[:, 0:2])')


def xy_to_XYZ_vectorise(xy):
    xy = as_array(xy, (-1, 2))
    return xyY_to_XYZ_vectorise(np.dstack((xy[:, 0],
                                           xy[:, 1],
                                           np.ones(xy.shape[0]))))


print('Reference:')
xy = (0.26414772236966133, 0.37770000704815188)
print(xy_to_XYZ(xy))

print('\n')

print('1d array:')
print(xy_to_XYZ_vectorise(xy))

print('\n')

print('2d array:')
xy = np.tile(xy, (5, 1))
print(xy_to_XYZ_vectorise(xy))

# get_ipython().magic(u'timeit xy_to_XYZ_vectorise(DATA1[:, 0:2])')

print('\n')

###############################################################################
#### colour.XYZ_to_xy
###############################################################################
message_box('XYZ_to_xy')


def XYZ_to_xy_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_xy(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_xy_2d(DATA1)')


def XYZ_to_xy_vectorise(XYZ,
                        illuminant=ILLUMINANTS.get(
                            'CIE 1931 2 Degree Standard Observer').get('D50')):
    xyY = as_array(XYZ_to_xyY_vectorise(XYZ, illuminant), (-1, 3))
    xy = np.squeeze(xyY[:, 0:2])

    return xy


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_xy(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_xy_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_xy_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_xy_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
### colour.models.cie_lab
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_Lab
###############################################################################
from colour.models.cie_lab import *

message_box('XYZ_to_Lab')


def XYZ_to_Lab_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_Lab(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_Lab_2d(DATA1)')


def XYZ_to_Lab_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))
    XYZ_r = np.resize(xy_to_XYZ_vectorise(illuminant), XYZ.shape)

    XYZ_f = XYZ / XYZ_r

    XYZ_f = np.where(XYZ_f > CIE_E,
                     np.power(XYZ_f, 1 / 3),
                     (CIE_K * XYZ_f + 16) / 116)

    X_f, Y_f, Z_f = XYZ_f[:, 0], XYZ_f[:, 1], XYZ_f[:, 2]

    L = 116 * Y_f - 16
    a = 500 * (X_f - Y_f)
    b = 200 * (Y_f - Z_f)

    Lab = as_stack((L, a, b), shape=shape)

    return Lab


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_Lab(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_Lab_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_Lab_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_Lab_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.Lab_to_XYZ
###############################################################################
message_box('Lab_to_XYZ')


def Lab_to_XYZ_2d(Lab):
    for i in range(len(Lab)):
        Lab_to_XYZ(Lab[i])


# get_ipython().magic(u'timeit Lab_to_XYZ_2d(DATA1)')


def Lab_to_XYZ_vectorise(Lab,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(Lab)
    Lab = as_array(Lab, (-1, 3))
    XYZ_r = np.resize(xy_to_XYZ_vectorise(illuminant), Lab.shape)

    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]

    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200

    x_r = np.where(f_x ** 3 > CIE_E, f_x ** 3, (116 * f_x - 16) / CIE_K)
    y_r = np.where(L > CIE_K * CIE_E, ((L + 16) / 116) ** 3, L / CIE_K)
    z_r = np.where(f_z ** 3 > CIE_E, f_z ** 3, (116 * f_z - 16) / CIE_K)

    XYZ = as_stack(as_stack((x_r, y_r, z_r)) * XYZ_r, direction='I',
                   shape=shape)

    return XYZ


print('Reference:')
Lab = np.array([37.9856291, -23.62302887, -4.41417036])
print(Lab_to_XYZ(Lab))

print('\n')

print('1d array:')
print(Lab_to_XYZ_vectorise(Lab))

print('\n')

print('2d array:')
Lab = np.tile(Lab, (5, 1))
print(Lab_to_XYZ_vectorise(Lab))

# get_ipython().magic(u'timeit Lab_to_XYZ_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.Lab_to_LCHab
###############################################################################
message_box('Lab_to_LCHab')


def Lab_to_LCHab_2d(Lab):
    for i in range(len(Lab)):
        Lab_to_LCHab(Lab[i])


# get_ipython().magic(u'timeit Lab_to_LCHab_2d(DATA1)')


def Lab_to_LCHab_vectorise(Lab):
    shape = as_shape(Lab)
    Lab = as_array(Lab, (-1, 3))
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]

    H = 180 * np.arctan2(b, a) / np.pi
    H[H < 0] += 360

    LCHab = as_stack((L, np.sqrt(a ** 2 + b ** 2), H), shape=shape)

    return LCHab


print('Reference:')
Lab = np.array([37.9856291, -23.62302887, -4.41417036])
print(Lab_to_LCHab(Lab))

print('\n')

print('1d array:')
print(Lab_to_LCHab_vectorise(Lab))

print('\n')

print('2d array:')
Lab = np.tile(Lab, (5, 1))
print(Lab_to_LCHab_vectorise(Lab))

# get_ipython().magic(u'timeit Lab_to_LCHab_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.LCHab_to_Lab
###############################################################################
message_box('LCHab_to_Lab')


def LCHab_to_Lab_2d(LCHab):
    for i in range(len(LCHab)):
        LCHab_to_Lab(LCHab[i])


# get_ipython().magic(u'timeit LCHab_to_Lab_2d(DATA1)')


def LCHab_to_Lab_vectorise(LCHab):
    shape = as_shape(LCHab)
    LCHab = as_array(LCHab, (-1, 3))
    L, C, H = LCHab[:, 0], LCHab[:, 1], LCHab[:, 2]

    return as_stack((L,
                     C * np.cos(np.radians(H)),
                     C * np.sin(np.radians(H))),
                    shape=shape)


print('Reference:')
LCHab = np.array([37.9856291, 24.03190365, 190.58415972])
print(LCHab_to_Lab(LCHab))

print('\n')

print('1d array:')
print(LCHab_to_Lab_vectorise(LCHab))

print('\n')

print('2d array:')
LCHab = np.tile(LCHab, (5, 1))
print(LCHab_to_Lab_vectorise(LCHab))

# get_ipython().magic(u'timeit LCHab_to_Lab_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
### colour.models.cie_luv
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_Luv
###############################################################################
from colour.models.cie_luv import *

message_box('XYZ_to_Luv')


def XYZ_to_Luv_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_Luv(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_Luv_2d(DATA1)')


def XYZ_to_Luv_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))
    XYZ_r = np.resize(xy_to_XYZ_vectorise(illuminant), XYZ.shape)

    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    X_r, Y_r, Z_r = XYZ_r[:, 0], XYZ_r[:, 1], XYZ_r[:, 2]

    y_r = Y / Y_r

    L = np.where(y_r > CIE_E, 116 * y_r ** (1 / 3) - 16, CIE_K * y_r)

    u = (13 * L * ((4 * X / (X + 15 * Y + 3 * Z)) -
                   (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r))))
    v = (13 * L * ((9 * Y / (X + 15 * Y + 3 * Z)) -
                   (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))))

    Luv = as_stack((L, u, v), shape=shape)

    return Luv


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_Luv(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_Luv_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_Luv_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_Luv_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.Luv_to_XYZ
###############################################################################
message_box('Luv_to_XYZ')


def Luv_to_XYZ_2d(Luv):
    for i in range(len(Luv)):
        Luv_to_XYZ(Luv[i])


# get_ipython().magic(u'timeit Luv_to_XYZ_2d(DATA1)')


def Luv_to_XYZ_vectorise(Luv,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(Luv)
    Luv = as_array(Luv, (-1, 3))
    XYZ_r = np.resize(xy_to_XYZ_vectorise(illuminant), Luv.shape)

    L, u, v = Luv[:, 0], Luv[:, 1], Luv[:, 2]
    X_r, Y_r, Z_r = XYZ_r[:, 0], XYZ_r[:, 1], XYZ_r[:, 2]

    Y = np.where(L > CIE_E * CIE_K, ((L + 16) / 116) ** 3, L / CIE_K)

    a = 1 / 3 * ((52 * L / (u + 13 * L *
                            (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L *
                       (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    XYZ = as_stack((X, Y, Z), shape=shape)

    return XYZ


print('Reference:')
Luv = np.array([37.9856291, -28.79229446, -1.3558195])
print(Luv_to_XYZ(Luv))

print('\n')

print('1d array:')
print(Luv_to_XYZ_vectorise(Luv))

print('\n')

print('2d array:')
Luv = np.tile(Luv, (5, 1))
print(Luv_to_XYZ_vectorise(Luv))

# get_ipython().magic(u'timeit Luv_to_XYZ_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.Luv_to_uv
###############################################################################
message_box('Luv_to_uv')


def Luv_to_uv_2d(Luv):
    for i in range(len(Luv)):
        Luv_to_uv(Luv[i])


# get_ipython().magic(u'timeit Luv_to_uv_2d(DATA1)')


def Luv_to_uv_vectorise(Luv,
                        illuminant=ILLUMINANTS.get(
                            'CIE 1931 2 Degree Standard Observer').get('D50')):
    XYZ = as_array(Luv_to_XYZ_vectorise(Luv, illuminant), (-1, 3))
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    uv = as_stack((4 * X / (X + 15 * Y + 3 * Z),
                   9 * Y / (X + 15 * Y + 3 * Z)))

    return uv


print('Reference:')
Luv = np.array([37.9856291, -28.79229446, -1.3558195])
print(Luv_to_uv(Luv))

print('\n')

print('1d array:')
print(Luv_to_uv_vectorise(Luv))

print('\n')

print('2d array:')
Luv = np.tile(Luv, (5, 1))
print(Luv_to_uv_vectorise(Luv))

# get_ipython().magic(u'timeit Luv_to_uv_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.Luv_uv_to_xy
###############################################################################
message_box('Luv_uv_to_xy')


def Luv_uv_to_xy_2d(uv):
    for i in range(len(uv)):
        Luv_uv_to_xy(uv[i])


# get_ipython().magic(u'timeit Luv_uv_to_xy_2d(DATA1[:, 0:2])')


def Luv_uv_to_xy_vectorise(uv):
    uv = as_array(uv, (-1, 2))
    u, v = uv[:, 0], uv[:, 1]

    xy = as_stack((9 * u / (6 * u - 16 * v + 12),
                   4 * v / (6 * u - 16 * v + 12)))
    return xy


print('Reference:')
uv = np.array([0.15085309882985695, 0.48532970854318019])
print(Luv_uv_to_xy(uv))

print('\n')

print('1d array:')
print(Luv_uv_to_xy_vectorise(uv))

print('\n')

print('2d array:')
uv = np.tile(uv, (5, 1))
print(Luv_uv_to_xy_vectorise(uv))

# get_ipython().magic(u'timeit Luv_uv_to_xy_vectorise(DATA1[:, 0:2])')

print('\n')

###############################################################################
#### colour.Luv_to_LCHuv
###############################################################################
message_box('Luv_to_LCHuv')


def Luv_to_LCHuv_2d(Luv):
    for i in range(len(Luv)):
        Luv_to_LCHuv(Luv[i])


# get_ipython().magic(u'timeit Luv_to_LCHuv_2d(DATA1)')


def Luv_to_LCHuv_vectorise(Luv):
    shape = as_shape(Luv)
    Luv = as_array(Luv, (-1, 3))
    L, u, v = Luv[:, 0], Luv[:, 1], Luv[:, 2]

    H = 180 * np.arctan2(v, u) / np.pi
    H[H < 0] += 360

    LCHuv = as_stack((L,
                      np.sqrt(u ** 2 + v ** 2),
                      H),
                     shape=shape)
    return LCHuv


print('Reference:')
Luv = np.array([37.9856291, -28.79229446, -1.3558195])
print(Luv_to_LCHuv(Luv))

print('\n')

print('1d array:')
print(Luv_to_LCHuv_vectorise(Luv))

print('\n')

print('2d array:')
Luv = np.tile(Luv, (5, 1))
print(Luv_to_LCHuv_vectorise(Luv))

# get_ipython().magic(u'timeit Luv_to_LCHuv_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.LCHuv_to_Luv
###############################################################################
message_box('LCHuv_to_Luv')


def LCHuv_to_Luv_2d(LCHuv):
    for i in range(len(LCHuv)):
        LCHuv_to_Luv(LCHuv[i])


# get_ipython().magic(u'timeit LCHuv_to_Luv_2d(DATA1)')


def LCHuv_to_Luv_vectorise(LCHuv):
    shape = as_shape(LCHuv)
    LCHuv = as_array(LCHuv, (-1, 3))
    L, C, H = LCHuv[:, 0], LCHuv[:, 1], LCHuv[:, 2]

    Luv = as_stack((L,
                    C * np.cos(np.radians(H)),
                    C * np.sin(np.radians(H))),
                   shape=shape)
    return Luv


print('Reference:')
LCHuv = np.array([37.9856291, 28.82419933, 182.69604747])
print(LCHuv_to_Luv(LCHuv))

print('\n')

print('1d array:')
print(LCHuv_to_Luv_vectorise(LCHuv))

print('\n')

print('2d array:')
LCHuv = np.tile(LCHuv, (5, 1))
print(LCHuv_to_Luv_vectorise(LCHuv))

# get_ipython().magic(u'timeit LCHuv_to_Luv_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
### colour.models.cie_ucs
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_UCS
###############################################################################
from colour.models.cie_ucs import *

message_box('XYZ_to_UCS')


def XYZ_to_UCS_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_UCS(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_UCS_2d(DATA1)')


def XYZ_to_UCS_vectorise(XYZ):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    UVW = as_stack((2 / 3 * X,
                    Y,
                    1 / 2 * (-X + 3 * Y + Z)),
                   shape=shape)

    return UVW


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_UCS(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_UCS_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_UCS_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_UCS_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.UCS_to_XYZ
###############################################################################
message_box('UCS_to_XYZ')


def UCS_to_XYZ_2d(UVW):
    for i in range(len(UVW)):
        UCS_to_XYZ(UVW[i])


# get_ipython().magic(u'timeit UCS_to_XYZ_2d(DATA1)')


def UCS_to_XYZ_vectorise(UVW):
    shape = as_shape(UVW)
    UVW = as_array(UVW, (-1, 3))
    U, V, W = UVW[:, 0], UVW[:, 1], UVW[:, 2]

    XYZ = as_stack((3 / 2 * U,
                    V,
                    3 / 2 * U - (3 * V) + (2 * W)),
                   shape=shape)

    return XYZ


print('Reference:')
UVW = np.array([0.04699689, 0.1008, 0.1637439])
print(UCS_to_XYZ(UVW))

print('\n')

print('1d array:')
print(UCS_to_XYZ_vectorise(UVW))

print('\n')

print('2d array:')
UVW = np.tile(UVW, (5, 1))
print(UCS_to_XYZ_vectorise(UVW))

# get_ipython().magic(u'timeit UCS_to_XYZ_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.UCS_to_uv
###############################################################################
message_box('UCS_to_uv')


def UCS_to_uv_2d(UVW):
    for i in range(len(UVW)):
        UCS_to_uv(UVW[i])


# get_ipython().magic(u'timeit UCS_to_uv_2d(DATA1)')


def UCS_to_uv_vectorise(UVW):
    UVW = as_array(UVW, (-1, 3))
    U, V, W = UVW[:, 0], UVW[:, 1], UVW[:, 2]

    uv = as_stack((U / (U + V + W), V / (U + V + W)))

    return uv


print('Reference:')
UVW = np.array([0.04699689, 0.1008, 0.1637439])
print(UCS_to_uv(UVW))

print('\n')

print('1d array:')
print(UCS_to_uv_vectorise(UVW))

print('\n')

print('2d array:')
UVW = np.tile(UVW, (5, 1))
print(UCS_to_uv_vectorise(UVW))

# get_ipython().magic(u'timeit UCS_to_uv_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.UCS_uv_to_xy
###############################################################################
message_box('UCS_uv_to_xy')


def UCS_uv_to_xy_2d(uv):
    for i in range(len(uv)):
        UCS_uv_to_xy(uv[i])


# get_ipython().magic(u'timeit UCS_uv_to_xy_2d(DATA1[:, 0:2])')


def UCS_uv_to_xy_vectorise(uv):
    uv = as_array(uv, (-1, 2))
    u, v = uv[:, 0], uv[:, 1]

    xy = as_stack((3 * u / (2 * u - 8 * v + 4),
                   2 * v / (2 * u - 8 * v + 4)))

    return xy


print('Reference:')
uv = np.array([0.15085308732766581, 0.3235531372954405])
print(UCS_uv_to_xy(uv))

print('\n')

print('1d array:')
print(UCS_uv_to_xy_vectorise(uv))

print('\n')

print('2d array:')
uv = np.tile(uv, (5, 1))
print(UCS_uv_to_xy_vectorise(uv))

# get_ipython().magic(u'timeit UCS_uv_to_xy_vectorise(DATA1[:, 0:2])')

print('\n')

###############################################################################
###############################################################################
### colour.models.cie_uvw
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_UVW
###############################################################################
from colour.models.cie_uvw import *


message_box('XYZ_to_UVW')


def XYZ_to_UVW_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_UVW(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_UVW_2d(DATA1)')


def XYZ_to_UVW_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    shape = as_shape(XYZ)
    xyY = as_array(XYZ_to_xyY_vectorise(XYZ, illuminant), (-1, 3))
    x, y, Y = xyY[:, 0], xyY[:, 1], xyY[:, 2]

    uv = as_array(UCS_to_uv_vectorise(
        as_array(XYZ_to_UCS_vectorise(XYZ),
                 (-1, 3))),
                  (-1, 2))
    u, v = uv[:, 0], uv[:, 1]

    uv_0 = as_array(UCS_to_uv_vectorise(
        as_array(XYZ_to_UCS_vectorise(
            as_array(xy_to_XYZ_vectorise(illuminant),
                     (-1, 3))),
                 (-1, 3))),
                    (-1, 2))
    u_0, v_0 = uv_0[:, 0], uv_0[:, 1]

    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u_0)
    V = 13 * W * (v - v_0)

    UVW = as_stack((U, V, W), shape=shape)

    return UVW


print('Reference:')
XYZ = np.array([0.07049534, 0.1008, 0.09558313])
print(XYZ_to_UVW(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_UVW_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_UVW_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_UVW_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
### colour.models.deprecated
###############################################################################
###############################################################################

###############################################################################
#### colour.models.deprecated.RGB_to_HSV
###############################################################################
from colour.models.deprecated import *

message_box('RGB_to_HSV')


def RGB_to_HSV_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_HSV(RGB[i])


# get_ipython().magic(u'timeit RGB_to_HSV_2d(DATA1)')


@handle_numpy_errors(divide='ignore', invalid='ignore')
def RGB_to_HSV_vectorise(RGB):
    shape = as_shape(RGB)
    RGB = as_array(RGB, (-1, 3))

    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    V = maximum

    R, G, B = RGB[:, 0], RGB[:, 1], RGB[:, 2]

    S = delta / maximum
    S[delta == 0] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[H < 0] += 1
    H[H > 1] -= 1
    H[delta == 0] = 0

    HSV = as_stack((H, S, V), shape=shape)

    return HSV


print('Reference:')
RGB = np.array([0.49019608, 0.98039216, 0.25098039])
print(RGB_to_HSV(RGB))

print('\n')

print('1d array:')
print(RGB_to_HSV_vectorise(RGB))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_HSV_vectorise(RGB))

# get_ipython().magic(u'timeit RGB_to_HSV_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.HSV_to_RGB
###############################################################################
message_box('HSV_to_RGB')


def HSV_to_RGB_2d(HSV):
    for i in range(len(HSV)):
        HSV_to_RGB(HSV[i])


# get_ipython().magic(u'timeit HSV_to_RGB_2d(DATA1)')


def HSV_to_RGB_vectorise(HSV):
    shape = as_shape(HSV)
    HSV = as_array(HSV, (-1, 3))

    H, S, V = HSV[:, 0], HSV[:, 1], HSV[:, 2]

    h = H * 6
    h[h == 6] = 0

    i = np.floor(h)
    j = V * (1 - S)
    k = V * (1 - S * (h - i))
    l = V * (1 - S * (1 - (h - i)))

    i = np.dstack((i, i, i)).astype(np.uint8)
    RGB = as_stack(np.choose(i,
                             (np.dstack((V, l, j)),
                              np.dstack((k, V, j)),
                              np.dstack((j, V, l)),
                              np.dstack((j, k, V)),
                              np.dstack((l, j, V)),
                              np.dstack((V, j, k)))),
                   shape=shape)

    return RGB


print('Reference:')
HSV = np.array([0.27867383, 0.744, 0.98039216])
print(HSV_to_RGB(HSV))

print('\n')

print('1d array:')
print(HSV_to_RGB_vectorise(HSV))

print('\n')

print('2d array:')
HSV = np.tile(HSV, (5, 1))
print(HSV_to_RGB_vectorise(HSV))

# get_ipython().magic(u'timeit HSV_to_RGB_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.RGB_to_HSL
###############################################################################
message_box('RGB_to_HSL')


def RGB_to_HSL_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_HSL(RGB[i])


# get_ipython().magic(u'timeit RGB_to_HSL_2d(DATA1)')


@handle_numpy_errors(divide='ignore', invalid='ignore')
def RGB_to_HSL_vectorise(RGB):
    shape = as_shape(RGB)
    RGB = as_array(RGB, (-1, 3))

    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    R, G, B = RGB[:, 0], RGB[:, 1], RGB[:, 2]

    L = (maximum + minimum) / 2

    S = np.where(L < 0.5,
                 delta / (maximum + minimum),
                 delta / (2 - maximum - minimum))
    S[delta == 0] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[H < 0] += 1
    H[H > 1] -= 1
    H[delta == 0] = 0

    HSL = as_stack((H, S, L), shape=shape)

    return HSL


print('Reference:')
RGB = np.array([0.49019608, 0.98039216, 0.25098039])
print(RGB_to_HSL(RGB))

print('\n')

print('1d array:')
print(RGB_to_HSL_vectorise(RGB))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_HSL_vectorise(RGB))

# get_ipython().magic(u'timeit RGB_to_HSL_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.HSL_to_RGB
###############################################################################
message_box('HSL_to_RGB')


def HSL_to_RGB_2d(HSL):
    for i in range(len(HSL)):
        HSL_to_RGB(HSL[i])


# get_ipython().magic(u'timeit HSL_to_RGB_2d(DATA1)')


def HSL_to_RGB_vectorise(HSL):
    shape = as_shape(HSL)
    HSL = as_array(HSL, (-1, 3))

    H, S, L = HSL[:, 0], HSL[:, 1], HSL[:, 2]

    def H_to_RGB(vi, vj, vH):
        """
        Converts *hue* value to *RGB* colourspace.
        """

        vH[vH < 0] += 1
        vH[vH > 1] -= 1

        v = np.full(vi.shape, np.nan)

        v = np.where(np.logical_and(6 * vH < 1, np.isnan(v)),
                     vi + (vj - vi) * 6 * vH,
                     v)
        v = np.where(np.logical_and(2 * vH < 1, np.isnan(v)),
                     vj,
                     v)
        v = np.where(np.logical_and(3 * vH < 2, np.isnan(v)),
                     vi + (vj - vi) * ((2 / 3) - vH) * 6,
                     v)
        v = np.where(np.isnan(v), vi, v)

        return v

    j = np.where(L < 0.5, L * (1 + S), (L + S) - (S * L))
    i = 2 * L - j

    R = H_to_RGB(i, j, H + (1 / 3))
    G = H_to_RGB(i, j, H)
    B = H_to_RGB(i, j, H - (1 / 3))

    R = np.where(S == 1, L, R)
    G = np.where(S == 1, L, G)
    B = np.where(S == 1, L, B)

    RGB = as_stack((R, G, B), shape=shape)

    return RGB


print('Reference:')
HSL = np.array([0.27867383, 0.9489796, 0.61568627])
print(HSL_to_RGB(HSL))

print('\n')

print('1d array:')
print(HSL_to_RGB_vectorise(HSL))

print('\n')

print('2d array:')
HSL = np.tile(HSL, (5, 1))
print(HSL_to_RGB_vectorise(HSL))

# get_ipython().magic(u'timeit HSL_to_RGB_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.RGB_to_CMY
###############################################################################
message_box('RGB_to_CMY')


def RGB_to_CMY_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_CMY(RGB[i])


# get_ipython().magic(u'timeit RGB_to_CMY_2d(DATA1)')


def RGB_to_CMY_vectorise(RGB):
    shape = as_shape(RGB)
    RGB = as_array(RGB, (-1, 3))

    CMY = as_stack(1 - RGB, shape=shape)

    return CMY


print('Reference:')
RGB = np.array([0.49019608, 0.98039216, 0.25098039])
print(RGB_to_CMY(RGB))

print('\n')

print('1d array:')
print(RGB_to_CMY_vectorise(RGB))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_CMY_vectorise(RGB))

# get_ipython().magic(u'timeit RGB_to_CMY_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.CMY_to_RGB
###############################################################################
message_box('CMY_to_RGB')


def CMY_to_RGB_2d(CMY):
    for i in range(len(CMY)):
        CMY_to_RGB(CMY[i])


# get_ipython().magic(u'timeit CMY_to_RGB_2d(DATA1)')


def CMY_to_RGB_vectorise(CMY):
    shape = as_shape(CMY)
    CMY = as_array(CMY, (-1, 3))

    RGB = as_stack(1 - CMY, shape=shape)

    return RGB


print('Reference:')
CMY = np.array([0.50980392, 0.01960784, 0.74901961])
print(CMY_to_RGB(CMY))

print('\n')

print('1d array:')
print(CMY_to_RGB_vectorise(CMY))

print('\n')

print('2d array:')
CMY = np.tile(CMY, (5, 1))
print(CMY_to_RGB_vectorise(CMY))

# get_ipython().magic(u'timeit CMY_to_RGB_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.CMY_to_CMYK
###############################################################################
message_box('CMY_to_CMYK')


def CMY_to_CMYK_2d(CMY):
    for i in range(len(CMY)):
        CMY_to_CMYK(CMY[i])


# get_ipython().magic(u'timeit CMY_to_CMYK_2d(DATA1)')


def CMY_to_CMYK_vectorise(CMY):
    CMY = as_array(CMY, (-1, 3))
    C, M, Y = CMY[:, 0], CMY[:, 1], CMY[:, 2]

    K = np.ones(C.shape)
    K = np.where(C < K, C, K)
    K = np.where(M < K, M, K)
    K = np.where(Y < K, Y, K)

    C = (C - K) / (1 - K)
    M = (M - K) / (1 - K)
    Y = (Y - K) / (1 - K)

    C[K == 1] = 0
    M[K == 1] = 0
    Y[K == 1] = 0

    CMYK = as_stack((C, M, Y, K))

    return CMYK


print('Reference:')
CMY = np.array([0.49019608, 0.98039216, 0.25098039])
print(CMY_to_CMYK(CMY))

print('\n')

print('1d array:')
print(CMY_to_CMYK_vectorise(CMY))

print('\n')

print('2d array:')
CMY = np.tile(CMY, (5, 1))
print(CMY_to_CMYK_vectorise(CMY))

# get_ipython().magic(u'timeit CMY_to_CMYK_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.models.deprecated.CMYK_to_CMY
###############################################################################
message_box('CMYK_to_CMY')


def CMYK_to_CMY_2d(CMYK):
    for i in range(len(CMYK)):
        CMYK_to_CMY(CMYK[i])


# get_ipython().magic(u'timeit CMYK_to_CMY_2d(np.resize(DATA1, (-1, 4)))')


def CMYK_to_CMY_vectorise(CMYK):
    CMYK = as_array(CMYK, (-1, 4))
    C, M, Y, K = CMYK[:, 0], CMYK[:, 1], CMYK[:, 2], CMYK[:, 3]

    CMY = as_stack((C * (1 - K) + K, M * (1 - K) + K, Y * (1 - K) + K))

    return CMY


print('Reference:')
CMYK = np.array([0.31937173, 0.97382199, 0., 0.25098039])
print(CMYK_to_CMY(CMYK))

print('\n')

print('1d array:')
print(CMYK_to_CMY_vectorise(CMYK))

print('\n')

print('2d array:')
CMYK = np.tile(CMYK, (5, 1))
print(CMYK_to_CMY_vectorise(CMYK))

# get_ipython().magic(u'timeit CMYK_to_CMY_vectorise(np.resize(DATA1, (-1, 4)))')

print('\n')

###############################################################################
###############################################################################
### colour.models.derivation
###############################################################################
###############################################################################

###############################################################################
#### colour.RGB_luminance
###############################################################################
from colour.models.derivation import *

message_box('RGB_luminance')


def RGB_luminance_2d(RGB):
    for i in range(len(RGB)):
        RGB_luminance(RGB[i],
                      np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010,
                                -0.07700]),
                      (0.32168, 0.33767))


# get_ipython().magic(u'timeit RGB_luminance_2d(DATA1)')


def RGB_luminance_vectorise(RGB, primaries, whitepoint):
    RGB = as_array(RGB, (-1, 3))
    X, Y, Z = np.ravel(normalised_primary_matrix(primaries,
                                                 whitepoint))[3:6]

    R, G, B = RGB[:, 0], RGB[:, 1], RGB[:, 2]

    return np.squeeze(X * R + Y * G + Z * B)


print('Reference:')
RGB = np.array([40.6, 4.2, 67.4])
P = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
W = (0.32168, 0.33767)
print(RGB_luminance(RGB, P, W))

print('\n')

print('1d array:')
print(RGB_luminance_vectorise(RGB, P, W))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_luminance_vectorise(RGB, P, W))

# get_ipython().magic(u'timeit RGB_luminance_vectorise(DATA1, P, W)')

print('\n')

###############################################################################
###############################################################################
### colour.models.ipt
###############################################################################
###############################################################################

###############################################################################
#### colour.XYZ_to_IPT
###############################################################################
from colour.models.ipt import *

message_box('XYZ_to_IPT')


def XYZ_to_IPT_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_IPT(XYZ[i])


# get_ipython().magic(u'timeit XYZ_to_IPT_2d(DATA1)')


def XYZ_to_IPT_vectorise(XYZ):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))

    LMS = np.einsum('...i,...ji', XYZ, IPT_XYZ_TO_LMS_MATRIX)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43
    IPT = as_stack(np.einsum('...i,...ji', LMS_prime, IPT_LMS_TO_IPT_MATRIX),
                   direction='I',
                   shape=shape)

    return IPT


print('Reference:')
XYZ = np.array([0.96907232, 1, 1.12179215])
print(XYZ_to_IPT(XYZ))

print('\n')

print('1d array:')
print(XYZ_to_IPT_vectorise(XYZ))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_IPT_vectorise(XYZ))

# get_ipython().magic(u'timeit XYZ_to_IPT_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
#### colour.IPT_to_XYZ
###############################################################################
###############################################################################
message_box('IPT_to_XYZ')


def IPT_to_XYZ_2d(IPT):
    for i in range(len(IPT)):
        IPT_to_XYZ(IPT[i])


# get_ipython().magic(u'timeit IPT_to_XYZ_2d(DATA1)')


def IPT_to_XYZ_vectorise(IPT):
    shape = as_shape(IPT)
    IPT = as_array(IPT, (-1, 3))

    LMS = np.einsum('...i,...ji', IPT, IPT_IPT_TO_LMS_MATRIX)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** (1 / 0.43)
    XYZ = as_stack(np.einsum('...i,...ji', LMS_prime, IPT_LMS_TO_XYZ_MATRIX),
                   direction='I',
                   shape=shape)

    return XYZ


print('Reference:')
IPT = np.array([1.00300825, 0.01906918, -0.01369292])
print(IPT_to_XYZ(IPT))

print('\n')

print('1d array:')
print(IPT_to_XYZ_vectorise(IPT))

print('\n')

print('2d array:')
IPT = np.tile(IPT, (5, 1))
print(IPT_to_XYZ_vectorise(IPT))

# get_ipython().magic(u'timeit IPT_to_XYZ_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.IPT_hue_angle
###############################################################################
message_box('IPT_hue_angle')


def IPT_hue_angle_2d(IPT):
    for i in range(len(IPT)):
        IPT_hue_angle(IPT[i])


# get_ipython().magic(u'timeit IPT_hue_angle_2d(DATA1)')


def IPT_hue_angle_vectorise(IPT):
    IPT = as_array(IPT, (-1, 3))

    hue = np.arctan2(IPT[:, 2], IPT[:, 1])

    return hue


print('Reference:')
IPT = np.array([0.96907232, 1., 1.12179215])
print(IPT_hue_angle(IPT))

print('\n')

print('1d array:')
print(IPT_hue_angle_vectorise(IPT))

print('\n')

print('2d array:')
IPT = np.tile(IPT, (5, 1))
print(IPT_hue_angle_vectorise(IPT))

# get_ipython().magic(u'timeit IPT_hue_angle_vectorise(DATA1)')

print('\n')

###############################################################################
###############################################################################
### colour.models.log
###############################################################################
###############################################################################

###############################################################################
#### colour.linear_to_cineon
###############################################################################
from colour.models.log import *

message_box('linear_to_cineon')


def linear_to_cineon_2d(value):
    for i in range(len(value)):
        linear_to_cineon(value[i])


DATA = np.linspace(0, 1, 1000000)

# get_ipython().magic(u'timeit linear_to_cineon_2d(DATA)')


def linear_to_cineon_vectorise(value, black_offset=10 ** ((95 - 685) / 300),
                               **kwargs):
    value = as_array(value)

    return as_numeric(((685 +
                        300 * np.log10(value * (
                            1 - black_offset) + black_offset)) / 1023))


print('Reference:')
print(linear_to_cineon(0.18))

print('\n')

print('1d array:')
print(linear_to_cineon_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_cineon_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_cineon_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.cineon_to_linear
###############################################################################
message_box('cineon_to_linear')


def cineon_to_linear_2d(value):
    for i in range(len(value)):
        cineon_to_linear(value[i])


# get_ipython().magic(u'timeit cineon_to_linear_2d(DATA)')


def cineon_to_linear_vectorise(value, black_offset=10 ** ((95 - 685) / 300),
                               **kwargs):
    value = as_array(value)

    return as_numeric(((10 ** ((1023 * value - 685) / 300) - black_offset) /
                       (1 - black_offset)))


print('Reference:')
print(cineon_to_linear(0.5))

print('\n')

print('1d array:')
print(cineon_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(cineon_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

###############################################################################
#### colour.linear_to_panalog
###############################################################################
message_box('linear_to_panalog')


def linear_to_panalog_2d(value):
    for i in range(len(value)):
        linear_to_panalog(value[i])


# get_ipython().magic(u'timeit linear_to_panalog_2d(DATA)')


def linear_to_panalog_vectorise(value, black_offset=10 ** ((64 - 681) / 444),
                                **kwargs):
    value = as_array(value)

    return as_numeric(((681 +
                        444 * np.log10(value * (
                            1 - black_offset) + black_offset)) / 1023))


print('Reference:')
print(linear_to_panalog(0.18))

print('\n')

print('1d array:')
print(linear_to_panalog_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_panalog_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_panalog_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.panalog_to_linear
###############################################################################
message_box('panalog_to_linear')


def panalog_to_linear_2d(value):
    for i in range(len(value)):
        panalog_to_linear(value[i])


# get_ipython().magic(u'timeit panalog_to_linear_2d(DATA)')


def panalog_to_linear_vectorise(value, black_offset=10 ** ((64 - 681) / 444),
                                **kwargs):
    value = as_array(value)

    return as_numeric(((10 ** ((1023 * value - 681) / 444) - black_offset) /
                       (1 - black_offset)))


print('Reference:')
print(panalog_to_linear(0.5))

print('\n')

print('1d array:')
print(panalog_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(panalog_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

# get_ipython().magic(u'timeit panalog_to_linear_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.linear_to_red_log
###############################################################################
message_box('linear_to_red_log')


def linear_to_red_log_2d(value):
    for i in range(len(value)):
        linear_to_red_log(value[i])


# get_ipython().magic(u'timeit linear_to_red_log(DATA)')


def linear_to_red_log_vectorise(value, black_offset=10 ** ((0 - 1023) / 511),
                                **kwargs):
    value = as_array(value)

    return as_numeric(((1023 +
                        511 * np.log10(value * (
                            1 - black_offset) + black_offset)) / 1023))


print('Reference:')
print(linear_to_red_log(0.18))

print('\n')

print('1d array:')
print(linear_to_red_log_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_red_log_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_red_log_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.red_log_to_linear
###############################################################################
message_box('red_log_to_linear')


def red_log_to_linear_2d(value):
    for i in range(len(value)):
        red_log_to_linear(value[i])


# get_ipython().magic(u'timeit red_log_to_linear_2d(DATA)')


def red_log_to_linear_vectorise(value, black_offset=10 ** ((0 - 1023) / 511),
                                **kwargs):
    value = as_array(value)

    return as_numeric((((10 ** ((1023 * value - 1023) / 511)) - black_offset) /
                       (1 - black_offset)))


print('Reference:')
print(red_log_to_linear(0.5))

print('\n')

print('1d array:')
print(red_log_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(red_log_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

# get_ipython().magic(u'timeit red_log_to_linear_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.linear_to_viper_log
###############################################################################
message_box('linear_to_viper_log')


def linear_to_viper_log_2d(value):
    for i in range(len(value)):
        linear_to_viper_log(value[i])


# get_ipython().magic(u'timeit linear_to_viper_log_2d(DATA)')


def linear_to_viper_log_vectorise(value, **kwargs):
    value = as_array(value)

    return as_numeric((1023 + 500 * np.log10(value)) / 1023)


print('Reference:')
print(linear_to_viper_log(0.18))

print('\n')

print('1d array:')
print(linear_to_viper_log_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_viper_log_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_viper_log_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.viper_log_to_linear
###############################################################################
message_box('viper_log_to_linear')


def viper_log_to_linear_2d(value):
    for i in range(len(value)):
        viper_log_to_linear(value[i])


# get_ipython().magic(u'timeit viper_log_to_linear_2d(DATA)')


def viper_log_to_linear_vectorise(value, **kwargs):
    value = as_array(value)

    return as_numeric(10 ** ((1023 * value - 1023) / 500))


print('Reference:')
print(viper_log_to_linear(0.5))

print('\n')

print('1d array:')
print(viper_log_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(viper_log_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

# get_ipython().magic(u'timeit viper_log_to_linear_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.linear_to_pivoted_log
###############################################################################
message_box('linear_to_pivoted_log')


def linear_to_pivoted_log_2d(value):
    for i in range(len(value)):
        linear_to_pivoted_log(value[i])


# get_ipython().magic(u'timeit linear_to_pivoted_log_2d(DATA)')


def linear_to_pivoted_log_vectorise(value,
                                    log_reference=445,
                                    linear_reference=0.18,
                                    negative_gamma=0.6,
                                    density_per_code_value=0.002):
    value = as_array(value)

    return as_numeric(((log_reference + np.log10(value / linear_reference) /
                        (density_per_code_value / negative_gamma)) / 1023))


print('Reference:')
print(linear_to_pivoted_log(0.18))

print('\n')

print('1d array:')
print(linear_to_pivoted_log_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_pivoted_log_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_pivoted_log_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.pivoted_log_to_linear
###############################################################################
message_box('pivoted_log_to_linear')


def pivoted_log_to_linear_2d(value):
    for i in range(len(value)):
        pivoted_log_to_linear(value[i])


# get_ipython().magic(u'timeit pivoted_log_to_linear_2d(DATA)')


def pivoted_log_to_linear_vectorise(value,
                                    log_reference=445,
                                    linear_reference=0.18,
                                    negative_gamma=0.6,
                                    density_per_code_value=0.002):
    value = as_array(value)

    return as_numeric((10 ** ((value * 1023 - log_reference) *
                              (density_per_code_value / negative_gamma)) *
                       linear_reference))


print('Reference:')
print(pivoted_log_to_linear(0.5))

print('\n')

print('1d array:')
print(pivoted_log_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(pivoted_log_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

# get_ipython().magic(u'timeit pivoted_log_to_linear_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.linear_to_c_log
###############################################################################
message_box('linear_to_c_log')


def linear_to_c_log_2d(value):
    for i in range(len(value)):
        linear_to_c_log(value[i])


# get_ipython().magic(u'timeit linear_to_c_log_2d(DATA)')


def linear_to_c_log_vectorise(value, **kwargs):
    value = as_array(value)

    return as_numeric(0.529136 * np.log10(10.1596 * value + 1) + 0.0730597)


print('Reference:')
print(linear_to_c_log(0.18))

print('\n')

print('1d array:')
print(linear_to_c_log_vectorise(0.18))

print('\n')

print('2d array:')
print(linear_to_c_log_vectorise(
    [0.18, 0.18, 0.18, 0.18, 0.18]))

# get_ipython().magic(u'timeit linear_to_c_log_vectorise(DATA)')

print('\n')

###############################################################################
#### colour.c_log_to_linear
###############################################################################
message_box('c_log_to_linear')


def c_log_to_linear_2d(value):
    for i in range(len(value)):
        c_log_to_linear(value[i])


# get_ipython().magic(u'timeit c_log_to_linear_2d(DATA)')


def c_log_to_linear_vectorise(value, **kwargs):
    value = as_array(value)

    return as_numeric(-0.071622555735168 * (
        1.3742747797867 - np.exp(1) ** (4.3515940948906 * value)))


print('Reference:')
print(c_log_to_linear(0.5))

print('\n')

print('1d array:')
print(c_log_to_linear_vectorise(0.5))

print('\n')

print('2d array:')
print(c_log_to_linear_vectorise(
    [0.5, 0.5, 0.5, 0.5, 0.5]))

# get_ipython().magic(u'timeit c_log_to_linear_vectorise(DATA)')

print('\n')

###############################################################################
###############################################################################
### colour.models.rgb
###############################################################################
###############################################################################

###############################################################################
#### OECF / OECF_i
###############################################################################
from colour.models.dataset.aces import _aces_cc_transfer_function, \
    _aces_cc_inverse_transfer_function
from colour.models.dataset.aces import *

message_box('OECF / OECF_i')

RGB = np.array([0.86969452, 1.00516431, 1.41715848])
RGB_t = np.tile(RGB, (5, 1))


def _aces_cc_transfer_function_vectorise(value):
    value = as_array(value)

    output = np.where(value < 0,
                      (np.log2(2 ** -15 * 0.5) + 9.72) / 17.52,
                      (np.log2(2 ** -16 + value * 0.5) + 9.72) / 17.52)
    output = np.where(value >= 2 ** -15,
                      (np.log2(value) + 9.72) / 17.52,
                      output)

    return as_numeric(output)


print(_aces_cc_transfer_function(RGB[0]))

print(_aces_cc_transfer_function_vectorise(RGB[0]))

print(_aces_cc_transfer_function_vectorise(RGB))

print(_aces_cc_transfer_function_vectorise(RGB_t))


def _aces_cc_inverse_transfer_function_vectorise(value):
    value = as_array(value)

    output = np.where(value < (9.72 - 15) / 17.52,
                      (2 ** (value * 17.52 - 9.72) - 2 ** -16) * 2,
                      2 ** (value * 17.52 - 9.72))
    output = np.where(value >= (np.log2(65504) + 9.72) / 17.52,
                      65504,
                      output)

    return as_numeric(output)


print(_aces_cc_inverse_transfer_function(RGB[0]))

print(_aces_cc_inverse_transfer_function_vectorise(RGB[0]))

print(_aces_cc_inverse_transfer_function_vectorise(RGB))

print(_aces_cc_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.aces import _aces_proxy_transfer_function, \
    _aces_proxy_inverse_transfer_function


def _aces_proxy_transfer_function_vectorise(value, bit_depth='10 Bit'):
    value = as_array(value)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    CV_min = np.resize(constants.CV_min, value.shape)
    CV_max = np.resize(constants.CV_max, value.shape)

    float_2_cv = lambda x: np.maximum(CV_min, np.minimum(CV_max, np.round(x)))

    output = np.where(value > 2 ** -9.72,
                      float_2_cv((np.log2(value) + constants.mid_log_offset) *
                                 constants.steps_per_stop + constants.mid_CV_offset),
                      np.resize(CV_min, value.shape))
    return as_numeric(output)


print(_aces_proxy_transfer_function(RGB[0]))

print(_aces_proxy_transfer_function_vectorise(RGB[0]))

print(_aces_proxy_transfer_function_vectorise(RGB))

print(_aces_proxy_transfer_function_vectorise(RGB_t))


def _aces_proxy_inverse_transfer_function_vectorise(value, bit_depth='10 Bit'):
    value = as_array(value)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    return as_numeric((2 ** (((value - constants.mid_CV_offset) /
                              constants.steps_per_stop - constants.mid_log_offset))))


print(_aces_proxy_inverse_transfer_function(RGB[0]))

print(_aces_proxy_inverse_transfer_function_vectorise(RGB[0]))

print(_aces_proxy_inverse_transfer_function_vectorise(RGB))

print(_aces_proxy_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.adobe_rgb_1998 import \
    _adobe_rgb_1998_transfer_function, \
    _adobe_rgb_1998_inverse_transfer_function
from colour.models.dataset.adobe_rgb_1998 import *


def _adobe_rgb_1998_transfer_function_vectorise(value):
    # Also valid for:
    # _adobe_wide_gamut_rgb_transfer_function
    return as_numeric(as_array(value) ** (1 / (563 / 256)))


print(_adobe_rgb_1998_transfer_function(RGB[0]))

print(_adobe_rgb_1998_transfer_function_vectorise(RGB[0]))

print(_adobe_rgb_1998_transfer_function_vectorise(RGB))

print(_adobe_rgb_1998_transfer_function_vectorise(RGB_t))


def _adobe_rgb_1998_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _adobe_wide_gamut_rgb_inverse_transfer_function
    return as_numeric(as_array(value) ** (563 / 256))


print(_adobe_rgb_1998_inverse_transfer_function(RGB[0]))

print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB[0]))

print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB))

print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.alexa_wide_gamut_rgb import \
    _alexa_wide_gamut_rgb_transfer_function, \
    _alexa_wide_gamut_rgb_inverse_transfer_function
from colour.models.dataset.alexa_wide_gamut_rgb import *


def _alexa_wide_gamut_rgb_transfer_function_vectorise(
        value,
        firmware='SUP 3.x',
        method='Linear Scene Exposure Factor',
        EI=800):
    value = as_array(value)

    cut, a, b, c, d, e, f, _ = ALEXA_LOG_C_CURVE_CONVERSION_DATA.get(
        firmware).get(method).get(EI)

    return as_numeric(
        np.where(value > cut, c * np.log10(a * value + b) + d, e * value + f))


print(_alexa_wide_gamut_rgb_transfer_function(RGB[0]))

print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB[0]))

print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB))

print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB_t))


def _alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(
        value,
        firmware='SUP 3.x',
        method='Linear Scene Exposure Factor',
        EI=800):
    value = as_array(value)

    cut, a, b, c, d, e, f, _ = (
        ALEXA_LOG_C_CURVE_CONVERSION_DATA.get(firmware).get(method).get(EI))

    return as_numeric(np.where(value > e * cut + f,
                               (np.power(10., (value - d) / c) - b) / a,
                               (value - f) / e))


print(_alexa_wide_gamut_rgb_inverse_transfer_function(RGB[0]))

print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB[0]))

print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB))

print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.apple_rgb import _apple_rgb_transfer_function, \
    _apple_rgb_inverse_transfer_function
from colour.models.dataset.apple_rgb import *


def _apple_rgb_transfer_function_vectorise(value):
    # Also valid for:
    # _color_match_rgb_transfer_function
    return as_numeric(as_array(value) ** (1 / 1.8))


print(_apple_rgb_transfer_function(RGB[0]))

print(_apple_rgb_transfer_function_vectorise(RGB[0]))

print(_apple_rgb_transfer_function_vectorise(RGB))

print(_apple_rgb_transfer_function_vectorise(RGB_t))


def _apple_rgb_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _color_match_rgb_inverse_transfer_function
    return as_numeric(as_array(value) ** 1.8)


print(_apple_rgb_inverse_transfer_function(RGB[0]))

print(_apple_rgb_inverse_transfer_function_vectorise(RGB[0]))

print(_apple_rgb_inverse_transfer_function_vectorise(RGB))

print(_apple_rgb_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.best_rgb import _best_rgb_transfer_function, \
    _best_rgb_inverse_transfer_function
from colour.models.dataset.best_rgb import *


def _best_rgb_transfer_function_vectorise(value):
    # Also valid for:
    # _beta_rgb_transfer_function
    # _cie_rgb_transfer_function
    # _don_rgb_4_transfer_function
    # _ekta_space_ps_5_transfer_function
    # _max_rgb_transfer_function
    # _ntsc_rgb_transfer_function
    # _russell_rgb_transfer_function
    # _smpte_c_rgb_transfer_function
    # _xtreme_rgb_transfer_function
    return as_numeric(as_array(value) ** (1 / 2.2))


print(_best_rgb_transfer_function(RGB[0]))

print(_best_rgb_transfer_function_vectorise(RGB[0]))

print(_best_rgb_transfer_function_vectorise(RGB))

print(_best_rgb_transfer_function_vectorise(RGB_t))


def _best_rgb_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _beta_rgb_inverse_transfer_function
    # _cie_rgb_inverse_transfer_function
    # _don_rgb_4_inverse_transfer_function
    # _ekta_space_ps_5_inverse_transfer_function
    # _max_rgb_inverse_transfer_function
    # _ntsc_rgb_inverse_transfer_function
    # _russell_rgb_inverse_transfer_function
    # _smpte_c_rgb_inverse_transfer_function
    # _xtreme_rgb_inverse_transfer_function
    return as_numeric(as_array(value) ** 2.2)


print(_best_rgb_inverse_transfer_function(RGB[0]))

print(_best_rgb_inverse_transfer_function_vectorise(RGB[0]))

print(_best_rgb_inverse_transfer_function_vectorise(RGB))

print(_best_rgb_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.dci_p3 import _dci_p3_transfer_function, \
    _dci_p3_inverse_transfer_function
from colour.models.dataset.dci_p3 import *


def _dci_p3_transfer_function_vectorise(value):
    return as_numeric(4095 * (as_array(value) / 52.37) ** (1 / 2.6))


print(_dci_p3_transfer_function(RGB[0]))

print(_dci_p3_transfer_function_vectorise(RGB[0]))

print(_dci_p3_transfer_function_vectorise(RGB))

print(_dci_p3_transfer_function_vectorise(RGB_t))


def _dci_p3_inverse_transfer_function_vectorise(value):
    return as_numeric(52.37 * (as_array(value) / 4095) ** 2.6)


print(_dci_p3_inverse_transfer_function(RGB[0]))

print(_dci_p3_inverse_transfer_function_vectorise(RGB[0]))

print(_dci_p3_inverse_transfer_function_vectorise(RGB))

print(_dci_p3_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.pal_secam_rgb import \
    _pal_secam_rgb_transfer_function, _pal_secam_rgb_inverse_transfer_function
from colour.models.dataset.pal_secam_rgb import *


def _pal_secam_rgb_transfer_function_vectorise(value):
    return as_numeric(as_array(value) ** (1 / 2.8))


print(_pal_secam_rgb_transfer_function(RGB[0]))

print(_pal_secam_rgb_transfer_function_vectorise(RGB[0]))

print(_pal_secam_rgb_transfer_function_vectorise(RGB))

print(_pal_secam_rgb_transfer_function_vectorise(RGB_t))


def _pal_secam_rgb_inverse_transfer_function_vectorise(value):
    return as_numeric(as_array(value) ** 2.8)


print(_pal_secam_rgb_inverse_transfer_function(RGB[0]))

print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB[0]))

print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB))

print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB_t))


def _prophoto_rgb_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(np.where(value < 0.001953,
                               value * 16,
                               value ** (1 / 1.8)))


print(_prophoto_rgb_transfer_function_vectorise(RGB))

print(_prophoto_rgb_transfer_function_vectorise(RGB_t))


def _prophoto_rgb_inverse_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(
        np.where(value < _prophoto_rgb_transfer_function_vectorise(0.001953),
                 value / 16,
                 value ** 1.8))


print(_prophoto_rgb_inverse_transfer_function_vectorise(RGB))

print(_prophoto_rgb_inverse_transfer_function_vectorise(RGB_t))


def _rec_709_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(np.where(value < 0.018,
                               value * 4.5,
                               1.099 * (value ** 0.45) - 0.099))


print(_rec_709_transfer_function_vectorise(RGB))

print(_rec_709_transfer_function_vectorise(RGB_t))


def _rec_709_inverse_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(
        np.where(value < _rec_709_transfer_function_vectorise(0.018),
                 value / 4.5,
                 ((value + 0.099) / 1.099) ** (1 / 0.45)))


print(_rec_709_inverse_transfer_function_vectorise(RGB))

print(_rec_709_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.rec_2020 import *


def _rec_2020_transfer_function_vectorise(value, is_10_bits_system=True):
    value = as_array(value)

    a = REC_2020_CONSTANTS.alpha(is_10_bits_system)
    b = REC_2020_CONSTANTS.beta(is_10_bits_system)
    return as_numeric(np.where(value < b,
                               value * 4.5,
                               a * (value ** 0.45) - (a - 1)))


print(_rec_2020_transfer_function_vectorise(RGB))

print(_rec_2020_transfer_function_vectorise(RGB_t))


def _rec_2020_inverse_transfer_function_vectorise(value,
                                                  is_10_bits_system=True):
    value = as_array(value)

    a = REC_2020_CONSTANTS.alpha(is_10_bits_system)
    b = REC_2020_CONSTANTS.beta(is_10_bits_system)
    return as_numeric(
        np.where(value < _rec_2020_transfer_function_vectorise(b),
                 value / 4.5,
                 ((value + (a - 1)) / a) ** (1 / 0.45)))


print(_rec_2020_inverse_transfer_function_vectorise(RGB))

print(_rec_2020_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.s_gamut import (
    _s_log_transfer_function,
    _s_log_inverse_transfer_function,
    _s_log2_transfer_function,
    _s_log2_inverse_transfer_function,
    _s_log3_transfer_function,
    _s_log3_inverse_transfer_function)
from colour.models.dataset.s_gamut import *


def _s_log_transfer_function_vectorise(value):
    return as_numeric(
        (0.432699 * np.log10(as_array(value) + 0.037584) + 0.616596) + 0.03)


print(_s_log_transfer_function(RGB[0]))

print(_s_log_transfer_function_vectorise(RGB[0]))

print(_s_log_transfer_function_vectorise(RGB))

print(_s_log_transfer_function_vectorise(RGB_t))


def _s_log_inverse_transfer_function_vectorise(value):
    return as_numeric(
        10 ** (((as_array(value) - 0.616596 - 0.03) / 0.432699)) - 0.037584)


print(_s_log_inverse_transfer_function(RGB[0]))

print(_s_log_inverse_transfer_function_vectorise(RGB[0]))

print(_s_log_inverse_transfer_function_vectorise(RGB))

print(_s_log_inverse_transfer_function_vectorise(RGB_t))


def _s_log2_transfer_function_vectorise(value):
    return as_numeric(((4 * (16 + 219 * (0.616596 + 0.03 + 0.432699 *
                                         (np.log10(0.037584 + as_array(
                                             value) / 0.9))))) / 1023))


print(_s_log2_transfer_function(RGB[0]))

print(_s_log2_transfer_function_vectorise(RGB[0]))

print(_s_log2_transfer_function_vectorise(RGB))

print(_s_log2_transfer_function_vectorise(RGB_t))


def _s_log2_inverse_transfer_function_vectorise(value):
    return as_numeric(
        ((10 ** (((((as_array(value) * 1023 / 4 - 16) / 219) - 0.616596 - 0.03)
                  / 0.432699)) - 0.037584) * 0.9))


print(_s_log2_inverse_transfer_function(RGB[0]))

print(_s_log2_inverse_transfer_function_vectorise(RGB[0]))

print(_s_log2_inverse_transfer_function_vectorise(RGB))

print(_s_log2_inverse_transfer_function_vectorise(RGB_t))


def _s_log3_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(np.where(value >= 0.01125000,
                               (420 + np.log10((value + 0.01) / (
                                   0.18 + 0.01)) * 261.5) / 1023,
                               (value * (
                                   171.2102946929 - 95) / 0.01125000 + 95) / 1023))


print(_s_log3_transfer_function(RGB[0]))

print(_s_log3_transfer_function_vectorise(RGB[0]))

print(_s_log3_transfer_function_vectorise(RGB))

print(_s_log3_transfer_function_vectorise(RGB_t))


def _s_log3_inverse_transfer_function_vectorise(value):
    value = as_array(value)

    return as_numeric(np.where(value >= 171.2102946929 / 1023,
                               ((10 ** ((value * 1023 - 420) / 261.5)) * (
                                   0.18 + 0.01) - 0.01),
                               (value * 1023 - 95) * 0.01125000 / (
                                   171.2102946929 - 95)))


print(_s_log3_inverse_transfer_function(RGB[0]))

print(_s_log3_inverse_transfer_function_vectorise(RGB[0]))

print(_s_log3_inverse_transfer_function_vectorise(RGB))

print(_s_log3_inverse_transfer_function_vectorise(RGB_t))

from colour.models.dataset.srgb import *


def _srgb_transfer_function(value):
    value = as_array(value)

    return as_numeric(np.where(value <= 0.0031308,
                               value * 12.92,
                               1.055 * (value ** (1 / 2.4)) - 0.055))


print(_srgb_transfer_function(RGB))

print(_srgb_transfer_function(RGB_t))


def _srgb_inverse_transfer_function(value):
    value = as_array(value)

    return as_numeric(np.where(value <= _srgb_transfer_function(0.0031308),
                               value / 12.92,
                               ((value + 0.055) / 1.055) ** 2.4))


print(_srgb_inverse_transfer_function(RGB))

print(_srgb_inverse_transfer_function(RGB_t))

###############################################################################
#### colour.XYZ_to_RGB
###############################################################################
from colour.models.rgb import *

XYZ = np.array([0.07049534, 0.1008, 0.09558313])
W_R = (0.34567, 0.35850)
W_T = (0.31271, 0.32902)
CAT = 'Bradford'
M = np.array([
    [3.24100326, -1.53739899, -0.49861587],
    [-0.96922426, 1.87592999, 0.04155422],
    [0.05563942, -0.2040112, 1.05714897]])

message_box('XYZ_to_RGB')


def XYZ_to_RGB_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_RGB(XYZ[i], W_R, W_T, M, CAT)


# get_ipython().magic(u'timeit XYZ_to_RGB_2d(DATA1)')


def XYZ_to_RGB_vectorise(XYZ,
                         illuminant_XYZ,
                         illuminant_RGB,
                         XYZ_to_RGB_matrix,
                         chromatic_adaptation_transform='CAT02',
                         transfer_function=None):
    shape = as_shape(XYZ)
    XYZ = as_array(XYZ, (-1, 3))

    M = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(illuminant_XYZ),
        xy_to_XYZ_vectorise(illuminant_RGB),
        transform=chromatic_adaptation_transform)

    XYZ_a = np.einsum('...i,...ji', XYZ, M)

    RGB = np.einsum('...i,...ji', XYZ_a, XYZ_to_RGB_matrix)

    if transfer_function is not None:
        RGB = transfer_function(RGB)

    RGB = as_stack(RGB, direction='I', shape=shape)

    return RGB


print('Reference:')
XYZ = np.array([0.96907232, 1, 1.12179215])
print(XYZ_to_RGB(XYZ, W_R, W_T, M, CAT))

print('\n')

print('1d array:')
print(XYZ_to_RGB_vectorise(XYZ, W_R, W_T, M, CAT))

print('\n')

print('2d array:')
XYZ = np.tile(XYZ, (5, 1))
print(XYZ_to_RGB_vectorise(XYZ, W_R, W_T, M, CAT))

# get_ipython().magic(u'timeit XYZ_to_RGB_vectorise(DATA1, W_R, W_T, M, CAT)')

print('\n')

###############################################################################
# #### colour.RGB_to_XYZ
###############################################################################
RGB = np.array([0.86969452, 1.00516431, 1.41715848])
W_R = (0.31271, 0.32902)
W_T = (0.34567, 0.35850)
CAT = 'Bradford'
M = np.array([
    [0.41238656, 0.35759149, 0.18045049],
    [0.21263682, 0.71518298, 0.0721802],
    [0.01933062, 0.11919716, 0.95037259]])

message_box('RGB_to_XYZ')


def RGB_to_XYZ_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_XYZ(RGB[i], W_R, W_T, M, CAT)


# get_ipython().magic(u'timeit RGB_to_XYZ_2d(DATA1)')


def RGB_to_XYZ_vectorise(RGB,
                         illuminant_RGB,
                         illuminant_XYZ,
                         RGB_to_XYZ_matrix,
                         chromatic_adaptation_transform='CAT02',
                         inverse_transfer_function=None):
    shape = as_shape(RGB)
    RGB = as_array(RGB, (-1, 3))

    if inverse_transfer_function is not None:
        RGB = inverse_transfer_function(RGB)

    XYZ = np.einsum('...i,...ji', RGB, RGB_to_XYZ_matrix)

    M = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(illuminant_RGB),
        xy_to_XYZ_vectorise(illuminant_XYZ),
        transform=chromatic_adaptation_transform)

    XYZ_a = np.einsum('...i,...ji', XYZ, M)

    XYZ_a = as_stack(XYZ_a, direction='I', shape=shape)

    return XYZ_a


print('Reference:')
RGB = np.array([0.86969452, 1.00516431, 1.41715848])
print(RGB_to_XYZ(RGB, W_R, W_T, M, CAT))

print('\n')

print('1d array:')
print(RGB_to_XYZ_vectorise(RGB, W_R, W_T, M, CAT))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_XYZ_vectorise(RGB, W_R, W_T, M, CAT))

# get_ipython().magic(u'timeit RGB_to_XYZ_vectorise(DATA1, W_R, W_T, M, CAT)')

print('\n')

###############################################################################
#### colour.RGB_to_RGB
###############################################################################
from colour import sRGB_COLOURSPACE

RGB = np.array([0.86969452, 1.00516431, 1.41715848])
C = sRGB_COLOURSPACE
CAT = 'Bradford'

message_box('RGB_to_RGB')


def RGB_to_RGB_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_RGB(RGB[i], C, C, CAT)


# get_ipython().magic(u'timeit RGB_to_RGB_2d(DATA1)')


def RGB_to_RGB_vectorise(RGB,
                         input_colourspace,
                         output_colourspace,
                         chromatic_adaptation_transform='CAT02'):
    shape = as_shape(RGB)
    RGB = as_array(RGB, (-1, 3))

    cat = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(input_colourspace.whitepoint),
        xy_to_XYZ_vectorise(output_colourspace.whitepoint),
        chromatic_adaptation_transform)

    M = np.einsum('...ij,...jk->...ik',
                  input_colourspace.RGB_to_XYZ_matrix,
                  cat)
    M = np.einsum('...ij,...jk->...ik',
                  M,
                  output_colourspace.XYZ_to_RGB_matrix)

    RGB = as_stack(np.einsum('...i,...ji', RGB, M),
                   direction='I',
                   shape=shape)

    return RGB


print('Reference:')
RGB = np.array([0.86969452, 1.00516431, 1.41715848])
print(RGB_to_RGB(RGB, C, C, CAT))

print('\n')

print('1d array:')
print(RGB_to_RGB_vectorise(RGB, C, C, CAT))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_RGB_vectorise(RGB, C, C, CAT))

# get_ipython().magic(u'timeit RGB_to_RGB_vectorise(DATA1, C, C, CAT)')

print('\n')

###############################################################################
###############################################################################
### colour.notation.munsell
###############################################################################
###############################################################################

###############################################################################
#### colour.munsell_value_Priest1920
###############################################################################
from colour.notation.munsell import *


message_box('munsell_value_Priest1920')


def munsell_value_Priest1920_2d(Y):
    for i in range(len(Y)):
        munsell_value_Priest1920(Y[i])


Y = np.linspace(0, 100, 1000000)

# get_ipython().magic(u'timeit munsell_value_Priest1920_2d(Y)')


def munsell_value_Priest1920_vectorise(Y):
    Y = as_array(Y)

    Y /= 100
    V = 10 * np.sqrt(Y)

    return as_numeric(V)


print('Reference:')
print(munsell_value_Priest1920(10.08))

print('\n')

print('1d array:')
print(munsell_value_Priest1920_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_Priest1920_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_Priest1920_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_Munsell1933
###############################################################################
message_box('munsell_value_Munsell1933')


def munsell_value_Munsell1933_2d(Y):
    for i in range(len(Y)):
        munsell_value_Munsell1933(Y[i])


# get_ipython().magic(u'timeit munsell_value_Munsell1933_2d(Y)')


def munsell_value_Munsell1933_vectorise(Y):
    Y = as_array(Y)

    V = np.sqrt(1.4742 * Y - 0.004743 * (Y * Y))

    return as_numeric(V)


print('Reference:')
print(munsell_value_Munsell1933(10.08))

print('\n')

print('1d array:')
print(munsell_value_Munsell1933_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_Munsell1933_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_Munsell1933_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_Moon1943
###############################################################################
message_box('munsell_value_Moon1943')


def munsell_value_Moon1943_2d(Y):
    for i in range(len(Y)):
        munsell_value_Moon1943(Y[i])


# get_ipython().magic(u'timeit munsell_value_Moon1943_2d(Y)')


def munsell_value_Moon1943_vectorise(Y):
    Y = as_array(Y)

    V = 1.4 * Y ** 0.426

    return as_numeric(V)


print('Reference:')
print(munsell_value_Moon1943(10.08))

print('\n')

print('1d array:')
print(munsell_value_Moon1943_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_Moon1943_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_Moon1943_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_Saunderson1944
###############################################################################
message_box('munsell_value_Saunderson1944')


def munsell_value_Saunderson1944_2d(Y):
    for i in range(len(Y)):
        munsell_value_Saunderson1944(Y[i])


# get_ipython().magic(u'timeit munsell_value_Saunderson1944_2d(Y)')


def munsell_value_Saunderson1944_vectorise(Y):
    Y = as_array(Y)

    V = 2.357 * (Y ** 0.343) - 1.52

    return as_numeric(V)


print('Reference:')
print(munsell_value_Saunderson1944(10.08))

print('\n')

print('1d array:')
print(munsell_value_Saunderson1944_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_Saunderson1944_vectorise(
    [10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_Saunderson1944_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_Ladd1955
###############################################################################
message_box('munsell_value_Ladd1955')


def munsell_value_Ladd1955_2d(Y):
    for i in range(len(Y)):
        munsell_value_Ladd1955(Y[i])


# get_ipython().magic(u'timeit munsell_value_Ladd1955_2d(Y)')


def munsell_value_Ladd1955_vectorise(Y):
    Y = as_array(Y)

    V = 2.468 * (Y ** (1 / 3)) - 1.636

    return as_numeric(V)


print('Reference:')
print(munsell_value_Ladd1955(10.08))

print('\n')

print('1d array:')
print(munsell_value_Ladd1955_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_Ladd1955_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_Ladd1955_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_McCamy1987
###############################################################################
message_box('munsell_value_McCamy1987')


def munsell_value_McCamy1987_2d(Y):
    for i in range(len(Y)):
        munsell_value_McCamy1987(Y[i])


# get_ipython().magic(u'timeit munsell_value_McCamy1987_2d(Y)')


@ignore_numpy_errors
def munsell_value_McCamy1987_vectorise(Y):
    Y = as_array(Y)

    V = np.where(Y <= 0.9,
                 0.87445 * (Y ** 0.9967),
                 (2.49268 * (Y ** (1 / 3)) - 1.5614 -
                  (0.985 / (((0.1073 * Y - 3.084) ** 2) + 7.54)) +
                  (0.0133 / (Y ** 2.3)) +
                  0.0084 * np.sin(4.1 * (Y ** (1 / 3)) + 1) +
                  (0.0221 / Y) * np.sin(0.39 * (Y - 2)) -
                  (0.0037 / (0.44 * Y)) * np.sin(1.28 * (Y - 0.53))))

    return as_numeric(V)


print('Reference:')
print(munsell_value_McCamy1987(10.08))

print('\n')

print('1d array:')
print(munsell_value_McCamy1987_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_McCamy1987_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_McCamy1987_vectorise(Y)')

print('\n')

###############################################################################
#### colour.munsell_value_ASTMD153508
###############################################################################
message_box('munsell_value_ASTMD153508')


def munsell_value_ASTMD153508_2d(Y):
    for i in range(len(Y)):
        munsell_value_ASTMD153508(Y[i])


# get_ipython().magic(u'timeit munsell_value_ASTMD153508_2d(Y)')

from colour.algebra import *

_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = None


def _munsell_value_ASTMD153508_interpolator():
    global _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE
    munsell_values = np.arange(0, 10, 0.001)
    if _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE is None:
        _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = Extrapolator1d(
            LinearInterpolator1d(
                luminance_ASTMD153508_vectorise(munsell_values),
                munsell_values))

    return _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE


def munsell_value_ASTMD153508_vectorise(Y):
    Y = as_array(Y)

    V = _munsell_value_ASTMD153508_interpolator()(Y)

    return as_numeric(V)


print('Reference:')
print(munsell_value_ASTMD153508(10.08))

print('\n')

print('1d array:')
print(munsell_value_ASTMD153508_vectorise(10.08))

print('\n')

print('2d array:')
print(munsell_value_ASTMD153508_vectorise([10.08, 10.08, 10.08, 10.08, 10.08]))

# get_ipython().magic(u'timeit munsell_value_ASTMD153508_vectorise(Y)')

print('\n')

###############################################################################
###############################################################################
### colour.notation.triplet
###############################################################################
###############################################################################

###############################################################################
#### colour.notation.triplet.RGB_to_HEX
###############################################################################
from colour.notation.triplet import *

RGB = np.array([0.86969452, 1.00516431, 1.41715848])

message_box('RGB_to_HEX')


def RGB_to_HEX_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_HEX(RGB[i])


# get_ipython().magic(u'timeit RGB_to_HEX_2d(DATA1)')


def RGB_to_HEX_vectorise(RGB):
    RGB = as_array(RGB, (-1, 3))

    to_HEX = np.vectorize('{0:02x}'.format)

    HEX = to_HEX((RGB * 255).astype(np.uint8)).astype(object)
    HEX = as_stack(np.array('#') + HEX[:, 0] + HEX[:, 1] + HEX[:, 2])

    return HEX


print('Reference:')
RGB = np.array([0.66666667, 0.86666667, 1])
print(RGB_to_HEX(RGB))

print('\n')

print('1d array:')
print(RGB_to_HEX_vectorise(RGB))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(RGB_to_HEX_vectorise(RGB))

# get_ipython().magic(u'timeit RGB_to_HEX_vectorise(DATA1)')

print('\n')

###############################################################################
#### colour.notation.triplet.HEX_to_RGB
###############################################################################
from colour.notation.triplet import *

HEX1 = ['#aaddff'] * (1920 * 1080)

message_box('HEX_to_RGB')


def HEX_to_RGB_2d(HEX):
    for i in range(len(HEX)):
        HEX_to_RGB(HEX[i])


# get_ipython().magic(u'timeit HEX_to_RGB_2d(HEX1)')


def HEX_to_RGB_vectorise(HEX):
    HEX = as_array(HEX, (-1, 1), str)
    HEX = np.core.defchararray.lstrip(HEX, '#')

    def to_RGB(x):
        length = len(x)
        return [int(x[i:i + length // 3], 16)
                for i in
                range(0, length, length // 3)]

    RGB = np.squeeze(np.array([to_RGB(x[0]) for x in HEX]) / 255)

    return RGB


print('Reference:')
RGB = '#aaddff'
print(HEX_to_RGB(RGB))

print('\n')

print('1d array:')
print(HEX_to_RGB_vectorise(RGB))

print('\n')

print('2d array:')
RGB = np.tile(RGB, (5, 1))
print(HEX_to_RGB_vectorise(RGB))

# get_ipython().magic(u'timeit HEX_to_RGB_vectorise(HEX1)')

print('\n')

###############################################################################
###############################################################################
### colour.phenomenons.rayleigh
###############################################################################
###############################################################################

###############################################################################
### colour.phenomenons.rayleigh.air_refraction_index_Penndorf1957
###############################################################################
from colour.phenomenons.rayleigh import *

message_box('air_refraction_index_Penndorf1957')


def air_refraction_index_Penndorf1957_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Penndorf1957(wl[i])


# get_ipython().magic(u'timeit air_refraction_index_Penndorf1957_2d(DATA1[:, 0])')


def air_refraction_index_Penndorf1957_vectorise(wavelength, *args):
    wl = as_array(wavelength)
    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n = as_numeric(n / 1.0e8 + 1)

    return n


print('Reference:')
print(air_refraction_index_Penndorf1957(0.555))

print('\n')

print('1d array:')
print(air_refraction_index_Penndorf1957_vectorise(0.555))

print('\n')

print('2d array:')
print(air_refraction_index_Penndorf1957_vectorise(
    [0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit air_refraction_index_Penndorf1957_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.air_refraction_index_Edlen1966
###############################################################################
message_box('air_refraction_index_Edlen1966')


def air_refraction_index_Edlen1966_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Edlen1966(wl[i])


# get_ipython().magic(u'timeit air_refraction_index_Edlen1966_2d(DATA1[:, 0])')


def air_refraction_index_Edlen1966_vectorise(wavelength, *args):
    wl = as_array(wavelength)
    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n = as_numeric(n / 1.0e8 + 1)

    return n


print('Reference:')
print(air_refraction_index_Edlen1966(0.555))

print('\n')

print('1d array:')
print(air_refraction_index_Edlen1966_vectorise(0.555))

print('\n')

print('2d array:')
print(air_refraction_index_Edlen1966_vectorise(
    [0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit air_refraction_index_Edlen1966_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.air_refraction_index_Peck1972
###############################################################################
message_box('air_refraction_index_Peck1972')


def air_refraction_index_Peck1972_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Peck1972(wl[i])


# get_ipython().magic(u'timeit air_refraction_index_Peck1972_2d(DATA1[:, 0])')


def air_refraction_index_Peck1972_vectorise(wavelength, *args):
    wl = as_array(wavelength)
    n = (8060.51 + 2480990 / (132.274 - wl ** (-2)) + 17455.7 /
         (39.32957 - wl ** (-2)))
    n = as_numeric(n / 1.0e8 + 1)

    return n


print('Reference:')
print(air_refraction_index_Peck1972(0.555))

print('\n')

print('1d array:')
print(air_refraction_index_Peck1972_vectorise(0.555))

print('\n')

print('2d array:')
print(air_refraction_index_Peck1972_vectorise(
    [0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit air_refraction_index_Peck1972_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.air_refraction_index_Bodhaine1999
###############################################################################
message_box('air_refraction_index_Bodhaine1999')


def air_refraction_index_Bodhaine1999_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Bodhaine1999(wl[i])


# get_ipython().magic(u'timeit air_refraction_index_Bodhaine1999_2d(DATA1[:, 0])')


def air_refraction_index_Bodhaine1999_vectorise(
        wavelength,
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    wl = as_array(wavelength)
    CCO2 = np.resize(as_array(CO2_concentration), wl.shape)

    n = as_numeric(((1 + 0.54 * ((CCO2 * 1e-6) - 300e-6)) *
                    (air_refraction_index_Peck1972(wl) - 1) + 1))

    return n


print('Reference:')
print(air_refraction_index_Bodhaine1999(0.555))

print('\n')

print('1d array:')
print(air_refraction_index_Bodhaine1999_vectorise(0.555))

print('\n')

print('2d array:')
print(air_refraction_index_Bodhaine1999_vectorise(
    [0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit air_refraction_index_Bodhaine1999_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.N2_depolarisation
###############################################################################
message_box('N2_depolarisation')


def N2_depolarisation_2d(wl):
    for i in range(len(wl)):
        N2_depolarisation(wl[i])


# get_ipython().magic(u'timeit N2_depolarisation_2d(DATA1[:, 0])')


def N2_depolarisation_vectorise(wavelength):
    wl = as_array(wavelength)
    N2 = as_numeric(1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2))

    return N2


print('Reference:')
print(N2_depolarisation(0.555))

print('\n')

print('1d array:')
print(N2_depolarisation_vectorise(0.555))

print('\n')

print('2d array:')
print(N2_depolarisation_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit N2_depolarisation_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.O2_depolarisation
###############################################################################
message_box('O2_depolarisation')


def O2_depolarisation_2d(wl):
    for i in range(len(wl)):
        O2_depolarisation(wl[i])


# get_ipython().magic(u'timeit O2_depolarisation_2d(DATA1[:, 0])')


def O2_depolarisation_vectorise(wavelength):
    wl = as_array(wavelength)
    O2 = as_numeric(1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) +
                    1.448 * 1.0e-4 * (1 / wl ** 4))

    return O2


print('Reference:')
print(O2_depolarisation(0.555))

print('\n')

print('1d array:')
print(O2_depolarisation_vectorise(0.555))

print('\n')

print('2d array:')
print(O2_depolarisation_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit O2_depolarisation_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.F_air_Penndorf1957
###############################################################################


def F_air_Penndorf1957_vectorise(wavelength, *args):
    wl = as_array(wavelength)

    return as_numeric(np.resize(np.array([1.0608]), wl.shape))


print('Reference:')
print(F_air_Penndorf1957(0.555))

print('\n')

print('1d array:')
print(F_air_Penndorf1957_vectorise(0.555))

print('\n')

print('2d array:')
print(F_air_Penndorf1957_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))

###############################################################################
#### colour.phenomenons.rayleigh.F_air_Young1981
###############################################################################


def F_air_Young1981_vectorise(wavelength, *args):
    wl = as_array(wavelength)

    return as_numeric(np.resize(np.array([1.0480]), wl.shape))


print('Reference:')
print(F_air_Young1981(0.555))

print('\n')

print('1d array:')
print(F_air_Young1981_vectorise(0.555))

print('\n')

print('2d array:')
print(F_air_Young1981_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))


###############################################################################
#### colour.phenomenons.rayleigh.F_air_Bates1984
###############################################################################
message_box('F_air_Bates1984')


def F_air_Bates1984_2d(wl):
    for i in range(len(wl)):
        F_air_Bates1984(wl[i])


# get_ipython().magic(u'timeit F_air_Bates1984_2d(DATA1[:, 0])')


def F_air_Bates1984_vectorise(wavelength, *args):
    wl = as_array(wavelength)
    O2 = O2_depolarisation_vectorise(wl)
    N2 = N2_depolarisation_vectorise(wl)
    Ar = 1.00
    CO2 = 1.15

    F_air = as_numeric((78.084 * N2 + 20.946 * O2 + CO2 + Ar) /
                       (78.084 + 20.946 + Ar + CO2))

    return F_air


print('Reference:')
print(F_air_Bates1984(0.555))

print('\n')

print('1d array:')
print(F_air_Bates1984_vectorise(0.555))

print('\n')

print('2d array:')
print(F_air_Bates1984_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit F_air_Bates1984_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.F_air_Bodhaine1999
###############################################################################
message_box('F_air_Bodhaine1999')


def F_air_Bodhaine1999_2d(wl):
    for i in range(len(wl)):
        F_air_Bodhaine1999(wl[i])


# get_ipython().magic(u'timeit F_air_Bodhaine1999_2d(DATA1[:, 0])')


def F_air_Bodhaine1999_vectorise(wavelength,
                                 CO2_concentration=STANDARD_CO2_CONCENTRATION):
    wl = as_array(wavelength)
    O2 = O2_depolarisation_vectorise(wl)
    N2 = N2_depolarisation_vectorise(wl)
    CO2_c = CO2_concentration

    F_air = as_numeric((78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) /
                       (78.084 + 20.946 + 0.934 + CO2_c))

    return F_air


print('Reference:')
print(F_air_Bodhaine1999(0.555))

print('\n')

print('1d array:')
print(F_air_Bodhaine1999_vectorise(0.555))

print('\n')

print('2d array:')
print(F_air_Bodhaine1999_vectorise([0.555, 0.555, 0.555, 0.555, 0.555]))

# get_ipython().magic(u'timeit F_air_Bodhaine1999_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.molecular_density
###############################################################################
message_box('molecular_density')


def molecular_density_2d(temperature):
    for i in range(len(temperature)):
        molecular_density(temperature[i])


# get_ipython().magic(u'timeit molecular_density_2d(DATA1[:, 0])')

from colour.constants import AVOGADRO_CONSTANT


def molecular_density_vectorise(temperature=STANDARD_AIR_TEMPERATURE,
                                avogadro_constant=AVOGADRO_CONSTANT):
    # Review doctests to use coherent temperature values.
    T = as_array(temperature)
    N_s = as_numeric((avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000))

    return N_s


print('Reference:')
print(molecular_density(15))

print('\n')

print('1d array:')
print(molecular_density_vectorise(15))

print('\n')

print('2d array:')
print(molecular_density_vectorise([15, 15, 15, 15, 15]))

# get_ipython().magic(u'timeit molecular_density_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.mean_molecular_weights
###############################################################################
message_box('mean_molecular_weights')


def mean_molecular_weights_2d(C):
    for i in range(len(C)):
        mean_molecular_weights(C[i])


# get_ipython().magic(u'timeit mean_molecular_weights_2d(DATA1[:, 0])')


def mean_molecular_weights_vectorise(
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    CO2_c = as_array(CO2_concentration) * 1.0e-6

    m_a = as_numeric(15.0556 * CO2_c + 28.9595)

    return m_a


print('Reference:')
print(mean_molecular_weights(300))

print('\n')

print('1d array:')
print(mean_molecular_weights_vectorise(300))

print('\n')

print('2d array:')
print(mean_molecular_weights_vectorise([300, 300, 300, 300, 300]))

# get_ipython().magic(u'timeit mean_molecular_weights_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.gravity_List1968
###############################################################################
message_box('gravity_List1968')


def gravity_List1968_2d(C):
    for i in range(len(C)):
        gravity_List1968(C[i])


# get_ipython().magic(u'timeit gravity_List1968_2d(DATA1[:, 0])')


def gravity_List1968_vectorise(latitude=DEFAULT_LATITUDE,
                               altitude=DEFAULT_ALTITUDE):
    latitude = as_array(latitude)
    altitude = np.resize(as_array(altitude), latitude.shape)

    cos2phi = np.cos(2 * np.radians(latitude))

    # Sea level acceleration of gravity.
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)

    g = as_numeric(g0 - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude +
                   (7.254e-11 + 1.0e-13 * cos2phi) * altitude ** 2 -
                   (1.517e-17 + 6e-20 * cos2phi) * altitude ** 3)

    return g


print('Reference:')
print(gravity_List1968(0, 0))

print('\n')

print('1d array:')
print(gravity_List1968_vectorise(0, 0))

print('\n')

print('2d array:')
print(gravity_List1968_vectorise([0, 0, 0, 0, 0], [0]))

# get_ipython().magic(u'timeit gravity_List1968_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.scattering_cross_section
###############################################################################
message_box('scattering_cross_section')


def scattering_cross_section_2d(wl):
    for i in range(len(wl)):
        scattering_cross_section(wl[i])


# get_ipython().magic(u'timeit scattering_cross_section_2d(DATA1[:, 0])')


def scattering_cross_section_vectorise(wavelength,
                                       CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                       temperature=STANDARD_AIR_TEMPERATURE,
                                       avogadro_constant=AVOGADRO_CONSTANT,
                                       n_s=air_refraction_index_Bodhaine1999,
                                       F_air=F_air_Bodhaine1999):
    wl = as_array(wavelength)
    temperature = np.resize(as_array(temperature), wl.shape)
    CO2_concentration = np.resize(as_array(CO2_concentration), wl.shape)

    wl_micrometers = wl * 10e3

    n_s = as_array(n_s(wl_micrometers))
    N_s = as_array(molecular_density(temperature, avogadro_constant))
    F_air = as_array(F_air(wl_micrometers, CO2_concentration))

    sigma = (24 * np.pi ** 3 * (n_s ** 2 - 1) ** 2 /
             (wl ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2))
    sigma = as_numeric(sigma * F_air)

    return sigma


print('Reference:')
print(scattering_cross_section(555 * 10e-8))

print('\n')

print('1d array:')
print(scattering_cross_section_vectorise(555 * 10e-8))

print('\n')

print('2d array:')
print(scattering_cross_section_vectorise(
    [555 * 10e-8, 555 * 10e-8, 555 * 10e-8, 555 * 10e-8, 555 * 10e-8]))

# get_ipython().magic(u'timeit scattering_cross_section_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.rayleigh_optical_depth
###############################################################################
message_box('rayleigh_optical_depth')


def rayleigh_optical_depth_2d(wl):
    for i in range(len(wl)):
        rayleigh_optical_depth(wl[i])


# get_ipython().magic(u'timeit rayleigh_optical_depth_2d(DATA1[:, 0])')


def rayleigh_optical_depth_vectorise(wavelength,
                                     CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                     temperature=STANDARD_AIR_TEMPERATURE,
                                     pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                                     latitude=DEFAULT_LATITUDE,
                                     altitude=DEFAULT_ALTITUDE,
                                     avogadro_constant=AVOGADRO_CONSTANT,
                                     n_s=air_refraction_index_Bodhaine1999,
                                     F_air=F_air_Bodhaine1999):
    wavelength = as_array(wavelength)
    CO2_c = np.resize(as_array(CO2_concentration), wavelength.shape)
    latitude = np.resize(as_array(latitude), wavelength.shape)
    altitude = np.resize(as_array(altitude), wavelength.shape)
    # Conversion from pascal to dyne/cm2.
    P = np.resize(as_array(pressure * 10), wavelength.shape)

    sigma = as_array(scattering_cross_section(wavelength,
                                              CO2_c,
                                              temperature,
                                              avogadro_constant,
                                              n_s,
                                              F_air))

    m_a = as_array(mean_molecular_weights(CO2_c))
    g = as_array(gravity_List1968(latitude, altitude))

    T_R = as_numeric(sigma * (P * avogadro_constant) / (m_a * g))

    return T_R


print('Reference:')
print(rayleigh_optical_depth(555 * 10e-8))

print('\n')

print('1d array:')
print(rayleigh_optical_depth_vectorise(555 * 10e-8))

print('\n')

print('2d array:')
print(rayleigh_optical_depth_vectorise(
    [555 * 10e-8, 555 * 10e-8, 555 * 10e-8, 555 * 10e-8, 555 * 10e-8]))

# get_ipython().magic(u'timeit rayleigh_optical_depth_vectorise(DATA1[:, 0])')

print('\n')

###############################################################################
#### colour.phenomenons.rayleigh.rayleigh_scattering_spd
###############################################################################
message_box('rayleigh_scattering_spd')


def rayleigh_scattering_spd_vectorise(shape=DEFAULT_SPECTRAL_SHAPE,
                                      CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                      temperature=STANDARD_AIR_TEMPERATURE,
                                      pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                                      latitude=DEFAULT_LATITUDE,
                                      altitude=DEFAULT_ALTITUDE,
                                      avogadro_constant=AVOGADRO_CONSTANT,
                                      n_s=air_refraction_index_Bodhaine1999,
                                      F_air=F_air_Bodhaine1999):
    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name=('Rayleigh Scattering - {0} ppm, {1} K, {2} Pa, {3} Degrees, '
              '{4} m').format(CO2_concentration,
                              temperature,
                              pressure,
                              latitude,
                              altitude),
        data=dict(zip(wavelengths,
                      rayleigh_optical_depth_vectorise(wavelengths * 10e-8,
                                                       CO2_concentration,
                                                       temperature,
                                                       pressure,
                                                       latitude,
                                                       altitude,
                                                       avogadro_constant,
                                                       n_s,
                                                       F_air))))


print(rayleigh_scattering_spd_vectorise().values)

###############################################################################
###############################################################################
### colour.quality.cqs
###############################################################################
###############################################################################

###############################################################################
### colour.quality.cqs.gamut_area
###############################################################################
from colour.quality.cqs import *

message_box('gamut_area_vectorise')


def gamut_area_vectorise(Lab):
    Lab = as_array(Lab)
    Lab_s = np.roll(np.copy(Lab), -3)

    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    L_s, a_s, b_s = Lab_s[:, 0], Lab_s[:, 1], Lab_s[:, 2]

    A = np.sqrt(a ** 2 + b ** 2)
    B = np.sqrt(a_s ** 2 + b_s ** 2)
    C = np.sqrt((a_s - a) ** 2 + (b_s - b) ** 2)
    t = (A + B + C) / 2
    S = np.sqrt(t * (t - A) * (t - B) * (t - C))

    return np.sum(S)


Lab = [
    np.array([39.94996006, 34.59018231, -19.86046321]),
    np.array([38.88395498, 21.44348519, -34.87805301]),
    np.array([36.60576301, 7.06742454, -43.21461177]),
    np.array([46.60142558, -15.90481586, -34.64616865]),
    np.array([56.50196523, -29.5465555, -20.50177194]),
    np.array([55.73912101, -43.39520959, -5.08956953]),
    np.array([56.2077687, -53.68997662, 20.2113441]),
    np.array([66.16683122, -38.64600327, 42.77396631]),
    np.array([76.7295211, -23.9214821, 61.04740432]),
    np.array([82.85370708, -3.98679065, 75.43320144]),
    np.array([69.26458861, 13.11066359, 68.83858372]),
    np.array([69.63154351, 28.24532497, 59.45609803]),
    np.array([61.26281449, 40.87950839, 44.97606172]),
    np.array([41.62567821, 57.34129516, 27.4671817]),
    np.array([40.52565174, 48.87449192, 3.4512168])]

print(gamut_area_vectorise(Lab))


###############################################################################
###############################################################################
### colour.temperature.cct
###############################################################################
###############################################################################

###############################################################################
#### colour.xy_to_CCT_McCamy1992
###############################################################################
from colour.temperature.cct import *

message_box('xy_to_CCT_McCamy1992')


def xy_to_CCT_McCamy1992_2d(xy):
    for i in range(len(xy)):
        xy_to_CCT_McCamy1992(xy[i])


# get_ipython().magic(u'timeit xy_to_CCT_McCamy1992_2d(DATA1[:, 0:2])')


def xy_to_CCT_McCamy1992_vectorise(xy):
    xy = as_array(xy, (-1, 2))
    x, y = xy[:, 0], xy[:, 1]

    n = (x - 0.3320) / (y - 0.1858)
    CCT = as_numeric(-449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33)

    return CCT


print('Reference:')
xy = np.array([0.31271, 0.32902])
print(xy_to_CCT_McCamy1992(xy))

print('\n')

print('1d array:')
print(xy_to_CCT_McCamy1992_vectorise(xy))

print('\n')

print('2d array:')
xy = np.tile(xy, (5, 1))
print(xy_to_CCT_McCamy1992_vectorise(xy))

# get_ipython().magic(u'timeit xy_to_CCT_McCamy1992_vectorise(DATA1[:, 0:2])')

print('\n')

###############################################################################
#### colour.xy_to_CCT_Hernandez1999
###############################################################################
message_box('xy_to_CCT_Hernandez1999')


def xy_to_CCT_Hernandez1999_2d(xy):
    for i in range(len(xy)):
        xy_to_CCT_Hernandez1999(xy[i])


# get_ipython().magic(u'timeit xy_to_CCT_Hernandez1999_2d(DATA1[:, 0:2])')


def xy_to_CCT_Hernandez1999_vectorise(xy):
    xy = as_array(xy, (-1, 2))
    x, y = xy[:, 0], xy[:, 1]

    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 +
           6253.80338 * np.exp(-n / 0.92159) +
           28.70599 * np.exp(-n / 0.20039) +
           0.00004 * np.exp(-n / 0.07125))

    n = np.where(CCT > 50000,
                 (x - 0.3356) / (y - 0.1691),
                 n)

    CCT = as_numeric(
        np.where(CCT > 50000,
                 36284.48953 +
                 0.00228 * np.exp(-n / 0.07861) +
                 5.4535e-36 * np.exp(-n / 0.01543),
                 CCT))

    return CCT


print('Reference:')
xy = np.array([0.31271, 0.32902])
print(xy_to_CCT_Hernandez1999(xy))

print('\n')

print('1d array:')
print(xy_to_CCT_Hernandez1999_vectorise(xy))

print('\n')

print('2d array:')
xy = np.tile(xy, (5, 1))
print(xy_to_CCT_Hernandez1999_vectorise(xy))

# get_ipython().magic(u'timeit xy_to_CCT_Hernandez1999_vectorise(DATA1[:, 0:2])')

print('\n')

###############################################################################
#### colour.CCT_to_xy_Kang2002
###############################################################################
message_box('CCT_to_xy_Kang2002')


def CCT_to_xy_Kang2002_2d(CCT):
    for i in range(len(CCT)):
        CCT_to_xy_Kang2002(CCT[i])


CCT = np.linspace(4000, 20000, 1000000)

# get_ipython().magic(u'timeit CCT_to_xy_Kang2002_2d(CCT)')


def CCT_to_xy_Kang2002_vectorise(CCT):
    CCT = as_array(CCT)

    if np.any(CCT[np.logical_or(CCT < 1667, CCT > 25000)]):
        warning(('Correlated colour temperature must be in domain '
                 '[1667, 25000], unpredictable results may occur!'))

    x = np.where(CCT <= 4000,
                 -0.2661239 * 10 ** 9 / CCT ** 3 -
                 0.2343589 * 10 ** 6 / CCT ** 2 +
                 0.8776956 * 10 ** 3 / CCT +
                 0.179910,
                 -3.0258469 * 10 ** 9 / CCT ** 3 +
                 2.1070379 * 10 ** 6 / CCT ** 2 +
                 0.2226347 * 10 ** 3 / CCT +
                 0.24039)

    y = np.select([CCT <= 2222,
                   np.logical_and(CCT > 2222, CCT <= 4000),
                   CCT > 4000],
                  [-1.1063814 * x ** 3 -
                   1.34811020 * x ** 2 +
                   2.18555832 * x -
                   0.20219683,
                   -0.9549476 * x ** 3 -
                   1.37418593 * x ** 2 +
                   2.09137015 * x -
                   0.16748867,
                   3.0817580 * x ** 3 -
                   5.8733867 * x ** 2 +
                   3.75112997 * x -
                   0.37001483])

    xy = as_stack((x, y))

    return xy


print('Reference:')
print(CCT_to_xy_Kang2002(6504.38938305))

print('\n')

print('1d array:')
print(CCT_to_xy_Kang2002_vectorise(6504.38938305))

print('\n')

print('2d array:')
print(CCT_to_xy_Kang2002_vectorise(
    [6504.38938305, 6504.38938305, 6504.38938305, 6504.38938305,
     6504.38938305]))

# get_ipython().magic(u'timeit CCT_to_xy_Kang2002_vectorise(CCT)')

print('\n')

###############################################################################
#### colour.CCT_to_xy_CIE_D
###############################################################################
message_box('CCT_to_xy_CIE_D')


def CCT_to_xy_CIE_D_2d(CCT):
    for i in range(len(CCT)):
        CCT_to_xy_CIE_D(CCT[i])


# get_ipython().magic(u'timeit CCT_to_xy_CIE_D_2d(CCT)')


def CCT_to_xy_CIE_D_vectorise(CCT):
    CCT = as_array(CCT)

    if np.any(CCT[np.logical_or(CCT < 4000, CCT > 25000)]):
        warning(('Correlated colour temperature must be in domain '
                 '[4000, 25000], unpredictable results may occur!'))

    x = np.where(CCT <= 7000,
                 -4.607 * 10 ** 9 / CCT ** 3 +
                 2.9678 * 10 ** 6 / CCT ** 2 +
                 0.09911 * 10 ** 3 / CCT +
                 0.244063,
                 -2.0064 * 10 ** 9 / CCT ** 3 +
                 1.9018 * 10 ** 6 / CCT ** 2 +
                 0.24748 * 10 ** 3 / CCT +
                 0.23704)

    y = -3 * x ** 2 + 2.87 * x - 0.275

    xy = as_stack((x, y))

    return xy


print('Reference:')
print(CCT_to_xy_CIE_D(6504.38938305))

print('\n')

print('1d array:')
print(CCT_to_xy_CIE_D_vectorise(6504.38938305))

print('\n')

print('2d array:')
print(CCT_to_xy_CIE_D_vectorise(
    [6504.38938305, 6504.38938305, 6504.38938305, 6504.38938305,
     6504.38938305]))

# get_ipython().magic(u'timeit CCT_to_xy_CIE_D_vectorise(CCT)')

print('\n')

###############################################################################
###############################################################################
### Ramblings
###############################################################################
###############################################################################

###############################################################################
#### Image Operations
###############################################################################

# get_ipython().magic(u'matplotlib inline')

# import pylab
# from OpenImageIO import FLOAT, ImageInput
#
# import colour
# from colour.plotting import *
#
#
# def read_image_as_array(path, bit_depth=FLOAT):
#     image = ImageInput.open(path)
#     specification = image.spec()
#
#     return np.array(image.read_image(bit_depth)).reshape((specification.height,
#                                                           specification.width,
#                                                           specification.nchannels))
#
#
# colour.sRGB_COLOURSPACE.transfer_function = _srgb_transfer_function
#
#
# def image_plot(image,
#                transfer_function=colour.sRGB_COLOURSPACE.transfer_function):
#     image = np.clip(transfer_function(Lab_to_XYZ_vectorise(image)), 0, 1)
#     pylab.imshow(image)
#
#     settings = {'no_ticks': True,
#                 'bounding_box': [0, 1, 0, 1],
#                 'bbox_inches': 'tight',
#                 'pad_inches': 0}
#
#     canvas(**{'figure_size': (16, 16)})
#     decorate(**settings)
#     display(**settings)
#
#
# marcie = read_image_as_array(
#     '/colour-science/colour-ramblings/resources/images/Digital_LAD_2048x1556.exr')
#
# image_plot(marcie)
