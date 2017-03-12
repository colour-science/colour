#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace & Transformations
=================================

Defines the :class:`RGB_Colourspace` class for the *RGB* colourspaces dataset
from :mod:`colour.models.dataset.aces_rgb`, etc... and the following *RGB*
colourspace transformations or helper definitions:

-   :func:`XYZ_to_RGB`
-   :func:`RGB_to_XYZ`
-   :func:`RGB_to_RGB_matrix`
-   :func:`RGB_to_RGB`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/blob/\
master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import (
    xy_to_XYZ,
    xy_to_xyY,
    xyY_to_XYZ)
from colour.models.rgb import normalised_primary_matrix
from colour.adaptation import chromatic_adaptation_matrix_VonKries
from colour.utilities import dot_matrix, dot_vector, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_Colourspace',
           'XYZ_to_RGB',
           'RGB_to_XYZ',
           'RGB_to_RGB_matrix',
           'RGB_to_RGB']


class RGB_Colourspace(object):
    """
    Implements support for the *RGB* colourspaces dataset from
    :mod:`colour.models.dataset.aces_rgb`, etc....

    Colour science literature related to *RGB* colourspaces and encodings
    defines their dataset using different degree of precision or rounding.
    While instances where a whitepoint is being defined with a value
    different than its canonical agreed one are rare, it is however very
    common to have normalised primary matrices rounded at different
    decimals. This can yield large discrepancies in computations.

    Such an occurrence is the *V-Gamut* colourspace white paper [1]_, that
    defines the *V-Gamut* to *Rec. 709* conversion matrix as follows::

        [[ 1.806576 -0.695697 -0.110879]
         [-0.170090  1.305955 -0.135865]
         [-0.025206 -0.154468  1.179674]]

    Computing this matrix using *Rec. 709* colourspace derived normalised
    primary matrix yields::

        [[ 1.8065736 -0.6956981 -0.1108786]
         [-0.1700890  1.3059548 -0.1358648]
         [-0.0252057 -0.1544678  1.1796737]]

    The latter matrix is almost equals with the former, however performing the
    same computation using *IEC 61966-2-1:1999* *sRGB* colourspace normalised
    primary matrix introduces severe disparities::

        [[ 1.8063853 -0.6956147 -0.1109453]
         [-0.1699311  1.3058387 -0.1358616]
         [-0.0251630 -0.1544899  1.1797117]]

    In order to provide support for both literature defined dataset and
    accurate computations enabling transformations without loss of precision,
    the :class:`RGB_Colourspace` class provides two sets of transformation
    matrices:

        -   Instantiation transformation matrices
        -   Derived transformation matrices

    Upon instantiation, the :class:`RGB_Colourspace` class stores the given
    `RGB_to_XYZ_matrix` and `XYZ_to_RGB_matrix` arguments and also
    computes their derived counterpart using the `primaries` and `whitepoint`
    arguments.

    Whether the initialisation or derived matrices are used in subsequent
    computations is dependent on the
    :attr:`RGB_Colourspace.use_derived_RGB_to_XYZ_matrix` and
    :attr:`RGB_Colourspace.use_derived_XYZ_to_RGB_matrix` attributes values.

    Parameters
    ----------
    name : unicode
        *RGB* colourspace name.
    primaries : array_like
        *RGB* colourspace primaries.
    whitepoint : array_like
        *RGB* colourspace whitepoint.
    illuminant : unicode, optional
        *RGB* colourspace whitepoint name as illuminant.
    RGB_to_XYZ_matrix : array_like, optional
        Transformation matrix from colourspace to *CIE XYZ* tristimulus values.
    XYZ_to_RGB_matrix : array_like, optional
        Transformation matrix from *CIE XYZ* tristimulus values to colourspace.
    encoding_cctf : object, optional
        Encoding colour component transfer function (Encoding CCTF) /
        opto-electronic transfer function (OETF / OECF) that maps estimated
        tristimulus values in a scene to :math:`R'G'B'` video component signal
        value.
    decoding_cctf : object, optional
        Decoding colour component transfer function (Decoding CCTF) /
        electro-optical transfer function (EOTF / EOCF) that maps an
        :math:`R'G'B'` video component signal value to tristimulus values at
        the display.
    use_derived_RGB_to_XYZ_matrix : bool, optional
        Whether to use the declaration time normalised primary matrix or to
        use a computed derived normalised primary matrix.
    use_derived_XYZ_to_RGB_matrix : bool, optional
        Whether to use the declaration time normalised primary matrix or to
        use a computed derived inverse normalised primary matrix.

    Attributes
    ----------
    name
    primaries
    whitepoint
    illuminant
    RGB_to_XYZ_matrix
    XYZ_to_RGB_matrix
    encoding_cctf
    decoding_cctf
    use_derived_RGB_to_XYZ_matrix
    use_derived_XYZ_to_RGB_matrix

    Methods
    -------
    use_derived_transformation_matrices

    Notes
    -----
    -   The normalised primary matrix defined by
        :attr:`RGB_Colourspace.RGB_to_XYZ_matrix` attribute is treated as the
        prime matrix from which the inverse will be calculated as required by
        the internal derivation mechanism. This behaviour has been chosen in
        accordance with literature where commonly a *RGB* colourspace is
        defined by its normalised primary matrix as it is directly computed
        from the chosen primaries and whitepoint.

    References
    ----------
    .. [1]  Panasonic. (2014). VARICAM V-Log/V-Gamut. Retrieved from
            http://pro-av.panasonic.net/en/varicam/common/pdf/\
VARICAM_V-Log_V-Gamut.pdf
    .. [2]  International Electrotechnical Commission. (1999). IEC
        61966-2-1:1999 - Multimedia systems and equipment - Colour measurement
        and management - Part 2-1: Colour management - Default RGB colour
        space - sRGB, 51. Retrieved from
        https://webstore.iec.ch/publication/6169

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> RGB_to_XYZ_matrix = np.identity(3)
    >>> XYZ_to_RGB_matrix = np.identity(3)
    >>> colourspace = RGB_Colourspace(
    ...     'RGB Colourspace',
    ...     p,
    ...     whitepoint,
    ...     'D60',
    ...     RGB_to_XYZ_matrix,
    ...     XYZ_to_RGB_matrix)
    >>> colourspace.RGB_to_XYZ_matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.XYZ_to_RGB_matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_transformation_matrices(True)
    True
    >>> colourspace.RGB_to_XYZ_matrix  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
    >>> colourspace.XYZ_to_RGB_matrix  # doctest: +ELLIPSIS
    array([[  1.0498110...e+00,   0.0000000...e+00,  -9.7484540...e-05],
           [ -4.9590302...e-01,   1.3733130...e+00,   9.8240036...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   9.9125201...e-01]])
    >>> colourspace.use_derived_RGB_to_XYZ_matrix = False
    >>> colourspace.RGB_to_XYZ_matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_XYZ_to_RGB_matrix = False
    >>> colourspace.XYZ_to_RGB_matrix
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 illuminant=None,
                 RGB_to_XYZ_matrix=None,
                 XYZ_to_RGB_matrix=None,
                 encoding_cctf=None,
                 decoding_cctf=None,
                 use_derived_RGB_to_XYZ_matrix=False,
                 use_derived_XYZ_to_RGB_matrix=False):
        self._derived_RGB_to_XYZ_matrix = None
        self._derived_XYZ_to_RGB_matrix = None

        self._name = None
        self.name = name
        self._primaries = None
        self.primaries = primaries
        self._whitepoint = None
        self.whitepoint = whitepoint
        self._illuminant = None
        self.illuminant = illuminant
        self._RGB_to_XYZ_matrix = None
        self.RGB_to_XYZ_matrix = RGB_to_XYZ_matrix
        self._XYZ_to_RGB_matrix = None
        self.XYZ_to_RGB_matrix = XYZ_to_RGB_matrix
        self._encoding_cctf = None
        self.encoding_cctf = encoding_cctf
        self._decoding_cctf = None
        self.decoding_cctf = decoding_cctf
        self._use_derived_RGB_to_XYZ_matrix = False
        self.use_derived_RGB_to_XYZ_matrix = use_derived_RGB_to_XYZ_matrix
        self._use_derived_XYZ_to_RGB_matrix = False
        self.use_derived_XYZ_to_RGB_matrix = use_derived_XYZ_to_RGB_matrix

    @property
    def name(self):
        """
        Property for **self._name** private attribute.

        Returns
        -------
        unicode
            self._name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for **self._name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('name', value))
        self._name = value

    @property
    def primaries(self):
        """
        Property for **self._primaries** private attribute.

        Returns
        -------
        array_like, (3, 2)
            self._primaries.
        """

        return self._primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for **self._primaries** private attribute.

        Parameters
        ----------
        value : array_like, (3, 2)
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._primaries = value

        self._derive_transformation_matrices()

    @property
    def whitepoint(self):
        """
        Property for **self._whitepoint** private attribute.

        Returns
        -------
        array_like
            self._whitepoint.
        """

        return self._whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for **self._whitepoint** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, (tuple, list, np.ndarray, np.matrix)), (
                ('"{0}" attribute: "{1}" is not a "tuple", "list", "ndarray" '
                 'or "matrix" instance!').format('whitepoint', value))
            value = np.asarray(value)
        self._whitepoint = value

        self._derive_transformation_matrices()

    @property
    def illuminant(self):
        """
        Property for **self._illuminant** private attribute.

        Returns
        -------
        unicode
            self._illuminant.
        """

        return self._illuminant

    @illuminant.setter
    def illuminant(self, value):
        """
        Setter for **self._illuminant** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('illuminant', value))
        self._illuminant = value

    @property
    def RGB_to_XYZ_matrix(self):
        """
        Property for **RGB_to_XYZ_matrix** attribute.

        Returns
        -------
        array_like, (3, 3)
            RGB_to_XYZ_matrix.
        """

        if not self._use_derived_RGB_to_XYZ_matrix:
            return self._RGB_to_XYZ_matrix
        else:
            return self._derived_RGB_to_XYZ_matrix

    @RGB_to_XYZ_matrix.setter
    def RGB_to_XYZ_matrix(self, value):
        """
        Setter for **RGB_to_XYZ_matrix** attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._RGB_to_XYZ_matrix = value

    @property
    def XYZ_to_RGB_matrix(self):
        """
        Property for **XYZ_to_RGB_matrix** attribute.

        Returns
        -------
        array_like, (3, 3)
            XYZ_to_RGB_matrix.
        """

        if not self._use_derived_XYZ_to_RGB_matrix:
            return self._XYZ_to_RGB_matrix
        else:
            return self._derived_XYZ_to_RGB_matrix

    @XYZ_to_RGB_matrix.setter
    def XYZ_to_RGB_matrix(self, value):
        """
        Setter for **XYZ_to_RGB_matrix** attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._XYZ_to_RGB_matrix = value

    @property
    def encoding_cctf(self):
        """
        Property for **self._encoding_cctf** private attribute.

        Returns
        -------
        callable
            self._encoding_cctf.
        """

        return self._encoding_cctf

    @encoding_cctf.setter
    def encoding_cctf(self, value):
        """
        Setter for **self._encoding_cctf** private attribute.

        Parameters
        ----------
        value : callable
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'encoding_cctf', value))
        self._encoding_cctf = value

    @property
    def decoding_cctf(self):
        """
        Property for **self._decoding_cctf** private attribute.

        Returns
        -------
        callable
            self._decoding_cctf.
        """

        return self._decoding_cctf

    @decoding_cctf.setter
    def decoding_cctf(self, value):
        """
        Setter for **self._decoding_cctf** private attribute.

        Parameters
        ----------
        value : callable
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'decoding_cctf', value))
        self._decoding_cctf = value

    @property
    def use_derived_RGB_to_XYZ_matrix(self):
        """
        Property for **self._use_derived_RGB_to_XYZ_matrix** private attribute.

        Returns
        -------
        bool
            self._use_derived_RGB_to_XYZ_matrix.
        """

        return self._use_derived_RGB_to_XYZ_matrix

    @use_derived_RGB_to_XYZ_matrix.setter
    def use_derived_RGB_to_XYZ_matrix(self, value):
        """
        Setter for **self._use_derived_RGB_to_XYZ_matrix** private attribute.

        Parameters
        ----------
        value : bool
            Attribute value.
        """

        # TODO: Revisit for potential behaviour / type checking.
        self._use_derived_RGB_to_XYZ_matrix = value

    @property
    def use_derived_XYZ_to_RGB_matrix(self):
        """
        Property for **self._use_derived_XYZ_to_RGB_matrix** private attribute.

        Returns
        -------
        bool
            self._use_derived_XYZ_to_RGB_matrix.
        """

        return self._use_derived_XYZ_to_RGB_matrix

    @use_derived_XYZ_to_RGB_matrix.setter
    def use_derived_XYZ_to_RGB_matrix(self, value):
        """
        Setter for **self._use_derived_XYZ_to_RGB_matrix** private attribute.

        Parameters
        ----------
        value : bool
            Attribute value.
        """

        # TODO: Revisit for potential behaviour / type checking.
        self._use_derived_XYZ_to_RGB_matrix = value

    def _derive_transformation_matrices(self):
        """
        Computes the derived transformations matrices, the normalised primary
        matrix and its inverse.
        """

        if hasattr(self, '_primaries') and hasattr(self, '_whitepoint'):
            if self._primaries is not None and self._whitepoint is not None:
                npm = normalised_primary_matrix(
                    self._primaries, self._whitepoint)

                self._derived_RGB_to_XYZ_matrix = npm
                self._derived_XYZ_to_RGB_matrix = np.linalg.inv(npm)

    def use_derived_transformation_matrices(self, usage=True):
        """
        Enables or disables usage of both derived transformations matrices,
        the normalised primary matrix and its inverse in subsequent
        computations.

        Parameters
        ----------
        usage : bool, optional
            Whether to use the derived transformations matrices.

        Returns
        -------
        bool
            Definition success.
        """

        self.use_derived_RGB_to_XYZ_matrix = usage
        self.use_derived_XYZ_to_RGB_matrix = usage

        return True


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               XYZ_to_RGB_matrix,
               chromatic_adaptation_transform='CAT02',
               encoding_cctf=None):
    """
    Converts from *CIE XYZ* tristimulus values to given *RGB* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant_XYZ : array_like
        *CIE XYZ* tristimulus values *illuminant* *xy* chromaticity coordinates
        or *CIE xyY* colourspace array.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* *xy* chromaticity coordinates or
        *CIE xyY* colourspace array.
    XYZ_to_RGB_matrix : array_like
        *Normalised primary matrix*.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    encoding_cctf : object, optional
        Encoding colour component transfer function (Encoding CCTF) or
        opto-electronic transfer function (OETF / OECF).

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *RGB* colourspace array is in range [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> XYZ_to_RGB_matrix = np.array([
    ...     [3.24062548, -1.53720797, -0.49862860],
    ...     [-0.96893071, 1.87575606, 0.04151752],
    ...     [0.05571012, -0.20402105, 1.05699594]])
    >>> XYZ_to_RGB(
    ...     XYZ,
    ...     illuminant_XYZ,
    ...     illuminant_RGB,
    ...     XYZ_to_RGB_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0110015...,  0.1273504...,  0.1163271...])
    """

    M = chromatic_adaptation_matrix_VonKries(
        xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
        xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
        transform=chromatic_adaptation_transform)

    XYZ_a = dot_vector(M, XYZ)

    RGB = dot_vector(XYZ_to_RGB_matrix, XYZ_a)

    if encoding_cctf is not None:
        RGB = encoding_cctf(RGB)

    return RGB


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               RGB_to_XYZ_matrix,
               chromatic_adaptation_transform='CAT02',
               decoding_cctf=None):
    """
    Converts from given *RGB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* chromaticity coordinates or *CIE xyY*
        colourspace array.
    illuminant_XYZ : array_like
        *CIE XYZ* tristimulus values *illuminant* chromaticity coordinates or
        *CIE xyY* colourspace array.
    RGB_to_XYZ_matrix : array_like
        *Normalised primary matrix*.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    decoding_cctf : object, optional
        Decoding colour component transfer function (Decoding CCTF) or
        electro-optical transfer function (EOTF / EOCF).

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *RGB* colourspace array is in domain [0, 1].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in range [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.01100154,  0.12735048,  0.11632713])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> RGB_to_XYZ_matrix = np.array([
    ...     [0.41240000, 0.35760000, 0.18050000],
    ...     [0.21260000, 0.71520000, 0.07220000],
    ...     [0.01930000, 0.11920000, 0.95050000]])
    >>> RGB_to_XYZ(
    ...     RGB,
    ...     illuminant_RGB,
    ...     illuminant_XYZ,
    ...     RGB_to_XYZ_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    if decoding_cctf is not None:
        RGB = decoding_cctf(RGB)

    M = chromatic_adaptation_matrix_VonKries(
        xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
        xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
        transform=chromatic_adaptation_transform)

    XYZ = dot_vector(RGB_to_XYZ_matrix, RGB)

    XYZ_a = dot_vector(M, XYZ)

    return XYZ_a


def RGB_to_RGB_matrix(input_colourspace,
                      output_colourspace,
                      chromatic_adaptation_transform='CAT02'):
    """
    Computes the matrix :math:`M` converting from given input *RGB*
    colourspace to output *RGB* colourspace using given *chromatic
    adaptation* method.

    Parameters
    ----------
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray
        Conversion matrix :math:`M`.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE
    >>> RGB = np.array([0.01103742, 0.12734226, 0.11632971])
    >>> RGB_to_RGB_matrix(
    ...     sRGB_COLOURSPACE,
    ...     PROPHOTO_RGB_COLOURSPACE)  # doctest: +ELLIPSIS
    array([[ 0.5288241...,  0.3340609...,  0.1373616...],
           [ 0.0975294...,  0.8790074...,  0.0233981...],
           [ 0.0163599...,  0.1066124...,  0.8772485...]])
    """

    cat = chromatic_adaptation_matrix_VonKries(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_transform)

    M = dot_matrix(cat, input_colourspace.RGB_to_XYZ_matrix)
    M = dot_matrix(output_colourspace.XYZ_to_RGB_matrix, M)

    return M


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_transform='CAT02',
               apply_decoding_cctf=False,
               apply_encoding_cctf=False):
    """
    Converts from given input *RGB* colourspace to output *RGB* colourspace
    using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    apply_decoding_cctf : bool, optional
        Apply input colourpace decoding colour component transfer function  /
        electro-optical transfer function.
    apply_encoding_cctf : bool, optional
        Apply output colourpace encoding colour component transfer function /
        opto-electronic transfer function.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input / output *RGB* colourspace arrays are in domain / range [0, 1].
    -   Input / output *RGB* colourspace arrays are assumed to be representing
        linear light values.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE
    >>> RGB = np.array([0.01103742, 0.12734226, 0.11632971])
    >>> RGB_to_RGB(
    ...     RGB,
    ...     sRGB_COLOURSPACE,
    ...     PROPHOTO_RGB_COLOURSPACE)  # doctest: +ELLIPSIS
    array([ 0.0643561...,  0.1157331...,  0.1158069...])
    """

    if apply_decoding_cctf:
        RGB = input_colourspace.decoding_cctf(RGB)

    M = RGB_to_RGB_matrix(input_colourspace,
                          output_colourspace,
                          chromatic_adaptation_transform)

    RGB = dot_vector(M, RGB)

    if apply_encoding_cctf:
        RGB = output_colourspace.encoding_cctf(RGB)

    return RGB
