"""
RGB Colourspace and Transformations
===================================

Define the :class:`colour.RGB_Colourspace` class for the *RGB* colourspaces
datasets from :mod:`colour.models.datasets.aces_rgb`, etc... and the following
*RGB* colourspace transformations or helper definitions:

-   :func:`colour.XYZ_to_RGB`
-   :func:`colour.RGB_to_XYZ`
-   :func:`colour.matrix_RGB_to_RGB`
-   :func:`colour.RGB_to_RGB`

References
----------
-   :cite:`InternationalElectrotechnicalCommission1999a` : International
    Electrotechnical Commission. (1999). IEC 61966-2-1:1999 - Multimedia
    systems and equipment - Colour measurement and management - Part 2-1:
    Colour management - Default RGB colour space - sRGB (p. 51).
    https://webstore.iec.ch/publication/6169
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut (pp.
    1-7).
    http://pro-av.panasonic.net/en/varicam/common/pdf/VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour.algebra import vecmul
from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    LiteralChromaticAdaptationTransform,
    LiteralRGBColourspace,
    NDArrayFloat,
    cast,
)
from colour.models import xy_to_xyY, xy_to_XYZ, xyY_to_XYZ
from colour.models.rgb import (
    chromatically_adapted_primaries,
    normalised_primary_matrix,
)
from colour.utilities import (
    as_float_array,
    attest,
    domain_range_scale,
    filter_kwargs,
    from_range_1,
    is_string,
    multiline_repr,
    multiline_str,
    optional,
    to_domain_1,
    usage_warning,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_Colourspace",
    "XYZ_to_RGB",
    "RGB_to_XYZ",
    "matrix_RGB_to_RGB",
    "RGB_to_RGB",
]


class RGB_Colourspace:
    """
    Implement support for the *RGB* colourspaces datasets from
    :mod:`colour.models.datasets.aces_rgb`, etc....

    Colour science literature related to *RGB* colourspaces and encodings
    defines their dataset using different degree of precision or rounding.
    While instances where a whitepoint is being defined with a value
    different from its canonical agreed one are rare, it is however very
    common to have normalised primary matrices rounded at different
    decimals. This can yield large discrepancies in computations.

    Such an occurrence is the *V-Gamut* colourspace white paper, that defines
    the *V-Gamut* to *ITU-R BT.709* conversion matrix as follows::

        [[ 1.806576 -0.695697 -0.110879]
         [-0.170090  1.305955 -0.135865]
         [-0.025206 -0.154468  1.179674]]

    Computing this matrix using *ITU-R BT.709* colourspace derived normalised
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
    the :class:`colour.RGB_Colourspace` class provides two sets of
    transformation matrices:

        -   Instantiation transformation matrices
        -   Derived transformation matrices

    Upon instantiation, the :class:`colour.RGB_Colourspace` class stores the
    given ``matrix_RGB_to_XYZ`` and ``matrix_XYZ_to_RGB`` arguments and also
    computes their derived counterpart using the ``primaries`` and
    ``whitepoint`` arguments.

    Whether the initialisation or derived matrices are used in subsequent
    computations is dependent on the
    :attr:`colour.RGB_Colourspace.use_derived_matrix_RGB_to_XYZ` and
    :attr:`colour.RGB_Colourspace.use_derived_matrix_XYZ_to_RGB` attribute
    values.

    Parameters
    ----------
    name
        *RGB* colourspace name.
    primaries
        *RGB* colourspace primaries.
    whitepoint
        *RGB* colourspace whitepoint.
    whitepoint_name
        *RGB* colourspace whitepoint name.
    matrix_RGB_to_XYZ
        Transformation matrix from colourspace to *CIE XYZ* tristimulus values.
    matrix_XYZ_to_RGB
        Transformation matrix from *CIE XYZ* tristimulus values to colourspace.
    cctf_encoding
        Encoding colour component transfer function (Encoding CCTF) /
        opto-electronic transfer function (OETF) that maps estimated
        tristimulus values in a scene to :math:`R'G'B'` video component signal
        value.
    cctf_decoding
        Decoding colour component transfer function (Decoding CCTF) /
        electro-optical transfer function (EOTF) that maps an
        :math:`R'G'B'` video component signal value to tristimulus values at
        the display.
    use_derived_matrix_RGB_to_XYZ
        Whether to use the instantiation time normalised primary matrix or to
        use a computed derived normalised primary matrix.
    use_derived_matrix_XYZ_to_RGB
        Whether to use the instantiation time inverse normalised primary
        matrix or to use a computed derived inverse normalised primary matrix.

    Attributes
    ----------
    -   :attr:`~colour.RGB_Colourspace.name`
    -   :attr:`~colour.RGB_Colourspace.primaries`
    -   :attr:`~colour.RGB_Colourspace.whitepoint`
    -   :attr:`~colour.RGB_Colourspace.whitepoint_name`
    -   :attr:`~colour.RGB_Colourspace.matrix_RGB_to_XYZ`
    -   :attr:`~colour.RGB_Colourspace.matrix_XYZ_to_RGB`
    -   :attr:`~colour.RGB_Colourspace.cctf_encoding`
    -   :attr:`~colour.RGB_Colourspace.cctf_decoding`
    -   :attr:`~colour.RGB_Colourspace.use_derived_matrix_RGB_to_XYZ`
    -   :attr:`~colour.RGB_Colourspace.use_derived_matrix_XYZ_to_RGB`

    Methods
    -------
    -   :attr:`~colour.RGB_Colourspace.__init__`
    -   :attr:`~colour.RGB_Colourspace.__str__`
    -   :attr:`~colour.RGB_Colourspace.__repr__`
    -   :attr:`~colour.RGB_Colourspace.use_derived_transformation_matrices`
    -   :attr:`~colour.RGB_Colourspace.chromatically_adapt`
    -   :attr:`~colour.RGB_Colourspace.copy`

    Notes
    -----
    -   The normalised primary matrix defined by
        :attr:`colour.RGB_Colourspace.matrix_RGB_to_XYZ` property is treated
        as the prime matrix from which the inverse will be calculated as
        required by the internal derivation mechanism. This behaviour has been
        chosen in accordance with literature where commonly a *RGB* colourspace
        is defined by its normalised primary matrix as it is directly computed
        from the chosen primaries and whitepoint.

    References
    ----------
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> matrix_RGB_to_XYZ = np.identity(3)
    >>> matrix_XYZ_to_RGB = np.identity(3)
    >>> colourspace = RGB_Colourspace(
    ...     "RGB Colourspace",
    ...     p,
    ...     whitepoint,
    ...     "ACES",
    ...     matrix_RGB_to_XYZ,
    ...     matrix_XYZ_to_RGB,
    ... )
    >>> colourspace.matrix_RGB_to_XYZ
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.matrix_XYZ_to_RGB
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_transformation_matrices(True)
    >>> colourspace.matrix_RGB_to_XYZ  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
    >>> colourspace.matrix_XYZ_to_RGB  # doctest: +ELLIPSIS
    array([[  1.0498110...e+00,   0.0000000...e+00,  -9.7484540...e-05],
           [ -4.9590302...e-01,   1.3733130...e+00,   9.8240036...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   9.9125201...e-01]])
    >>> colourspace.use_derived_matrix_RGB_to_XYZ = False
    >>> colourspace.matrix_RGB_to_XYZ
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_matrix_XYZ_to_RGB = False
    >>> colourspace.matrix_XYZ_to_RGB
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """

    def __init__(
        self,
        name: str,
        primaries: ArrayLike,
        whitepoint: ArrayLike,
        whitepoint_name: str | None = None,
        matrix_RGB_to_XYZ: ArrayLike | None = None,
        matrix_XYZ_to_RGB: ArrayLike | None = None,
        cctf_encoding: Callable | None = None,
        cctf_decoding: Callable | None = None,
        use_derived_matrix_RGB_to_XYZ: bool = False,
        use_derived_matrix_XYZ_to_RGB: bool = False,
    ) -> None:
        self._derived_matrix_RGB_to_XYZ: NDArrayFloat = np.array([])
        self._derived_matrix_XYZ_to_RGB: NDArrayFloat = np.array([])

        self._name: str = f"{self.__class__.__name__} ({id(self)})"
        self.name = name
        self._primaries: NDArrayFloat = np.array([])
        self.primaries = primaries
        self._whitepoint: NDArrayFloat = np.array([])
        self.whitepoint = whitepoint
        self._whitepoint_name: str | None = None
        self.whitepoint_name = whitepoint_name
        self._matrix_RGB_to_XYZ: NDArrayFloat | None = None
        self.matrix_RGB_to_XYZ = matrix_RGB_to_XYZ
        self._matrix_XYZ_to_RGB: NDArrayFloat | None = None
        self.matrix_XYZ_to_RGB = matrix_XYZ_to_RGB
        self._cctf_encoding: Callable | None = None
        self.cctf_encoding = cctf_encoding
        self._cctf_decoding: Callable | None = None
        self.cctf_decoding = cctf_decoding
        self._use_derived_matrix_RGB_to_XYZ: bool = False
        self.use_derived_matrix_RGB_to_XYZ = use_derived_matrix_RGB_to_XYZ
        self._use_derived_matrix_XYZ_to_RGB: bool = False
        self.use_derived_matrix_XYZ_to_RGB = use_derived_matrix_XYZ_to_RGB

    @property
    def name(self) -> str:
        """
        Getter and setter property for the name.

        Parameters
        ----------
        value
            Value to set the name with.

        Returns
        -------
        :class:`str`
            *RGB* colourspace name.
        """

        return self._name

    @name.setter
    def name(self, value: str):
        """Setter for the **self.name** property."""

        attest(
            is_string(value),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def primaries(self) -> NDArrayFloat:
        """
        Getter and setter property for the primaries.

        Parameters
        ----------
        value
            Value to set the primaries with.

        Returns
        -------
        :class:`numpy.ndarray`
            *RGB* colourspace primaries.
        """

        return self._primaries

    @primaries.setter
    def primaries(self, value: ArrayLike):
        """Setter for the **self.primaries** property."""

        attest(
            isinstance(value, (tuple, list, np.ndarray, np.matrix)),
            f'"matrix_XYZ_to_RGB" property: "{value!r}" is not a "tuple", '
            f'"list", "ndarray" or "matrix" instance!',
        )

        value = as_float_array(value)

        value = np.reshape(value, (3, 2))

        self._primaries = value

        self._derived_matrix_XYZ_to_RGB = np.array([])
        self._derived_matrix_RGB_to_XYZ = np.array([])

    @property
    def whitepoint(self) -> NDArrayFloat:
        """
        Getter and setter property for the whitepoint.

        Parameters
        ----------
        value
            Value to set the whitepoint with.

        Returns
        -------
        :class:`numpy.ndarray`
            *RGB* colourspace whitepoint.
        """

        return self._whitepoint

    @whitepoint.setter
    def whitepoint(self, value: ArrayLike):
        """Setter for the **self.whitepoint** property."""

        attest(
            isinstance(value, (tuple, list, np.ndarray, np.matrix)),
            f'"matrix_XYZ_to_RGB" property: "{value!r}" is not a "tuple", '
            f'"list", "ndarray" or "matrix" instance!',
        )

        value = as_float_array(value)

        self._whitepoint = value
        self._derived_matrix_XYZ_to_RGB = np.array([])
        self._derived_matrix_RGB_to_XYZ = np.array([])

    @property
    def whitepoint_name(self) -> str | None:
        """
        Getter and setter property for the whitepoint_name.

        Parameters
        ----------
        value
            Value to set the whitepoint_name with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            *RGB* colourspace whitepoint name.
        """

        return self._whitepoint_name

    @whitepoint_name.setter
    def whitepoint_name(self, value: str | None):
        """Setter for the **self.whitepoint_name** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"whitepoint_name" property: "{value}" type is not "str"!',
            )

        self._whitepoint_name = value

    @property
    def matrix_RGB_to_XYZ(self) -> NDArrayFloat:
        """
        Getter and setter property for the transformation matrix from
        colourspace to *CIE XYZ* tristimulus values.

        Parameters
        ----------
        value
            Transformation matrix from colourspace to *CIE XYZ* tristimulus
            values.

        Returns
        -------
        :class:`numpy.ndarray`
            Transformation matrix from colourspace to *CIE XYZ* tristimulus
            values.
        """

        if self._matrix_RGB_to_XYZ is None or self._use_derived_matrix_RGB_to_XYZ:
            if self._derived_matrix_RGB_to_XYZ.size == 0:
                self._derive_transformation_matrices()
            return self._derived_matrix_RGB_to_XYZ
        else:
            return self._matrix_RGB_to_XYZ

    @matrix_RGB_to_XYZ.setter
    def matrix_RGB_to_XYZ(self, value: ArrayLike | None):
        """Setter for the **self.matrix_RGB_to_XYZ** property."""

        if value is not None:
            attest(
                isinstance(value, (tuple, list, np.ndarray, np.matrix)),
                f'"matrix_RGB_to_XYZ" property: "{value!r}" is not a "tuple", '
                f'"list", "ndarray" or "matrix" instance!',
            )

            value = as_float_array(value)

        self._matrix_RGB_to_XYZ = value

    @property
    def matrix_XYZ_to_RGB(self) -> NDArrayFloat:
        """
        Getter and setter property for the transformation matrix from *CIE XYZ*
        tristimulus values to colourspace.

        Parameters
        ----------
        value
            Transformation matrix from *CIE XYZ* tristimulus values to
            colourspace.

        Returns
        -------
        :class:`numpy.ndarray`
            Transformation matrix from *CIE XYZ* tristimulus values to
            colourspace.
        """

        if self._matrix_XYZ_to_RGB is None or self._use_derived_matrix_XYZ_to_RGB:
            if self._derived_matrix_XYZ_to_RGB.size == 0:
                self._derive_transformation_matrices()
            return self._derived_matrix_XYZ_to_RGB
        else:
            return self._matrix_XYZ_to_RGB

    @matrix_XYZ_to_RGB.setter
    def matrix_XYZ_to_RGB(self, value: ArrayLike | None):
        """Setter for the **self.matrix_XYZ_to_RGB** property."""

        if value is not None:
            attest(
                isinstance(value, (tuple, list, np.ndarray, np.matrix)),
                f'"matrix_XYZ_to_RGB" property: "{value!r}" is not a "tuple", '
                f'"list", "ndarray" or "matrix" instance!',
            )

            value = as_float_array(value)

        self._matrix_XYZ_to_RGB = value

    @property
    def cctf_encoding(self) -> Callable | None:
        """
        Getter and setter property for the encoding colour component transfer
        function (Encoding CCTF) / opto-electronic transfer function
        (OETF).

        Parameters
        ----------
        value
            Encoding colour component transfer function (Encoding CCTF) /
            opto-electronic transfer function (OETF).

        Returns
        -------
        :py:data:`None` or Callable
            Encoding colour component transfer function (Encoding CCTF) /
            opto-electronic transfer function (OETF).
        """

        return self._cctf_encoding

    @cctf_encoding.setter
    def cctf_encoding(self, value: Callable | None):
        """Setter for the **self.cctf_encoding** property."""

        if value is not None:
            attest(
                callable(value),
                f'"cctf_encoding" property: "{value}" is not callable!',
            )

        self._cctf_encoding = value

    @property
    def cctf_decoding(self) -> Callable | None:
        """
        Getter and setter property for the decoding colour component transfer
        function (Decoding CCTF) / electro-optical transfer function
        (EOTF).

        Parameters
        ----------
        value
            Decoding colour component transfer function (Decoding CCTF) /
            electro-optical transfer function (EOTF).

        Returns
        -------
        :py:data:`None` or Callable
            Decoding colour component transfer function (Decoding CCTF) /
            electro-optical transfer function (EOTF).
        """

        return self._cctf_decoding

    @cctf_decoding.setter
    def cctf_decoding(self, value: Callable | None):
        """Setter for the **self.cctf_decoding** property."""

        if value is not None:
            attest(
                callable(value),
                f'"cctf_decoding" property: "{value}" is not callable!',
            )

        self._cctf_decoding = value

    @property
    def use_derived_matrix_RGB_to_XYZ(self) -> bool:
        """
        Getter and setter property for whether to use the instantiation time
        normalised primary matrix or to use a computed derived normalised
        primary matrix.

        Parameters
        ----------
        value
            Whether to use the instantiation time normalised primary matrix or
            to use a computed derived normalised primary matrix.

        Returns
        -------
        :class:`bool`
            Whether to use the instantiation time normalised primary matrix or
            to use a computed derived normalised primary matrix.
        """

        return self._use_derived_matrix_RGB_to_XYZ

    @use_derived_matrix_RGB_to_XYZ.setter
    def use_derived_matrix_RGB_to_XYZ(self, value: bool):
        """Setter for the **self.use_derived_matrix_RGB_to_XYZ** property."""

        attest(
            isinstance(value, (bool, np.bool_)),
            f'"use_derived_matrix_RGB_to_XYZ" property: "{value}" is not a "bool"!',
        )

        self._use_derived_matrix_RGB_to_XYZ = value

    @property
    def use_derived_matrix_XYZ_to_RGB(self) -> bool:
        """
        Getter and setter property for Whether to use the instantiation time
        inverse normalised primary matrix or to use a computed derived inverse
        normalised primary matrix.

        Parameters
        ----------
        value
            Whether to use the instantiation time inverse normalised primary
            matrix or to use a computed derived inverse normalised primary
            matrix.

        Returns
        -------
        :class:`bool`
            Whether to use the instantiation time inverse normalised primary
            matrix or to use a computed derived inverse normalised primary
            matrix.
        """

        return self._use_derived_matrix_XYZ_to_RGB

    @use_derived_matrix_XYZ_to_RGB.setter
    def use_derived_matrix_XYZ_to_RGB(self, value: bool):
        """Setter for the **self.use_derived_matrix_XYZ_to_RGB** property."""

        attest(
            isinstance(value, (bool, np.bool_)),
            f'"use_derived_matrix_XYZ_to_RGB" property: "{value}" is not a "bool"!',
        )

        self._use_derived_matrix_XYZ_to_RGB = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *RGB* colourspace.

        Returns
        -------
        :class:`str`
            Formatted string representation.

        Examples
        --------
        >>> p = np.array(
        ...     [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]
        ... )
        >>> whitepoint = np.array([0.32168, 0.33767])
        >>> matrix_RGB_to_XYZ = np.identity(3)
        >>> matrix_XYZ_to_RGB = np.identity(3)
        >>> cctf_encoding = lambda x: x
        >>> cctf_decoding = lambda x: x
        >>> print(
        ...     RGB_Colourspace(
        ...         "RGB Colourspace",
        ...         p,
        ...         whitepoint,
        ...         "ACES",
        ...         matrix_RGB_to_XYZ,
        ...         matrix_XYZ_to_RGB,
        ...         cctf_encoding,
        ...         cctf_decoding,
        ...     )
        ... )
        ... # doctest: +ELLIPSIS
        RGB Colourspace
        ---------------
        <BLANKLINE>
        Primaries          : [[  7.34700000e-01   2.65300000e-01]
                              [  0.00000000e+00   1.00000000e+00]
                              [  1.00000000e-04  -7.70000000e-02]]
        Whitepoint         : [ 0.32168  0.33767]
        Whitepoint Name    : ACES
        Encoding CCTF      : <function <lambda> at 0x...>
        Decoding CCTF      : <function <lambda> at 0x...>
        NPM                : [[ 1.  0.  0.]
                              [ 0.  1.  0.]
                              [ 0.  0.  1.]]
        NPM -1             : [[ 1.  0.  0.]
                              [ 0.  1.  0.]
                              [ 0.  0.  1.]]
        Derived NPM        : \
[[  9.5255239...e-01   0.0000000...e+00   9.3678631...e-05]
                             \
 [  3.4396645...e-01   7.2816609...e-01  -7.2132546...e-02]
                             \
 [  0.0000000...e+00   0.0000000...e+00   1.0088251...e+00]]
        Derived NPM -1     : \
[[  1.0498110...e+00   0.0000000...e+00  -9.7484540...e-05]
                             \
 [ -4.9590302...e-01   1.3733130...e+00   9.8240036...e-02]
                             \
 [  0.0000000...e+00   0.0000000...e+00   9.9125201...e-01]]
        Use Derived NPM    : False
        Use Derived NPM -1 : False
        """
        if self._derived_matrix_XYZ_to_RGB.size == 0:
            self._derive_transformation_matrices()
        return multiline_str(
            self,
            [
                {"name": "name", "section": True},
                {"line_break": True},
                {"name": "primaries", "label": "Primaries"},
                {"name": "whitepoint", "label": "Whitepoint"},
                {"name": "whitepoint_name", "label": "Whitepoint Name"},
                {"name": "cctf_encoding", "label": "Encoding CCTF"},
                {"name": "cctf_decoding", "label": "Decoding CCTF"},
                {"name": "_matrix_RGB_to_XYZ", "label": "NPM"},
                {"name": "_matrix_XYZ_to_RGB", "label": "NPM -1"},
                {
                    "name": "_derived_matrix_RGB_to_XYZ",
                    "label": "Derived NPM",
                },
                {
                    "name": "_derived_matrix_XYZ_to_RGB",
                    "label": "Derived NPM -1",
                },
                {
                    "name": "use_derived_matrix_RGB_to_XYZ",
                    "label": "Use Derived NPM",
                },
                {
                    "name": "use_derived_matrix_XYZ_to_RGB",
                    "label": "Use Derived NPM -1",
                },
            ],
        )

    def __repr__(self) -> str:
        """
        Return an (almost) evaluable string representation of the *RGB*
        colourspace.

        Returns
        -------
        :class`str`
            (Almost) evaluable string representation.

        Examples
        --------
        >>> from colour.models import linear_function
        >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = np.array([0.32168, 0.33767])
        >>> matrix_RGB_to_XYZ = np.identity(3)
        >>> matrix_XYZ_to_RGB = np.identity(3)
        >>> RGB_Colourspace(
        ...     "RGB Colourspace",
        ...     p,
        ...     whitepoint,
        ...     "ACES",
        ...     matrix_RGB_to_XYZ,
        ...     matrix_XYZ_to_RGB,
        ...     linear_function,
        ...     linear_function,
        ... )
        ... # doctest: +ELLIPSIS
        RGB_Colourspace('RGB Colourspace',
                        [[  7.34700000e-01,   2.65300000e-01],
                         [  0.00000000e+00,   1.00000000e+00],
                         [  1.00000000e-04,  -7.70000000e-02]],
                        [ 0.32168,  0.33767],
                        'ACES',
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        linear_function,
                        linear_function,
                        False,
                        False)
        """

        return multiline_repr(
            self,
            [
                {"name": "name"},
                {"name": "primaries"},
                {"name": "whitepoint"},
                {"name": "whitepoint_name"},
                {"name": "matrix_RGB_to_XYZ"},
                {"name": "matrix_XYZ_to_RGB"},
                {
                    "name": "cctf_encoding",
                    "formatter": lambda x: (  # noqa: ARG005
                        None
                        if self.cctf_encoding is None
                        else (
                            self.cctf_encoding.__name__
                            if hasattr(self.cctf_encoding, "__name__")
                            else str(self.cctf_encoding)
                        )
                    ),
                },
                {
                    "name": "cctf_decoding",
                    "formatter": lambda x: (  # noqa: ARG005
                        None
                        if self.cctf_decoding is None
                        else (
                            self.cctf_decoding.__name__
                            if hasattr(self.cctf_decoding, "__name__")
                            else str(self.cctf_decoding)
                        )
                    ),
                },
                {"name": "use_derived_matrix_RGB_to_XYZ"},
                {"name": "use_derived_matrix_XYZ_to_RGB"},
            ],
        )

    def _derive_transformation_matrices(self):
        """
        Compute the derived transformations matrices, the normalised primary
        matrix and its inverse.
        """

        if self._primaries is not None and self._whitepoint is not None:
            npm = normalised_primary_matrix(self._primaries, self._whitepoint)

            self._derived_matrix_RGB_to_XYZ = npm
            self._derived_matrix_XYZ_to_RGB = np.linalg.inv(npm)

    def use_derived_transformation_matrices(self, usage: bool = True):
        """
        Enable or disables usage of both derived transformations matrices,
        the normalised primary matrix and its inverse in subsequent
        computations.

        Parameters
        ----------
        usage
            Whether to use the derived transformations matrices.
        """

        self.use_derived_matrix_RGB_to_XYZ = usage
        self.use_derived_matrix_XYZ_to_RGB = usage

    def chromatically_adapt(
        self,
        whitepoint: ArrayLike,
        whitepoint_name: str | None = None,
        chromatic_adaptation_transform: (
            LiteralChromaticAdaptationTransform | str
        ) = "CAT02",
    ) -> RGB_Colourspace:
        """
        Chromatically adapt the *RGB* colourspace *primaries* :math:`xy`
        chromaticity coordinates from *RGB* colourspace whitepoint to reference
        ``whitepoint``.

        Parameters
        ----------
        whitepoint
            Reference illuminant / whitepoint :math:`xy` chromaticity
            coordinates.
        whitepoint_name
            Reference illuminant / whitepoint name.
        chromatic_adaptation_transform
            *Chromatic adaptation* transform.

        Returns
        -------
        :class:`colour.RGB_Colourspace`
            Chromatically adapted *RGB* colourspace.

        Examples
        --------
        >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> w_t = np.array([0.32168, 0.33767])
        >>> w_r = np.array([0.31270, 0.32900])
        >>> colourspace = RGB_Colourspace("RGB Colourspace", p, w_t, "D65")
        >>> print(colourspace.chromatically_adapt(w_r, "D50", "Bradford"))
        ... # doctest: +ELLIPSIS
        RGB Colourspace - Chromatically Adapted to 'D50'
        ------------------------------------------------
        <BLANKLINE>
        Primaries          : [[ 0.73485524  0.26422533]
                              [-0.00617091  1.01131496]
                              [ 0.01596756 -0.0642355 ]]
        Whitepoint         : [ 0.3127  0.329 ]
        Whitepoint Name    : D50
        Encoding CCTF      : None
        Decoding CCTF      : None
        NPM                : None
        NPM -1             : None
        Derived NPM        : [[ 0.93827985 -0.00445145  0.01662752]
                              [ 0.33736889  0.72952157 -0.06689046]
                              [ 0.00117395 -0.00371071  1.09159451]]
        Derived NPM -1     : [[ 1.06349549  0.00640891 -0.01580679]
                              [-0.49207413  1.36822341  0.09133709]
                              [-0.00281646  0.00464417  0.91641857]]
        Use Derived NPM    : True
        Use Derived NPM -1 : True
        """

        colourspace = self.copy()

        colourspace.primaries = chromatically_adapted_primaries(
            colourspace.primaries,
            colourspace.whitepoint,
            whitepoint,
            chromatic_adaptation_transform,
        )
        colourspace.whitepoint = whitepoint
        colourspace.whitepoint_name = whitepoint_name

        colourspace._matrix_RGB_to_XYZ = None
        colourspace._matrix_XYZ_to_RGB = None
        colourspace._derive_transformation_matrices()
        colourspace.use_derived_transformation_matrices()

        colourspace.name = (
            f"{colourspace.name} - Chromatically Adapted to "
            f"{cast(str, optional(whitepoint_name, whitepoint))!r}"
        )

        return colourspace

    def copy(self) -> RGB_Colourspace:
        """
        Return a copy of the *RGB* colourspace.

        Returns
        -------
        :class:`colour.RGB_Colourspace`
            *RGB* colourspace copy.
        """

        return deepcopy(self)


def XYZ_to_RGB(
    XYZ: ArrayLike,
    colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    illuminant: ArrayLike | None = None,
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
    apply_cctf_encoding: bool = False,
    *args: Any,
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace array.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    colourspace
        Output *RGB* colourspace.
    illuminant
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the input *CIE XYZ* tristimulus values.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    apply_cctf_encoding
        Apply the *RGB* colourspace encoding colour component transfer
        function / opto-electronic transfer function.

    Other Parameters
    ----------------
    args
        Arguments for deprecation management.
    kwargs
        Keywords arguments for deprecation management.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----
    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``XYZ``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_XYZ`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_RGB`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> XYZ = np.array([0.21638819, 0.12570000, 0.03847493])
    >>> illuminant = np.array([0.34570, 0.35850])
    >>> XYZ_to_RGB(XYZ, RGB_COLOURSPACE_sRGB, illuminant, "Bradford")
    ... # doctest: +ELLIPSIS
    array([ 0.4559528...,  0.0304078...,  0.0408731...])
    >>> XYZ_to_RGB(XYZ, "sRGB", illuminant, "Bradford")
    ... # doctest: +ELLIPSIS
    array([ 0.4559528...,  0.0304078...,  0.0408731...])
    """

    from colour.models import RGB_COLOURSPACES

    XYZ = to_domain_1(XYZ)

    if not isinstance(colourspace, (RGB_Colourspace, str)):
        usage_warning(
            'The "colour.XYZ_to_RGB" definition signature has changed with '
            '"Colour 0.4.3". The used call arguments are deprecated, '
            "please refer to the documentation for more information about the "
            "new signature."
        )
        illuminant_XYZ = kwargs.pop("illuminant_XYZ", colourspace)
        illuminant_RGB = kwargs.pop("illuminant_RGB", illuminant)
        matrix_XYZ_to_RGB = kwargs.pop(
            "matrix_XYZ_to_RGB", chromatic_adaptation_transform
        )
        chromatic_adaptation_transform = kwargs.pop(
            "chromatic_adaptation_transform",
            (
                apply_cctf_encoding
                if not isinstance(apply_cctf_encoding, bool)
                else "CAT02"
            ),
        )
        cctf_encoding = kwargs.pop("cctf_encoding", args[0] if len(args) == 1 else None)
        apply_cctf_encoding = True
    else:
        if isinstance(colourspace, str):
            colourspace = validate_method(
                colourspace,
                tuple(RGB_COLOURSPACES),
                '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
            )
            colourspace = RGB_COLOURSPACES[colourspace]

        illuminant_XYZ = optional(
            illuminant,
            colourspace.whitepoint,  # pyright: ignore
        )
        illuminant_RGB = colourspace.whitepoint  # pyright: ignore
        matrix_XYZ_to_RGB = colourspace.matrix_XYZ_to_RGB  # pyright: ignore
        cctf_encoding = colourspace.cctf_encoding  # pyright: ignore

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
            transform=chromatic_adaptation_transform,
        )

        XYZ = vecmul(M_CAT, XYZ)

    RGB = vecmul(matrix_XYZ_to_RGB, XYZ)

    if apply_cctf_encoding and cctf_encoding is not None:
        with domain_range_scale("ignore"):
            RGB = cctf_encoding(RGB)

    return from_range_1(RGB)


def RGB_to_XYZ(
    RGB: ArrayLike,
    colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    illuminant: ArrayLike | None = None,
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
    apply_cctf_decoding: bool = False,
    *args,
    **kwargs,
) -> NDArrayFloat:
    """
    Convert given *RGB* colourspace array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    colourspace
        Input *RGB* colourspace.
    illuminant
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the output *CIE XYZ* tristimulus values.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    apply_cctf_decoding
        Apply the *RGB* colourspace decoding colour component transfer
        function / opto-electronic transfer function.

    Other Parameters
    ----------------
    args
        Arguments for deprecation management.
    kwargs
        Keywords arguments for deprecation management.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_XYZ`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_RGB`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``XYZ``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> illuminant = np.array([0.34570, 0.35850])
    >>> RGB_to_XYZ(RGB, RGB_COLOURSPACE_sRGB, illuminant, "Bradford")
    ... # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])
    >>> RGB_to_XYZ(RGB, "sRGB", illuminant, "Bradford")
    ... # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])
    """

    from colour.models import RGB_COLOURSPACES

    RGB = to_domain_1(RGB)

    if not isinstance(colourspace, (RGB_Colourspace, str)):
        usage_warning(
            'The "colour.RGB_to_XYZ" definition signature has changed with '
            '"Colour 0.4.3". The used call arguments are deprecated, '
            "please refer to the documentation for more information about the "
            "new signature."
        )
        illuminant_RGB = kwargs.pop("illuminant_RGB", colourspace)
        illuminant_XYZ = kwargs.pop("illuminant_XYZ", illuminant)
        matrix_RGB_to_XYZ = kwargs.pop(
            "matrix_RGB_to_XYZ", chromatic_adaptation_transform
        )
        chromatic_adaptation_transform = kwargs.pop(
            "chromatic_adaptation_transform",
            (
                apply_cctf_decoding
                if not isinstance(apply_cctf_decoding, bool)
                else "CAT02"
            ),
        )
        cctf_decoding = kwargs.pop("cctf_decoding", args[0] if len(args) == 1 else None)
        apply_cctf_decoding = True
    else:
        if isinstance(colourspace, str):
            colourspace = validate_method(
                colourspace,
                tuple(RGB_COLOURSPACES),
                '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
            )
            colourspace = RGB_COLOURSPACES[colourspace]

        illuminant_XYZ = optional(
            illuminant,
            colourspace.whitepoint,  # pyright: ignore
        )
        illuminant_RGB = colourspace.whitepoint  # pyright: ignore
        matrix_RGB_to_XYZ = colourspace.matrix_RGB_to_XYZ  # pyright: ignore
        cctf_decoding = colourspace.cctf_decoding  # pyright: ignore

    if apply_cctf_decoding and cctf_decoding is not None:
        with domain_range_scale("ignore"):
            RGB = cctf_decoding(RGB)

    XYZ = vecmul(matrix_RGB_to_XYZ, RGB)

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
            transform=chromatic_adaptation_transform,
        )

        XYZ = vecmul(M_CAT, XYZ)

    return from_range_1(XYZ)


def matrix_RGB_to_RGB(
    input_colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    output_colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
) -> NDArrayFloat:
    """
    Compute the matrix :math:`M` converting from given input *RGB*
    colourspace to output *RGB* colourspace using given *chromatic
    adaptation* method.

    Parameters
    ----------
    input_colourspace
        *RGB* input colourspace.
    output_colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.

    Returns
    -------
    :class:`numpy.ndarray`
        Conversion matrix :math:`M`.

    Examples
    --------
    >>> from colour.models import (
    ...     RGB_COLOURSPACE_sRGB,
    ...     RGB_COLOURSPACE_PROPHOTO_RGB,
    ... )
    >>> matrix_RGB_to_RGB(RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    ... # doctest: +ELLIPSIS
    array([[ 0.5288241...,  0.3340609...,  0.1373616...],
           [ 0.0975294...,  0.8790074...,  0.0233981...],
           [ 0.0163599...,  0.1066124...,  0.8772485...]])
    >>> matrix_RGB_to_RGB("sRGB", "ProPhoto RGB")
    ... # doctest: +ELLIPSIS
    array([[ 0.5288241...,  0.3340609...,  0.1373616...],
           [ 0.0975294...,  0.8790074...,  0.0233981...],
           [ 0.0163599...,  0.1066124...,  0.8772485...]])
    """

    from colour.models import RGB_COLOURSPACES

    if isinstance(input_colourspace, str):
        input_colourspace = validate_method(
            input_colourspace,
            tuple(RGB_COLOURSPACES),
            '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
        )
        input_colourspace = cast(RGB_Colourspace, RGB_COLOURSPACES[input_colourspace])

    if isinstance(output_colourspace, str):
        output_colourspace = validate_method(
            output_colourspace,
            tuple(RGB_COLOURSPACES),
            '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
        )
        output_colourspace = cast(RGB_Colourspace, RGB_COLOURSPACES[output_colourspace])

    M = input_colourspace.matrix_RGB_to_XYZ

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xy_to_XYZ(input_colourspace.whitepoint),
            xy_to_XYZ(output_colourspace.whitepoint),
            chromatic_adaptation_transform,
        )

        M = np.matmul(M_CAT, input_colourspace.matrix_RGB_to_XYZ)

    M = np.matmul(output_colourspace.matrix_XYZ_to_RGB, M)

    return M


def RGB_to_RGB(
    RGB: ArrayLike,
    input_colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    output_colourspace: RGB_Colourspace | LiteralRGBColourspace | str,
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
    apply_cctf_decoding: bool = False,
    apply_cctf_encoding: bool = False,
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Convert given *RGB* colourspace array from given input *RGB* colourspace
    to output *RGB* colourspace using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    input_colourspace
        *RGB* input colourspace.
    output_colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    apply_cctf_decoding
        Apply the input colourspace decoding colour component transfer function
        / electro-optical transfer function.
    apply_cctf_encoding
        Apply the output colourspace encoding colour component transfer
        function / opto-electronic transfer function.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments for the colour component transfer functions.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.

    Notes
    -----
    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.models import (
    ...     RGB_COLOURSPACE_sRGB,
    ...     RGB_COLOURSPACE_PROPHOTO_RGB,
    ... )
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> RGB_to_RGB(RGB, RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    ... # doctest: +ELLIPSIS
    array([ 0.2568891...,  0.0721446...,  0.0465553...])
    >>> RGB_to_RGB(RGB, "sRGB", "ProPhoto RGB")
    ... # doctest: +ELLIPSIS
    array([ 0.2568891...,  0.0721446...,  0.0465553...])
    """

    from colour.models import RGB_COLOURSPACES

    if isinstance(input_colourspace, str):
        input_colourspace = validate_method(
            input_colourspace,
            tuple(RGB_COLOURSPACES),
            '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
        )
        input_colourspace = cast(RGB_Colourspace, RGB_COLOURSPACES[input_colourspace])

    if isinstance(output_colourspace, str):
        output_colourspace = validate_method(
            output_colourspace,
            tuple(RGB_COLOURSPACES),
            '"{0}" "RGB" colourspace is invalid, it must be one of {1}!',
        )
        output_colourspace = cast(RGB_Colourspace, RGB_COLOURSPACES[output_colourspace])

    RGB = to_domain_1(RGB)

    if apply_cctf_decoding and input_colourspace.cctf_decoding is not None:
        with domain_range_scale("ignore"):
            RGB = input_colourspace.cctf_decoding(
                RGB, **filter_kwargs(input_colourspace.cctf_decoding, **kwargs)
            )

    M = matrix_RGB_to_RGB(
        input_colourspace, output_colourspace, chromatic_adaptation_transform
    )

    RGB = vecmul(M, RGB)

    if apply_cctf_encoding and output_colourspace.cctf_encoding is not None:
        with domain_range_scale("ignore"):
            RGB = output_colourspace.cctf_encoding(
                RGB,
                **filter_kwargs(output_colourspace.cctf_encoding, **kwargs),
            )

    return from_range_1(RGB)
