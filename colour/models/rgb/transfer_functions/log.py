# -*- coding: utf-8 -*-
"""
Common Log Encodings
====================

Defines the common log encodings:

-   :func:`colour.models.log_encoding_Log2`
-   :func:`colour.models.log_decoding_Log2`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciencesa` :
    The Academy of Motion Picture Arts and Sciences,
    Science and Technology Council,
    & Academy Color Encoding System (ACES) Project Subcommittee.(n.d.).
    ACESutil.Lin_to_Log2_param.ctl. Retrieved June 14, 2020,
    from https://github.com/ampas/aces-dev/blob/\
518c27f577e99cdecfddf2ebcfaa53444b1f9343/transforms/ctl/utilities/\
ACESutil.Lin_to_Log2_param.ctl
-   :cite:`TheAcademyofMotionPictureArtsandSciencesb` :
    The Academy of Motion Picture Arts and Sciences,
    Science and Technology Council,
    & Academy Color Encoding System (ACES) Project Subcommittee.(n.d.).
    ACESutil.Log2_to_Lin_param.ctl. Retrieved June 14, 2020,
    from https://github.com/ampas/aces-dev/blob/\
518c27f577e99cdecfddf2ebcfaa53444b1f9343/transforms/ctl/utilities/\
ACESutil.Log2_to_Lin_param.ctl
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.utilities import from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['log_encoding_Log2', 'log_decoding_Log2']


def log_encoding_Log2(lin,
                      middle_grey=0.18,
                      min_exposure=0.18 * 2 ** -6.5,
                      max_exposure=0.18 * 2 ** 6.5):
    """
    Defines the common *Log2* encoding function.

    Parameters
    ----------
    lin : numeric or array_like
          Linear data to undergo encoding.
    middle_grey : numeric, optional
          *Middle Grey* exposure value.
    min_exposure : numeric, optional
          Minimum exposure level.
    max_exposure : numeric, optional
          Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Non-linear *Log2* encoded data.

    Notes
    -----

    The common *Log2* encoding function can be used
    to build linear to logarithmic shapers in the
    ACES OCIO configuration.

    A (48-nits OCIO) shaper having values in a linear
    domain, can be encoded to a logarithmic domain:

    +-------------------+-------------------+
    | **Shaper Domain** | **Shaper Range**  |
    +===================+===================+
    | [0.002, 16.291]   | [0, 1]            |
    +-------------------+-------------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciencesa`

    Examples
    --------
    Linear numeric input is encoded as follows:

    >>> log_encoding_Log2(18)
    0.40773288970434662

    Linear array-like input is encoded as follows:

    >>> log_encoding_Log2(np.linspace(1, 2, 3))
    array([ 0.15174832,  0.18765817,  0.21313661])
    """

    lin = to_domain_1(lin)

    lg2 = np.log2(lin / middle_grey)
    log_norm = (lg2 - min_exposure) / (max_exposure - min_exposure)

    return from_range_1(log_norm)


def log_decoding_Log2(log_norm,
                      middle_grey=0.18,
                      min_exposure=0.18 * 2 ** -6.5,
                      max_exposure=0.18 * 2 ** 6.5):
    """
    Defines the common *Log2* decoding function.

    Parameters
    ----------
    log_norm : numeric or array_like
               Logarithmic data to undergo decoding.
    middle_grey : numeric, optional
               *Middle Grey* exposure value.
    min_exposure : numeric, optional
               Minimum exposure level.
    max_exposure : numeric, optional
               Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Linear *Log2* decoded data.

    Notes
    -----

    The common *Log2* decoding function can be used
    to build logarithmic to linear shapers in the
    ACES OCIO configuration.

    The shaper with logarithmic encoded values can be
    decoded back to linear domain:

    +-------------------+-------------------+
    | **Shaper Range**  | **Shaper Domain** |
    +===================+===================+
    | [0, 1]            | [0.002, 16.291]   |
    +-------------------+-------------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciencesb`

    Examples
    --------
    Logarithmic input is decoded as follows:

    >>> log_decoding_Log2(0.40773288970434662)
    17.999999999999993

    Linear array-like input is encoded as follows:

    >>> log_decoding_Log2(np.linspace(0, 1, 4))
    array([  1.80248299e-01,   7.77032379e+00,   3.34970882e+02,
             1.44402595e+04])
    """

    log_norm = to_domain_1(log_norm)

    lg2 = log_norm * (max_exposure - min_exposure) + min_exposure
    lin = (2 ** lg2) * middle_grey

    return from_range_1(lin)
