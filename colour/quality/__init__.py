from __future__ import annotations

from colour.colorimetry import SpectralDistribution
from colour.hints import Floating, Literal, Union

from .datasets import *  # noqa
from . import datasets
from .cfi2017 import (
    ColourRendering_Specification_CIE2017,
    colour_fidelity_index_CIE2017,
)
from .cri import ColourRendering_Specification_CRI, colour_rendering_index
from .cqs import (
    COLOUR_QUALITY_SCALE_METHODS,
    ColourRendering_Specification_CQS,
    colour_quality_scale,
)
from .ssi import spectral_similarity_index
from .tm3018 import (
    ColourQuality_Specification_ANSIIESTM3018,
    colour_fidelity_index_ANSIIESTM3018,
)
from colour.utilities import CaseInsensitiveMapping, validate_method

__all__ = []
__all__ += datasets.__all__
__all__ += [
    "ColourRendering_Specification_CIE2017",
    "colour_fidelity_index_CIE2017",
]
__all__ += [
    "ColourQuality_Specification_ANSIIESTM3018",
    "colour_fidelity_index_ANSIIESTM3018",
]
__all__ += [
    "ColourRendering_Specification_CRI",
    "colour_rendering_index",
]
__all__ += [
    "ColourRendering_Specification_CQS",
    "COLOUR_QUALITY_SCALE_METHODS",
    "colour_quality_scale",
]
__all__ += [
    "spectral_similarity_index",
]

COLOUR_FIDELITY_INDEX_METHODS = CaseInsensitiveMapping(
    {
        "CIE 2017": colour_fidelity_index_CIE2017,
        "ANSI/IES TM-30-18": colour_fidelity_index_ANSIIESTM3018,
    }
)
COLOUR_FIDELITY_INDEX_METHODS.__doc__ = """
Supported *Colour Fidelity Index* (CFI) computation methods.

References
----------
:cite:`CIETC1-902017`, :cite:`ANSI2018`
"""


def colour_fidelity_index(
    sd_test: SpectralDistribution,
    additional_data=False,
    method: Union[Literal["CIE 2017", "ANSI/IES TM-30-18"], str] = "CIE 2017",
) -> Union[
    Floating,
    ColourRendering_Specification_CIE2017,
    ColourQuality_Specification_ANSIIESTM3018,
]:
    """
    Return the *Colour Fidelity Index* (CFI) :math:`R_f` of given spectral
    distribution using given method.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or \
:class:`colour.quality.ColourRendering_Specification_CIE2017` or \
:class:`colour.quality.ColourQuality_Specification_ANSIIESTM3018`
        *Colour Fidelity Index* (CFI) :math:`R_f`.

    References
    ----------
    :cite:`CIETC1-902017`, :cite:`ANSI2018`

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_fidelity_index(sd)  # doctest: +ELLIPSIS
    70.1208254...
    """

    method = validate_method(method, COLOUR_FIDELITY_INDEX_METHODS)

    function = COLOUR_FIDELITY_INDEX_METHODS[method]

    return function(sd_test, additional_data)


__all__ += [
    "COLOUR_FIDELITY_INDEX_METHODS",
    "colour_fidelity_index",
]
