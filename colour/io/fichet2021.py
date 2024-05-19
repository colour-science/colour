"""
OpenEXR Layout for Spectral Images - Fichet, Pacanowski and Wilkie (2021)
=========================================================================

Defines the *Fichet et al. (2021)* spectral image input / output objects.

References
----------
-   :cite:`Fichet2021` : Fichet, A., Pacanowski, R., & Wilkie, A. (2021). An
    OpenEXR Layout for Spectral Images. 10(3). Retrieved April 26, 2024, from
    http://jcgt.org/published/0010/03/01/
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    sds_and_msds_to_msds,
)
from colour.constants import CONSTANT_LIGHT_SPEED
from colour.hints import (
    Callable,
    Dict,
    List,
    Literal,
    NDArrayFloat,
    Sequence,
    Tuple,
    Union,
)
from colour.io.image import (
    MAPPING_BIT_DEPTH,
    Image_Specification_Attribute,
    add_attributes_to_image_specification_OpenImageIO,
)
from colour.models import RGB_COLOURSPACE_sRGB, XYZ_to_RGB
from colour.utilities import (
    as_float_array,
    interval,
    required,
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
    "MAPPING_UNIT_CONVERSION",
    "PATTERN_FICHET2021",
    "ComponentsFichet2021",
    "match_groups_to_nm",
    "sd_to_spectrum_attribute_Fichet2021",
    "spectrum_attribute_to_sd_Fichet2021",
    "Specification_Fichet2021",
    "read_spectral_image_Fichet2021",
    "sds_and_msds_to_components_Fichet2021",
    "components_to_sRGB_Fichet2021",
    "write_spectral_image_Fichet2021",
]

MAPPING_UNIT_CONVERSION: dict = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "": 1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}
"""
Unit conversion mapping.

References
----------
:cite:`Fichet2021`
"""

PATTERN_FICHET2021: str = (
    r"(\d*,?\d*([eE][-+]?\d+)?)(Y|Z|E|P|T|G|M|k|h|da|d|c|m|u|n|p|f|a|z|y)?(m|Hz)"
)
"""
Regex pattern for numbers and quantities.

References
----------
:cite:`Fichet2021`
"""


ComponentsFichet2021 = Dict[Union[str, float], Tuple[NDArrayFloat, NDArrayFloat]]


def match_groups_to_nm(
    number: str,
    multiplier: Literal[
        "Y",
        "Z",
        "E",
        "P",
        "T",
        "G",
        "M",
        "k",
        "h",
        "da",
        "",
        "d",
        "c",
        "m",
        "u",
        "n",
        "p",
    ]
    | str,
    units: Literal["m", "Hz"] | str,
) -> float:
    """
    Convert match groups of a wavelength (or frequency) to the nanometer value.

    Parameters
    ----------
    number
        Wavelength (or frequency) number to convert.
    multiplier
        Unit multiplier.
    units
        Frequency or wavelength.

    Returns
    -------
    :class:`float`
        Nanometer value.

    Examples
    --------
    >>> match_groups_to_nm("555.5", "n", "m")
    555.5
    >>> match_groups_to_nm("555.5", "", "m")
    555500000000.0
    >>> from colour.constants import CONSTANT_LIGHT_SPEED
    >>> match_groups_to_nm(str(CONSTANT_LIGHT_SPEED / (555 * 1e-9)), "", "Hz")
    ... # doctest: +ELLIPSIS
    555.0000000...
    """

    multiplier = validate_method(
        multiplier, tuple(MAPPING_UNIT_CONVERSION), as_lowercase=False
    )
    units = validate_method(units, ("m", "Hz"), as_lowercase=False)

    v = float(number.replace(",", "."))

    if multiplier == "n" and units == "m":
        return v

    v *= MAPPING_UNIT_CONVERSION[multiplier]

    if units == "m":
        v *= 1e9
    elif units == "Hz":
        v = CONSTANT_LIGHT_SPEED / v * 1e9

    return v


def sd_to_spectrum_attribute_Fichet2021(
    sd: SpectralDistribution, decimals: int = 7
) -> str:
    """
    Convert a spectral distribution to a spectrum attribute value according to
    *Fichet et al. (2021)*.

    Parameters
    ----------
    sd
        Spectral distribution to convert.
    decimals
        Formatting decimals.

    Returns
    -------
    :class:`str`
        Spectrum attribute value.

    References
    ----------
    :cite:`Fichet2021`

    Examples
    --------
    >>> sd_to_spectrum_attribute_Fichet2021(SDS_ILLUMINANTS["D65"], 2)[:56]
    '300.00nm:0.03;305.00nm:1.66;310.00nm:3.29;315.00nm:11.77'
    """

    return ";".join(
        f"{wavelength:.{decimals}f}nm:{value:.{decimals}f}"
        for wavelength, value in zip(sd.wavelengths, sd.values)
    )


def spectrum_attribute_to_sd_Fichet2021(
    spectrum_attribute: str,
) -> SpectralDistribution:
    """
    Convert a spectrum attribute value to a spectral distribution according to
    *Fichet et al. (2021)*.

    Parameters
    ----------
    spectrum_attribute
        Spectrum attribute value to convert.

    Returns
    -------
    :class:`SpectralDistribution`
        Spectral distribution.

    References
    ----------
    :cite:`Fichet2021`

    Examples
    --------
    >>> spectrum_attribute_to_sd_Fichet2021(
    ...     "300.00nm:0.03;305.00nm:1.66;310.00nm:3.29;315.00nm:11.77"
    ... )  # doctest: +SKIP
    SpectralDistribution([[  3.0000000...e+02,   3.0000000...e-02],
                          [  3.0500000...e+02,   1.6600000...e+00],
                          [  3.1000000...e+02,   3.2900000...e+00],
                          [  3.1500000...e+02,   1.1770000...e+01]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    data = {}
    pattern = re.compile(PATTERN_FICHET2021)
    parts = spectrum_attribute.split(";")
    for part in parts:
        domain, range_ = part.split(":")
        match = pattern.match(domain.replace(".", ","))
        if match is not None:
            multiplier, units = match.group(3, 4)
            wavelength = match_groups_to_nm(match.group(1), multiplier, units)
            data[wavelength] = float(range_)

    return SpectralDistribution(data)


@dataclass
class Specification_Fichet2021:
    """
    Define the *Fichet et al. (2021)* spectral image specification.

    Parameters
    ----------
    path
        Path of the spectral image.
    components
        Components of the spectral image, e.g., *S0*, *S1*, *S2*, *S3*, *T*, or
        any wavelength number for bi-spectral images.
    is_emissive
        Whether the image is emissive, i.e, using the *S0* component.
    is_polarised
        Whether the image is polarised, i.e, using the *S0*, *S1*, *S2*, and
        *S3* components.
    is_bispectral
        Whether the image is bi-spectral, i.e, using the *T*, and any
        wavelength number.
    attributes
        An array of :class:`colour.io.Image_Specification_Attribute` class
        instances used to set attributes of the image.

    Methods
    -------
    -   :meth:`~colour.Specification_Fichet2021.from_spectral_image`

    References
    ----------
    :cite:`Fichet2021`
    """  # noqa: D405, D407, D410, D411

    path: str | None = field(default_factory=lambda: None)
    components: defaultdict = field(default_factory=lambda: defaultdict(dict))
    is_emissive: bool = field(default_factory=lambda: False)
    is_polarised: bool = field(default_factory=lambda: False)
    is_bispectral: bool = field(default_factory=lambda: False)
    attributes: List | None = field(default_factory=lambda: None)

    @staticmethod
    @required("OpenImageIO")
    def from_spectral_image(path: str | Path) -> Specification_Fichet2021:
        """
        Create a *Fichet et al. (2021)* spectral image specification from given
        image path.

        Parameters
        ----------
        path
            Image path

        Returns
        -------
        :class:`Specification_Fichet2021`
            *Fichet et al. (2021)* spectral image specification.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> path = os.path.join(
        ...     colour.__path__[0],
        ...     "io",
        ...     "tests",
        ...     "resources",
        ...     "D65.exr",
        ... )
        >>> specification = Specification_Fichet2021.from_spectral_image(path)
        ... # doctest: +SKIP
        >>> specification.is_emissive  # doctest: +SKIP
        True
        """

        from OpenImageIO import ImageInput

        path = str(path)

        components = defaultdict(dict)
        is_emissive = False
        is_polarised = False
        is_bispectral = False

        pattern_emissive = re.compile(rf"^(S[0-3])\.*{PATTERN_FICHET2021}$")
        pattern_reflective = re.compile(rf"^T\.*{PATTERN_FICHET2021}$")
        pattern_bispectral = re.compile(
            rf"^T\.*{PATTERN_FICHET2021}\.*{PATTERN_FICHET2021}$"
        )

        image_specification = ImageInput.open(path).spec()
        channels = image_specification.channelnames

        for i, channel in enumerate(channels):
            match = pattern_emissive.match(channel)
            if match:
                is_emissive = True

                component = match.group(1)
                multiplier, units = match.group(4, 5)
                wavelength = match_groups_to_nm(match.group(2), multiplier, units)
                components[component][wavelength] = i

                if len(components) > 1:
                    is_polarised = True

            match = pattern_bispectral.match(channel)
            if match:
                is_bispectral = True

                input_multiplier, input_units = match.group(3, 4)
                input_wavelength = match_groups_to_nm(
                    match.group(1), input_multiplier, input_units
                )
                output_multiplier, output_units = match.group(7, 8)
                output_wavelength = match_groups_to_nm(
                    match.group(5), output_multiplier, output_units
                )
                components[input_wavelength][output_wavelength] = i

            match = pattern_reflective.match(channel)
            if match:
                multiplier, units = match.group(3, 4)
                wavelength = match_groups_to_nm(match.group(1), multiplier, units)
                components["T"][wavelength] = i

        attributes = []
        for attribute in image_specification.extra_attribs:
            attributes.append(
                Image_Specification_Attribute(
                    attribute.name, attribute.value, attribute.type
                )
            )

        return Specification_Fichet2021(
            path,
            components,
            is_emissive,
            is_polarised,
            is_bispectral,
            attributes,
        )


@required("OpenImageIO")
def read_spectral_image_Fichet2021(
    path: str | Path,
    bit_depth: Literal["float16", "float32"] = "float32",
    additional_data: bool = False,
) -> ComponentsFichet2021 | Tuple[ComponentsFichet2021, Specification_Fichet2021]:
    """
    Read the *Fichet et al. (2021)* spectral image at given path using
    *OpenImageIO*.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit-depth.
    additional_data
        Whether to return additional data.

    Returns
    -------
    :class:`dict` or :class:`tuple`
        Dictionary of component names and their corresponding tuple of
        wavelengths and values or tuple of the aforementioned dictionary and
        :class:`colour.Specification_Fichet2021` class instance.

    Notes
    -----
    -   Spectrum attributes are not parsed but can be converted to spectral
        distribution using the :func:`colour.io.spectrum_attribute_to_sd_Fichet2021`
        definition.

    References
    ----------
    :cite:`Fichet2021`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "D65.exr",
    ... )
    >>> msds, specification = read_spectral_image_Fichet2021(
    ...     path, additional_data=True
    ... )  # doctest: +SKIP
    >>> components.keys()  # doctest: +SKIP
    dict_keys(['S0'])
    >>> components["S0"][0].shape  # doctest: +SKIP
    (97,)
    >>> components["S0"][1].shape  # doctest: +SKIP
    (1, 1, 97)
    >>> specification.is_emissive  # doctest: +SKIP
    True
    """

    from OpenImageIO import ImageInput

    path = str(path)

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    specification = Specification_Fichet2021.from_spectral_image(path)
    image = ImageInput.open(path).read_image(bit_depth_specification.openimageio)

    components = {}
    for component, wavelengths_indexes in specification.components.items():
        wavelengths, indexes = zip(*wavelengths_indexes.items())
        values = as_float_array(
            image[:, :, indexes],
            dtype=bit_depth_specification.numpy,
        )
        components[component] = (
            as_float_array(wavelengths),
            np.array(values, dtype=bit_depth_specification.numpy),
        )

    if additional_data:
        return components, specification
    else:
        return components


def sds_and_msds_to_components_Fichet2021(
    sds: Sequence[SpectralDistribution | MultiSpectralDistributions]
    | SpectralDistribution
    | MultiSpectralDistributions,
    specification: Specification_Fichet2021 = Specification_Fichet2021(),
    **kwargs,
) -> ComponentsFichet2021:
    """
    Convert given spectral and multi-spectral distributions to
    *Fichet et al. (2021)* components.

    The spectral and multi-spectral distributions will be aligned to the
    intersection of their spectral shapes.

    Parameters
    ----------
    sds
        Spectral and multi-spectral distributions to convert to
        *Fichet et al. (2021)* components.
    specification
        *Fichet et al. (2021)* spectral image specification, used to generate
        the proper component type, i.e., emissive or other.

    Other Parameters
    ----------------
    shape
        Optional shape the *Fichet et al. (2021)* components should take: Used
        when converting spectral distributions of a colour
        rendition chart to create a rectangular image rather than a single
        line of values.

    Returns
    -------
    :class:`dict`
        Dictionary of component names and their corresponding tuple of
        wavelengths and values.

    References
    ----------
    :cite:`Fichet2021`

    Examples
    --------
    >>> components = sds_and_msds_to_components_Fichet2021(SDS_ILLUMINANTS["D65"])
    >>> components.keys()
    dict_keys(['T'])
    >>> components = sds_and_msds_to_components_Fichet2021(
    ...     SDS_ILLUMINANTS["D65"], Specification_Fichet2021(is_emissive=True)
    ... )
    >>> components.keys()
    dict_keys(['S0'])
    >>> components["S0"][0].shape
    (97,)
    >>> components["S0"][1].shape
    (1, 1, 97)
    """

    msds = sds_and_msds_to_msds(sds)
    component = "S0" if specification.is_emissive else "T"

    wavelengths = msds.wavelengths
    values = np.transpose(msds.values)
    values = np.reshape(values, (1, -1, values.shape[-1]))

    if "shape" in kwargs:
        values = np.reshape(values, kwargs["shape"])

    return {component: (wavelengths, values)}


@required("OpenImageIO")
def components_to_sRGB_Fichet2021(
    components: ComponentsFichet2021,
    specification: Specification_Fichet2021 = Specification_Fichet2021(),
) -> Tuple[NDArrayFloat | None, Sequence[Image_Specification_Attribute]]:
    """
    Convert given *Fichet et al. (2021)* components to *sRGB* colourspace values.

    Parameters
    ----------
    components
        *Fichet et al. (2021)* components to convert.
    specification
        *Fichet et al. (2021)* spectral image specification, used to perform
        the proper conversion to *sRGB* colourspace values.

    Returns
    -------
    :class:`tuple`
        Tuple of *sRGB* colourspace values and list of
        :class:`colour.io.Image_Specification_Attribute` class instances.

    Warnings
    --------
    -   This definition currently assumes a uniform wavelength interval.
    -   This definition currently does not support integration of bi-spectral
        component.

    Notes
    -----
    -   When an emissive component is given, its exposure will be normalised so
        that its median is 0.18.

    References
    ----------
    :cite:`Fichet2021`

    Examples
    --------
    >>> specification = Specification_Fichet2021(is_emissive=True)
    >>> components = sds_and_msds_to_components_Fichet2021(
    ...     SDS_ILLUMINANTS["D65"],
    ...     specification,
    ... )
    >>> RGB, attributes = components_to_sRGB_Fichet2021(
    ...     components["S0"], specification
    ... )  # doctest: +SKIP
    >>> RGB  # doctest: +SKIP
    array([[[ 0.1799829...,  0.1800080...,  0.1800090...]]])
    >>> for attribute in attributes:
    ...     print(attribute.name)  # doctest: +SKIP
    X
    Y
    Z
    illuminant
    chromaticities
    EV
    """

    from OpenImageIO import TypeDesc

    component = components.get("S0", components.get("T"))

    if component is None:
        return None, []

    # TODO: Implement support for integration of bi-spectral component.
    if specification.is_bispectral:
        usage_warning(
            "Bi-spectral components conversion to *sRGB* colourspace values "
            "is unsupported!"
        )

    # TODO: Implement support for re-binning component with non-uniform interval.
    if len(interval(component[0])) != 1:
        usage_warning(
            "Components have a non-uniform interval, unexpected results might occur!"
        )

    msds = component[1]
    shape = SpectralShape(component[0][0], component[0][-1], interval(component[0])[0])

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    colourspace = RGB_COLOURSPACE_sRGB

    if specification.is_emissive:
        illuminant = SDS_ILLUMINANTS["E"]

        XYZ = msds_to_XYZ(msds, cmfs=cmfs, method="Integration", shape=shape)
    else:
        illuminant = SDS_ILLUMINANTS["D65"]

        XYZ = (
            msds_to_XYZ(
                msds,
                cmfs=cmfs,
                illuminant=illuminant,
                method="Integration",
                shape=shape,
            )
            / 100
        )

    RGB = XYZ_to_RGB(XYZ, colourspace)

    chromaticities = np.ravel(
        np.vstack([colourspace.primaries, colourspace.whitepoint])
    ).tolist()

    attributes = [
        Image_Specification_Attribute(
            "X", sd_to_spectrum_attribute_Fichet2021(cmfs.signals["x_bar"])
        ),
        Image_Specification_Attribute(
            "Y", sd_to_spectrum_attribute_Fichet2021(cmfs.signals["y_bar"])
        ),
        Image_Specification_Attribute(
            "Z", sd_to_spectrum_attribute_Fichet2021(cmfs.signals["z_bar"])
        ),
        Image_Specification_Attribute(
            "illuminant", sd_to_spectrum_attribute_Fichet2021(illuminant)
        ),
        Image_Specification_Attribute(
            "chromaticities", chromaticities, TypeDesc("float[8]")
        ),
    ]

    if specification.is_emissive:
        EV = np.mean(RGB) / 0.18
        RGB /= EV
        attributes.append(
            Image_Specification_Attribute("EV", np.log2(EV)),
        )

    return RGB, attributes


@required("OpenImageIO")
def write_spectral_image_Fichet2021(
    components: Sequence[SpectralDistribution | MultiSpectralDistributions]
    | SpectralDistribution
    | MultiSpectralDistributions
    | ComponentsFichet2021,
    path: str | Path,
    bit_depth: Literal["float16", "float32"] = "float32",
    specification: Specification_Fichet2021 = Specification_Fichet2021(),
    components_to_RGB_callable: Callable = components_to_sRGB_Fichet2021,
    **kwargs,
):
    """
    Write given *Fichet et al. (2021)* components to given path using *OpenImageIO*.

    Parameters
    ----------
    components
        *Fichet et al. (2021)* components.
    path
        Image path.
    bit_depth
        Bit-depth to write the image at, the bit-depth conversion behaviour is
        ruled directly by *OpenImageIO*.
    specification
        *Fichet et al. (2021)* spectral image specification.
    components_to_RGB_callable
        Callable converting the components to a preview *RGB* image.

    Other Parameters
    ----------------
    shape
        Optional shape the *Fichet et al. (2021)* components should take: Used
        when converting spectral distributions of a colour
        rendition chart to create a rectangular image rather than a single
        line of values.

    Returns
    -------
    :class:`bool`:
        Definition success.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "BabelColorAverage.exr",
    ... )
    >>> msds = list(colour.SDS_COLOURCHECKERS["BabelColor Average"].values())
    >>> specification = Specification_Fichet2021(is_emissive=False)
    >>> write_spectral_image_Fichet2021(
    ...     msds,
    ...     path,
    ...     "float16",
    ...     specification,
    ...     shape=(4, 6, len(msds[0].shape.wavelengths)),
    ... )  # doctest: +SKIP
    True
    """

    from OpenImageIO import ImageBuf, ImageBufAlgo

    path = str(path)

    if isinstance(
        components, (Sequence, SpectralDistribution, MultiSpectralDistributions)
    ):
        components = sds_and_msds_to_components_Fichet2021(
            components, specification, **kwargs
        )

    if specification.attributes is None:
        specification.attributes = [
            Image_Specification_Attribute("spectralLayoutVersion", "1.0")
        ]

        if specification.is_emissive:
            specification.attributes.extend(
                [
                    Image_Specification_Attribute("polarisationHandedness", "right"),
                    Image_Specification_Attribute("emissiveUnits", "W.m^-2.sr^-1"),
                ]
            )

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    channels = {}

    RGB, attributes = components_to_RGB_callable(components, specification)
    if RGB is not None:
        channels.update({"R": RGB[..., 0], "G": RGB[..., 1], "B": RGB[..., 2]})

    for component, wavelengths_values in components.items():
        wavelengths, values = wavelengths_values
        for i, wavelength in enumerate(wavelengths):
            component_type = str(component)[0]
            if component_type == "S":  # Emissive Component Type # noqa: SIM114
                channel_name = f'{component}.{str(wavelength).replace(".", ",")}nm'
            elif component_type == "T":  # Reflectance et al. Component Type
                channel_name = f'{component}.{str(wavelength).replace(".", ",")}nm'
            else:  # Bi-spectral Component Type
                channel_name = (
                    f'T.{str(component).replace(".", ",")}nm.'
                    f'{str(wavelength).replace(".", ",")}nm'
                )

            channels[channel_name] = values[..., i]

    image_buffer = ImageBuf()
    for channel_name, channel_data in channels.items():
        channel_buffer = ImageBuf(channel_data.astype(bit_depth_specification.numpy))
        channel_specification = channel_buffer.specmod()
        channel_specification.channelnames = [channel_name]
        image_buffer = ImageBufAlgo.channel_append(image_buffer, channel_buffer)

    add_attributes_to_image_specification_OpenImageIO(
        image_buffer.specmod(), [*specification.attributes, *attributes]
    )

    image_buffer.write(path)

    return True
