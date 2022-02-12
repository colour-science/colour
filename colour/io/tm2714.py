"""
IES TM-27-14 Data Input / Output
================================

Defines the :class:`colour.SpectralDistribution_IESTM2714` class handling *IES
TM-27-14* spectral data *XML* files.

References
----------
-   :cite:`IESComputerCommittee2014a` : IES Computer Committee, & TM-27-14
    Working Group. (2014). IES Standard Format for the Electronic Transfer of
    Spectral Data Electronic Transfer of Spectral Data. Illuminating
    Engineering Society. ISBN:978-0-87995-295-2
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree  # nosec
from xml.dom import minidom  # nosec

from colour.colorimetry import SpectralDistribution
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    Floating,
    Integer,
    Literal,
    Optional,
    cast,
)
from colour.utilities import (
    Structure,
    as_float_array,
    as_float_scalar,
    attest,
    optional,
    is_numeric,
    is_string,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "VERSION_IESTM2714",
    "NAMESPACE_IESTM2714",
    "Element_Specification_IESTM2714",
    "Header_IESTM2714",
    "SpectralDistribution_IESTM2714",
]

VERSION_IESTM2714: str = "1.0"

NAMESPACE_IESTM2714: str = "http://www.ies.org/iestm2714"


@dataclass
class Element_Specification_IESTM2714:
    """
    *IES TM-27-14* spectral data *XML* file element specification.

    Parameters
    ----------
    element
        Element name.
    attribute
        Associated attribute name.
    type_
        Element type.
    required
        Is element required.
    read_conversion
        Method to convert from *XML* to type on reading.
    write_conversion
        Method to convert from type to *XML* on writing.
    """

    element: str
    attribute: str
    type_: Any = field(default_factory=str)
    required: Boolean = field(default_factory=lambda: False)
    read_conversion: Callable = field(
        default_factory=lambda: lambda x: None if x == "None" else str(x)
    )
    write_conversion: Callable = field(default_factory=lambda: str)


class Header_IESTM2714:
    """
    Define the header object for a *IES TM-27-14* spectral distribution.

    Parameters
    ----------
    manufacturer
        Manufacturer of the device under test.
    catalog_number
        Manufacturer's product catalog number.
    description
        Description of the spectral data in the spectral data *XML* file.
    document_creator
        Creator of the spectral data *XML* file, which may be a
        test lab, a research group, a standard body, a company or an
        individual.
    unique_identifier
        Unique identifier to the product under test or the spectral data in the
        document.
    measurement_equipment
        Description of the equipment used to measure the spectral data.
    laboratory
        Testing laboratory name that performed the spectral data measurements.
    report_number
        Testing laboratory report number.
    report_date
        Testing laboratory report date using the *XML DateTime Data Type*,
        *YYYY-MM-DDThh:mm:ss*.
    document_creation_date
        Spectral data *XML* file creation date using the
        *XML DateTime Data Type*, *YYYY-MM-DDThh:mm:ss*.
    comments
        Additional information relating to the tested and reported data.

    Attributes
    ----------
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.mapping`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.manufacturer`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.catalog_number`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.description`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.document_creator`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.unique_identifier`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.measurement_equipment`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.laboratory`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.report_number`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.report_date`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.document_creation_date`
    -   :attr:`~colour.io.ies_tm2714.Header_IESTM2714.comments`

    Methods
    -------
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__init__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__hash__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__eq__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__ne__`

    Examples
    --------
    >>> Header_IESTM2714('colour-science')  # doctest: +ELLIPSIS
    <...Header_IESTM2714 object at 0x...>
    >>> Header_IESTM2714('colour-science').manufacturer  # doctest: +SKIP
    'colour-science'
    """

    def __init__(
        self,
        manufacturer: Optional[str] = None,
        catalog_number: Optional[str] = None,
        description: Optional[str] = None,
        document_creator: Optional[str] = None,
        unique_identifier: Optional[str] = None,
        measurement_equipment: Optional[str] = None,
        laboratory: Optional[str] = None,
        report_number: Optional[str] = None,
        report_date: Optional[str] = None,
        document_creation_date: Optional[str] = None,
        comments: Optional[str] = None,
    ):

        self._mapping: Structure = Structure(
            **{
                "element": "Header",
                "elements": (
                    Element_Specification_IESTM2714(
                        "Manufacturer", "manufacturer"
                    ),
                    Element_Specification_IESTM2714(
                        "CatalogNumber", "catalog_number"
                    ),
                    Element_Specification_IESTM2714(
                        "Description", "description", required=True
                    ),
                    Element_Specification_IESTM2714(
                        "DocumentCreator", "document_creator", required=True
                    ),
                    Element_Specification_IESTM2714(
                        "UniqueIdentifier", "unique_identifier"
                    ),
                    Element_Specification_IESTM2714(
                        "MeasurementEquipment", "measurement_equipment"
                    ),
                    Element_Specification_IESTM2714(
                        "Laboratory", "laboratory"
                    ),
                    Element_Specification_IESTM2714(
                        "ReportNumber", "report_number"
                    ),
                    Element_Specification_IESTM2714(
                        "ReportDate", "report_date"
                    ),
                    Element_Specification_IESTM2714(
                        "DocumentCreationDate",
                        "document_creation_date",
                        required=True,
                    ),
                    Element_Specification_IESTM2714(
                        "Comments", "comments", False
                    ),
                ),
            }
        )

        self._manufacturer: Optional[str] = None
        self.manufacturer = manufacturer
        self._catalog_number: Optional[str] = None
        self.catalog_number = catalog_number
        self._description: Optional[str] = None
        self.description = description
        self._document_creator: Optional[str] = None
        self.document_creator = document_creator
        self._unique_identifier: Optional[str] = None
        self.unique_identifier = unique_identifier
        self._measurement_equipment: Optional[str] = None
        self.measurement_equipment = measurement_equipment
        self._laboratory: Optional[str] = None
        self.laboratory = laboratory
        self._report_number: Optional[str] = None
        self.report_number = report_number
        self._report_date: Optional[str] = None
        self.report_date = report_date
        self._document_creation_date: Optional[str] = None
        self.document_creation_date = document_creation_date
        self._comments: Optional[str] = None
        self.comments = comments

    @property
    def mapping(self) -> Structure:
        """
        Getter property for the mapping structure.

        Returns
        -------
        :class:`colour.utilities.Structure`
            Mapping structure.
        """

        return self._mapping

    @property
    def manufacturer(self) -> Optional[str]:
        """
        Getter and setter property for the manufacturer.

        Parameters
        ----------
        value
            Value to set the manufacturer with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Manufacturer.
        """

        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, value: Optional[str]):
        """Setter for the **self.manufacturer** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"manufacturer" property: "{value}" type is not "str"!',
            )

        self._manufacturer = value

    @property
    def catalog_number(self) -> Optional[str]:
        """
        Getter and setter property for the catalog number.

        Parameters
        ----------
        value
            Value to set the catalog number with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Catalog number.
        """

        return self._catalog_number

    @catalog_number.setter
    def catalog_number(self, value: Optional[str]):
        """Setter for the **self.catalog_number** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"catalog_number" property: "{value}" type is not "str"!',
            )

        self._catalog_number = value

    @property
    def description(self) -> Optional[str]:
        """
        Getter and setter property for the description.

        Parameters
        ----------
        value
            Value to set the description with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Description.
        """

        return self._description

    @description.setter
    def description(self, value: Optional[str]):
        """Setter for the **self.description** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"description" property: "{value}" type is not "str"!',
            )

        self._description = value

    @property
    def document_creator(self) -> Optional[str]:
        """
        Getter and setter property for the document creator.

        Parameters
        ----------
        value
            Value to set the document creator with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Document creator.
        """

        return self._document_creator

    @document_creator.setter
    def document_creator(self, value: Optional[str]):
        """Setter for the **self.document_creator** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"document_creator" property: "{value}" type is not "str"!',
            )

        self._document_creator = value

    @property
    def unique_identifier(self) -> Optional[str]:
        """
        Getter and setter property for the unique identifier.

        Parameters
        ----------
        value
            Value to set the unique identifier with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Unique identifier.
        """

        return self._unique_identifier

    @unique_identifier.setter
    def unique_identifier(self, value: Optional[str]):
        """Setter for the **self.unique_identifier** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"unique_identifier" property: "{value}" type is not "str"!',
            )

        self._unique_identifier = value

    @property
    def measurement_equipment(self) -> Optional[str]:
        """
        Getter and setter property for the measurement equipment.

        Parameters
        ----------
        value
            Value to set the measurement equipment with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Measurement equipment.
        """

        return self._measurement_equipment

    @measurement_equipment.setter
    def measurement_equipment(self, value: Optional[str]):
        """Setter for the **self.measurement_equipment** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"measurement_equipment" property: "{value}" type is not "str"!',
            )

        self._measurement_equipment = value

    @property
    def laboratory(self) -> Optional[str]:
        """
        Getter and setter property for the laboratory.

        Parameters
        ----------
        value
            Value to set the laboratory with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Laboratory.
        """

        return self._laboratory

    @laboratory.setter
    def laboratory(self, value: Optional[str]):
        """Setter for the **self.measurement_equipment** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"laboratory" property: "{value}" type is not "str"!',
            )

        self._laboratory = value

    @property
    def report_number(self) -> Optional[str]:
        """
        Getter and setter property for the report number.

        Parameters
        ----------
        value
            Value to set the report number with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Report number.
        """

        return self._report_number

    @report_number.setter
    def report_number(self, value: Optional[str]):
        """Setter for the **self.report_number** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"report_number" property: "{value}" type is not "str"!',
            )

        self._report_number = value

    @property
    def report_date(self) -> Optional[str]:
        """
        Getter and setter property for the report date.

        Parameters
        ----------
        value
            Value to set the report date with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Report date.
        """

        return self._report_date

    @report_date.setter
    def report_date(self, value: Optional[str]):
        """Setter for the **self.report_date** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"report_date" property: "{value}" type is not "str"!',
            )

        self._report_date = value

    @property
    def document_creation_date(self) -> Optional[str]:
        """
        Getter and setter property for the document creation date.

        Parameters
        ----------
        value
            Value to set the document creation date with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Document creation date.
        """

        return self._document_creation_date

    @document_creation_date.setter
    def document_creation_date(self, value: Optional[str]):
        """Setter for the **self.document_creation_date** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"document_creation_date" property: "{value}" type is not "str"!',
            )

        self._document_creation_date = value

    @property
    def comments(self) -> Optional[str]:
        """
        Getter and setter property for the comments.

        Parameters
        ----------
        value
            Value to set the comments with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Comments.
        """

        return self._comments

    @comments.setter
    def comments(self, value: Optional[str]):
        """Setter for the **self.comments** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"comments" property: "{value}" type is not "str"!',
            )

        self._comments = value

    def __hash__(self) -> Integer:
        """
        Return the header hash.

        Returns
        -------
        :class:`numpy.integer`
            Object hash.
        """

        return hash(
            (
                self._manufacturer,
                self._catalog_number,
                self._description,
                self._document_creator,
                self._unique_identifier,
                self._measurement_equipment,
                self._laboratory,
                self._report_number,
                self._report_date,
                self._document_creation_date,
                self._comments,
            )
        )

    def __eq__(self, other: Any) -> Boolean:
        """
        Return whether the header is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the header.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the header.

        Examples
        --------
        >>> Header_IESTM2714('Foo') == Header_IESTM2714('Foo')
        True
        >>> Header_IESTM2714('Foo') == Header_IESTM2714('Bar')
        False
        """

        if isinstance(other, Header_IESTM2714):
            return all(
                [
                    self._manufacturer == other.manufacturer,
                    self._catalog_number == other.catalog_number,
                    self._description == other.description,
                    self._document_creator == other.document_creator,
                    self._unique_identifier == other.unique_identifier,
                    self._measurement_equipment == other.measurement_equipment,
                    self._laboratory == other.laboratory,
                    self._report_number == other.report_number,
                    self._report_date == other.report_date,
                    self._document_creation_date
                    == other.document_creation_date,
                    self._comments == other.comments,
                ]
            )
        return False

    def __ne__(self, other: Any) -> Boolean:
        """
        Return whether the header is not equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the header.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the header.

        Examples
        --------
        >>> Header_IESTM2714('Foo') != Header_IESTM2714('Foo')
        False
        >>> Header_IESTM2714('Foo') != Header_IESTM2714('Bar')
        True
        """

        return not (self == other)


class SpectralDistribution_IESTM2714(SpectralDistribution):
    """
    Define a *IES TM-27-14* spectral distribution.

    This class can read and write *IES TM-27-14* spectral data *XML* files.

    Parameters
    ----------
    path
        Spectral data *XML* file path.
    header
        *IES TM-27-14* spectral distribution header.
    spectral_quantity
        Quantity of measurement for each element of the spectral data.
    reflection_geometry
        Spectral reflectance factors geometric conditions.
    transmission_geometry
        Spectral transmittance factors geometric conditions.
    bandwidth_FWHM
        Spectroradiometer full-width half-maximum bandwidth in nanometers.
    bandwidth_corrected
        Specifies if bandwidth correction has been applied to the measured
        data.

    Other Parameters
    ----------------
    data
        Data to be stored in the spectral distribution.
    domain
        Values to initialise the
        :attr:`colour.SpectralDistribution.wavelength` property with.
        If both ``data`` and ``domain`` arguments are defined, the latter will
        be used to initialise the
        :attr:`colour.SpectralDistribution.wavelength` property.
    extrapolator
        Extrapolator class type to use as extrapolating function.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function.
    interpolator
        Interpolator class type to use as interpolating function.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function.
    name
        Spectral distribution name.
    strict_name
        Spectral distribution name for figures, default to
        :attr:`colour.SpectralDistribution.name` property value.

    Notes
    -----
    *Reflection Geometry*

    -   di:8: Diffuse / eight-degree, specular component included.
    -   de:8: Diffuse / eight-degree, specular component excluded.
    -   8:di: Eight-degree / diffuse, specular component included.
    -   8:de: Eight-degree / diffuse, specular component excluded.
    -   d:d: Diffuse / diffuse.
    -   d:0: Alternative diffuse.
    -   45a:0: Forty-five degree annular / normal.
    -   45c:0: Forty-five degree circumferential / normal.
    -   0:45a: Normal / forty-five degree annular.
    -   45x:0: Forty-five degree directional / normal.
    -   0:45x: Normal / forty-five degree directional.
    -   other: User-specified in comments.

    *Transmission Geometry*

    -   0:0: Normal / normal.
    -   di:0: Diffuse / normal, regular component included.
    -   de:0: Diffuse / normal, regular component excluded.
    -   0:di: Normal / diffuse, regular component included.
    -   0:de: Normal / diffuse, regular component excluded.
    -   d:d: Diffuse / diffuse.
    -   other: User-specified in comments.

    Attributes
    ----------
    -   :attr:`~colour.SpectralDistribution_IESTM2714.mapping`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.path`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.header`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.spectral_quantity`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.reflection_geometry`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.transmission_geometry`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.bandwidth_FWHM`
    -   :attr:`~colour.SpectralDistribution_IESTM2714.bandwidth_corrected`

    Methods
    -------
    -   :meth:`~colour.SpectralDistribution_IESTM2714.__init__`
    -   :meth:`~colour.SpectralDistribution_IESTM2714.read`
    -   :meth:`~colour.SpectralDistribution_IESTM2714.write`

    References
    ----------
    :cite:`IESComputerCommittee2014a`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> directory = join(dirname(__file__), 'tests', 'resources')
    >>> sd = SpectralDistribution_IESTM2714(
    ...     join(directory, 'Fluorescent.spdx')).read()
    >>> sd.name  # doctest: +SKIP
    'Unknown - N/A - Rare earth fluorescent lamp'
    >>> sd.header.comments
    'Ambient temperature 25 degrees C.'
    >>> sd[501.7]  # doctest: +ELLIPSIS
    0.0950000...
    """

    def __init__(
        self,
        path: Optional[str] = None,
        header: Optional[Header_IESTM2714] = None,
        spectral_quantity: Optional[
            Literal[
                "absorptance",
                "exitance",
                "flux",
                "intensity",
                "irradiance",
                "radiance",
                "reflectance",
                "relative",
                "transmittance",
                "R-Factor",
                "T-Factor",
                "other",
            ]
        ] = None,
        reflection_geometry: Optional[
            Literal[
                "di:8",
                "de:8",
                "8:di",
                "8:de",
                "d:d",
                "d:0",
                "45a:0",
                "45c:0",
                "0:45a",
                "45x:0",
                "0:45x",
                "other",
            ]
        ] = None,
        transmission_geometry: Optional[
            Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"]
        ] = None,
        bandwidth_FWHM: Optional[Floating] = None,
        bandwidth_corrected: Optional[Boolean] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self._mapping: Structure = Structure(
            **{
                "element": "SpectralDistribution",
                "elements": (
                    Element_Specification_IESTM2714(
                        "SpectralQuantity", "spectral_quantity", required=True
                    ),
                    Element_Specification_IESTM2714(
                        "ReflectionGeometry", "reflection_geometry"
                    ),
                    Element_Specification_IESTM2714(
                        "TransmissionGeometry", "transmission_geometry"
                    ),
                    Element_Specification_IESTM2714(
                        "BandwidthFWHM",
                        "bandwidth_FWHM",
                        read_conversion=as_float_scalar,
                    ),
                    Element_Specification_IESTM2714(
                        "BandwidthCorrected",
                        "bandwidth_corrected",
                        read_conversion=(
                            lambda x: True if x == "true" else False
                        ),
                        write_conversion=(
                            lambda x: "true" if x is True else "False"
                        ),
                    ),
                ),
                "data": Element_Specification_IESTM2714(
                    "SpectralData", "wavelength", required=True
                ),
            }
        )

        self._path: Optional[str] = None
        self.path = path
        self._header: Header_IESTM2714 = Header_IESTM2714()
        self.header = optional(header, self._header)
        self._spectral_quantity: Optional[
            Literal[
                "absorptance",
                "exitance",
                "flux",
                "intensity",
                "irradiance",
                "radiance",
                "reflectance",
                "relative",
                "transmittance",
                "R-Factor",
                "T-Factor",
                "other",
            ]
        ] = None
        self.spectral_quantity = spectral_quantity
        self._reflection_geometry: Optional[
            Literal[
                "di:8",
                "de:8",
                "8:di",
                "8:de",
                "d:d",
                "d:0",
                "45a:0",
                "45c:0",
                "0:45a",
                "45x:0",
                "0:45x",
                "other",
            ]
        ] = None
        self.reflection_geometry = reflection_geometry
        self._transmission_geometry: Optional[
            Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"]
        ] = None
        self.transmission_geometry = transmission_geometry
        self._bandwidth_FWHM: Optional[Floating] = None
        self.bandwidth_FWHM = bandwidth_FWHM
        self._bandwidth_corrected: Optional[Boolean] = None
        self.bandwidth_corrected = bandwidth_corrected

    @property
    def mapping(self) -> Structure:
        """
        Getter property for the mapping structure.

        Returns
        -------
        :class:`colour.utilities.Structure`
            Mapping structure.
        """

        return self._mapping

    @property
    def path(self) -> Optional[str]:
        """
        Getter and setter property for the path.

        Parameters
        ----------
        value
            Value to set the path with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Path.
        """

        return self._path

    @path.setter
    def path(self, value: Optional[str]):
        """Setter for the **self.path** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"path" property: "{value}" type is not "str"!',
            )

        self._path = value

    @property
    def header(self) -> Header_IESTM2714:
        """
        Getter and setter property for the header.

        Parameters
        ----------
        value
            Value to set the header with.

        Returns
        -------
        :class:`colour.io.tm2714.Header_IESTM2714`
            Header.
        """

        return self._header

    @header.setter
    def header(self, value: Header_IESTM2714):
        """Setter for the **self.header** property."""

        attest(
            isinstance(value, Header_IESTM2714),
            f'"header" property: "{value}" type is not "Header_IESTM2714"!',
        )

        self._header = value

    @property
    def spectral_quantity(
        self,
    ) -> Optional[
        Literal[
            "absorptance",
            "exitance",
            "flux",
            "intensity",
            "irradiance",
            "radiance",
            "reflectance",
            "relative",
            "transmittance",
            "R-Factor",
            "T-Factor",
            "other",
        ]
    ]:
        """
        Getter and setter property for the spectral quantity.

        Parameters
        ----------
        value
            Value to set the spectral quantity with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Spectral quantity.
        """

        return self._spectral_quantity

    @spectral_quantity.setter
    def spectral_quantity(
        self,
        value: Optional[
            Literal[
                "absorptance",
                "exitance",
                "flux",
                "intensity",
                "irradiance",
                "radiance",
                "reflectance",
                "relative",
                "transmittance",
                "R-Factor",
                "T-Factor",
                "other",
            ]
        ],
    ):
        """Setter for the **self.spectral_quantity** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"spectral_quantity" property: "{value}" type is not "str"!',
            )

        self._spectral_quantity = value

    @property
    def reflection_geometry(
        self,
    ) -> Optional[
        Literal[
            "di:8",
            "de:8",
            "8:di",
            "8:de",
            "d:d",
            "d:0",
            "45a:0",
            "45c:0",
            "0:45a",
            "45x:0",
            "0:45x",
            "other",
        ]
    ]:
        """
        Getter and setter property for the reflection geometry.

        Parameters
        ----------
        value
            Value to set the reflection geometry with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Reflection geometry.
        """

        return self._reflection_geometry

    @reflection_geometry.setter
    def reflection_geometry(
        self,
        value: Optional[
            Literal[
                "di:8",
                "de:8",
                "8:di",
                "8:de",
                "d:d",
                "d:0",
                "45a:0",
                "45c:0",
                "0:45a",
                "45x:0",
                "0:45x",
                "other",
            ]
        ],
    ):
        """Setter for the **self.reflection_geometry** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"reflection_geometry" property: "{value}" type is not "str"!',
            )

        self._reflection_geometry = value

    @property
    def transmission_geometry(
        self,
    ) -> Optional[
        Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"]
    ]:
        """
        Getter and setter property for the transmission geometry.

        Parameters
        ----------
        value
            Value to set the transmission geometry with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Transmission geometry.
        """

        return self._transmission_geometry

    @transmission_geometry.setter
    def transmission_geometry(
        self,
        value: Optional[
            Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"]
        ],
    ):
        """Setter for the **self.transmission_geometry** property."""

        if value is not None:
            attest(
                is_string(value),
                f'"transmission_geometry" property: "{value}" type is not "str"!',
            )

        self._transmission_geometry = value

    @property
    def bandwidth_FWHM(self) -> Optional[Floating]:
        """
        Getter and setter property for the full-width half-maximum bandwidth.

        Parameters
        ----------
        value
            Value to set the full-width half-maximum bandwidth with.

        Returns
        -------
        :py:data:`None` or :class:`numpy.floating`
            Full-width half-maximum bandwidth.
        """

        return self._bandwidth_FWHM

    @bandwidth_FWHM.setter
    def bandwidth_FWHM(self, value: Optional[Floating]):
        """Setter for the **self.bandwidth_FWHM** property."""

        if value is not None:
            attest(
                is_numeric(value),
                f'"bandwidth_FWHM" property: "{value}" is not a "number"!',
            )

            value = as_float_scalar(value)

        self._bandwidth_FWHM = value

    @property
    def bandwidth_corrected(self) -> Optional[Boolean]:
        """
        Getter and setter property for whether bandwidth correction has been
        applied to the measured data.

        Parameters
        ----------
        value
            Whether bandwidth correction has been applied to the measured data.

        Returns
        -------
        :py:data:`None` or :class:`bool`
            Whether bandwidth correction has been applied to the measured data.
        """

        return self._bandwidth_corrected

    @bandwidth_corrected.setter
    def bandwidth_corrected(self, value: Optional[Boolean]):
        """Setter for the **self.bandwidth_corrected** property."""

        if value is not None:
            attest(
                isinstance(value, bool),
                f'"bandwidth_corrected" property: "{value}" type is not "bool"!',
            )

        self._bandwidth_corrected = value

    def read(self) -> SpectralDistribution_IESTM2714:
        """
        Read and parses the spectral data *XML* file path.

        Returns
        -------
        :class:`colour.SpectralDistribution_IESTM2714`
            *IES TM-27-14* spectral distribution.

        Raises
        ------
        ValueError
            If the *IES TM-27-14* spectral distribution path is undefined.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_IESTM2714(
        ...     join(directory, 'Fluorescent.spdx')).read()
        >>> sd.name  # doctest: +SKIP
        'Unknown - N/A - Rare earth fluorescent lamp'
        >>> sd.header.comments
        'Ambient temperature 25 degrees C.'
        >>> sd[400]  # doctest: +ELLIPSIS
        0.0340000...
        """

        if self._path is not None:
            formatter = "./{{{0}}}{1}/{{{0}}}{2}"

            tree = ElementTree.parse(self._path)  # nosec
            root = tree.getroot()

            match = re.match("{(.*)}", root.tag)
            if match:
                namespace = match.group(1)
            else:
                raise ValueError(
                    'The "IES TM-27-14" spectral distribution namespace '
                    "was not found!"
                )

            self.name = os.path.splitext(os.path.basename(self._path))[0]

            iterator = root.iter

            for header_element in (self.header, self):
                mapping = header_element.mapping  # type: ignore[attr-defined]
                for specification in mapping.elements:
                    element = root.find(
                        formatter.format(
                            namespace, mapping.element, specification.element
                        )
                    )
                    if element is not None:
                        setattr(
                            header_element,
                            specification.attribute,
                            specification.read_conversion(element.text),
                        )

            # Reading spectral data.
            wavelengths = []
            values = []
            for spectral_data in iterator(
                f"{{{namespace}}}{self.mapping.data.element}"
            ):
                wavelengths.append(
                    spectral_data.attrib[self.mapping.data.attribute]
                )
                values.append(spectral_data.text)

            components = [
                component
                for component in (
                    self.header.manufacturer,
                    self.header.catalog_number,
                    self.header.description,
                )
                if component is not None
            ]
            self.name = (
                "Undefined" if len(components) == 0 else " - ".join(components)
            )

            self.wavelengths = as_float_array(wavelengths)
            self.values = as_float_array(cast(ArrayLike, values))

            return self
        else:
            raise ValueError(
                'The "IES TM-27-14" spectral distribution path is undefined!'
            )

    def write(self) -> Boolean:
        """
        Write the spectral distribution spectral data to *XML* file path.

        Returns
        -------
        :class:`bool`
            Definition success.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from shutil import rmtree
        >>> from tempfile import mkdtemp
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_IESTM2714(
        ...     join(directory, 'Fluorescent.spdx')).read()
        >>> temporary_directory = mkdtemp()
        >>> sd.path = join(temporary_directory, 'Fluorescent.spdx')
        >>> sd.write()
        True
        >>> rmtree(temporary_directory)
        """

        if self._path is not None:
            root = ElementTree.Element("IESTM2714")
            root.attrib = {
                "xmlns": NAMESPACE_IESTM2714,
                "version": VERSION_IESTM2714,
            }

            spectral_distribution = ElementTree.Element("")
            for header_element in (self.header, self):
                mapping = header_element.mapping  # type: ignore[attr-defined]
                element = ElementTree.SubElement(root, mapping.element)
                for specification in mapping.elements:
                    element_child = ElementTree.SubElement(
                        element, specification.element
                    )
                    value = getattr(header_element, specification.attribute)
                    element_child.text = specification.write_conversion(value)

                if header_element is self:
                    spectral_distribution = element

            # Writing spectral data.
            for (wavelength, value) in tstack([self.wavelengths, self.values]):
                element_child = ElementTree.SubElement(
                    spectral_distribution, mapping.data.element
                )
                element_child.text = mapping.data.write_conversion(value)
                element_child.attrib = {
                    mapping.data.attribute: mapping.data.write_conversion(
                        wavelength
                    )
                }

            xml = minidom.parseString(
                ElementTree.tostring(root)
            ).toprettyxml()  # nosec

            with open(self._path, "w") as file:
                file.write(xml)

            return True
        else:
            raise ValueError(
                'The "IES TM-27-14" spectral distribution path is undefined!'
            )
