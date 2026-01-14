"""
IES TM-27-14 Data Input / Output
================================

Define classes and utilities for handling *IES TM-27-14* spectral data *XML*
files.

This module provides the :class:`colour.SpectralDistribution_IESTM2714` class
for reading and writing spectral distribution data according to the *IES
TM-27-14* standard format for electronic transfer of spectral data.

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
import typing
from dataclasses import dataclass, field
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from colour.colorimetry import SpectralDistribution

if typing.TYPE_CHECKING:
    from colour.hints import Any, Callable, Literal, PathLike

from colour.utilities import (
    Structure,
    as_float_array,
    as_float_scalar,
    attest,
    is_numeric,
    multiline_repr,
    multiline_str,
    optional,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
    Define *IES TM-27-14* spectral data *XML* file element specification.

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
    required: bool = field(default_factory=lambda: False)
    read_conversion: Callable = field(
        default_factory=lambda: lambda x: None if x == "None" else str(x)
    )
    write_conversion: Callable = field(default_factory=lambda: str)


class Header_IESTM2714:
    """
    Define the header object for an *IES TM-27-14* spectral distribution.

    Parameters
    ----------
    manufacturer
        Manufacturer of the device under test.
    catalog_number
        Manufacturer's product catalog number.
    description
        Description of the spectral data in the spectral data *XML* file.
    document_creator
        Creator of the spectral data *XML* file, which may be a test lab,
        a research group, a standard body, a company or an individual.
    unique_identifier
        Unique identifier to the product under test or the spectral data
        in the document.
    measurement_equipment
        Description of the equipment used to measure the spectral data.
    laboratory
        Testing laboratory name that performed the spectral data
        measurements.
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
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__str__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__repr__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__hash__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__eq__`
    -   :meth:`~colour.io.ies_tm2714.Header_IESTM2714.__ne__`

    Examples
    --------
    >>> Header_IESTM2714("colour-science")  # doctest: +ELLIPSIS
    Header_IESTM2714('colour-science',
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None)
    >>> Header_IESTM2714("colour-science").manufacturer  # doctest: +SKIP
    'colour-science'
    """

    def __init__(
        self,
        manufacturer: str | None = None,
        catalog_number: str | None = None,
        description: str | None = None,
        document_creator: str | None = None,
        unique_identifier: str | None = None,
        measurement_equipment: str | None = None,
        laboratory: str | None = None,
        report_number: str | None = None,
        report_date: str | None = None,
        document_creation_date: str | None = None,
        comments: str | None = None,
    ) -> None:
        self._mapping: Structure = Structure(
            element="Header",
            elements=(
                Element_Specification_IESTM2714("Manufacturer", "manufacturer"),
                Element_Specification_IESTM2714("CatalogNumber", "catalog_number"),
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
                Element_Specification_IESTM2714("Laboratory", "laboratory"),
                Element_Specification_IESTM2714("ReportNumber", "report_number"),
                Element_Specification_IESTM2714("ReportDate", "report_date"),
                Element_Specification_IESTM2714(
                    "DocumentCreationDate",
                    "document_creation_date",
                    required=True,
                ),
                Element_Specification_IESTM2714("Comments", "comments", False),
            ),
        )

        self._manufacturer: str | None = None
        self.manufacturer = manufacturer
        self._catalog_number: str | None = None
        self.catalog_number = catalog_number
        self._description: str | None = None
        self.description = description
        self._document_creator: str | None = None
        self.document_creator = document_creator
        self._unique_identifier: str | None = None
        self.unique_identifier = unique_identifier
        self._measurement_equipment: str | None = None
        self.measurement_equipment = measurement_equipment
        self._laboratory: str | None = None
        self.laboratory = laboratory
        self._report_number: str | None = None
        self.report_number = report_number
        self._report_date: str | None = None
        self.report_date = report_date
        self._document_creation_date: str | None = None
        self.document_creation_date = document_creation_date
        self._comments: str | None = None
        self.comments = comments

    @property
    def mapping(self) -> Structure:
        """
        Getter for the mapping structure.

        Returns
        -------
        :class:`colour.utilities.Structure`
            Mapping structure.
        """

        return self._mapping

    @property
    def manufacturer(self) -> str | None:
        """
        Getter and setter for the manufacturer name.

        Parameters
        ----------
        value
            Value to set the manufacturer with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Manufacturer name.
        """

        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, value: str | None) -> None:
        """Setter for the **self.manufacturer** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"manufacturer" property: "{value}" type is not "str"!',
            )

        self._manufacturer = value

    @property
    def catalog_number(self) -> str | None:
        """
        Getter and setter for the catalog number.

        Parameters
        ----------
        value
            Value to set the catalog number with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Catalog number.
        """

        return self._catalog_number

    @catalog_number.setter
    def catalog_number(self, value: str | None) -> None:
        """Setter for the **self.catalog_number** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"catalog_number" property: "{value}" type is not "str"!',
            )

        self._catalog_number = value

    @property
    def description(self) -> str | None:
        """
        Getter and setter for the description.

        Parameters
        ----------
        value
            Value to set the description with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Description.
        """

        return self._description

    @description.setter
    def description(self, value: str | None) -> None:
        """Setter for the **self.description** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"description" property: "{value}" type is not "str"!',
            )

        self._description = value

    @property
    def document_creator(self) -> str | None:
        """
        Getter and setter for the document creator.

        Parameters
        ----------
        value
            Value to set the document creator with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Document creator.
        """

        return self._document_creator

    @document_creator.setter
    def document_creator(self, value: str | None) -> None:
        """Setter for the **self.document_creator** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"document_creator" property: "{value}" type is not "str"!',
            )

        self._document_creator = value

    @property
    def unique_identifier(self) -> str | None:
        """
        Getter and setter for the unique identifier.

        Parameters
        ----------
        value
            Value to set the unique identifier with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Unique identifier.
        """

        return self._unique_identifier

    @unique_identifier.setter
    def unique_identifier(self, value: str | None) -> None:
        """Setter for the **self.unique_identifier** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"unique_identifier" property: "{value}" type is not "str"!',
            )

        self._unique_identifier = value

    @property
    def measurement_equipment(self) -> str | None:
        """
        Getter and setter for the measurement equipment.

        Parameters
        ----------
        value
            Value to set the measurement equipment with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Measurement equipment.
        """

        return self._measurement_equipment

    @measurement_equipment.setter
    def measurement_equipment(self, value: str | None) -> None:
        """Setter for the **self.measurement_equipment** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"measurement_equipment" property: "{value}" type is not "str"!',
            )

        self._measurement_equipment = value

    @property
    def laboratory(self) -> str | None:
        """
        Getter and setter for the laboratory information.

        Parameters
        ----------
        value
            Value to set the laboratory with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Laboratory.
        """

        return self._laboratory

    @laboratory.setter
    def laboratory(self, value: str | None) -> None:
        """Setter for the **self.laboratory** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"laboratory" property: "{value}" type is not "str"!',
            )

        self._laboratory = value

    @property
    def report_number(self) -> str | None:
        """
        Getter and setter for the report number.

        Parameters
        ----------
        value
            Value to set the report number with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Report number.
        """

        return self._report_number

    @report_number.setter
    def report_number(self, value: str | None) -> None:
        """Setter for the **self.report_number** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"report_number" property: "{value}" type is not "str"!',
            )

        self._report_number = value

    @property
    def report_date(self) -> str | None:
        """
        Getter and setter for the report date.

        Parameters
        ----------
        value
            Value to set the report date with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Report date.
        """

        return self._report_date

    @report_date.setter
    def report_date(self, value: str | None) -> None:
        """Setter for the **self.report_date** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"report_date" property: "{value}" type is not "str"!',
            )

        self._report_date = value

    @property
    def document_creation_date(self) -> str | None:
        """
        Getter and setter for the document creation date.

        Parameters
        ----------
        value
            Value to set the document creation date with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Document creation date.
        """

        return self._document_creation_date

    @document_creation_date.setter
    def document_creation_date(self, value: str | None) -> None:
        """Setter for the **self.document_creation_date** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"document_creation_date" property: "{value}" type is not "str"!',
            )

        self._document_creation_date = value

    @property
    def comments(self) -> str | None:
        """
        Getter and setter for the comments associated with the object.

        Parameters
        ----------
        value
            Value to set the comments with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Comments.
        """

        return self._comments

    @comments.setter
    def comments(self, value: str | None) -> None:
        """Setter for the **self.comments** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"comments" property: "{value}" type is not "str"!',
            )

        self._comments = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *IES TM-27-14* header.

        Returns
        -------
        :class:`str`
            Formatted string representation displaying the header attributes.

        Examples
        --------
        >>> print(Header_IESTM2714("colour-science"))
        Manufacturer           : colour-science
        Catalog Number         : None
        Description            : None
        Document Creator       : None
        Unique Identifier      : None
        Measurement Equipment  : None
        Laboratory             : None
        Report Number          : None
        Report Date            : None
        Document Creation Date : None
        Comments               : None
        """

        return multiline_str(
            self,
            [
                {"name": "_manufacturer", "label": "Manufacturer"},
                {"name": "_catalog_number", "label": "Catalog Number"},
                {"name": "_description", "label": "Description"},
                {"name": "_document_creator", "label": "Document Creator"},
                {"name": "_unique_identifier", "label": "Unique Identifier"},
                {
                    "name": "_measurement_equipment",
                    "label": "Measurement Equipment",
                },
                {"name": "_laboratory", "label": "Laboratory"},
                {"name": "_report_number", "label": "Report Number"},
                {"name": "_report_date", "label": "Report Date"},
                {
                    "name": "_document_creation_date",
                    "label": "Document Creation Date",
                },
                {"name": "_comments", "label": "Comments"},
            ],
        )

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the header.

        Returns
        -------
        :class:`str`
            Evaluable string representation.

        Examples
        --------
        >>> Header_IESTM2714("colour-science")
        Header_IESTM2714('colour-science',
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None)
        """

        return multiline_repr(
            self,
            [
                {"name": "_manufacturer"},
                {"name": "_catalog_number"},
                {"name": "_description"},
                {"name": "_document_creator"},
                {"name": "_unique_identifier"},
                {"name": "_measurement_equipment"},
                {"name": "_laboratory"},
                {"name": "_report_number"},
                {"name": "_report_date"},
                {"name": "_document_creation_date"},
                {"name": "_comments"},
            ],
        )

    def __hash__(self) -> int:
        """
        Return the header hash.

        Returns
        -------
        :class:`int`
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

    def __eq__(self, other: object) -> bool:
        """
        Determine whether the header is equal to the specified other object.

        Compare all header attributes to determine equality between this header
        and the specified object. Two headers are considered equal when all
        their attributes match exactly.

        Parameters
        ----------
        other
            Object to test whether it is equal to the header.

        Returns
        -------
        :class:`bool`
            Whether the specified object is equal to the header.

        Examples
        --------
        >>> Header_IESTM2714("Foo") == Header_IESTM2714("Foo")
        True
        >>> Header_IESTM2714("Foo") == Header_IESTM2714("Bar")
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
                    self._document_creation_date == other.document_creation_date,
                    self._comments == other.comments,
                ]
            )

        return False

    def __ne__(self, other: object) -> bool:
        """
        Determine whether the header is not equal to the specified other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the header.

        Returns
        -------
        :class:`bool`
            Whether specified object is not equal to the header.

        Examples
        --------
        >>> Header_IESTM2714("Foo") != Header_IESTM2714("Foo")
        False
        >>> Header_IESTM2714("Foo") != Header_IESTM2714("Bar")
        True
        """

        return not (self == other)


class SpectralDistribution_IESTM2714(SpectralDistribution):
    """
    Define an *IES TM-27-14* spectral distribution for electronic transfer of
    spectral data.

    This class provides functionality to read and write spectral distribution
    data using the *IES TM-27-14* standard format. The standard defines
    an *XML* schema for exchanging spectral measurement data between systems,
    ensuring consistent representation of wavelength-dependent measurements
    across different applications and instruments.

    Parameters
    ----------
    path
        Spectral data *XML* file path.
    header
        *IES TM-27-14* spectral distribution header containing metadata.
    spectral_quantity
        Quantity of measurement for each element of the spectral data (e.g.,
        "reflectance", "transmittance", "radiance").
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
    display_name
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
    -   :meth:`~colour.SpectralDistribution_IESTM2714.__str__`
    -   :meth:`~colour.SpectralDistribution_IESTM2714.__repr__`
    -   :meth:`~colour.SpectralDistribution_IESTM2714.read`
    -   :meth:`~colour.SpectralDistribution_IESTM2714.write`

    References
    ----------
    :cite:`IESComputerCommittee2014a`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> directory = join(dirname(__file__), "tests", "resources")
    >>> sd = SpectralDistribution_IESTM2714(join(directory, "Fluorescent.spdx"))
    >>> sd.name  # doctest: +SKIP
    'Unknown - N/A - Rare earth fluorescent lamp'
    >>> sd.header.comments
    'Ambient temperature 25 degrees C.'
    >>> sd[501.7]  # doctest: +ELLIPSIS
    np.float64(0.09...)
    """

    def __init__(
        self,
        path: str | PathLike | None = None,
        header: Header_IESTM2714 | None = None,
        spectral_quantity: (
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
            | None
        ) = None,
        reflection_geometry: (
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
            | None
        ) = None,
        transmission_geometry: (
            Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"] | None
        ) = None,
        bandwidth_FWHM: float | None = None,
        bandwidth_corrected: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._mapping: Structure = Structure(
            element="SpectralDistribution",
            elements=(
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
                    read_conversion=(
                        lambda x: (None if x == "None" else as_float_scalar(x))
                    ),
                ),
                Element_Specification_IESTM2714(
                    "BandwidthCorrected",
                    "bandwidth_corrected",
                    read_conversion=(lambda x: bool(x == "true")),
                    write_conversion=(lambda x: "true" if x is True else "false"),
                ),
            ),
            data=Element_Specification_IESTM2714(
                "SpectralData", "wavelength", required=True
            ),
        )

        self._path: str | None = None
        self.path = path
        self._header: Header_IESTM2714 = Header_IESTM2714()
        self.header = optional(header, self._header)
        self._spectral_quantity: (
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
            | None
        ) = None
        self.spectral_quantity = spectral_quantity
        self._reflection_geometry: (
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
            | None
        ) = None
        self.reflection_geometry = reflection_geometry
        self._transmission_geometry: (
            Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"] | None
        ) = None
        self.transmission_geometry = transmission_geometry
        self._bandwidth_FWHM: float | None = None
        self.bandwidth_FWHM = bandwidth_FWHM
        self._bandwidth_corrected: bool | None = None
        self.bandwidth_corrected = bandwidth_corrected

        if self.path is not None and os.path.exists(
            self.path  # pyright: ignore
        ):
            self.read()

    @property
    def mapping(self) -> Structure:
        """
        Getter for the structure containing file mappings.

        Returns
        -------
        :class:`colour.utilities.Structure`
            Mapping structure.
        """

        return self._mapping

    @property
    def path(self) -> str | None:
        """
        Getter and setter for the resource path.

        Parameters
        ----------
        value
            Value to set the path with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Path to the resource.
        """

        return self._path

    @path.setter
    def path(self, value: str | PathLike | None) -> None:
        """Setter for the **self.path** property."""

        if value is not None:
            attest(
                isinstance(value, (str, Path)),
                f'"path" property: "{value}" type is not "str" or "Path"!',
            )

            value = str(value)

        self._path = value

    @property
    def header(self) -> Header_IESTM2714:
        """
        Getter and setter for the *IES TM-27-14* spectral distribution header.

        Parameters
        ----------
        value
            Value to set the header with.

        Returns
        -------
        :class:`colour.io.tm2714.Header_IESTM2714`
            Header object containing spectral distribution metadata.
        """

        return self._header

    @header.setter
    def header(self, value: Header_IESTM2714) -> None:
        """Setter for the **self.header** property."""

        attest(
            isinstance(value, Header_IESTM2714),
            f'"header" property: "{value}" type is not "Header_IESTM2714"!',
        )

        self._header = value

    @property
    def spectral_quantity(
        self,
    ) -> (
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
        | None
    ):
        """
        Getter and setter for the spectral quantity.

        Parameters
        ----------
        value
            Value to set the spectral quantity with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Spectral quantity.
        """

        return self._spectral_quantity

    @spectral_quantity.setter
    def spectral_quantity(
        self,
        value: (
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
            | None
        ),
    ) -> None:
        """Setter for the **self.spectral_quantity** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"spectral_quantity" property: "{value}" type is not "str"!',
            )

        self._spectral_quantity = value

    @property
    def reflection_geometry(
        self,
    ) -> (
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
        | None
    ):
        """
        Getter and setter for the reflection geometry.

        Parameters
        ----------
        value
            Value to set the reflection geometry with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Reflection geometry.
        """

        return self._reflection_geometry

    @reflection_geometry.setter
    def reflection_geometry(
        self,
        value: (
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
            | None
        ),
    ) -> None:
        """Setter for the **self.reflection_geometry** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"reflection_geometry" property: "{value}" type is not "str"!',
            )

        self._reflection_geometry = value

    @property
    def transmission_geometry(
        self,
    ) -> Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"] | None:
        """
        Getter and setter for the transmission geometry.

        Parameters
        ----------
        value
            Value to set the transmission geometry with.

        Returns
        -------
        :class:`str` or :py:data:`None`
            Transmission geometry.
        """

        return self._transmission_geometry

    @transmission_geometry.setter
    def transmission_geometry(
        self,
        value: (Literal["0:0", "di:0", "de:0", "0:di", "0:de", "d:d", "other"] | None),
    ) -> None:
        """Setter for the **self.transmission_geometry** property."""

        if value is not None:
            attest(
                isinstance(value, str),
                f'"transmission_geometry" property: "{value}" type is not "str"!',
            )

        self._transmission_geometry = value

    @property
    def bandwidth_FWHM(self) -> float | None:
        """
        Getter and setter for the full-width half-maximum (FWHM) bandwidth of
        the spectral measurement.

        Parameters
        ----------
        value
            Value to set the full-width half-maximum bandwidth with.

        Returns
        -------
        :class:`float` or :py:data:`None`
            Full-width half-maximum bandwidth.
        """

        return self._bandwidth_FWHM

    @bandwidth_FWHM.setter
    def bandwidth_FWHM(self, value: float | None) -> None:
        """Setter for the **self.bandwidth_FWHM** property."""

        if value is not None:
            attest(
                is_numeric(value),
                f'"bandwidth_FWHM" property: "{value}" is not a "number"!',
            )

            value = as_float_scalar(value)

        self._bandwidth_FWHM = value

    @property
    def bandwidth_corrected(self) -> bool | None:
        """
        Getter and setter for whether bandwidth correction has been applied to
        the measured data.

        Parameters
        ----------
        value
            Whether bandwidth correction has been applied to the measured
            data.

        Returns
        -------
        :class:`bool` or :py:data:`None`
            Whether bandwidth correction has been applied to the measured
            data.
        """

        return self._bandwidth_corrected

    @bandwidth_corrected.setter
    def bandwidth_corrected(self, value: bool | None) -> None:
        """Setter for the **self.bandwidth_corrected** property."""

        if value is not None:
            attest(
                isinstance(value, bool),
                f'"bandwidth_corrected" property: "{value}" type is not "bool"!',
            )

        self._bandwidth_corrected = value

    def __str__(self) -> str:
        """
        Generate a formatted string representation of the *IES TM-27-14*
        spectral distribution.

        Returns
        -------
        :class:`str`
            Formatted string representation containing path, spectral
            metadata, header details, and spectral data values.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> print(SpectralDistribution_IESTM2714(join(directory, "Fluorescent.spdx")))
        ... # doctest: +ELLIPSIS
        IES TM-27-14 Spectral Distribution
        ==================================
        <BLANKLINE>
        Path                  : ...
        Spectral Quantity     : relative
        Reflection Geometry   : other
        Transmission Geometry : other
        Bandwidth (FWHM)      : 2.0
        Bandwidth Corrected   : True
        <BLANKLINE>
        Header
        ------
        <BLANKLINE>
        Manufacturer           : Unknown
        Catalog Number         : N/A
        Description            : Rare earth fluorescent lamp
        Document Creator       : byHeart Consultants
        Unique Identifier      : C3567553-C75B-4354-961E-35CEB9FEB42C
        Measurement Equipment  : None
        Laboratory             : N/A
        Report Number          : N/A
        Report Date            : N/A
        Document Creation Date : 2014-06-23
        Comments               : Ambient temperature 25 degrees C.
        <BLANKLINE>
        Spectral Data
        -------------
        <BLANKLINE>
        [[4.000e+02 3.400e-02]
         [4.031e+02 3.700e-02]
         [4.055e+02 6.900e-02]
         [4.075e+02 3.700e-02]
         [4.206e+02 4.200e-02]
         [4.310e+02 4.900e-02]
         [4.337e+02 6.000e-02]
         [4.370e+02 3.570e-01]
         [4.389e+02 6.000e-02]
         [4.600e+02 6.800e-02]
         [4.770e+02 7.500e-02]
         [4.810e+02 8.500e-02]
         [4.882e+02 2.040e-01]
         [4.926e+02 1.660e-01]
         [5.017e+02 9.500e-02]
         [5.076e+02 7.800e-02]
         [5.176e+02 7.100e-02]
         [5.299e+02 7.600e-02]
         [5.354e+02 9.900e-02]
         [5.399e+02 4.230e-01]
         [5.432e+02 8.020e-01]
         [5.444e+02 7.130e-01]
         [5.472e+02 9.990e-01]
         [5.487e+02 5.730e-01]
         [5.502e+02 3.400e-01]
         [5.538e+02 2.080e-01]
         [5.573e+02 1.390e-01]
         [5.637e+02 1.290e-01]
         [5.748e+02 1.310e-01]
         [5.780e+02 1.980e-01]
         [5.792e+02 1.900e-01]
         [5.804e+02 2.050e-01]
         [5.848e+02 2.440e-01]
         [5.859e+02 2.360e-01]
         [5.875e+02 2.560e-01]
         [5.903e+02 1.800e-01]
         [5.935e+02 2.180e-01]
         [5.955e+02 1.590e-01]
         [5.970e+02 1.470e-01]
         [5.994e+02 1.700e-01]
         [6.022e+02 1.340e-01]
         [6.046e+02 1.210e-01]
         [6.074e+02 1.400e-01]
         [6.094e+02 2.290e-01]
         [6.102e+02 4.650e-01]
         [6.120e+02 9.520e-01]
         [6.146e+02 4.770e-01]
         [6.169e+02 2.080e-01]
         [6.185e+02 1.350e-01]
         [6.221e+02 1.500e-01]
         [6.256e+02 1.550e-01]
         [6.284e+02 1.340e-01]
         [6.312e+02 1.680e-01]
         [6.332e+02 8.700e-02]
         [6.356e+02 6.800e-02]
         [6.427e+02 5.800e-02]
         [6.487e+02 5.800e-02]
         [6.507e+02 7.400e-02]
         [6.526e+02 6.300e-02]
         [6.562e+02 5.300e-02]
         [6.570e+02 5.600e-02]
         [6.606e+02 4.900e-02]
         [6.626e+02 5.900e-02]
         [6.642e+02 4.800e-02]
         [6.860e+02 4.100e-02]
         [6.876e+02 4.800e-02]
         [6.892e+02 3.900e-02]
         [6.924e+02 3.800e-02]
         [6.935e+02 4.400e-02]
         [6.955e+02 3.400e-02]
         [7.023e+02 3.600e-02]
         [7.067e+02 4.200e-02]
         [7.071e+02 6.100e-02]
         [7.102e+02 6.100e-02]
         [7.110e+02 4.100e-02]
         [7.122e+02 5.200e-02]
         [7.142e+02 3.300e-02]
         [7.484e+02 3.400e-02]
         [7.579e+02 3.100e-02]
         [7.607e+02 3.900e-02]
         [7.639e+02 2.900e-02]
         [8.088e+02 2.900e-02]
         [8.107e+02 3.900e-02]
         [8.127e+02 3.000e-02]
         [8.501e+02 3.000e-02]]
        """

        try:
            str_parent = super().__str__()

            return multiline_str(
                self,
                [
                    {
                        "label": "IES TM-27-14 Spectral Distribution",
                        "header": True,
                    },
                    {"line_break": True},
                    {"name": "path", "label": "Path"},
                    {
                        "name": "spectral_quantity",
                        "label": "Spectral Quantity",
                    },
                    {
                        "name": "reflection_geometry",
                        "label": "Reflection Geometry",
                    },
                    {
                        "name": "transmission_geometry",
                        "label": "Transmission Geometry",
                    },
                    {"name": "bandwidth_FWHM", "label": "Bandwidth (FWHM)"},
                    {
                        "name": "bandwidth_corrected",
                        "label": "Bandwidth Corrected",
                    },
                    {"line_break": True},
                    {"label": "Header", "section": True},
                    {"line_break": True},
                    {"formatter": lambda x: str(self.header)},  # noqa: ARG005
                    {"line_break": True},
                    {"label": "Spectral Data", "section": True},
                    {"line_break": True},
                    {"formatter": lambda x: str_parent},  # noqa: ARG005
                ],
            )
        except TypeError:  # pragma: no cover
            return super().__str__()

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *IES TM-27-14*
        spectral distribution.

        Returns
        -------
        :class:`str`
            Evaluable string representation.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> SpectralDistribution_IESTM2714(
        ...     join(directory, "Fluorescent.spdx")
        ... )  # doctest: +SKIP
        SpectralDistribution_IESTM2714(...)  # doctest: +SKIP
        """

        try:
            return multiline_repr(
                self,
                [
                    {"name": "path"},
                    {"name": "header"},
                    {"name": "spectral_quantity"},
                    {"name": "reflection_geometry"},
                    {"name": "transmission_geometry"},
                    {"name": "bandwidth_FWHM"},
                    {"name": "bandwidth_corrected"},
                    {
                        "formatter": lambda x: repr(  # noqa: ARG005
                            tstack([self.domain, self.range])
                        ),
                    },
                    {
                        "name": "interpolator",
                        "formatter": lambda x: (  # noqa: ARG005
                            self.interpolator.__name__
                        ),
                    },
                    {"name": "interpolator_kwargs"},
                    {
                        "name": "extrapolator",
                        "formatter": lambda x: (  # noqa: ARG005
                            self.extrapolator.__name__
                        ),
                    },
                    {"name": "extrapolator_kwargs"},
                ],
            )
        except TypeError:  # pragma: no cover
            return super().__repr__()

    def read(self) -> SpectralDistribution_IESTM2714:
        """
        Read and parse the spectral data from the specified *XML* file path.

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
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_IESTM2714(join(directory, "Fluorescent.spdx"))
        >>> sd.name  # doctest: +SKIP
        'Unknown - N/A - Rare earth fluorescent lamp'
        >>> sd.header.comments
        'Ambient temperature 25 degrees C.'
        >>> sd[400]  # doctest: +ELLIPSIS
        np.float64(0.03...)
        """

        if self._path is not None:
            formatter = "./{{{0}}}{1}/{{{0}}}{2}"

            tree = ET.parse(self._path)  # noqa: S314
            root = tree.getroot()

            match = re.match("{(.*)}", root.tag)
            if match:
                namespace = match.group(1)
            else:
                error = (
                    'The "IES TM-27-14" spectral distribution namespace was not found!'
                )

                raise ValueError(error)

            self.name = os.path.splitext(os.path.basename(self._path))[0]

            iterator = root.iter

            for header_element in (self.header, self):
                mapping = header_element.mapping
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
                wavelengths.append(spectral_data.attrib[self.mapping.data.attribute])
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
            self.name = "Undefined" if len(components) == 0 else " - ".join(components)

            self.wavelengths = as_float_array(wavelengths)
            self.values = as_float_array(values)

            return self

        error = 'The "IES TM-27-14" spectral distribution path is undefined!'

        raise ValueError(error)

    def write(self) -> bool:
        """
        Write the spectral distribution spectral data to the specified *XML*
        file path.

        Returns
        -------
        :class:`bool`
            Definition success.

        Raises
        ------
        ValueError
            If the *IES TM-27-14* spectral distribution path is undefined.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from shutil import rmtree
        >>> from tempfile import mkdtemp
        >>> directory = join(dirname(__file__), "tests", "resources")
        >>> sd = SpectralDistribution_IESTM2714(join(directory, "Fluorescent.spdx"))
        >>> temporary_directory = mkdtemp()
        >>> sd.path = join(temporary_directory, "Fluorescent.spdx")
        >>> sd.write()
        True
        >>> rmtree(temporary_directory)
        """

        if self._path is not None:
            root = ET.Element("IESTM2714")
            root.attrib = {
                "xmlns": NAMESPACE_IESTM2714,
                "version": VERSION_IESTM2714,
            }

            spectral_distribution = ET.Element("")
            for header_element in (self.header, self):
                mapping = header_element.mapping
                element = ET.SubElement(root, mapping.element)
                for specification in mapping.elements:
                    element_child = ET.SubElement(element, specification.element)
                    value = getattr(header_element, specification.attribute)
                    element_child.text = specification.write_conversion(value)

                if header_element is self:
                    spectral_distribution = element

            # Writing spectral data.
            for wavelength, value in tstack([self.wavelengths, self.values]):
                element_child = ET.SubElement(
                    spectral_distribution, mapping.data.element
                )
                element_child.text = mapping.data.write_conversion(value)
                element_child.attrib = {
                    mapping.data.attribute: mapping.data.write_conversion(wavelength)
                }

            xml = minidom.parseString(  # noqa: S318
                ET.tostring(root)
            ).toprettyxml()

            with open(self._path, "w") as file:
                file.write(xml)

            return True

        error = 'The "IES TM-27-14" spectral distribution path is undefined!'

        raise ValueError(error)
