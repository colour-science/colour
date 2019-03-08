# -*- coding: utf-8 -*-
"""
IES TM-27-14 Data Input / Output
================================

Defines the :class:`colour.SpectralDistribution_IESTM2714` class handling
*IES TM-27-14* spectral data XML files.

References
----------
-   :cite:`IESComputerCommittee2014a` : IES Computer Committee, & TM-27-14
    Working Group. (2014). IES Standard Format for the Electronic Transfer of
    Spectral Data Electronic Transfer of Spectral Data. ISBN:978-0879952952
"""

from __future__ import division, unicode_literals

import os
import re
from collections import namedtuple
from xml.etree import ElementTree  # nosec
from xml.dom import minidom  # nosec

from colour.colorimetry import SpectralDistribution
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import Structure, is_numeric, is_string, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'IES_TM2714_VERSION', 'IES_TM2714_NAMESPACE',
    'IES_TM2714_ElementSpecification', 'IES_TM2714_Header',
    'SpectralDistribution_IESTM2714'
]

IES_TM2714_VERSION = '1.0'
IES_TM2714_NAMESPACE = 'http://www.ies.org/iestm2714'


class IES_TM2714_ElementSpecification(
        namedtuple('IES_TM2714_ElementSpecification',
                   ('element', 'attribute', 'type', 'required',
                    'read_conversion', 'write_conversion'))):
    """
    *IES TM-27-14* spectral data XML file element specification.

    Parameters
    ----------
    attribute : unicode
        Associated attribute name.
    element : unicode
        Element name.
    type_ : unicode
        Element type.
    attribute : object
        Element type.
    required : bool
        Is element required.
    read_conversion : object
        Method to convert from XML to type on reading.
    write_conversion : object
        Method to convert from type to XML on writing.
    """

    def __new__(cls,
                element,
                attribute,
                type_=str,
                required=False,
                read_conversion=format,
                write_conversion=(
                    lambda x: format(x) if x is not None else 'N/A')):
        """
        Returns a new instance of the
        :class:`colour.io.ies_tm2714.IES_TM2714_Element` class.
        """

        return super(IES_TM2714_ElementSpecification, cls).__new__(
            cls, element, attribute, type_, required, read_conversion,
            write_conversion)


class IES_TM2714_Header(object):
    """
    Defines the header object for a *IES TM-27-14* spectral distribution.

    Parameters
    ----------
    manufacturer : unicode, optional
        Manufacturer of the device under test.
    catalog_number : unicode, optional
        Manufacturer's product catalog number.
    description : unicode, optional
        Description of the spectral data in the spectral data XML file.
    document_creator : unicode, optional
        Creator of the spectral data XML file, which may be a
        test lab, a research group, a standard body, a company or an
        individual.
    unique_identifier : unicode, optional
        Description of the equipment used to measure the spectral data.
    measurement_equipment : unicode, optional
        Description of the equipment used to measure the spectral data.
    laboratory : unicode, optional
        Testing laboratory name that performed the spectral data measurements.
    report_number : unicode, optional
        Testing laboratory report number.
    report_date : unicode, optional
        Testing laboratory report date using the *XML DateTime Data Type*,
        *YYYY-MM-DDThh:mm:ss*.
    document_creation_date : unicode, optional
        Spectral data XML file creation date using the
        *XML DateTime Data Type*, *YYYY-MM-DDThh:mm:ss*.
    comments : unicode, optional
        Additional information relating to the tested and reported data.

    Attributes
    ----------
    mapping
    manufacturer
    catalog_number
    description
    document_creator
    unique_identifier
    measurement_equipment
    laboratory
    report_number
    report_date
    document_creation_date
    comments

    Examples
    --------
    >>> IES_TM2714_Header('colour-science')  # doctest: +ELLIPSIS
    <...IES_TM2714_Header object at 0x...>
    >>> IES_TM2714_Header('colour-science').manufacturer  # doctest: +SKIP
    'colour-science'
    """

    def __init__(self,
                 manufacturer=None,
                 catalog_number=None,
                 description=None,
                 document_creator=None,
                 unique_identifier=None,
                 measurement_equipment=None,
                 laboratory=None,
                 report_number=None,
                 report_date=None,
                 document_creation_date=None,
                 comments=None):

        self._mapping = Structure(
            **{
                'element':
                    'Header',
                'elements':
                    (IES_TM2714_ElementSpecification('Manufacturer',
                                                     'manufacturer'),
                     IES_TM2714_ElementSpecification('CatalogNumber',
                                                     'catalog_number'),
                     IES_TM2714_ElementSpecification(
                         'Description', 'description', required=True),
                     IES_TM2714_ElementSpecification(
                         'DocumentCreator', 'document_creator', required=True),
                     IES_TM2714_ElementSpecification('UniqueIdentifier',
                                                     'unique_identifier'),
                     IES_TM2714_ElementSpecification('MeasurementEquipment',
                                                     'measurement_equipment'),
                     IES_TM2714_ElementSpecification('Laboratory',
                                                     'laboratory'),
                     IES_TM2714_ElementSpecification('ReportNumber',
                                                     'report_number'),
                     IES_TM2714_ElementSpecification('ReportDate',
                                                     'report_date'),
                     IES_TM2714_ElementSpecification(
                         'DocumentCreationDate',
                         'document_creation_date',
                         required=True),
                     IES_TM2714_ElementSpecification('Comments', 'comments',
                                                     False))
            })

        self._manufacturer = None
        self.manufacturer = manufacturer
        self._catalog_number = None
        self.catalog_number = catalog_number
        self._description = None
        self.description = description
        self._document_creator = None
        self.document_creator = document_creator
        self._unique_identifier = None
        self.unique_identifier = unique_identifier
        self._measurement_equipment = None
        self.measurement_equipment = measurement_equipment
        self._laboratory = None
        self.laboratory = laboratory
        self._report_number = None
        self.report_number = report_number
        self._report_date = None
        self.report_date = report_date
        self._document_creation_date = None
        self.document_creation_date = document_creation_date
        self._comments = None
        self.comments = comments

    @property
    def mapping(self):
        """
        Getter and setter property for the mapping structure.

        Parameters
        ----------
        value : Structure
            Value to set the mapping structure with.

        Returns
        -------
        Structure
            Mapping structure.
        """

        return self._mapping

    @property
    def manufacturer(self):
        """
        Getter and setter property for the manufacturer.

        Parameters
        ----------
        value : unicode
            Value to set the manufacturer with.

        Returns
        -------
        unicode
            Manufacturer.
        """

        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, value):
        """
        Setter for the **self.manufacturer** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'manufacturer', value))
        self._manufacturer = value

    @property
    def catalog_number(self):
        """
        Getter and setter property for the catalog number.

        Parameters
        ----------
        value : unicode
            Value to set the catalog number with.

        Returns
        -------
        unicode
            Catalog number.
        """

        return self._catalog_number

    @catalog_number.setter
    def catalog_number(self, value):
        """
        Setter for the **self.catalog_number** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'catalog_number', value))
        self._catalog_number = value

    @property
    def description(self):
        """
        Getter and setter property for the description.

        Parameters
        ----------
        value : unicode
            Value to set the description with.

        Returns
        -------
        unicode
            Description.
        """

        return self._description

    @description.setter
    def description(self, value):
        """
        Setter for the **self.description** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'description', value))
        self._description = value

    @property
    def document_creator(self):
        """
        Getter and setter property for the document creator.

        Parameters
        ----------
        value : unicode
            Value to set the document creator with.

        Returns
        -------
        unicode
            Document creator.
        """

        return self._document_creator

    @document_creator.setter
    def document_creator(self, value):
        """
        Setter for the **self.document_creator** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'document_creator', value))
        self._document_creator = value

    @property
    def unique_identifier(self):
        """
        Getter and setter property for the unique identifier.

        Parameters
        ----------
        value : unicode
            Value to set the unique identifier with.

        Returns
        -------
        unicode
            Unique identifier.
        """

        return self._unique_identifier

    @unique_identifier.setter
    def unique_identifier(self, value):
        """
        Setter for the **self.unique_identifier** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'unique_identifier', value))
        self._unique_identifier = value

    @property
    def measurement_equipment(self):
        """
        Getter and setter property for the measurement equipment.

        Parameters
        ----------
        value : unicode
            Value to set the measurement equipment with.

        Returns
        -------
        unicode
            Measurement equipment.
        """

        return self._measurement_equipment

    @measurement_equipment.setter
    def measurement_equipment(self, value):
        """
        Setter for the **self.measurement_equipment** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'measurement_equipment', value))
        self._measurement_equipment = value

    @property
    def laboratory(self):
        """
        Getter and setter property for the laboratory.

        Parameters
        ----------
        value : unicode
            Value to set the laboratory with.

        Returns
        -------
        unicode
            Laboratory.
        """

        return self._laboratory

    @laboratory.setter
    def laboratory(self, value):
        """
        Setter for the **self.measurement_equipment** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'laboratory', value))
        self._laboratory = value

    @property
    def report_number(self):
        """
        Getter and setter property for the report number.

        Parameters
        ----------
        value : unicode
            Value to set the report number with.

        Returns
        -------
        unicode
            Report number.
        """

        return self._report_number

    @report_number.setter
    def report_number(self, value):
        """
        Setter for the **self.report_number** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'report_number', value))
        self._report_number = value

    @property
    def report_date(self):
        """
        Getter and setter property for the report date.

        Parameters
        ----------
        value : unicode
            Value to set the report date with.

        Returns
        -------
        unicode
            Report date.
        """

        return self._report_date

    @report_date.setter
    def report_date(self, value):
        """
        Setter for the **self.report_date** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'report_date', value))
        self._report_date = value

    @property
    def document_creation_date(self):
        """
        Getter and setter property for the document creation date.

        Parameters
        ----------
        value : unicode
            Value to set the document creation date with.

        Returns
        -------
        unicode
            Document creation date.
        """

        return self._document_creation_date

    @document_creation_date.setter
    def document_creation_date(self, value):
        """
        Setter for the **self.document_creation_date** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'document_creation_date', value))
        self._document_creation_date = value

    @property
    def comments(self):
        """
        Getter and setter property for the comments.

        Parameters
        ----------
        value : unicode
            Value to set the comments with.

        Returns
        -------
        unicode
            Comments.
        """

        return self._comments

    @comments.setter
    def comments(self, value):
        """
        Setter for the **self.comments** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'comments', value))
        self._comments = value


class SpectralDistribution_IESTM2714(SpectralDistribution):
    """
    Defines a *IES TM-27-14* spectral distribution.

    This class can read and write *IES TM-27-14* spectral data XML files.

    Parameters
    ----------
    path : unicode, optional
        Spectral data XML file path.
    header : IES_TM2714_Header, optional
        *IES TM-27-14* spectral distribution header.
    spectral_quantity : unicode, optional
        **{'flux', 'absorptance', 'transmittance', 'reflectance', 'intensity',
        'irradiance', 'radiance', 'exitance', 'R-Factor', 'T-Factor',
        'relative', 'other'}**,
        Quantity of measurement for each element of the spectral data.
    reflection_geometry : unicode, optional
        **{'di:8', 'de:8', '8:di', '8:de', 'd:d', 'd:0', '45a:0', '45c:0',
        '0:45a', '45x:0', '0:45x', 'other'}**,
        Spectral reflectance factors geometric conditions.
    transmission_geometry : unicode, optional
        **{'0:0', 'di:0', 'de:0', '0:di', '0:de', 'd:d', 'other'}**,
        Spectral transmittance factors geometric conditions.
    bandwidth_FWHM : numeric, optional
        Spectroradiometer full-width half-maximum bandwidth in nanometers.
    bandwidth_corrected : bool, optional
        Specifies if bandwidth correction has been applied to the measured
        data.

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
    mapping
    path
    header
    spectral_quantity
    reflection_geometry
    transmission_geometry
    bandwidth_FWHM
    bandwidth_corrected

    Methods
    -------
    read
    write

    References
    ----------
    :cite:`IESComputerCommittee2014a`

    Examples
    --------
    >>> from os.path import dirname, join
    >>> directory = join(dirname(__file__), 'tests', 'resources')
    >>> sd = SpectralDistribution_IESTM2714(
    ...     join(directory, 'Fluorescent.spdx'))
    >>> sd.read()
    True
    >>> sd.header.manufacturer
    'Unknown'
    >>> # Doctests ellipsis for Python 2.x compatibility.
    >>> sd[501.7]  # doctest: +ELLIPSIS
    0.0950000...
    """

    def __init__(self,
                 path=None,
                 header=None,
                 spectral_quantity=None,
                 reflection_geometry=None,
                 transmission_geometry=None,
                 bandwidth_FWHM=None,
                 bandwidth_corrected=None):

        super(SpectralDistribution_IESTM2714, self).__init__(
            data=None, domain=None)

        self._mapping = Structure(
            **{
                'element':
                    'SpectralDistribution',
                'elements':
                    (IES_TM2714_ElementSpecification(
                        'SpectralQuantity', 'spectral_quantity',
                        required=True),
                     IES_TM2714_ElementSpecification('ReflectionGeometry',
                                                     'reflection_geometry'),
                     IES_TM2714_ElementSpecification('TransmissionGeometry',
                                                     'transmission_geometry'),
                     IES_TM2714_ElementSpecification(
                         'BandwidthFWHM',
                         'bandwidth_FWHM',
                         read_conversion=DEFAULT_FLOAT_DTYPE),
                     IES_TM2714_ElementSpecification(
                         'BandwidthCorrected',
                         'bandwidth_corrected',
                         read_conversion=(
                             lambda x: True if x == 'true' else False),
                         write_conversion=(
                             lambda x: 'true' if x is True else 'False'))),
                'data':
                    IES_TM2714_ElementSpecification(
                        'SpectralData', 'wavelength', required=True)
            })

        self._path = None
        self.path = path
        self._header = None
        self.header = header if header is not None else IES_TM2714_Header()
        self._spectral_quantity = None
        self.spectral_quantity = spectral_quantity
        self._reflection_geometry = None
        self.reflection_geometry = reflection_geometry
        self._transmission_geometry = None
        self.transmission_geometry = transmission_geometry
        self._bandwidth_FWHM = None
        self.bandwidth_FWHM = bandwidth_FWHM
        self._bandwidth_corrected = None
        self.bandwidth_corrected = bandwidth_corrected

    @property
    def mapping(self):
        """
        Getter and setter property for the mapping structure.

        Parameters
        ----------
        value : Structure
            Value to set the mapping structure with.

        Returns
        -------
        Structure
            Mapping structure.
        """

        return self._mapping

    @property
    def path(self):
        """
        Getter and setter property for the path.

        Parameters
        ----------
        value : unicode
            Value to set the path with.

        Returns
        -------
        unicode
            Path.
        """

        return self._path

    @path.setter
    def path(self, value):
        """
        Setter for the **self.path** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'path', value))
        self._path = value

    @property
    def header(self):
        """
        Getter and setter property for the header.

        Parameters
        ----------
        value : IES_TM2714_Header
            Value to set the header with.

        Returns
        -------
        IES_TM2714_Header
            Header.
        """

        return self._header

    @header.setter
    def header(self, value):
        """
        Setter for the **self.header** property.
        """

        if value is not None:
            assert isinstance(value, IES_TM2714_Header), (
                '"{0}" attribute: "{1}" is not a "IES_TM2714_Header" '
                'instance!'.format('header', value))
        self._header = value

    @property
    def spectral_quantity(self):
        """
        Getter and setter property for the spectral quantity.

        Parameters
        ----------
        value : unicode
            Value to set the spectral quantity with.

        Returns
        -------
        unicode
            Spectral quantity.
        """

        return self._spectral_quantity

    @spectral_quantity.setter
    def spectral_quantity(self, value):
        """
        Setter for the **self.spectral_quantity** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'spectral_quantity', value))
        self._spectral_quantity = value

    @property
    def reflection_geometry(self):
        """
        Getter and setter property for the reflection geometry.

        Parameters
        ----------
        value : unicode
            Value to set the reflection geometry with.

        Returns
        -------
        unicode
            Reflection geometry.
        """

        return self._reflection_geometry

    @reflection_geometry.setter
    def reflection_geometry(self, value):
        """
        Setter for the **self.reflection_geometry** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'reflection_geometry', value))
        self._reflection_geometry = value

    @property
    def transmission_geometry(self):
        """
        Getter and setter property for the transmission geometry.

        Parameters
        ----------
        value : unicode
            Value to set the transmission geometry with.

        Returns
        -------
        unicode
            Transmission geometry.
        """

        return self._transmission_geometry

    @transmission_geometry.setter
    def transmission_geometry(self, value):
        """
        Setter for the **self.transmission_geometry** property.
        """

        if value is not None:
            assert is_string(value), (
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'transmission_geometry', value))
        self._transmission_geometry = value

    @property
    def bandwidth_FWHM(self):
        """
        Getter and setter property for the full-width half-maximum bandwidth.

        Parameters
        ----------
        value : numeric
            Value to set the full-width half-maximum bandwidth with.

        Returns
        -------
        numeric
            Full-width half-maximum bandwidth.
        """

        return self._bandwidth_FWHM

    @bandwidth_FWHM.setter
    def bandwidth_FWHM(self, value):
        """
        Setter for the **self.bandwidth_FWHM** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!'.format(
                    'bandwidth_FWHM', value))

        self._bandwidth_FWHM = value

    @property
    def bandwidth_corrected(self):
        """
        Getter and setter property for whether bandwidth correction has been
        applied to the measured data.

        Parameters
        ----------
        value : bool
            Whether bandwidth correction has been applied to the measured data.

        Returns
        -------
        bool
            Whether bandwidth correction has been applied to the measured data.
        """

        return self._bandwidth_corrected

    @bandwidth_corrected.setter
    def bandwidth_corrected(self, value):
        """
        Setter for the **self.bandwidth_corrected** property.
        """

        if value is not None:
            assert isinstance(value, bool), (
                '"{0}" attribute: "{1}" is not a "bool" instance!'.format(
                    'bandwidth_corrected', value))

        self._bandwidth_corrected = value

    def read(self):
        """
        Reads and parses the spectral data XML file path.

        Returns
        -------
        bool
            Definition success.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_IESTM2714(
        ...     join(directory, 'Fluorescent.spdx'))
        >>> sd.read()
        True
        >>> sd.header.description
        'Rare earth fluorescent lamp'
        >>> # Doctests ellipsis for Python 2.x compatibility.
        >>> sd[400]  # doctest: +ELLIPSIS
        0.0339999...
        """

        formatter = './{{{0}}}{1}/{{{0}}}{2}'

        tree = ElementTree.parse(self._path)  # nosec
        root = tree.getroot()

        namespace = re.match('{(.*)}', root.tag).group(1)

        self.name = os.path.splitext(os.path.basename(self._path))[0]

        iterator = root.iter

        for header_element in (self.header, self):
            mapping = header_element.mapping
            for specification in mapping.elements:
                element = root.find(
                    formatter.format(namespace, mapping.element,
                                     specification.element))
                if element is not None:
                    setattr(header_element, specification.attribute,
                            specification.read_conversion(element.text))

        # Reading spectral data.
        wavelengths = []
        values = []
        for spectral_data in iterator('{{{0}}}{1}'.format(
                namespace, self.mapping.data.element)):
            wavelengths.append(
                DEFAULT_FLOAT_DTYPE(
                    spectral_data.attrib[self.mapping.data.attribute]))
            values.append(DEFAULT_FLOAT_DTYPE(spectral_data.text))

        self.wavelengths = wavelengths
        self.values = values

        return True

    def write(self):
        """
        Write the spectral distribution spectral data to XML file path.

        Returns
        -------
        bool
            Definition success.

        Examples
        --------
        >>> from os.path import dirname, join
        >>> from shutil import rmtree
        >>> from tempfile import mkdtemp
        >>> directory = join(dirname(__file__), 'tests', 'resources')
        >>> sd = SpectralDistribution_IESTM2714(
        ...     join(directory, 'Fluorescent.spdx'))
        >>> sd.read()
        True
        >>> temporary_directory = mkdtemp()
        >>> sd.path = join(temporary_directory, 'Fluorescent.spdx')
        >>> sd.write()
        True
        >>> rmtree(temporary_directory)
        """

        root = ElementTree.Element('IESTM2714')
        root.attrib = {
            'xmlns': IES_TM2714_NAMESPACE,
            'version': IES_TM2714_VERSION
        }

        spectral_distribution = None
        for header_element in (self.header, self):
            mapping = header_element.mapping
            element = ElementTree.SubElement(root, mapping.element)
            for specification in mapping.elements:
                element_child = ElementTree.SubElement(element,
                                                       specification.element)
                value = getattr(header_element, specification.attribute)
                element_child.text = specification.write_conversion(value)

            if header_element is self:
                spectral_distribution = element

        # Writing spectral data.
        for (wavelength, value) in tstack([self.wavelengths, self.values]):
            element_child = ElementTree.SubElement(spectral_distribution,
                                                   mapping.data.element)
            element_child.text = mapping.data.write_conversion(value)
            element_child.attrib = {
                mapping.data.attribute:
                    mapping.data.write_conversion(wavelength)
            }

        xml = minidom.parseString(
            ElementTree.tostring(root)).toprettyxml()  # nosec

        with open(self._path, 'w') as file:
            file.write(xml)

        return True
