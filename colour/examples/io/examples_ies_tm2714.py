# -*- coding: utf-8 -*-
"""
Showcases input / output *IES TM-27-14* spectral data XML files related
examples.
"""

import os

import colour
from colour.utilities import message_box

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

message_box('"IES TM-27-14" Spectral Data "XML" File IO')

message_box('Reading spectral data from "IES TM-27-14" "XML" file.')
sd_tm2714 = colour.SpectralDistribution_IESTM2714(
    os.path.join(RESOURCES_DIRECTORY,
                 'TM27 Sample Spectral Data.spdx')).read()
print(sd_tm2714)

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file header:')
print('Manufacturer: {0}'.format(sd_tm2714.header.manufacturer))
print('Catalog Number: {0}'.format(sd_tm2714.header.catalog_number))
print('Description: {0}'.format(sd_tm2714.header.description))
print('Document Creator: {0}'.format(sd_tm2714.header.document_creator))
print('Unique Identifier: {0}'.format(sd_tm2714.header.unique_identifier))
print('Measurement Equipment: {0}'.format(
    sd_tm2714.header.measurement_equipment))
print('Laboratory: {0}'.format(sd_tm2714.header.laboratory))
print('Report Number: {0}'.format(sd_tm2714.header.report_number))
print('Report Date: {0}'.format(sd_tm2714.header.report_date))
print('Document Creation Date: {0}'.format(
    sd_tm2714.header.document_creation_date))
print('Comments: {0}'.format(sd_tm2714.header.comments))

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file spectral distribution:')
print('Spectral Quantity: {0}'.format(sd_tm2714.spectral_quantity))
print('Reflection Geometry: {0}'.format(sd_tm2714.reflection_geometry))
print('Transmission Geometry: {0}'.format(sd_tm2714.transmission_geometry))
print('Bandwidth FWHM: {0}'.format(sd_tm2714.bandwidth_FWHM))
print('Bandwidth Corrected: {0}'.format(sd_tm2714.bandwidth_corrected))

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file spectral data:')
print(sd_tm2714)
