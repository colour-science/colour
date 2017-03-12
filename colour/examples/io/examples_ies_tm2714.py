#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases input / output *IES TM-27-14* spectral data XML files related
examples.
"""

import os
from pprint import pprint

import colour
from colour.utilities.verbose import message_box

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

message_box('"IES TM-27-14" Spectral Data "XML" File IO')

message_box('Reading spectral data from "IES TM-27-14" "XML" file.')
spd = colour.IES_TM2714_Spd(os.path.join(RESOURCES_DIRECTORY,
                                         'TM27 Sample Spectral Data.spdx'))
spd.read()
print(spd)

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file header:')
print('Manufacturer: {0}'.format(spd.header.manufacturer))
print('Catalog Number: {0}'.format(spd.header.catalog_number))
print('Description: {0}'.format(spd.header.description))
print('Document Creator: {0}'.format(spd.header.document_creator))
print('Unique Identifier: {0}'.format(spd.header.unique_identifier))
print('Measurement Equipment: {0}'.format(spd.header.measurement_equipment))
print('Laboratory: {0}'.format(spd.header.laboratory))
print('Report Number: {0}'.format(spd.header.report_number))
print('Report Date: {0}'.format(spd.header.report_date))
print('Document Creation Date: {0}'.format(spd.header.document_creation_date))
print('Comments: {0}'.format(spd.header.comments))

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file spectral distribution:')
print('Spectral Quantity: {0}'.format(spd.spectral_quantity))
print('Reflection Geometry: {0}'.format(spd.reflection_geometry))
print('Transmission Geometry: {0}'.format(spd.transmission_geometry))
print('Bandwidth FWHM: {0}'.format(spd.bandwidth_FWHM))
print('Bandwidth Corrected: {0}'.format(spd.bandwidth_corrected))

print('\n')

message_box('"IES TM-27-14" spectral data "XML" file spectral data:')
pprint(list(spd.items))
