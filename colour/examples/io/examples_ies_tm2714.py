"""Showcases *IES TM-27-14* spectral data *XML* files input / output examples."""

import os

import colour
from colour.utilities import message_box

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), "resources")

message_box('"IES TM-27-14" Spectral Data "XML" File IO')

message_box('Reading spectral data from "IES TM-27-14" "XML" file.')
sd_tm2714 = colour.SpectralDistribution_IESTM2714(
    os.path.join(RESOURCES_DIRECTORY, "TM27 Sample Spectral Data.spdx")
).read()
print(sd_tm2714)

print("\n")

message_box('"IES TM-27-14" spectral data "XML" file header:')
print(f"Manufacturer: {sd_tm2714.header.manufacturer}")
print(f"Catalog Number: {sd_tm2714.header.catalog_number}")
print(f"Description: {sd_tm2714.header.description}")
print(f"Document Creator: {sd_tm2714.header.document_creator}")
print(f"Unique Identifier: {sd_tm2714.header.unique_identifier}")
print(f"Measurement Equipment: {sd_tm2714.header.measurement_equipment}")
print(f"Laboratory: {sd_tm2714.header.laboratory}")
print(f"Report Number: {sd_tm2714.header.report_number}")
print(f"Report Date: {sd_tm2714.header.report_date}")
print(f"Document Creation Date: {sd_tm2714.header.document_creation_date}")
print(f"Comments: {sd_tm2714.header.comments}")

print("\n")

message_box('"IES TM-27-14" spectral data "XML" file spectral distribution:')
print(f"Spectral Quantity: {sd_tm2714.spectral_quantity}")
print(f"Reflection Geometry: {sd_tm2714.reflection_geometry}")
print(f"Transmission Geometry: {sd_tm2714.transmission_geometry}")
print(f"Bandwidth FWHM: {sd_tm2714.bandwidth_FWHM}")
print(f"Bandwidth Corrected: {sd_tm2714.bandwidth_corrected}")

print("\n")

message_box('"IES TM-27-14" spectral data "XML" file spectral data:')
print(sd_tm2714)
