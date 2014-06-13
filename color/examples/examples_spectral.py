# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package color *spectrum* related examples.
"""

import color

# Defining a sample spectral power distribution data.
sample_spd_data = {380: 0.048,
                   385: 0.051,
                   390: 0.055,
                   395: 0.06,
                   400: 0.065,
                   405: 0.068,
                   410: 0.068,
                   415: 0.067,
                   420: 0.064,
                   425: 0.062,
                   430: 0.059,
                   435: 0.057,
                   440: 0.055,
                   445: 0.054,
                   450: 0.053,
                   455: 0.053,
                   460: 0.052,
                   465: 0.052,
                   470: 0.052,
                   475: 0.053,
                   480: 0.054,
                   485: 0.055,
                   490: 0.057,
                   495: 0.059,
                   500: 0.061,
                   505: 0.062,
                   510: 0.065,
                   515: 0.067,
                   520: 0.07,
                   525: 0.072,
                   530: 0.074,
                   535: 0.075,
                   540: 0.076,
                   545: 0.078,
                   550: 0.079,
                   555: 0.082,
                   560: 0.087,
                   565: 0.092,
                   570: 0.1,
                   575: 0.107,
                   580: 0.115,
                   585: 0.122,
                   590: 0.129,
                   595: 0.134,
                   600: 0.138,
                   605: 0.142,
                   610: 0.146,
                   615: 0.15,
                   620: 0.154,
                   625: 0.158,
                   630: 0.163,
                   635: 0.167,
                   640: 0.173,
                   645: 0.18,
                   650: 0.188,
                   655: 0.196,
                   660: 0.204,
                   665: 0.213,
                   670: 0.222,
                   675: 0.231,
                   680: 0.242,
                   685: 0.251,
                   690: 0.261,
                   695: 0.271,
                   700: 0.282,
                   705: 0.294,
                   710: 0.305,
                   715: 0.318,
                   720: 0.334,
                   725: 0.354,
                   730: 0.372,
                   735: 0.392,
                   740: 0.409,
                   745: 0.42,
                   750: 0.436,
                   755: 0.45,
                   760: 0.462,
                   765: 0.465,
                   770: 0.448,
                   775: 0.432,
                   780: 0.421}

# Creating the sample spectral power distribution.
spd = color.SpectralPowerDistribution(name="Sample", spd=sample_spd_data)

# Displaying the sample spectral power distribution shape.
print(spd.shape)

# Checking the sample spectral power distribution uniformity.
print(spd.is_uniform())

# Cloning the sample spectral power distribution.
clone_spd = spd.clone()

# Interpolating the cloned sample spectral power distribution.
clone_spd.interpolate(start=360, end=780, steps=1)
print(clone_spd[666])

# Extrapolating the cloned sample spectral power distribution.
clone_spd.extrapolate(start=340, end=830)
print(clone_spd[340], clone_spd[360])

# Aligning the cloned sample spectral power distribution.
clone_spd.align(start=400, end=700, steps=5)
print(clone_spd[400], clone_spd[700])

cmfs = color.CMFS["CIE 1931 2 Degree Standard Observer"]
illuminant = color.ILLUMINANTS_RELATIVE_SPDS["A"]

# Calculating the sample spectral power distribution *CIE XYZ* tristimulus values.
print(color.spectral_to_XYZ(spd, cmfs, illuminant))

# Calculating *A* illuminant chromaticity coordinates under *CIE 1931 2 Degree Standard Observer*.
print(color.XYZ_to_xy(color.spectral_to_XYZ(illuminant, cmfs)))

# From '546.1 nm' wavelength to *CIE XYZ* colorspace tristimulus values.
print(color.wavelength_to_XYZ(546.1, color.CMFS["CIE 1931 2 Degree Standard Observer"]))

# Correcting spectral bandpass.
print color.bandpass_correction(spd).values