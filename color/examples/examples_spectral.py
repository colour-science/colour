# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Color** package color *spectral* related examples.
"""

import color

# From '480 nm' wavelength to *CIE XYZ* colorspace tristimulus values.
print(color.wavelength_to_XYZ(480,
                              color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS["Standard CIE 1931 2 Degree Observer"]))

# Defining a sample spectral power distribution data.
SAMPLE_SPD_DATA = {340: 0.0000,
                   345: 0.0000,
                   350: 0.0000,
                   355: 0.0000,
                   360: 0.0000,
                   365: 0.0000,
                   370: 0.0000,
                   375: 0.0000,
                   380: 0.0000,
                   385: 0.0000,
                   390: 0.0000,
                   395: 0.0000,
                   400: 0.0641,
                   405: 0.0650,
                   410: 0.0654,
                   415: 0.0652,
                   420: 0.0645,
                   425: 0.0629,
                   430: 0.0605,
                   435: 0.0581,
                   440: 0.0562,
                   445: 0.0551,
                   450: 0.0543,
                   455: 0.0539,
                   460: 0.0537,
                   465: 0.0538,
                   470: 0.0541,
                   475: 0.0547,
                   480: 0.0559,
                   485: 0.0578,
                   490: 0.0603,
                   495: 0.0629,
                   500: 0.0651,
                   505: 0.0667,
                   510: 0.0680,
                   515: 0.0691,
                   520: 0.0705,
                   525: 0.0720,
                   530: 0.0736,
                   535: 0.0753,
                   540: 0.0772,
                   545: 0.0791,
                   550: 0.0809,
                   555: 0.0833,
                   560: 0.0870,
                   565: 0.0924,
                   570: 0.0990,
                   575: 0.1061,
                   580: 0.1128,
                   585: 0.1190,
                   590: 0.1251,
                   595: 0.1308,
                   600: 0.1360,
                   605: 0.1403,
                   610: 0.1439,
                   615: 0.1473,
                   620: 0.1511,
                   625: 0.1550,
                   630: 0.1590,
                   635: 0.1634,
                   640: 0.1688,
                   645: 0.1753,
                   650: 0.1828,
                   655: 0.1909,
                   660: 0.1996,
                   665: 0.2088,
                   670: 0.2187,
                   675: 0.2291,
                   680: 0.2397,
                   685: 0.2505,
                   690: 0.2618,
                   695: 0.2733,
                   700: 0.2852,
                   705: 0.0000,
                   710: 0.0000,
                   715: 0.0000,
                   720: 0.0000,
                   725: 0.0000,
                   730: 0.0000,
                   735: 0.0000,
                   740: 0.0000,
                   745: 0.0000,
                   750: 0.0000,
                   755: 0.0000,
                   760: 0.0000,
                   765: 0.0000,
                   770: 0.0000,
                   775: 0.0000,
                   780: 0.0000,
                   785: 0.0000,
                   790: 0.0000,
                   795: 0.0000,
                   800: 0.0000,
                   805: 0.0000,
                   810: 0.0000,
                   815: 0.0000,
                   820: 0.0000,
                   825: 0.0000,
                   830: 0.0000}

# Creating the sample spectral power distribution.
spd = color.SpectralPowerDistribution(name="Sample", spd=SAMPLE_SPD_DATA)

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

cmfs = color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS["Standard CIE 1931 2 Degree Observer"]
illuminant = color.ILLUMINANTS_RELATIVE_SPD["A"]

# Calculating the sample spectral power distribution *CIE XYZ* tristimulus values.
print(color.spectral_to_XYZ(spd,
                            cmfs,
                            illuminant))

# Calculating *A* illuminant chromaticity coordinates under *Standard CIE 1931 2 Degree Observer*.
print(color.XYZ_to_xy(color.spectral_to_XYZ(illuminant,
                                            cmfs)))

# Correcting spectral bandpass.
print color.bandpass_correction(spd).values