.. raw:: html

    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/colour-science/colour-branding/master/images/Colour_Logo_Dark_001.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/colour-science/colour-branding/master/images/Colour_Logo_001.svg">
        <img style="background:rgb(0, 0, 0, 0) !important;" src="https://raw.githubusercontent.com/colour-science/colour-branding/master/images/Colour_Logo_001.svg">
    </picture>

|

`Colour <https://github.com/colour-science/colour>`__ is an open-source
`Python <https://www.python.org>`__ package providing a comprehensive number
of algorithms and datasets for colour science.

It is freely available under the
`BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

**Colour** is an affiliated project of `NumFOCUS <https://numfocus.org>`__, a
501(c)(3) nonprofit in the United States.

.. sectnum::

Draft Release Notes
-------------------

The draft release notes from the
`develop <https://github.com/colour-science/colour/tree/develop>`__
branch are available at this
`url <https://gist.github.com/KelSolaar/4a6ebe9ec3d389f0934b154fec8df51d>`__.

Sponsors
--------

We are grateful for the support of our
`sponsors <https://github.com/colour-science/colour/blob/develop/SPONSORS.rst>`__.
If you'd like to join them, please consider
`becoming a sponsor on OpenCollective <https://opencollective.com/colour-science>`__.

Features
--------

Most of the objects are available from the ``colour`` namespace:

.. code-block:: python

    import colour

Automatic Colour Conversion Graph - ``colour.graph``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with version *0.3.14*, **Colour** implements an automatic colour
conversion graph enabling easier colour conversions.

..  image:: https://colour.readthedocs.io/en/develop/_static/Examples_Colour_Automatic_Conversion_Graph.png

.. code-block:: python

    sd = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"]
    colour.convert(sd, "Spectral Distribution", "sRGB", verbose={"mode": "Short"})

.. code-block:: text

    ===============================================================================
    *                                                                             *
    *   [ Conversion Path ]                                                       *
    *                                                                             *
    *   "sd_to_XYZ" --> "XYZ_to_sRGB"                                             *
    *                                                                             *
    ===============================================================================
    array([ 0.45675795,  0.30986982,  0.24861924])

.. code-block:: python

    illuminant = colour.SDS_ILLUMINANTS["FL2"]
    colour.convert(
        sd,
        "Spectral Distribution",
        "sRGB",
        sd_to_XYZ={"illuminant": illuminant},
    )

.. code-block:: text

    array([ 0.47924575,  0.31676968,  0.17362725])

Chromatic Adaptation - ``colour.adaptation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    XYZ = [0.20654008, 0.12197225, 0.05136952]
    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    A = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["A"]
    colour.chromatic_adaptation(XYZ, colour.xy_to_XYZ(D65), colour.xy_to_XYZ(A))

.. code-block:: text

    array([ 0.2533053 ,  0.13765138,  0.01543307])


.. code-block:: python

    sorted(colour.CHROMATIC_ADAPTATION_METHODS)

.. code-block:: text

    ['CIE 1994', 'CMCCAT2000', 'Fairchild 1990', 'Von Kries', 'Zhai 2018', 'vK20']


Algebra - ``colour.algebra``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel Interpolation
********************

.. code-block:: python

    y = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
    x = range(len(y))
    colour.KernelInterpolator(x, y)([0.25, 0.75, 5.50])

.. code-block:: text

    array([  6.18062083,   8.08238488,  57.85783403])

Sprague (1880) Interpolation
****************************

.. code-block:: python

    y = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
    x = range(len(y))
    colour.SpragueInterpolator(x, y)([0.25, 0.75, 5.50])

.. code-block:: text

    array([  6.72951612,   7.81406251,  43.77379185])

Colour Appearance Models - ``colour.appearance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    XYZ_w = [95.05, 100.00, 108.88]
    L_A = 318.31
    Y_b = 20.0
    colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b)

.. code-block:: text

    CAM_Specification_CIECAM02(J=34.434525727858997, C=67.365010921125943, h=22.279164147957065, s=62.81485585332716, Q=177.47124941102123, M=70.024939419291414, H=2.6896085344238898, HC=None)

.. code-block:: python

    colour.XYZ_to_CIECAM16(XYZ, XYZ_w, L_A, Y_b)

.. code-block:: text

    CAM_Specification_CIECAM16(J=34.434525727858997, C=67.365010921125943, h=22.279164147957065, s=62.81485585332716, Q=177.47124941102123, M=70.024939419291414, H=2.6896085344238898, HC=None)

.. code-block:: python

    colour.XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b)

.. code-block:: text

    CAM_Specification_CAM16(J=33.880368498111686, C=69.444353357408033, h=19.510887327451748, s=64.03612114840314, Q=176.03752758512178, M=72.18638534116765, H=399.52975599115319, HC=None)

.. code-block:: python

    colour.XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A)

.. code-block:: text

    CAM_Specification_Hellwig2022(J=33.880368498111686, C=40.347043294550311, h=19.510887327451748, s=117.38555017188679, Q=45.34489577734751, M=53.228355383108031, H=399.52975599115319, HC=None)

.. code-block:: python

    colour.XYZ_to_Kim2009(XYZ, XYZ_w, L_A)

.. code-block:: text

    CAM_Specification_Kim2009(J=19.879918542450902, C=55.839055250876946, h=22.013388165090046, s=112.97979354939129, Q=36.309026130161449, M=46.346415858227864, H=2.3543198369639931, HC=None)

.. code-block:: python

    colour.XYZ_to_ZCAM(XYZ, XYZ_w, L_A, Y_b)

.. code-block:: text

    CAM_Specification_ZCAM(J=38.347186278956357, C=21.12138989208518, h=33.711578931095197, s=81.444585609489536, Q=76.986725284523772, M=42.403805833900506, H=0.45779200212219573, HC=None, V=43.623590687423544, K=43.20894953152817, W=34.829588380192149)

Colour Blindness - ``colour.blindness``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    cmfs = colour.LMS_CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"]
    colour.msds_cmfs_anomalous_trichromacy_Machado2009(cmfs, np.array([15, 0, 0]))[450]

.. code-block:: text

    array([ 0.08912884,  0.0870524 ,  0.955393  ])

.. code-block:: python

    primaries = colour.MSDS_DISPLAY_PRIMARIES["Apple Studio Display"]
    d_LMS = (15, 0, 0)
    colour.matrix_anomalous_trichromacy_Machado2009(cmfs, primaries, d_LMS)

.. code-block:: text

    array([[-0.27774652,  2.65150084, -1.37375432],
           [ 0.27189369,  0.20047862,  0.52762768],
           [ 0.00644047,  0.25921579,  0.73434374]])

Colour Correction - ``colour characterisation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    RGB = [0.17224810, 0.09170660, 0.06416938]
    M_T = np.random.random((24, 3))
    M_R = M_T + (np.random.random((24, 3)) - 0.5) * 0.5
    colour.colour_correction(RGB, M_T, M_R)

.. code-block:: text

    array([ 0.1806237 ,  0.07234791,  0.07848845])

.. code-block:: python

    sorted(colour.COLOUR_CORRECTION_METHODS)

.. code-block:: text

    ['Cheung 2004', 'Finlayson 2015', 'Vandermonde']

ACES Input Transform - ``colour characterisation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sensitivities = colour.MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]
    illuminant = colour.SDS_ILLUMINANTS["D55"]
    colour.matrix_idt(sensitivities, illuminant)

.. code-block:: text

    (array([[ 0.59368175,  0.30418371,  0.10213454],
           [ 0.00457979,  1.14946003, -0.15403982],
           [ 0.03552213, -0.16312291,  1.12760077]]), array([ 1.58214188,  1.        ,  1.28910346]))

Colorimetry - ``colour.colorimetry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spectral Computations
*********************

.. code-block:: python

    colour.sd_to_XYZ(colour.SDS_LIGHT_SOURCES["Neodimium Incandescent"])

.. code-block:: text

    array([ 36.94726204,  32.62076174,  13.0143849 ])

.. code-block:: python

    sorted(colour.SPECTRAL_TO_XYZ_METHODS)

.. code-block:: text

    ['ASTM E308', 'Integration', 'astm2015']


Multi-Spectral Computations
***************************

.. code-block:: python

    msds = np.array(
        [
            [
                [
                    0.01367208,
                    0.09127947,
                    0.01524376,
                    0.02810712,
                    0.19176012,
                    0.04299992,
                ],
                [
                    0.00959792,
                    0.25822842,
                    0.41388571,
                    0.22275120,
                    0.00407416,
                    0.37439537,
                ],
                [
                    0.01791409,
                    0.29707789,
                    0.56295109,
                    0.23752193,
                    0.00236515,
                    0.58190280,
                ],
            ],
            [
                [
                    0.01492332,
                    0.10421912,
                    0.02240025,
                    0.03735409,
                    0.57663846,
                    0.32416266,
                ],
                [
                    0.04180972,
                    0.26402685,
                    0.03572137,
                    0.00413520,
                    0.41808194,
                    0.24696727,
                ],
                [
                    0.00628672,
                    0.11454948,
                    0.02198825,
                    0.39906919,
                    0.63640803,
                    0.01139849,
                ],
            ],
            [
                [
                    0.04325933,
                    0.26825359,
                    0.23732357,
                    0.05175860,
                    0.01181048,
                    0.08233768,
                ],
                [
                    0.02484169,
                    0.12027161,
                    0.00541695,
                    0.00654612,
                    0.18603799,
                    0.36247808,
                ],
                [
                    0.03102159,
                    0.16815442,
                    0.37186235,
                    0.08610666,
                    0.00413520,
                    0.78492409,
                ],
            ],
            [
                [
                    0.11682307,
                    0.78883040,
                    0.74468607,
                    0.83375293,
                    0.90571451,
                    0.70054168,
                ],
                [
                    0.06321812,
                    0.41898224,
                    0.15190357,
                    0.24591440,
                    0.55301750,
                    0.00657664,
                ],
                [
                    0.00305180,
                    0.11288624,
                    0.11357290,
                    0.12924391,
                    0.00195315,
                    0.21771573,
                ],
            ],
        ]
    )
    colour.msds_to_XYZ(
        msds,
        method="Integration",
        shape=colour.SpectralShape(400, 700, 60),
    )

.. code-block:: text

    array([[[  7.68544647,   4.09414317,   8.49324254],
            [ 17.12567298,  27.77681821,  25.52573685],
            [ 19.10280411,  34.45851476,  29.76319628]],
           [[ 18.03375827,   8.62340812,   9.71702574],
            [ 15.03110867,   6.54001068,  24.53208465],
            [ 37.68269495,  26.4411103 ,  10.66361816]],
           [[  8.09532373,  12.75333339,  25.79613956],
            [  7.09620297,   2.79257389,  11.15039854],
            [  8.933163  ,  19.39985815,  17.14915636]],
           [[ 80.00969553,  80.39810464,  76.08184429],
            [ 33.27611427,  24.38947838,  39.34919287],
            [  8.89425686,  11.05185138,  10.86767594]]])

.. code-block:: python

    sorted(colour.MSDS_TO_XYZ_METHODS)

.. code-block:: text

    ['ASTM E308', 'Integration', 'astm2015']

Blackbody Spectral Radiance Computation
***************************************

.. code-block:: python

    colour.sd_blackbody(5000)

.. code-block:: text

    SpectralDistribution([[  3.60000000e+02,   6.65427827e+12],
                          [  3.61000000e+02,   6.70960528e+12],
                          [  3.62000000e+02,   6.76482512e+12],
                          ...
                          [  7.78000000e+02,   1.06068004e+13],
                          [  7.79000000e+02,   1.05903327e+13],
                          [  7.80000000e+02,   1.05738520e+13]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={'right': None, 'method': 'Constant', 'left': None})

Dominant, Complementary Wavelength & Colour Purity Computation
**************************************************************

.. code-block:: python

    xy = [0.54369557, 0.32107944]
    xy_n = [0.31270000, 0.32900000]
    colour.dominant_wavelength(xy, xy_n)

.. code-block:: text

    (array(616.0),
     array([ 0.68354746,  0.31628409]),
     array([ 0.68354746,  0.31628409]))

Lightness Computation
*********************

.. code-block:: python

    colour.lightness(12.19722535)

.. code-block:: text

    41.527875844653451

.. code-block:: python

    sorted(colour.LIGHTNESS_METHODS)

.. code-block:: text

    ['Abebe 2017',
     'CIE 1976',
     'Fairchild 2010',
     'Fairchild 2011',
     'Glasser 1958',
     'Lstar1976',
     'Wyszecki 1963']

Luminance Computation
*********************

.. code-block:: python

    colour.luminance(41.52787585)

.. code-block:: text

    12.197225353400775

.. code-block:: python

    sorted(colour.LUMINANCE_METHODS)

.. code-block:: text

    ['ASTM D1535',
     'CIE 1976',
     'Fairchild 2010',
     'Fairchild 2011',
     'Newhall 1943',
     'astm2008',
     'cie1976']

Whiteness Computation
*********************

.. code-block:: python

    XYZ = [95.00000000, 100.00000000, 105.00000000]
    XYZ_0 = [94.80966767, 100.00000000, 107.30513595]
    colour.whiteness(XYZ, XYZ_0)

.. code-block:: text

    array([ 93.756     ,  -1.33000001])

.. code-block:: python

    sorted(colour.WHITENESS_METHODS)

.. code-block:: text

    ['ASTM E313',
     'Berger 1959',
     'CIE 2004',
     'Ganz 1979',
     'Stensby 1968',
     'Taube 1960',
     'cie2004']

Yellowness Computation
**********************

.. code-block:: python

    XYZ = [95.00000000, 100.00000000, 105.00000000]
    colour.yellowness(XYZ)

.. code-block:: text

    4.3400000000000034

.. code-block:: python

    sorted(colour.YELLOWNESS_METHODS)

.. code-block:: text

    ['ASTM D1925', 'ASTM E313', 'ASTM E313 Alternative']

Luminous Flux, Efficiency & Efficacy Computation
************************************************

.. code-block:: python

    sd = colour.SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    colour.luminous_flux(sd)

.. code-block:: text

    23807.655527367202

.. code-block:: python

    sd = colour.SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    colour.luminous_efficiency(sd)

.. code-block:: text

    0.19943935624521045

.. code-block:: python

    sd = colour.SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    colour.luminous_efficacy(sd)

.. code-block:: text

    136.21708031547874

Contrast Sensitivity Function - ``colour.contrast``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    colour.contrast_sensitivity_function(u=4, X_0=60, E=65)

.. code-block:: text

    358.51180789884984

.. code-block:: python

    sorted(colour.CONTRAST_SENSITIVITY_METHODS)

.. code-block:: text

    ['Barten 1999']

Colour Difference - ``colour.difference``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    Lab_1 = [100.00000000, 21.57210357, 272.22819350]
    Lab_2 = [100.00000000, 426.67945353, 72.39590835]
    colour.delta_E(Lab_1, Lab_2)

.. code-block:: text

    94.035649026659485

.. code-block:: python

    sorted(colour.DELTA_E_METHODS)

.. code-block:: text

    ['CAM02-LCD',
     'CAM02-SCD',
     'CAM02-UCS',
     'CAM16-LCD',
     'CAM16-SCD',
     'CAM16-UCS',
     'CIE 1976',
     'CIE 1994',
     'CIE 2000',
     'CMC',
     'DIN99',
     'ITP',
     'cie1976',
     'cie1994',
     'cie2000']

IO - ``colour.io``
~~~~~~~~~~~~~~~~~~

Images
******

.. code-block:: python

    RGB = colour.read_image("Ishihara_Colour_Blindness_Test_Plate_3.png")
    RGB.shape

.. code-block:: text

    (276, 281, 3)

Spectral Images - Fichet et al. (2021)
**************************************

.. code-block:: python

    components = colour.read_spectral_image_Fichet2021("Polarised.exr")
    list(components.keys())

.. code-block:: text

    ['S0', 'S1', 'S2', 'S3']

Look Up Table (LUT) Data
************************

.. code-block:: python

    LUT = colour.read_LUT("ACES_Proxy_10_to_ACES.cube")
    print(LUT)

.. code-block:: text

    LUT3x1D - ACES Proxy 10 to ACES
    -------------------------------
    Dimensions : 2
    Domain     : [[0 0 0]
                  [1 1 1]]
    Size       : (32, 3)

.. code-block:: python

    RGB = [0.17224810, 0.09170660, 0.06416938]
    LUT.apply(RGB)

.. code-block:: text

    array([ 0.00575674,  0.00181493,  0.00121419])

Colour Models - ``colour.models``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CIE xyY Colourspace
*******************

.. code-block:: python

    colour.XYZ_to_xyY([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.54369557,  0.32107944,  0.12197225])

CIE L*a*b* Colourspace
**********************

.. code-block:: python

    colour.XYZ_to_Lab([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 41.52787529,  52.63858304,  26.92317922])

CIE L*u*v* Colourspace
**********************

.. code-block:: python

    colour.XYZ_to_Luv([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 41.52787529,  96.83626054,  17.75210149])

CIE 1960 UCS Colourspace
************************

.. code-block:: python

    colour.XYZ_to_UCS([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.13769339,  0.12197225,  0.1053731 ])

CIE 1964 U*V*W* Colourspace
***************************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    colour.XYZ_to_UVW(XYZ)

.. code-block:: text

    array([ 94.55035725,  11.55536523,  40.54757405])

CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces - Luo, Cui and Li (2006)
*************************************************************************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    XYZ_w = [95.05, 100.00, 108.88]
    L_A = 318.31
    Y_b = 20.0
    surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]
    specification = colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
    JMh = (specification.J, specification.M, specification.h)
    colour.JMh_CIECAM02_to_CAM02UCS(JMh)

.. code-block:: text

    array([ 47.16899898,  38.72623785,  15.8663383 ])

.. code-block:: python

    XYZ = [0.20654008, 0.12197225, 0.05136952]
    XYZ_w = [95.05 / 100, 100.00 / 100, 108.88 / 100]
    colour.XYZ_to_CAM02UCS(XYZ, XYZ_w=XYZ_w, L_A=L_A, Y_b=Y_b)

.. code-block:: text

    array([ 47.16899898,  38.72623785,  15.8663383 ])

CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces - Li et al. (2017)
*******************************************************************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    XYZ_w = [95.05, 100.00, 108.88]
    L_A = 318.31
    Y_b = 20.0
    surround = colour.VIEWING_CONDITIONS_CAM16["Average"]
    specification = colour.XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)
    JMh = (specification.J, specification.M, specification.h)
    colour.JMh_CAM16_to_CAM16UCS(JMh)

.. code-block:: text

    array([ 46.55542238,  40.22460974,  14.25288392])

.. code-block:: python

    XYZ = [0.20654008, 0.12197225, 0.05136952]
    XYZ_w = [95.05 / 100, 100.00 / 100, 108.88 / 100]
    colour.XYZ_to_CAM16UCS(XYZ, XYZ_w=XYZ_w, L_A=L_A, Y_b=Y_b)

.. code-block:: text

    array([ 46.55542238,  40.22460974,  14.25288392])

DIN99 Colourspace and DIN99b, DIN99c, DIN99d Refined Formulas
*************************************************************

.. code-block:: python

    Lab = [41.52787529, 52.63858304, 26.92317922]
    colour.Lab_to_DIN99(Lab)

.. code-block:: text

    array([ 53.22821988,  28.41634656,   3.89839552])

ICaCb Colourspace
******************

.. code-block:: python

    XYZ_to_ICaCb(np.array([0.20654008, 0.12197225, 0.05136952]))

.. code-block:: text

    array([ 0.06875297,  0.05753352,  0.02081548])

IgPgTg Colourspace
******************

.. code-block:: python

    colour.XYZ_to_IgPgTg([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.42421258,  0.18632491,  0.10689223])

IPT Colourspace
***************

.. code-block:: python

    colour.XYZ_to_IPT([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.38426191,  0.38487306,  0.18886838])

Jzazbz Colourspace
******************

.. code-block:: python

    colour.XYZ_to_Jzazbz([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.00535048,  0.00924302,  0.00526007])

hdr-CIELAB Colourspace
**********************

.. code-block:: python

    colour.XYZ_to_hdr_CIELab([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 51.87002062,  60.4763385 ,  32.14551912])

hdr-IPT Colourspace
*******************

.. code-block:: python

    colour.XYZ_to_hdr_IPT([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 25.18261761, -22.62111297,   3.18511729])

Hunter L,a,b Colour Scale
*************************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    colour.XYZ_to_Hunter_Lab(XYZ)

.. code-block:: text

    array([ 34.92452577,  47.06189858,  14.38615107])

Hunter Rd,a,b Colour Scale
**************************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    colour.XYZ_to_Hunter_Rdab(XYZ)

.. code-block:: text

    array([ 12.197225  ,  57.12537874,  17.46241341])

Oklab Colourspace
*****************

.. code-block:: python

    colour.XYZ_to_Oklab([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.51634019,  0.154695  ,  0.06289579])

OSA UCS Colourspace
*******************

.. code-block:: python

    XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    colour.XYZ_to_OSA_UCS(XYZ)

.. code-block:: text

    array([-3.0049979 ,  2.99713697, -9.66784231])

ProLab Colourspace
******************

.. code-block:: python

    colour.XYZ_to_ProLab([0.51634019, 0.15469500, 0.06289579])

.. code-block:: text

    array([1.24610688, 2.39525236, 0.41902126])

Ragoo and Farup (2021) Optimised IPT Colourspace
************************************************

.. code-block:: python

    colour.XYZ_to_IPT_Ragoo2021([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.42248243,  0.2910514 ,  0.20410663])

Yrg Colourspace - Kirk (2019)
*****************************

.. code-block:: python

    colour.XYZ_to_Yrg([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    array([ 0.13137801,  0.49037645,  0.37777388])

Y'CbCr Colour Encoding
**********************

.. code-block:: python

    colour.RGB_to_YCbCr([1.0, 1.0, 1.0])

.. code-block:: text

    array([ 0.92156863,  0.50196078,  0.50196078])

YCoCg Colour Encoding
*********************

.. code-block:: python

    colour.RGB_to_YCoCg([0.75, 0.75, 0.0])

.. code-block:: text

    array([ 0.5625,  0.375 ,  0.1875])

ICtCp Colour Encoding
*********************

.. code-block:: python

    colour.RGB_to_ICtCp([0.45620519, 0.03081071, 0.04091952])

.. code-block:: text

    array([ 0.07351364,  0.00475253,  0.09351596])

HSV Colourspace
***************

.. code-block:: python

    colour.RGB_to_HSV([0.45620519, 0.03081071, 0.04091952])

.. code-block:: text

    array([ 0.99603944,  0.93246304,  0.45620519])

IHLS Colourspace
****************

.. code-block:: python

    colour.RGB_to_IHLS([0.45620519, 0.03081071, 0.04091952])

.. code-block:: text

    array([ 6.26236117,  0.12197943,  0.42539448])

Prismatic Colourspace
*********************

.. code-block:: python

    colour.RGB_to_Prismatic([0.25, 0.50, 0.75])

.. code-block:: text

    array([ 0.75      ,  0.16666667,  0.33333333,  0.5       ])

RGB Colourspace and Transformations
***********************************

.. code-block:: python

    XYZ = [0.21638819, 0.12570000, 0.03847493]
    illuminant_XYZ = [0.34570, 0.35850]
    illuminant_RGB = [0.31270, 0.32900]
    chromatic_adaptation_transform = "Bradford"
    matrix_XYZ_to_RGB = [
        [3.24062548, -1.53720797, -0.49862860],
        [-0.96893071, 1.87575606, 0.04151752],
        [0.05571012, -0.20402105, 1.05699594],
    ]
    colour.XYZ_to_RGB(
        XYZ,
        illuminant_XYZ,
        illuminant_RGB,
        matrix_XYZ_to_RGB,
        chromatic_adaptation_transform,
    )

.. code-block:: text

    array([ 0.45595571,  0.03039702,  0.04087245])

RGB Colourspace Derivation
**************************

.. code-block:: python

    p = [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]
    w = [0.32168, 0.33767]
    colour.normalised_primary_matrix(p, w)

.. code-block:: text

    array([[  9.52552396e-01,   0.00000000e+00,   9.36786317e-05],
           [  3.43966450e-01,   7.28166097e-01,  -7.21325464e-02],
           [  0.00000000e+00,   0.00000000e+00,   1.00882518e+00]])

RGB Colourspaces
****************

.. code-block:: python

    sorted(colour.RGB_COLOURSPACES)

.. code-block:: text

    ['ACES2065-1',
     'ACEScc',
     'ACEScct',
     'ACEScg',
     'ACESproxy',
     'ARRI Wide Gamut 3',
     'ARRI Wide Gamut 4',
     'Adobe RGB (1998)',
     'Adobe Wide Gamut RGB',
     'Apple RGB',
     'Best RGB',
     'Beta RGB',
     'Blackmagic Wide Gamut',
     'CIE RGB',
     'Cinema Gamut',
     'ColorMatch RGB',
     'DCDM XYZ',
     'DCI-P3',
     'DCI-P3-P',
     'DJI D-Gamut',
     'DRAGONcolor',
     'DRAGONcolor2',
     'DaVinci Wide Gamut',
     'Display P3',
     'Don RGB 4',
     'EBU Tech. 3213-E',
     'ECI RGB v2',
     'ERIMM RGB',
     'Ekta Space PS 5',
     'F-Gamut',
     'FilmLight E-Gamut',
     'ITU-R BT.2020',
     'ITU-R BT.470 - 525',
     'ITU-R BT.470 - 625',
     'ITU-R BT.709',
     'ITU-T H.273 - 22 Unspecified',
     'ITU-T H.273 - Generic Film',
     'Max RGB',
     'N-Gamut',
     'NTSC (1953)',
     'NTSC (1987)',
     'P3-D65',
     'PLASA ANSI E1.54',
     'Pal/Secam',
     'ProPhoto RGB',
     'Protune Native',
     'REDWideGamutRGB',
     'REDcolor',
     'REDcolor2',
     'REDcolor3',
     'REDcolor4',
     'RIMM RGB',
     'ROMM RGB',
     'Russell RGB',
     'S-Gamut',
     'S-Gamut3',
     'S-Gamut3.Cine',
     'SMPTE 240M',
     'SMPTE C',
     'Sharp RGB',
     'V-Gamut',
     'Venice S-Gamut3',
     'Venice S-Gamut3.Cine',
     'Xtreme RGB',
     'aces',
     'adobe1998',
     'prophoto',
     'sRGB']


OETFs
*****

.. code-block:: python

    sorted(colour.OETFS)

.. code-block:: text

    ['ARIB STD-B67',
     'Blackmagic Film Generation 5',
     'DaVinci Intermediate',
     'ITU-R BT.2020',
     'ITU-R BT.2100 HLG',
     'ITU-R BT.2100 PQ',
     'ITU-R BT.601',
     'ITU-R BT.709',
     'ITU-T H.273 IEC 61966-2',
     'ITU-T H.273 Log',
     'ITU-T H.273 Log Sqrt',
     'SMPTE 240M']


EOTFs
*****

.. code-block:: python

    sorted(colour.EOTFS)

.. code-block:: text

    ['DCDM',
     'DICOM GSDF',
     'ITU-R BT.1886',
     'ITU-R BT.2100 HLG',
     'ITU-R BT.2100 PQ',
     'ITU-T H.273 ST.428-1',
     'SMPTE 240M',
     'ST 2084',
     'sRGB']

OOTFs
*****

.. code-block:: python

    sorted(colour.OOTFS)

.. code-block:: text

    ['ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ']


Log Encoding / Decoding
***********************

.. code-block:: python

    sorted(colour.LOG_ENCODINGS)

.. code-block:: text

    ['ACEScc',
     'ACEScct',
     'ACESproxy',
     'ARRI LogC3',
     'ARRI LogC4',
     'Canon Log',
     'Canon Log 2',
     'Canon Log 3',
     'Cineon',
     'D-Log',
     'ERIMM RGB',
     'F-Log',
     'F-Log2',
     'Filmic Pro 6',
     'L-Log',
     'Log2',
     'Log3G10',
     'Log3G12',
     'N-Log',
     'PLog',
     'Panalog',
     'Protune',
     'REDLog',
     'REDLogFilm',
     'S-Log',
     'S-Log2',
     'S-Log3',
     'T-Log',
     'V-Log',
     'ViperLog']

CCTFs Encoding / Decoding
*************************

.. code-block:: python

    sorted(colour.CCTF_ENCODINGS)

.. code-block:: text

    ['ACEScc',
     'ACEScct',
     'ACESproxy',
     'Apple Log Profile',
     'ARRI LogC3',
     'ARRI LogC4',
     'ARIB STD-B67',
     'Canon Log',
     'Canon Log 2',
     'Canon Log 3',
     'Cineon',
     'D-Log',
     'DCDM',
     'DICOM GSDF',
     'ERIMM RGB',
     'F-Log',
     'F-Log2',
     'Filmic Pro 6',
     'Gamma 2.2',
     'Gamma 2.4',
     'Gamma 2.6',
     'ITU-R BT.1886',
     'ITU-R BT.2020',
     'ITU-R BT.2100 HLG',
     'ITU-R BT.2100 PQ',
     'ITU-R BT.601',
     'ITU-R BT.709',
     'Log2',
     'Log3G10',
     'Log3G12',
     'PLog',
     'Panalog',
     'ProPhoto RGB',
     'Protune',
     'REDLog',
     'REDLogFilm',
     'RIMM RGB',
     'ROMM RGB',
     'S-Log',
     'S-Log2',
     'S-Log3',
     'SMPTE 240M',
     'ST 2084',
     'T-Log',
     'V-Log',
     'ViperLog',
     'sRGB']

Recommendation ITU-T H.273 Code points for Video Signal Type Identification
***************************************************************************

.. code-block:: python

    colour.COLOUR_PRIMARIES_ITUTH273.keys()

.. code-block:: text

    dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23])

.. code-block:: python

    colour.models.describe_video_signal_colour_primaries(1)

.. code-block:: text

    ===============================================================================
    *                                                                             *
    *   Colour Primaries: 1                                                       *
    *   -------------------                                                       *
    *                                                                             *
    *   Primaries        : [[ 0.64  0.33]                                         *
    *                       [ 0.3   0.6 ]                                         *
    *                       [ 0.15  0.06]]                                        *
    *   Whitepoint       : [ 0.3127  0.329 ]                                      *
    *   Whitepoint Name  : D65                                                    *
    *   NPM              : [[ 0.4123908   0.35758434  0.18048079]                 *
    *                       [ 0.21263901  0.71516868  0.07219232]                 *
    *                       [ 0.01933082  0.11919478  0.95053215]]                *
    *   NPM -1           : [[ 3.24096994 -1.53738318 -0.49861076]                 *
    *                       [-0.96924364  1.8759675   0.04155506]                 *
    *                       [ 0.05563008 -0.20397696  1.05697151]]                *
    *   FFmpeg Constants : ['AVCOL_PRI_BT709', 'BT709']                           *
    *                                                                             *
    ===============================================================================

.. code-block:: python

    colour.TRANSFER_CHARACTERISTICS_ITUTH273.keys()

.. code-block:: text

    dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

.. code-block:: python

    colour.models.describe_video_signal_transfer_characteristics(1)

.. code-block:: text

    ===============================================================================
    *                                                                             *
    *   Transfer Characteristics: 1                                               *
    *   ---------------------------                                               *
    *                                                                             *
    *   Function         : <function oetf_BT709 at 0x165bb3550>                   *
    *   FFmpeg Constants : ['AVCOL_TRC_BT709', 'BT709']                           *
    *                                                                             *
    ===============================================================================

.. code-block:: python

    colour.MATRIX_COEFFICIENTS_ITUTH273.keys()

.. code-block:: text

    dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

.. code-block:: python

    colour.models.describe_video_signal_matrix_coefficients(1)

.. code-block:: text

    ===============================================================================
    *                                                                             *
    *   Matrix Coefficients: 1                                                    *
    *   ----------------------                                                    *
    *                                                                             *
    *   Matrix Coefficients : [ 0.2126  0.0722]                                   *
    *   FFmpeg Constants    : ['AVCOL_SPC_BT709', 'BT709']                        *
    *                                                                             *
    ===============================================================================

Colour Notation Systems - ``colour.notation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Munsell Value
*************

.. code-block:: python

    colour.munsell_value(12.23634268)

.. code-block:: text

    4.0824437076525664

.. code-block:: python

    sorted(colour.MUNSELL_VALUE_METHODS)

.. code-block:: text

    ['ASTM D1535',
     'Ladd 1955',
     'McCamy 1987',
     'Moon 1943',
     'Munsell 1933',
     'Priest 1920',
     'Saunderson 1944',
     'astm2008']

Munsell Colour
**************

.. code-block:: python

    colour.xyY_to_munsell_colour([0.38736945, 0.35751656, 0.59362000])

.. code-block:: text

    '4.2YR 8.1/5.3'

.. code-block:: python

    colour.munsell_colour_to_xyY("4.2YR 8.1/5.3")

.. code-block:: text

    array([ 0.38736945,  0.35751656,  0.59362   ])

Optical Phenomena - ``colour.phenomena``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    colour.rayleigh_scattering_sd()

.. code-block:: text

    SpectralDistribution([[  3.60000000e+02,   5.99101337e-01],
                          [  3.61000000e+02,   5.92170690e-01],
                          [  3.62000000e+02,   5.85341006e-01],
                          ...
                          [  7.78000000e+02,   2.55208377e-02],
                          [  7.79000000e+02,   2.53887969e-02],
                          [  7.80000000e+02,   2.52576106e-02]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={'right': None, 'method': 'Constant', 'left': None})

Light Quality - ``colour.quality``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Colour Fidelity Index
*********************

.. code-block:: python

    colour.colour_fidelity_index(colour.SDS_ILLUMINANTS["FL2"])

.. code-block:: text

    70.120825477833037

.. code-block:: python

    sorted(colour.COLOUR_FIDELITY_INDEX_METHODS)

.. code-block:: text

    ['ANSI/IES TM-30-18', 'CIE 2017']

Colour Quality Scale
********************

.. code-block:: python

    colour.colour_quality_scale(colour.SDS_ILLUMINANTS["FL2"])

.. code-block:: text

    64.111703163816699

.. code-block:: python

    sorted(colour.COLOUR_QUALITY_SCALE_METHODS)

.. code-block:: text

    ['NIST CQS 7.4', 'NIST CQS 9.0']

Colour Rendering Index
**********************

.. code-block:: python

    colour.colour_rendering_index(colour.SDS_ILLUMINANTS["FL2"])

.. code-block:: text

    64.233724121664807

Academy Spectral Similarity Index (SSI)
***************************************

.. code-block:: python

    colour.spectral_similarity_index(
        colour.SDS_ILLUMINANTS["C"], colour.SDS_ILLUMINANTS["D65"]
    )

.. code-block:: text

    94.0

Spectral Up-Sampling & Recovery - ``colour.recovery``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reflectance Recovery
********************

.. code-block:: python

    colour.XYZ_to_sd([0.20654008, 0.12197225, 0.05136952])

.. code-block:: text

    SpectralDistribution([[  3.60000000e+02,   8.40144095e-02],
                          [  3.65000000e+02,   8.41264236e-02],
                          [  3.70000000e+02,   8.40057597e-02],
                          ...
                          [  7.70000000e+02,   4.46743012e-01],
                          [  7.75000000e+02,   4.46817187e-01],
                          [  7.80000000e+02,   4.46857696e-01]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})

.. code-block:: python

    sorted(colour.REFLECTANCE_RECOVERY_METHODS)

.. code-block:: text

    ['Jakob 2019', 'Mallett 2019', 'Meng 2015', 'Otsu 2018', 'Smits 1999']

Camera RGB Sensitivities Recovery
*********************************

.. code-block:: python

    illuminant = colour.colorimetry.SDS_ILLUMINANTS["D65"]
    sensitivities = colour.characterisation.MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]
    reflectances = [
        sd.copy().align(colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017)
        for sd in colour.characterisation.SDS_COLOURCHECKERS["BabelColor Average"].values()
    ]
    reflectances = colour.colorimetry.sds_and_msds_to_msds(reflectances)
    RGB = colour.colorimetry.msds_to_XYZ(
        reflectances,
        method="Integration",
        cmfs=sensitivities,
        illuminant=illuminant,
        k=0.01,
        shape=colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    )
    colour.recovery.RGB_to_msds_camera_sensitivities_Jiang2013(
        RGB,
        illuminant,
        reflectances,
        colour.recovery.BASIS_FUNCTIONS_DYER2017,
        colour.recovery.SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    )

.. code-block:: text

    RGB_CameraSensitivities([[  4.00000000e+02,   7.22815777e-03,   9.22506480e-03,
                               -9.88368972e-03],
                             [  4.10000000e+02,  -8.50457609e-03,   1.12777480e-02,
                                3.86248655e-03],
                             [  4.20000000e+02,   4.58191132e-02,   7.15520948e-02,
                                4.04068293e-01],
                             ...
                             [  6.80000000e+02,   4.08276173e-02,   5.55290476e-03,
                                1.39907862e-03],
                             [  6.90000000e+02,  -3.71437574e-03,   2.50935640e-03,
                                3.97652622e-04],
                             [  7.00000000e+02,  -5.62256563e-03,   1.56433970e-03,
                                5.84726936e-04]],
                            ['red', 'green', 'blue'],
                            SpragueInterpolator,
                            {},
                            Extrapolator,
                            {'method': 'Constant', 'left': None, 'right': None})

Correlated Colour Temperature Computation Methods - ``colour.temperature``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    colour.uv_to_CCT([0.1978, 0.3122])

.. code-block:: text

    array([  6.50751282e+03,   3.22335875e-03])

.. code-block:: python

    sorted(colour.UV_TO_CCT_METHODS)

.. code-block:: text

    ['Krystek 1985', 'Ohno 2013', 'Planck 1900', 'Robertson 1968', 'ohno2013', 'robertson1968']

.. code-block:: python

    sorted(colour.XY_TO_CCT_METHODS)

.. code-block:: text

    ['CIE Illuminant D Series',
     'Hernandez 1999',
     'Kang 2002',
     'McCamy 1992',
     'daylight',
     'hernandez1999',
     'kang2002',
     'mccamy1992']

Colour Volume - ``colour.volume``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    colour.RGB_colourspace_volume_MonteCarlo(colour.RGB_COLOURSPACE_RGB["sRGB"])

.. code-block:: text

    821958.30000000005

Geometry Primitives Generation - ``colour.geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    colour.primitive("Grid")

.. code-block:: text

 (array([ ([-0.5,  0.5,  0. ], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  0.,  1.]),
           ([ 0.5,  0.5,  0. ], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  0.,  1.]),
           ([-0.5, -0.5,  0. ], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  0.,  1.]),
           ([ 0.5, -0.5,  0. ], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  0.,  1.])],
          dtype=[('position', '<f4', (3,)), ('uv', '<f4', (2,)), ('normal', '<f4', (3,)), ('colour', '<f4', (4,))]), array([[0, 2, 1],
           [2, 3, 1]], dtype=uint32), array([[0, 2],
           [2, 3],
           [3, 1],
           [1, 0]], dtype=uint32))

.. code-block:: python

    sorted(colour.PRIMITIVE_METHODS)

.. code-block:: text

    ['Cube', 'Grid']

.. code-block:: python

    colour.primitive_vertices("Quad MPL")

.. code-block:: text

    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.],
           [ 0.,  1.,  0.]])
    sorted(colour.PRIMITIVE_VERTICES_METHODS)
    ['Cube MPL', 'Grid MPL', 'Quad MPL', 'Sphere']

Plotting - ``colour.plotting``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the objects are available from the ``colour.plotting`` namespace:

.. code-block:: python

    from colour.plotting import *

    colour_style()

Visible Spectrum
****************

.. code-block:: python

    plot_visible_spectrum("CIE 1931 2 Degree Standard Observer")

..  image:: _static/Examples_Plotting_Visible_Spectrum.png

Spectral Distribution
*********************

.. code-block:: python

    plot_single_illuminant_sd("FL1")

..  image:: _static/Examples_Plotting_Illuminant_F1_SD.png

Blackbody
*********

.. code-block:: python

    blackbody_sds = [
        colour.sd_blackbody(i, colour.SpectralShape(0, 10000, 10))
        for i in range(1000, 15000, 1000)
    ]
    plot_multi_sds(
        blackbody_sds,
        y_label="W / (sr m$^2$) / m",
        plot_kwargs={"use_sd_colours": True, "normalise_sd_colours": True},
        legend_location="upper right",
        bounding_box=(0, 1250, 0, 2.5e6),
    )
..  image:: _static/Examples_Plotting_Blackbodies.png

Colour Matching Functions
*************************

.. code-block:: python

    plot_single_cmfs(
        "Stockman & Sharpe 2 Degree Cone Fundamentals",
        y_label="Sensitivity",
        bounding_box=(390, 870, 0, 1.1),
    )

..  image:: _static/Examples_Plotting_Cone_Fundamentals.png

Luminous Efficiency
*******************

.. code-block:: python

    sd_mesopic_luminous_efficiency_function = (
        colour.sd_mesopic_luminous_efficiency_function(0.2)
    )
    plot_multi_sds(
        (
            sd_mesopic_luminous_efficiency_function,
            colour.PHOTOPIC_LEFS["CIE 1924 Photopic Standard Observer"],
            colour.SCOTOPIC_LEFS["CIE 1951 Scotopic Standard Observer"],
        ),
        y_label="Luminous Efficiency",
        legend_location="upper right",
        y_tighten=True,
        margins=(0, 0, 0, 0.1),
    )

..  image:: _static/Examples_Plotting_Luminous_Efficiency.png

Colour Checker
**************

.. code-block:: python

    from colour.characterisation.dataset.colour_checkers.sds import (
        COLOURCHECKER_INDEXES_TO_NAMES_MAPPING,
    )

    plot_multi_sds(
        [
            colour.SDS_COLOURCHECKERS["BabelColor Average"][value]
            for key, value in sorted(COLOURCHECKER_INDEXES_TO_NAMES_MAPPING.items())
        ],
        plot_kwargs={
            "use_sd_colours": True,
        },
        title=("BabelColor Average - " "Spectral Distributions"),
    )

..  image:: _static/Examples_Plotting_BabelColor_Average.png

.. code-block:: python

    plot_single_colour_checker("ColorChecker 2005", text_kwargs={"visible": False})

..  image:: _static/Examples_Plotting_ColorChecker_2005.png

Chromaticities Prediction
*************************

.. code-block:: python

    plot_corresponding_chromaticities_prediction(2, "Von Kries", "Bianco 2010")

..  image:: _static/Examples_Plotting_Chromaticities_Prediction.png

Chromaticities
**************

.. code-block:: python

    import numpy as np

    RGB = np.random.random((32, 32, 3))
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
        RGB,
        "ITU-R BT.709",
        colourspaces=["ACEScg", "S-Gamut"],
        show_pointer_gamut=True,
    )

..  image:: _static/Examples_Plotting_Chromaticities_CIE_1931_Chromaticity_Diagram.png

Colour Rendering Index Bars
***************************

.. code-block:: python

    plot_single_sd_colour_rendering_index_bars(colour.SDS_ILLUMINANTS["FL2"])

..  image:: _static/Examples_Plotting_CRI.png

ANSI/IES TM-30-18 Colour Rendition Report
*****************************************

.. code-block:: python

    plot_single_sd_colour_rendition_report(colour.SDS_ILLUMINANTS["FL2"])

..  image:: _static/Examples_Plotting_Colour_Rendition_Report.png

Gamut Section
*************

.. code-block:: python

    plot_visible_spectrum_section(section_colours="RGB", section_opacity=0.15)

..  image:: _static/Examples_Plotting_Plot_Visible_Spectrum_Section.png

.. code-block:: python

    plot_RGB_colourspace_section("sRGB", section_colours="RGB", section_opacity=0.15)

..  image:: _static/Examples_Plotting_Plot_RGB_Colourspace_Section.png

Colour Temperature
******************

.. code-block:: python

    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(["A", "B", "C"])

..  image:: _static/Examples_Plotting_CCT_CIE_1960_UCS_Chromaticity_Diagram.png

User Guide
----------

.. toctree::
    :maxdepth: 2

    user-guide

API Reference
-------------

.. toctree::
    :maxdepth: 2

    reference

See Also
--------

Software
~~~~~~~~

**Python**

- `ColorAide <https://facelessuser.github.io/coloraide>`__ by Muse, I.
- `ColorPy <http://markkness.net/colorpy/ColorPy.html>`__ by Kness, M.
- `Colorspacious <https://colorspacious.readthedocs.io>`__ by Smith, N. J., et al.
- `python-colormath <https://python-colormath.readthedocs.io>`__ by Taylor, G., et al.

**Go**

- `go-colorful <https://github.com/lucasb-eyer/go-colorful>`__  by Beyer, L., et al.

**.NET**

- `Colourful <https://github.com/tompazourek/Colourful>`__ by Pažourek, T., et al.

**Julia**

- `Colors.jl <https://github.com/JuliaGraphics/Colors.jl>`__ by Holy, T., et al.

**Matlab & Octave**

- `COLORLAB <https://www.uv.es/vista/vistavalencia/software/colorlab.html>`__ by Malo, J., et al.
- `Psychtoolbox <http://psychtoolbox.org>`__ by Brainard, D., et al.
- `The Munsell and Kubelka-Munk Toolbox <http://www.munsellcolourscienceforpainters.com/MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html>`__ by Centore, P.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`__,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct>`__ page.

Contact & Social
----------------

The *Colour Developers* can be reached via different means:

- `Email <mailto:colour-developers@colour-science.org>`__
- `Facebook <https://www.facebook.com/python.colour.science>`__
- `Github Discussions <https://github.com/colour-science/colour/discussions>`__
- `Gitter <https://gitter.im/colour-science/colour>`__
- `Twitter <https://twitter.com/colour_science>`__

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
