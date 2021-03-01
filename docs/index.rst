..  image:: https://raw.githubusercontent.com/colour-science/colour-branding/master/images/Colour_Logo_Medium_001.png

`Colour <https://github.com/colour-science/colour>`__ is an open-source
`Python <https://www.python.org/>`__ package providing a comprehensive number
of algorithms and datasets for colour science.

It is freely available under the
`New BSD License <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

**Colour** is an affiliated project of `NumFOCUS <https://numfocus.org/>`__, a
501(c)(3) nonprofit in the United States.

.. contents:: Table of Contents
    :local:
    :depth: 3

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

**Colour** features a rich dataset and collection of objects, please see the
`features <https://www.colour-science.org/features/>`__ page for more
information.

Installation
------------

**Colour** and its primary dependencies can be easily installed from the
`Python Package Index <https://pypi.org/project/colour-science/>`__
by issuing this command in a shell:

.. code-block:: bash

    $ pip install --user colour-science

The detailed installation procedure for the secondary dependencies is
described in the `Installation Guide <https://www.colour-science.org/installation-guide/>`__.

**Colour** is also available for `Anaconda <https://www.continuum.io/downloads>`__
from *Continuum Analytics* via `conda-forge <https://conda-forge.org/>`__:

.. code-block:: bash

    $ conda install -c conda-forge colour-science

Documentation
-------------

Tutorial
~~~~~~~~

The `static tutorial <https://colour.readthedocs.io/en/develop/tutorial.html>`__
provides an introduction to **Colour**. An interactive version is available via
`Google Colab <https://colab.research.google.com/notebook#fileId=1Im9J7or9qyClQCv5sPHmKdyiQbG4898K&offline=true&sandboxMode=true>`__.

How-To Guide
~~~~~~~~~~~~

The `How-To <https://colab.research.google.com/notebook#fileId=1NRcdXSCshivkwoU2nieCvC3y14fx1X4X&offline=true&sandboxMode=true>`__
guide for **Colour** shows various techniques to solve specific problems and
highlights some interesting use cases.

API Reference
~~~~~~~~~~~~~

The main technical reference for **Colour** and its API is the
`Colour Manual <https://colour.readthedocs.io/en/latest/manual.html>`__.

.. toctree::
    :maxdepth: 4

    manual

Examples
~~~~~~~~

Most of the objects are available from the ``colour`` namespace:

.. code-block:: python

    >>> import colour

Automatic Colour Conversion Graph - ``colour.graph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with version *0.3.14*, **Colour** implements an automatic colour
conversion graph enabling easier colour conversions.

..  image:: https://colour.readthedocs.io/en/develop/_static/Examples_Colour_Automatic_Conversion_Graph.png

.. code-block:: python

    >>> sd = colour.SDS_COLOURCHECKERS['ColorChecker N Ohta']['dark skin']
    >>> colour.convert(sd, 'Spectral Distribution', 'sRGB', verbose={'mode': 'Short'})

::

    ===============================================================================
    *                                                                             *
    *   [ Conversion Path ]                                                       *
    *                                                                             *
    *   "sd_to_XYZ" --> "XYZ_to_sRGB"                                             *
    *                                                                             *
    ===============================================================================
    array([ 0.45675795,  0.30986982,  0.24861924])

.. code-block:: python

    >>> illuminant = colour.SDS_ILLUMINANTS['FL2']
    >>> colour.convert(sd, 'Spectral Distribution', 'sRGB', sd_to_XYZ={'illuminant': illuminant})
    array([ 0.47924575,  0.31676968,  0.17362725])

Chromatic Adaptation - ``colour.adaptation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> XYZ = [0.20654008, 0.12197225, 0.05136952]
    >>> D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    >>> A = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['A']
    >>> colour.chromatic_adaptation(
    ...     XYZ, colour.xy_to_XYZ(D65), colour.xy_to_XYZ(A))
    array([ 0.2533053 ,  0.13765138,  0.01543307])
    >>> sorted(colour.CHROMATIC_ADAPTATION_METHODS)
    ['CIE 1994', 'CMCCAT2000', 'Fairchild 1990', 'Von Kries']

Algebra - ``colour.algebra``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kernel Interpolation
********************

.. code-block:: python

    >>> y = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
    >>> x = range(len(y))
    >>> colour.KernelInterpolator(x, y)([0.25, 0.75, 5.50])
    array([  6.18062083,   8.08238488,  57.85783403])

Sprague (1880) Interpolation
****************************

.. code-block:: python

    >>> y = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
    >>> x = range(len(y))
    >>> colour.SpragueInterpolator(x, y)([0.25, 0.75, 5.50])
    array([  6.72951612,   7.81406251,  43.77379185])

Colour Appearance Models - ``colour.appearance``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> XYZ_w = [95.05, 100.00, 108.88]
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> colour.XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b)
    CAM_Specification_CIECAM02(J=34.434525727858997, C=67.365010921125915, h=22.279164147957076, s=62.814855853327131, Q=177.47124941102123, M=70.024939419291385, H=2.689608534423904, HC=None)

Colour Blindness - ``colour.blindness``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import numpy as np
    >>> cmfs = colour.LMS_CMFS['Stockman & Sharpe 2 Degree Cone Fundamentals']
    >>> colour.msds_cmfs_anomalous_trichromacy_Machado2009(cmfs, np.array([15, 0, 0]))[450]
    array([ 0.08912884,  0.0870524 ,  0.955393  ])
    >>> primaries = colour.MSDS_DISPLAY_PRIMARIES['Apple Studio Display']
    >>> d_LMS = (15, 0, 0)
    >>> colour.matrix_anomalous_trichromacy_Machado2009(cmfs, primaries, d_LMS)
    array([[-0.27774652,  2.65150084, -1.37375432],
           [ 0.27189369,  0.20047862,  0.52762768],
           [ 0.00644047,  0.25921579,  0.73434374]])

Colour Correction - ``colour characterisation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import numpy as np
    >>> RGB = [0.17224810, 0.09170660, 0.06416938]
    >>> M_T = np.random.random((24, 3))
    >>> M_R = M_T + (np.random.random((24, 3)) - 0.5) * 0.5
    >>> colour.colour_correction(RGB, M_T, M_R)
    array([ 0.1806237 ,  0.07234791,  0.07848845])
    >>> sorted(colour.COLOUR_CORRECTION_METHODS)
    ['Cheung 2004', 'Finlayson 2015', 'Vandermonde']

ACES Input Transform - ``colour characterisation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> sensitivities = colour.MSDS_CAMERA_SENSITIVITIES['Nikon 5100 (NPL)']
    >>> illuminant = colour.SDS_ILLUMINANTS['D55']
    >>> colour.matrix_idt(sensitivities, illuminant)
    array([[ 0.46579991,  0.13409239,  0.01935141],
           [ 0.01786094,  0.77557292, -0.16775555],
           [ 0.03458652, -0.16152926,  0.74270359]])

Colorimetry - ``colour.colorimetry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spectral Computations
*********************

.. code-block:: python

    >>> colour.sd_to_XYZ(colour.SDS_LIGHT_SOURCES['Neodimium Incandescent'])
    array([ 36.94726204,  32.62076174,  13.0143849 ])
    >>> sorted(colour.SPECTRAL_TO_XYZ_METHODS)
    ['ASTM E308', 'Integration', 'astm2015']


Multi-Spectral Computations
***************************

.. code-block:: python

    >>> msds = np.array([
    ...     [[0.01367208, 0.09127947, 0.01524376, 0.02810712, 0.19176012, 0.04299992],
    ...      [0.00959792, 0.25822842, 0.41388571, 0.22275120, 0.00407416, 0.37439537],
    ...      [0.01791409, 0.29707789, 0.56295109, 0.23752193, 0.00236515, 0.58190280]],
    ...     [[0.01492332, 0.10421912, 0.02240025, 0.03735409, 0.57663846, 0.32416266],
    ...      [0.04180972, 0.26402685, 0.03572137, 0.00413520, 0.41808194, 0.24696727],
    ...      [0.00628672, 0.11454948, 0.02198825, 0.39906919, 0.63640803, 0.01139849]],
    ...     [[0.04325933, 0.26825359, 0.23732357, 0.05175860, 0.01181048, 0.08233768],
    ...      [0.02484169, 0.12027161, 0.00541695, 0.00654612, 0.18603799, 0.36247808],
    ...      [0.03102159, 0.16815442, 0.37186235, 0.08610666, 0.00413520, 0.78492409]],
    ...     [[0.11682307, 0.78883040, 0.74468607, 0.83375293, 0.90571451, 0.70054168],
    ...      [0.06321812, 0.41898224, 0.15190357, 0.24591440, 0.55301750, 0.00657664],
    ...      [0.00305180, 0.11288624, 0.11357290, 0.12924391, 0.00195315, 0.21771573]],
    ... ])
    >>> colour.msds_to_XYZ(msds, method='Integration',
    ...                    shape=colour.SpectralShape(400, 700, 60))
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
    >>> sorted(colour.MSDS_TO_XYZ_METHODS)
    ['ASTM E308', 'Integration', 'astm2015']

Blackbody Spectral Radiance Computation
***************************************

.. code-block:: python

    >>> colour.sd_blackbody(5000)
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

    >>> xy = [0.54369557, 0.32107944]
    >>> xy_n = [0.31270000, 0.32900000]
    >>> colour.dominant_wavelength(xy, xy_n)
    (array(616.0),
     array([ 0.68354746,  0.31628409]),
     array([ 0.68354746,  0.31628409]))

Lightness Computation
*********************

.. code-block:: python

    >>> colour.lightness(12.19722535)
    41.527875844653451
    >>> sorted(colour.LIGHTNESS_METHODS)
    ['CIE 1976',
     'Fairchild 2010',
     'Fairchild 2011',
     'Glasser 1958',
     'Lstar1976',
     'Wyszecki 1963']

Luminance Computation
*********************

.. code-block:: python

    >>> colour.luminance(41.52787585)
    12.197225353400775
    >>> sorted(colour.LUMINANCE_METHODS)
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

    >>> XYZ = [95.00000000, 100.00000000, 105.00000000]
    >>> XYZ_0 = [94.80966767, 100.00000000, 107.30513595]
    >>> colour.whiteness(XYZ, XYZ_0)
    array([ 93.756     ,  -1.33000001])
    >>> sorted(colour.WHITENESS_METHODS)
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

    >>> XYZ = [95.00000000, 100.00000000, 105.00000000]
    >>> colour.yellowness(XYZ)
    4.3400000000000034
    >>> sorted(colour.YELLOWNESS_METHODS)
    ['ASTM D1925', 'ASTM E313', 'ASTM E313 Alternative']

Luminous Flux, Efficiency & Efficacy Computation
************************************************

.. code-block:: python

    >>> sd = colour.SDS_LIGHT_SOURCES['Neodimium Incandescent']
    >>> colour.luminous_flux(sd)
    23807.655527367202
    >>> sd = colour.SDS_LIGHT_SOURCES['Neodimium Incandescent']
    >>> colour.luminous_efficiency(sd)
    0.19943935624521045
    >>> sd = colour.SDS_LIGHT_SOURCES['Neodimium Incandescent']
    >>> colour.luminous_efficacy(sd)
    136.21708031547874

Contrast Sensitivity Function - ``colour.contrast``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.contrast_sensitivity_function(u=4, X_0=60, E=65)
    358.51180789884984
    >>> sorted(colour.CONTRAST_SENSITIVITY_METHODS)
    ['Barten 1999']


Colour Difference - ``colour.difference``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> Lab_1 = [100.00000000, 21.57210357, 272.22819350]
    >>> Lab_2 = [100.00000000, 426.67945353, 72.39590835]
    >>> colour.delta_E(Lab_1, Lab_2)
    94.035649026659485
    >>> sorted(colour.DELTA_E_METHODS)
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
     'cie1976',
     'cie1994',
     'cie2000']

IO - ``colour.io``
^^^^^^^^^^^^^^^^^^

Images
******

.. code-block:: python

    >>> RGB = colour.read_image('Ishihara_Colour_Blindness_Test_Plate_3.png')
    >>> RGB.shape
    (276, 281, 3)

Look Up Table (LUT) Data
************************

.. code-block:: python

    >>> LUT = colour.read_LUT('ACES_Proxy_10_to_ACES.cube')
    >>> print(LUT)

::

    LUT3x1D - ACES Proxy 10 to ACES
    -------------------------------
    Dimensions : 2
    Domain     : [[0 0 0]
                  [1 1 1]]
    Size       : (32, 3)

.. code-block:: python

    >>> RGB = [0.17224810, 0.09170660, 0.06416938]
    >>> LUT.apply(RGB)
    array([ 0.00575674,  0.00181493,  0.00121419])

Colour Models - ``colour.models``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CIE xyY Colourspace
*******************

.. code-block:: python

    >>> colour.XYZ_to_xyY([0.20654008, 0.12197225, 0.05136952])
    array([ 0.54369557,  0.32107944,  0.12197225])

CIE L*a*b* Colourspace
**********************

.. code-block:: python

    >>> colour.XYZ_to_Lab([0.20654008, 0.12197225, 0.05136952])
    array([ 41.52787529,  52.63858304,  26.92317922])

CIE L*u*v* Colourspace
**********************

.. code-block:: python

    >>> colour.XYZ_to_Luv([0.20654008, 0.12197225, 0.05136952])
    array([ 41.52787529,  96.83626054,  17.75210149])

CIE 1960 UCS Colourspace
************************

.. code-block:: python

    >>> colour.XYZ_to_UCS([0.20654008, 0.12197225, 0.05136952])
    array([ 0.13769339,  0.12197225,  0.1053731 ])

CIE 1964 U*V*W* Colourspace
***************************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> colour.XYZ_to_UVW(XYZ)
    array([ 94.55035725,  11.55536523,  40.54757405])

Hunter L,a,b Colour Scale
*************************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> colour.XYZ_to_Hunter_Lab(XYZ)
    array([ 34.92452577,  47.06189858,  14.38615107])

Hunter Rd,a,b Colour Scale
**************************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> colour.XYZ_to_Hunter_Rdab(XYZ)
    array([ 12.197225  ,  57.12537874,  17.46241341])

CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces - Luo, Cui and Li (2006)
*************************************************************************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> XYZ_w = [95.05, 100.00, 108.88]
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = colour.VIEWING_CONDITIONS_CIECAM02['Average']
    >>> specification = colour.XYZ_to_CIECAM02(
            XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> colour.JMh_CIECAM02_to_CAM02UCS(JMh)
    array([ 47.16899898,  38.72623785,  15.8663383 ])
    >>> XYZ = [0.20654008, 0.12197225, 0.05136952]
    >>> XYZ_w = [95.05 / 100, 100.00 / 100, 108.88 / 100]
    >>> colour.XYZ_to_CAM02UCS(XYZ, XYZ_w=XYZ_w, L_A=L_A, Y_b=Y_b)
    array([ 47.16899898,  38.72623785,  15.8663383 ])

CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces - Li et al. (2017)
*******************************************************************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> XYZ_w = [95.05, 100.00, 108.88]
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = colour.VIEWING_CONDITIONS_CAM16['Average']
    >>> specification = colour.XYZ_to_CAM16(
            XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> colour.JMh_CAM16_to_CAM16UCS(JMh)
    array([ 46.55542238,  40.22460974,  14.25288392]
    >>> XYZ = [0.20654008, 0.12197225, 0.05136952]
    >>> XYZ_w = [95.05 / 100, 100.00 / 100, 108.88 / 100]
    >>> colour.XYZ_to_CAM16UCS(XYZ, XYZ_w=XYZ_w, L_A=L_A, Y_b=Y_b)
    array([ 46.55542238,  40.22460974,  14.25288392])

IgPgTg Colourspace
******************

.. code-block:: python

    >>> colour.XYZ_to_IgPgTg([0.20654008, 0.12197225, 0.05136952])
    array([ 0.42421258,  0.18632491,  0.10689223])

IPT Colourspace
***************

.. code-block:: python

    >>> colour.XYZ_to_IPT([0.20654008, 0.12197225, 0.05136952])
    array([ 0.38426191,  0.38487306,  0.18886838])

DIN99 Colourspace
*****************

.. code-block:: python

    >>> Lab = [41.52787529, 52.63858304, 26.92317922]
    >>> colour.Lab_to_DIN99(Lab)
    array([ 53.22821988,  28.41634656,   3.89839552])

hdr-CIELAB Colourspace
**********************

.. code-block:: python

    >>> colour.XYZ_to_hdr_CIELab([0.20654008, 0.12197225, 0.05136952])
    array([ 51.87002062,  60.4763385 ,  32.14551912])

hdr-IPT Colourspace
*******************

.. code-block:: python

    >>> colour.XYZ_to_hdr_IPT([0.20654008, 0.12197225, 0.05136952])
    array([ 25.18261761, -22.62111297,   3.18511729])

Oklab Colourspace
*****************

.. code-block:: python

    >>> colour.XYZ_to_Oklab([0.20654008, 0.12197225, 0.05136952])
    array([ 0.51634019,  0.154695  ,  0.06289579])

OSA UCS Colourspace
*******************

.. code-block:: python

    >>> XYZ = [0.20654008 * 100, 0.12197225 * 100, 0.05136952 * 100]
    >>> colour.XYZ_to_OSA_UCS(XYZ)
    array([-3.0049979 ,  2.99713697, -9.66784231])

JzAzBz Colourspace
******************

.. code-block:: python

    >>> colour.XYZ_to_JzAzBz([0.20654008, 0.12197225, 0.05136952])
    array([ 0.00535048,  0.00924302,  0.00526007])

Y'CbCr Colour Encoding
**********************

.. code-block:: python

    >>> colour.RGB_to_YCbCr([1.0, 1.0, 1.0])
    array([ 0.92156863,  0.50196078,  0.50196078])

YCoCg Colour Encoding
*********************

.. code-block:: python

    >>> colour.RGB_to_YCoCg([0.75, 0.75, 0.0])
    array([ 0.5625,  0.375 ,  0.1875])

ICtCp Colour Encoding
*********************

.. code-block:: python

    >>> colour.RGB_to_ICtCp([0.45620519, 0.03081071, 0.04091952])
    array([ 0.07351364,  0.00475253,  0.09351596])

HSV Colourspace
***************

.. code-block:: python

    >>> colour.RGB_to_HSV([0.45620519, 0.03081071, 0.04091952])
    array([ 0.99603944,  0.93246304,  0.45620519])

Prismatic Colourspace
*********************

.. code-block:: python

    >>> colour.RGB_to_Prismatic([0.25, 0.50, 0.75])
    array([ 0.75      ,  0.16666667,  0.33333333,  0.5       ])

RGB Colourspace and Transformations
***********************************

.. code-block:: python

    >>> XYZ = [0.21638819, 0.12570000, 0.03847493]
    >>> illuminant_XYZ = [0.34570, 0.35850]
    >>> illuminant_RGB = [0.31270, 0.32900]
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> matrix_XYZ_to_RGB = [
             [3.24062548, -1.53720797, -0.49862860],
             [-0.96893071, 1.87575606, 0.04151752],
             [0.05571012, -0.20402105, 1.05699594]]
    >>> colour.XYZ_to_RGB(
             XYZ,
             illuminant_XYZ,
             illuminant_RGB,
             matrix_XYZ_to_RGB,
             chromatic_adaptation_transform)
    array([ 0.45595571,  0.03039702,  0.04087245])

RGB Colourspace Derivation
**************************

.. code-block:: python

    >>> p = [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]
    >>> w = [0.32168, 0.33767]
    >>> colour.normalised_primary_matrix(p, w)
    array([[  9.52552396e-01,   0.00000000e+00,   9.36786317e-05],
           [  3.43966450e-01,   7.28166097e-01,  -7.21325464e-02],
           [  0.00000000e+00,   0.00000000e+00,   1.00882518e+00]])

RGB Colourspaces
****************

.. code-block:: python

    >>> sorted(colour.RGB_COLOURSPACES)
    ['ACES2065-1',
     'ACEScc',
     'ACEScct',
     'ACEScg',
     'ACESproxy',
     'ALEXA Wide Gamut',
     'Adobe RGB (1998)',
     'Adobe Wide Gamut RGB',
     'Apple RGB',
     'Best RGB',
     'Beta RGB',
     'CIE RGB',
     'Cinema Gamut',
     'ColorMatch RGB',
     'DaVinci Wide Gamut',
     'DCDM XYZ',
     'DCI-P3',
     'DCI-P3+',
     'DJI D-Gamut',
     'DRAGONcolor',
     'DRAGONcolor2',
     'Display P3',
     'Don RGB 4',
     'ECI RGB v2',
     'ERIMM RGB',
     'Ekta Space PS 5',
     'F-Gamut',
     'FilmLight E-Gamut',
     'ITU-R BT.2020',
     'ITU-R BT.470 - 525',
     'ITU-R BT.470 - 625',
     'ITU-R BT.709',
     'Max RGB',
     'NTSC (1953)',
     'NTSC (1987)',
     'P3-D65',
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

OETFs
*****

.. code-block:: python

    >>> sorted(colour.OETFS)
    ['ARIB STD-B67',
     'DaVinci Intermediate',
     'ITU-R BT.2100 HLG',
     'ITU-R BT.2100 PQ',
     'ITU-R BT.601',
     'ITU-R BT.709',
     'SMPTE 240M']

EOTFs
*****

.. code-block:: python

    >>> sorted(colour.EOTFS)
    ['DCDM',
     'DICOM GSDF',
     'ITU-R BT.1886',
     'ITU-R BT.2020',
     'ITU-R BT.2100 HLG',
     'ITU-R BT.2100 PQ',
     'SMPTE 240M',
     'ST 2084',
     'sRGB']

OOTFs
*****

.. code-block:: python

    >>> sorted(colour.OOTFS)
    ['ITU-R BT.2100 HLG', 'ITU-R BT.2100 PQ']


Log Encoding / Decoding
***********************

.. code-block:: python

    >>> sorted(colour.LOG_ENCODINGS)
    ['ACEScc',
     'ACEScct',
     'ACESproxy',
     'ALEXA Log C',
     'Canon Log',
     'Canon Log 2',
     'Canon Log 3',
     'Cineon',
     'D-Log',
     'ERIMM RGB',
     'F-Log',
     'Filmic Pro 6',
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

    >>> sorted(colour.CCTF_ENCODINGS)
    ['ACEScc',
     'ACEScct',
     'ACESproxy',
     'ALEXA Log C',
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

Colour Notation Systems - ``colour.notation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Munsell Value
*************

.. code-block:: python

    >>> colour.munsell_value(12.23634268)
    4.0824437076525664
    >>> sorted(colour.MUNSELL_VALUE_METHODS)
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

    >>> colour.xyY_to_munsell_colour([0.38736945, 0.35751656, 0.59362000])
    '4.2YR 8.1/5.3'
    >>> colour.munsell_colour_to_xyY('4.2YR 8.1/5.3')
    array([ 0.38736945,  0.35751656,  0.59362   ])

Optical Phenomena - ``colour.phenomena``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.rayleigh_scattering_sd()
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Colour Fidelity Index
*********************

.. code-block:: python

    >>> colour.colour_fidelity_index(colour.SDS_ILLUMINANTS['FL2'])
    70.120825477833037
    >>> sorted(colour.COLOUR_FIDELITY_INDEX_METHODS)
    ['ANSI/IES TM-30-18', 'CIE 2017']

Colour Rendering Index
**********************

.. code-block:: python

    >>> colour.colour_quality_scale(colour.SDS_ILLUMINANTS['FL2'])
    64.111703163816699
    >>> sorted(colour.COLOUR_QUALITY_SCALE_METHODS)
    ['NIST CQS 7.4', 'NIST CQS 9.0']

Colour Quality Scale
********************

.. code-block:: python

    >>> colour.colour_rendering_index(colour.SDS_ILLUMINANTS['FL2'])
    64.233724121664807

Academy Spectral Similarity Index (SSI)
***************************************

.. code-block:: python

    >>> colour.spectral_similarity_index(colour.SDS_ILLUMINANTS['C'], colour.SDS_ILLUMINANTS['D65'])
    94.0

Spectral Up-Sampling & Reflectance Recovery - ``colour.recovery``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.XYZ_to_sd([0.20654008, 0.12197225, 0.05136952])
    SpectralDistribution([[  3.60000000e+02,   8.37868873e-02],
                          [  3.65000000e+02,   8.39337988e-02],
                          ...
                          [  7.70000000e+02,   4.46793405e-01],
                          [  7.75000000e+02,   4.46872853e-01],
                          [  7.80000000e+02,   4.46914431e-01]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={'method': 'Constant', 'left': None, 'right': None})

    >>> sorted(colour.REFLECTANCE_RECOVERY_METHODS)
    ['Jakob 2019', 'Mallett 2019', 'Meng 2015', 'Otsu 2018', 'Smits 1999']

Correlated Colour Temperature Computation Methods - ``colour.temperature``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.uv_to_CCT([0.1978, 0.3122])
    array([  6.50751282e+03,   3.22335875e-03])
    >>> sorted(colour.UV_TO_CCT_METHODS)
    ['Krystek 1985', 'Ohno 2013', 'Robertson 1968', 'ohno2013', 'robertson1968']
    >>> sorted(colour.XY_TO_CCT_METHODS)
    ['CIE Illuminant D Series',
     'Hernandez 1999',
     'Kang 2002',
     'McCamy 1992',
     'daylight',
     'hernandez1999',
     'kang2002',
     'mccamy1992']

Colour Volume - ``colour.volume``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.RGB_colourspace_volume_MonteCarlo(colour.RGB_COLOURSPACE_RGB['sRGB'])
    821958.30000000005

Geometry Primitives Generation - ``colour.geometry``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> colour.primitive('Grid')
    (array([ ([-0.5,  0.5,  0. ], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  0.,  1.]),
           ([ 0.5,  0.5,  0. ], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  0.,  1.]),
           ([-0.5, -0.5,  0. ], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  0.,  1.]),
           ([ 0.5, -0.5,  0. ], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  0.,  1.])],
          dtype=[('position', '<f4', (3,)), ('uv', '<f4', (2,)), ('normal', '<f4', (3,)), ('colour', '<f4', (4,))]), array([[0, 2, 1],
           [2, 3, 1]], dtype=uint32), array([[0, 2],
           [2, 3],
           [3, 1],
           [1, 0]], dtype=uint32))
    >>> sorted(colour.PRIMITIVE_METHODS)
    ['Cube', 'Grid']
    >>> colour.primitive_vertices('Quad MPL')
    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.],
           [ 0.,  1.,  0.]])
    >>> sorted(colour.PRIMITIVE_VERTICES_METHODS)
    ['Cube MPL', 'Grid MPL', 'Quad MPL', 'Sphere']

Plotting - ``colour.plotting``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the objects are available from the ``colour.plotting`` namespace:

.. code-block:: python

    >>> from colour.plotting import *
    >>> colour_style()

Visible Spectrum
****************

.. code-block:: python

    >>> plot_visible_spectrum('CIE 1931 2 Degree Standard Observer')

..  image:: _static/Examples_Plotting_Visible_Spectrum.png

Spectral Distribution
*********************

.. code-block:: python

    >>> plot_single_illuminant_sd('FL1')

..  image:: _static/Examples_Plotting_Illuminant_F1_SD.png

Blackbody
*********

.. code-block:: python

    >>> blackbody_sds = [
    ...     colour.sd_blackbody(i, colour.SpectralShape(0, 10000, 10))
    ...     for i in range(1000, 15000, 1000)
    ... ]
    >>> plot_multi_sds(
    ...     blackbody_sds,
    ...     y_label='W / (sr m$^2$) / m',
    ...     plot_kwargs={
    ...         use_sd_colours=True,
    ...         normalise_sd_colours=True,
    ...     },
    ...     legend_location='upper right',
    ...     bounding_box=(0, 1250, 0, 2.5e15))

..  image:: _static/Examples_Plotting_Blackbodies.png

Colour Matching Functions
*************************

.. code-block:: python

    >>> plot_single_cmfs(
    ...     'Stockman & Sharpe 2 Degree Cone Fundamentals',
    ...     y_label='Sensitivity',
    ...     bounding_box=(390, 870, 0, 1.1))

..  image:: _static/Examples_Plotting_Cone_Fundamentals.png

Luminous Efficiency
*******************

.. code-block:: python

    >>> sd_mesopic_luminous_efficiency_function = (
    ...     colour.sd_mesopic_luminous_efficiency_function(0.2))
    >>> plot_multi_sds(
    ...     (sd_mesopic_luminous_efficiency_function,
    ...      colour.PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
    ...      colour.SCOTOPIC_LEFS['CIE 1951 Scotopic Standard Observer']),
    ...     y_label='Luminous Efficiency',
    ...     legend_location='upper right',
    ...     y_tighten=True,
    ...     margins=(0, 0, 0, .1))

..  image:: _static/Examples_Plotting_Luminous_Efficiency.png

Colour Checker
**************

.. code-block:: python

    >>> from colour.characterisation.dataset.colour_checkers.sds import (
    ...     COLOURCHECKER_INDEXES_TO_NAMES_MAPPING)
    >>> plot_multi_sds(
    ...     [
    ...         colour.SDS_COLOURCHECKERS['BabelColor Average'][value]
    ...         for key, value in sorted(
    ...             COLOURCHECKER_INDEXES_TO_NAMES_MAPPING.items())
    ...     ],
    ...     plot_kwargs={
    ...         use_sd_colours=True,
    ...     },
    ...     title=('BabelColor Average - '
    ...            'Spectral Distributions'))

..  image:: _static/Examples_Plotting_BabelColor_Average.png

.. code-block:: python

    >>> plot_single_colour_checker(
    ...     'ColorChecker 2005', text_kwargs={'visible': False})

..  image:: _static/Examples_Plotting_ColorChecker_2005.png

Chromaticities Prediction
*************************

.. code-block:: python

    >>> plot_corresponding_chromaticities_prediction(
    ...     2, 'Von Kries', 'Bianco 2010')

..  image:: _static/Examples_Plotting_Chromaticities_Prediction.png

Colour Temperature
******************

.. code-block:: python

    >>> plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(['A', 'B', 'C'])

..  image:: _static/Examples_Plotting_CCT_CIE_1960_UCS_Chromaticity_Diagram.png


Chromaticities
**************

.. code-block:: python

    >>> import numpy as np
    >>> RGB = np.random.random((32, 32, 3))
    >>> plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
    ...     RGB, 'ITU-R BT.709',
    ...     colourspaces=['ACEScg', 'S-Gamut'], show_pointer_gamut=True)

..  image:: _static/Examples_Plotting_Chromaticities_CIE_1931_Chromaticity_Diagram.png

Colour Rendering Index
**********************

.. code-block:: python

    >>> plot_single_sd_colour_rendering_index_bars(
    ...     colour.SDS_ILLUMINANTS['FL2'])

..  image:: _static/Examples_Plotting_CRI.png

ANSI/IES TM-30-18 Colour Rendition Report
*****************************************

.. code-block:: python

    >>> plot_single_sd_colour_rendition_report(
    ...     colour.SDS_ILLUMINANTS['FL2'])

..  image:: _static/Examples_Plotting_Colour_Rendition_Report.png

Contributing
------------

If you would like to contribute to **Colour**, please refer to the following
`Contributing <https://www.colour-science.org/contributing>`__ guide.

Changes
-------

The changes are viewable on the `Releases <https://github.com/colour-science/colour/releases>`__ page.

Bibliography
------------

The bibliography is available on the `Bibliography <https://www.colour-science.org/bibliography/>`__ page.

It is also viewable directly from the repository in
`BibTeX <https://github.com/colour-science/colour/blob/develop/BIBLIOGRAPHY.bib>`__
format.

See Also
--------

Here is a list of notable colour science packages sorted by languages:

**Python**

- `Colorio <https://github.com/nschloe/colorio/>`__  by Schlömer, N.
- `ColorPy <http://markkness.net/colorpy/ColorPy.html>`__ by Kness, M.
- `Colorspacious <https://colorspacious.readthedocs.io/>`__ by Smith, N. J., et al.
- `python-colormath <https://python-colormath.readthedocs.io/>`__ by Taylor, G., et al.

**Go**

- `go-colorful <https://github.com/lucasb-eyer/go-colorful/>`__  by Beyer, L., et al.

**.NET**

- `Colourful <https://github.com/tompazourek/Colourful>`__ by Pažourek, T., et al.

**Julia**

- `Colors.jl <https://github.com/JuliaGraphics/Colors.jl>`__ by Holy, T., et al.

**Matlab & Octave**

- `COLORLAB <https://www.uv.es/vista/vistavalencia/software/colorlab.html>`__ by Malo, J., et al.
- `Psychtoolbox <http://psychtoolbox.org/>`__ by Brainard, D., et al.
- `The Munsell and Kubelka-Munk Toolbox <http://www.munsellcolourscienceforpainters.com/MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html>`__ by Centore, P.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`__,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct/>`__ page.

Contact & Social
----------------

The *Colour Developers* can be reached via different means:

- `Email <mailto:colour-developers@colour-science.org>`__
- `Discourse <https://colour-science.discourse.group/>`__
- `Facebook <https://www.facebook.com/python.colour.science>`__
- `Github Discussions <https://github.com/colour-science/colour/discussions>`__
- `Gitter <https://gitter.im/colour-science/colour>`__
- `Twitter <https://twitter.com/colour_science>`__

About
-----

| **Colour** by Colour Developers
| Copyright © 2013-2021 – Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
