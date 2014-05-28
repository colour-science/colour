Color
=====

..  image:: https://secure.travis-ci.org/KelSolaar/Color.png?branch=master
..  image:: https://gemnasium.com/KelSolaar/Color.png

Introduction
------------

**Color** is a **Python** color science package implementing a comprehensive number of color transformations and manipulations objects.

Highlights:

-  RGB and XYZ color matching functions spectral data:

  -  Stiles & Burch 1955 2° Observer RGB color matching functions.
  -  Stiles & Burch 1959 10° Observer RGB color matching functions.
  -  Standard CIE 1931 2° Observer XYZ color matching functions.
  -  Standard CIE 1964 10° Observer XYZ color matching functions.
  -  Standard CIE 2006 2° Observer XYZ color matching functions.
  -  Standard CIE 2006 10° Observer XYZ color matching functions.

-  Illuminants spectral data:

  -  A
  -  C
  -  D50
  -  D55
  -  D60
  -  D65
  -  D75
  -  F1
  -  F2
  -  F3
  -  F4
  -  F5
  -  F6
  -  F7
  -  F8
  -  F9
  -  F10
  -  F11
  -  F12

- Photopic & Scotopic luminous efficiency functions spectral data:

  -  CIE 1924 2 Degree Observer
  -  Judd Modified CIE 1951 2 Degree Observer
  -  Judd-Vos Modified CIE 1978 2 Degree Observer
  -  Stockman, Jagle, Pirzer & Sharpe CIE 2008 2 Degree Observe
  -  Stockman, Jagle, Pirzer & Sharpe CIE 2008 10 Degree Observer
  -  Wald & Crawford CIE 1951 2 Degree Observer

-  Correlated color temperature calculation:

  -  Wyszecki & Roberston method implementation.
  -  Yoshi Ohno method implementation.

-  Spectral power distribution data manipulation and conversion to color.
-  Blackbody spectral radiance calculation.
-  Spectral bandpass correction.
-  Sprague interpolation.
-  Chromatic adaptation with following methods:

  -  XYZ Scaling.
  -  Bradford.
  -  Von Kries.
  -  CAT02.

-  Luminance, Munsell value and Lightness calculations:

  -  Luminance 1943
  -  Luminance 1976
  -  Munsell Value 1920
  -  Munsell Value 1933
  -  Munsell Value 1943
  -  Munsell Value 1944
  -  Munsell Value 1955
  -  Lightness 1958
  -  Lightness 1964
  -  Lightness 1976

-  RGB Colorspaces support:

  -  ACES RGB
  -  ACES RGB Log
  -  ACES RGB Proxy 10
  -  ACES RGB Proxy 12
  -  Adobe RGB 1998
  -  Adobe Wide Gamut RGB
  -  ALEXA Wide Gamut RGB
  -  Apple RGB
  -  Best RGB
  -  Beta RGB
  -  CIE RGB
  -  ColorMatch RGB
  -  DCI-P3
  -  Don RGB 4
  -  ECI RGB v2
  -  Ekta Space PS 5
  -  Max RGB
  -  NTSC RGB
  -  Pal/Secam RGB
  -  ProPhoto RGB
  -  Rec. 709
  -  Rec. 2020
  -  Russell RGB
  -  S-Log
  -  SMPTE-C RGB
  -  Xtreme RGB
  -  sRGB

-  Colorspaces transformations:

  -  Wavelength to XYZ.
  -  Spectral to XYZ.
  -  XYZ to xyY.
  -  xyY to XYZ.
  -  xy to XYZ.
  -  XYZ to xy.
  -  XYZ to RGB.
  -  RGB to XYZ.
  -  xyY to RGB.
  -  RGB to xyY.
  -  XYZ to UVW.
  -  UVW to XYZ.
  -  UVW to uv.
  -  UVW uv to xy.
  -  XYZ to Luv.
  -  Luv to XYZ.
  -  Luv to uv.
  -  Luv uv to xy.
  -  Luv to LCHuv.
  -  LCHuv to Luv.
  -  XYZ to Lab.
  -  Lab to XYZ.
  -  Lab to LCHab.
  -  LCHab to Lab.
  -  uv to cct, duv.
  -  cct, duv to uv.

Convenience deprecated transformations:

  -  RGB to HSV.
  -  HSV to RGB.
  -  RGB to HSL.
  -  HSL to RGB.
  -  RGB to CMY.
  -  CMY to RGB.
  -  CMY to CMYK.
  -  CMYK to CMY.
  -  RGB to HEX.
  -  HEX to RGB.

-  Illuminants chromaticity coordinates data.
-  Correlated color temperature calculation.
-  Colorspaces derivation.
-  Color difference calculation with following methods:

  -  ΔE CIE 1976.
  -  ΔE CIE 1994.
  -  ΔE CIE 2000.
  -  ΔE CMC.

-  Color rendition chart data.
-  Colorspaces visualisation within **Autodesk Maya**.
-  First order color fit.
-  Comprehensive plotting capabilities.

Installation
------------

The following dependencies are needed:

-  **Python 2.6.7** or **Python 2.7.3**: http://www.python.org/

To install **Color** from the `Python Package Index <http://pypi.python.org/pypi/ColorScience>`_ you can issue this command in a shell::

    pip install ColorScience

or this alternative command::

    easy_install ColorScience

You can also install directly from `Github <http://github.com/KelSolaar/Color>`_ source repository::

	git clone git://github.com/KelSolaar/Color.git
	cd Color
	python setup.py install

If you want to build the documentation you will also need:

-  **Tidy** http://tidy.sourceforge.net/

Usage
-----

Api
---

Acknowledgements
----------------

-  **Yoshi Ohno** for helping me pinpointing the root cause of calculation discrepancies in my implementation of his CCT & Duv calculation method.
-  **Charles Poynton** for taking time to reply to my questions.
-  **Michael Parsons** for all the continuous technical advices.

References
----------

**Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN: 0-471-39918-3

**Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN: 978-0-470-66569-5

**Edward J. Giorgianni & Thomas E. Madden**, *Digital Color Management: Encoding Solutions - Second Edition*, Wiley, published November 2008, ISBN: 978-0-470-99436-8

**Charles Poynton**, *Digital Video and HD: Algorithms and Interfaces*, The Morgan Kaufmann Series in Computer Graphics, published 2 December 2012, ISBN: 978-0123919267

**Charles Poynton**, `Color FAQ <http://www.poynton.com/ColorFAQ.html>`_

**Charles Poynton**, `Gamma FAQ <http://www.poynton.com/GammaFAQ.html>`_

Algebra
-------

`Sprague Interpolation <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, Page 33. (Last accessed 28 May 2014)

Chromatic Adaptation
--------------------

**Bruce Lindbloom**, `XYZ Scaling Chromatic Adaptation <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Bradford Chromatic Adaptation <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Von Kries Chromatic Adaptation <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)

`CAT02 Chromatic Adaptation <http://en.wikipedia.org/wiki/CIECAM02#CAT0>`_ (Last accessed 24 February 2014)

Color Rendition Charts
----------------------

`Babel Color ColorChecker RGB and Spectral Data <http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls>`_ (Last accessed 24 February 2014)

Colorspace Derivation
---------------------

`Colorspace Derivation <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_ (Last accessed 24 February 2014)

Color Difference
----------------

**Bruce Lindbloom**, `ΔE CIE 1976 <http://brucelindbloom.com/Eqn_DeltaE_CIE76.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `ΔE CIE 1994 <http://brucelindbloom.com/Eqn_DeltaE_CIE94.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `ΔE CIE 2000 <http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `ΔE CMC <http://brucelindbloom.com/Eqn_DeltaE_CMC.html>`_ (Last accessed 24 February 2014)

Color Matching Functions
------------------------

`Stiles & Burch 1955 2 Degree Observer <http://www.cvrl.org/stilesburch2_ind.htm>`_ (Last accessed 24 February 2014)

`Stiles & Burch 1959 10 Degree Observer <http://www.cvrl.org/stilesburch10_ind.htm>`_ (Last accessed 24 February 2014)

`Standard CIE 1931 2 Degree Observer <http://cvrl.ioo.ucl.ac.uk/cie.htm>`_ (Last accessed 24 February 2014)

`Standard CIE 1964 10 Degree Observer <http://cvrl.ioo.ucl.ac.uk/cie.htm>`_ (Last accessed 24 February 2014)

`Standard CIE 2006 2 Degree Observer <http://cvrl.ioo.ucl.ac.uk/ciexyzpr.htm>`_ (Last accessed 24 February 2014)

`Standard CIE 2006 10 Degree Observer <http://cvrl.ioo.ucl.ac.uk/ciexyzpr.htm>`_ (Last accessed 24 February 2014)

Deprecated Transformations
--------------------------

`RGB to HSV <http://www.easyrgb.com/index.php?X=MATH&H=20#text20>`_ (Last accessed 18 May 2014)

`HSV to RGB <http://www.easyrgb.com/index.php?X=MATH&H=21#text21>`_ (Last accessed 18 May 2014)

`RGB to HSL <http://www.easyrgb.com/index.php?X=MATH&H=18#text18>`_ (Last accessed 18 May 2014)

`HSL to RGB <http://www.easyrgb.com/index.php?X=MATH&H=21#text21>`_ (Last accessed 18 May 2014)

`RGB to CMY <http://www.easyrgb.com/index.php?X=MATH&H=11#text11>`_ (Last accessed 18 May 2014)

`CMY to RGB <http://www.easyrgb.com/index.php?X=MATH&H=12#text12>`_ (Last accessed 18 May 2014)

`CMY to CMYK <http://www.easyrgb.com/index.php?X=MATH&H=13#text13>`_ (Last accessed 18 May 2014)

`CMYK to CMY <http://www.easyrgb.com/index.php?X=MATH&H=14#text14>`_ (Last accessed 18 May 2014)

Illuminants Relative Spectral Power Distributions
-------------------------------------------------

`A <http://files.cie.co.at/204.xls>`_ (Last accessed 24 February 2014)

`D65 <http://files.cie.co.at/204.xls>`_ (Last accessed 24 February 2014)

`C <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`D50 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`D55 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`D75 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F1 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F2 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F3 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F4 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F5 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F6 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F7 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F8 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F9 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F10 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F11 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

`F12 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `D60 <http://www.brucelindbloom.com/Eqn_DIlluminant.html>`_ (Last accessed 5 April 2014)

Illuminants Chromaticity Coordinates
------------------------------------

`Illuminants Chromaticity Coordinates <http://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants>`_ (Last accessed 24 February 2014)

Luminance, Munsell value, Lightness
-----------------------------------

`Luminance <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_ (Last accessed 24 February 2014)

`Luminance 1943 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Luminance 1976 <http://www.poynton.com/PDFs/GammaFAQ.pdf>`_ (Last accessed 12 April 2014)

`Munsell Value 1920 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Munsell Value 1933 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Munsell Value 1943 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Munsell Value 1944 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Munsell Value 1955 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Lightness 1958 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

`Lightness 1964 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)

**Charles Poynton**, `Lightness 1976 <http://www.poynton.com/PDFs/GammaFAQ.pdf>`_ (Last accessed 12 April 2014)

Luminous Efficiency Functions
-----------------------------

`CIE 1924 2 Degree Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

`Judd Modified CIE 1951 2 Degree Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

`Judd-Vos Modified CIE 1978 2 Degree Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

`Stockman, Jagle, Pirzer & Sharpe CIE 2008 2 Degree Observe <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

`Stockman, Jagle, Pirzer & Sharpe CIE 2008 10 Degree Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

`Wald & Crawford CIE 1951 2 Degree Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)

Planc's Law
-----------

`Planck's Law <http://en.wikipedia.org/wiki/Planck's_law>`_ (Last accessed 24 February 2014)

RGB Colorspaces
---------------

`ACES RGB Colorspace <http://www.oscars.org/science-technology/council/projects/aces.html>`_ (Last accessed 24 February 2014)

`ACES RGB Log Colorspace <http://www.dropbox.com/sh/iwd09buudm3lfod/AAA-X1nVs_XLjWlzNhfhqiIna/ACESlog_v1.0.pdf>`_ (Last accessed 17 May 2014)

`ACES RGB Proxy 10 Colorspace <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_ (Last accessed 17 May 2014)

`ACES RGB Proxy 12 Colorspace <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_ (Last accessed 17 May 2014)

`Adobe RGB 1998 Colorspace <http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf>`_ (Last accessed 24 February 2014)

`Adobe Wide Gamut RGB Colorspace <http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space>`_ (Last accessed 13 April 2014)

`ALEXA Wide Gamut RGB Colorspace <http://www.arri.com/?eID=registration&file_uid=8026>`_ (Last accessed 13 April 2014)

`Apple RGB Colorspace <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 11 April 2014)

`Best RGB Colorspace <http://www.hutchcolor.com/profiles/BestRGB.zip>`_ (Last accessed 11 April 2014)

`Beta RGB Colorspace <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 11 April 2014)

`CIE RGB Colorspace <http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data>`_ (Last accessed 24 February 2014)

`C-Log Colorspace <http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf>`_ (Last accessed 18 April 2014)

`ColorMatch Colorspace <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 12 April 2014)

`DCI-P3 Colorspace <http://www.hp.com/united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf>`_ (Last accessed 24 February 2014)

`Don RGB 4 Colorspace <http://www.hutchcolor.com/profiles/DonRGB4.zip>`_ (Last accessed 12 April 2014)

`ECI RGB v2 Colorspace <http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip>`_ (Last accessed 13 April 2014)

`Ekta Space PS 5 Colorspace <http://www.josephholmes.com/Ekta_Space.zip>`_ (Last accessed 13 April 2014)

`Max RGB Colorspace <http://www.hutchcolor.com/profiles/MaxRGB.zip>`_ (Last accessed 12 April 2014)

`NTSC RGB Colorspace <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf>`_ (Last accessed 13 April 2014)

`Pal/Secam RGB Colorspace <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf>`_ (Last accessed 13 April 2014)

`Pointer's Gamut <http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls>`_ (Last accessed 24 February 2014)

`ProPhoto RGB Colorspace <http://www.color.org/ROMMRGB.pdf>`_ (Last accessed 24 February 2014)

`Rec. 709 Colorspace <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf>`_ (Last accessed 24 February 2014)

`Rec. 2020 Colorspace <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf>`_ (Last accessed 13 April 2014)

`Russell RGB Colorspace <http://www.russellcottrell.com/photo/RussellRGB.htm>`_ (Last accessed 11 April 2014)

`S-Log Colorspace <http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf>`_ (Last accessed 13 April 2014)

`SMPTE-C RGB Colorspace <http://standards.smpte.org/content/978-1-61482-164-9/rp-145-2004/SEC1.body.pdf>`_ (Last accessed 13 April 2014)

`sRGB Colorspace <http://www.color.org/srgb.pdf>`_ (Last accessed 24 February 2014)

`Xtreme RGB Colorspace <http://www.hutchcolor.com/profiles/MaxRGB.zip>`_ (Last accessed 12 April 2014)

Spectral
--------

**Spectral to XYZ**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 158.

**Spectral Bandpass Correction**, Stearns, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, Page 38.

`Extrapolation <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_ (Last accessed 28 May 2014)

Temperature
-----------

**Yoshi Ohno**, `Practical Use and Calculation of CCT and Duv <http://dx.doi.org/10.1080/15502724.2014.839020>`_ (Last accessed 3 March 2014)

Transformations
---------------

**Bruce Lindbloom**, `XYZ to xyY <http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `xyY to XYZ <http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html>`_ (Last accessed 24 February 2014)

`XYZ to UVW <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)

`UVW to XYZ <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)

`UVW to uv <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)

`UVW uv to xy <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `XYZ to Luv <http://brucelindbloom.com/Eqn_XYZ_to_Luv.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Luv to XYZ <http://brucelindbloom.com/Eqn_Luv_to_XYZ.html>`_ (Last accessed 24 February 2014)

`Luv to uv <http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation>`_ (Last accessed 24 February 2014)

`Luv uv to xy <http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Luv to LCHuv <http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `LCHuv to Luv <http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `XYZ to Lab <http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Lab to XYZ <http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `Lab to LCHab <http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html>`_ (Last accessed 24 February 2014)

**Bruce Lindbloom**, `LCHab to Lab <http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html>`_ (Last accessed 24 February 2014)

About
-----

| **Color** by Thomas Mansencal - Michael Parsons - 2013 - 2014
| Copyright © 2013 - 2014 – Thomas Mansencal – `thomas.mansencal@gmail.com <mailto:thomas.mansencal@gmail.com>`_
| This software is released under terms of GNU GPL V3 license: http://www.gnu.org/licenses/
| `http://www.thomasmansencal.com/ <http://www.thomasmansencal.com/>`_