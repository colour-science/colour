Colour - Manual - Help File
===========================

.. raw:: html

    <br/>

Table Of Content
=================

.. .tocTree

-  `Introduction`_

   -  `History`_
   -  `Highlights`_

-  `Installation`_
-  `Usage`_
-  `Api`_
-  `Acknowledgements`_
-  `Changes`_
-  `References`_
-  `About`_

.. raw:: html

    <br/>

.. .introduction

_`Introduction`
===============

**Colour** is a **Python** colour science package implementing a comprehensive number of colour theory transformations and algorithms.

_`History`
----------

**Colour** started as a raw conversion building block for `The Moving Picture Company <http://www.moving-picture.com>`_ stills ingestion pipeline.

Generic objects have been extracted, reorganised and are now provided as a nice packaged API while keeping the undisclosable code private. The original `MPC <http://www.moving-picture.com>`_ *camelCase* naming convention and code style has been changed for *Pep8* compliance.

Matplotlib implementation idea is coming from the excellent *Mark Kness*'s `ColorPy <http://markkness.net/colorpy/ColourPy.html>`_ **Python** package.

_`Highlights`
-------------

-  RGB and XYZ colour matching functions spectral data:

  -  Wright & Guild 1931 2 Degree RGB CMFs RGB colour matching functions.
  -  Stiles & Burch 1955 2 Degree RGB CMFs RGB colour matching functions.
  -  Stiles & Burch 1959 10 Degree RGB CMFs RGB colour matching functions.
  -  CIE 1931 2 Degree Standard Observer XYZ colour matching functions.
  -  CIE 1964 10 Degree Standard Observer XYZ colour matching functions.
  -  CIE 2012 2 Degree Standard Observer XYZ colour matching functions.
  -  CIE 2012 10 Degree Standard Observer XYZ colour matching functions.

-  Cone fundamentals spectral data:

  -  Stockman & Sharpe 2 Degree Cone Fundamentals.
  -  Stockman & Sharpe 10 Degree Cone Fundamentals.

- Photopic & Scotopic luminous efficiency functions spectral data:

  -  CIE 1924 Photopic Standard Observer
  -  Judd Modified CIE 1951 Photopic Standard Observer
  -  Judd-Vos Modified CIE 1978 Photopic Standard Observer
  -  CIE 1964 Photopic 10 Degree Standard Observer
  -  CIE 2008 2 Degree Physiologically Relevant LEF
  -  CIE 2008 10 Degree Physiologically Relevant LEF
  -  CIE 1951 Scotopic Standard Observer

-  Illuminants spectral data:

  -  A
  -  B
  -  C
  -  D50
  -  D55
  -  D60
  -  D65
  -  D75
  -  E
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
  -  FL3.1
  -  FL3.2
  -  FL3.3
  -  FL3.4
  -  FL3.5
  -  FL3.6
  -  FL3.7
  -  FL3.8
  -  FL3.9
  -  FL3.10
  -  FL3.11
  -  FL3.12
  -  FL3.13
  -  FL3.14
  -  FL3.15
  -  HP1
  -  HP2
  -  HP3
  -  HP4
  -  HP5

-  Colour rendition charts spectral data.
-  Colour appearance models:

  -  CIECAM02

-  Correlated colour temperature calculation:

  -  Roberston method implementation.
  -  Yoshi Ohno method implementation.
  -  McCamy method implementation.
  -  Hernandez-Andres, Lee & Romero method implementation.
  -  Kang, Moon, Hong, Lee, Cho and Kim implementation.
  -  CIE Illuminant D Series implementation.

-  Colour matching functions conversions for educational purpose:

  -  Wright & Guild 1931 2 Degree RGB CMFs to CIE 1931 2 Degree Standard Observer
  -  Stiles & Burch 1959 10 Degree RGB CMFs to CIE 1964 10 Degree Standard Observer
  -  Stiles & Burch 1959 10 Degree RGB CMFs to Stockman & Sharpe 10 Degree Cone Fundamentals
  -  Stockman & Sharpe 2 Degree Cone Fundamentals to CIE 2012 2 Degree Standard Observer
  -  Stockman & Sharpe 10 Degree Cone Fundamentals to CIE 2012 10 Degree Standard Observer

-  Spectral power distribution data manipulation and conversion to tristimulus values.
-  Blackbody spectral radiance calculation.
-  Spectral bandpass correction.
-  Sprague interpolation.
-  Chromatic adaptation with following methods:

  -  XYZ Scaling.
  -  Bradford.
  -  Von Kries.
  -  CAT02.

-  Luminance, Munsell value and Lightness calculations:

  -  Luminance Newhall 1943
  -  Luminance 1976
  -  Luminance ASTM D1535-08
  -  Munsell Value Priest 1920
  -  Munsell Value Munsell 1933
  -  Munsell Value Moon 1943
  -  Munsell Value Saunderson 1944
  -  Munsell Value Ladd 1955
  -  Munsell Value McCamy 1987
  -  Munsell Value ASTM D1535-08
  -  Lightness Glasser 1958
  -  Lightness Wyszecki 1964
  -  Lightness 1976

-  RGB Colourspaces support:

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

-  Colourspaces transformations:

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
  -  RGB to RGB.
  -  XYZ to UCS.
  -  UCS to XYZ.
  -  UCS to uv.
  -  UCS uv to xy.
  -  XYZ to UVW.
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
  -  uv to CCT, Duv.
  -  CCT, Duv to uv.
  -  xyY to Munsell Colour.
  -  Munsell Colour to xyY.

Convenience deprecated colourspaces transformations:

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
-  Colourspaces derivation.
-  Colour difference calculation with following methods:

  -  ΔE CIE 1976.
  -  ΔE CIE 1994.
  -  ΔE CIE 2000.
  -  ΔE CMC.

-  Colour rendering index calculation.
-  Colour rendition chart data.
-  RGB colourspaces visualisation within **Autodesk Maya**.
-  First order colour fit.
-  Comprehensive plotting capabilities.

.. raw:: html

    <br/>

.. .installation

_`Installation`
===============

The following dependencies are needed:

-  **Python 2.6.7** or **Python 2.7.3**: http://www.python.org/

To install **Colour** from the `Python Package Index <http://pypi.python.org/pypi/ColourScience>`_ you can issue this command in a shell::

    pip install ColourScience

or this alternative command::

    easy_install ColourScience

You can also install directly from `Github <http://github.com/KelSolaar/Colour>`_ source repository::

	git clone git://github.com/KelSolaar/Colour.git
	cd Colour
	python setup.py install

If you want to build the documentation you will also need:

-  `Oncilla <https://pypi.python.org/pypi/Oncilla/>`_
-  `Tidy <http://tidy.sourceforge.net/>`_

.. raw:: html

    <br/>

.. .usage

_`Usage`
========

.. raw:: html

    <br/>

.. .api

_`Api`
======

.. raw:: html

    <br/>

.. .changes

_`Changes`
==========

.. raw:: html

    <br/>

.. .acknowledgements

_`Acknowledgements`
===================

-  **Yoshi Ohno** for helping me pinpointing the root cause of calculation discrepancies in my implementation of his CCT & Duv calculation method.
-  **Charles Poynton** for replying to my questions.
-  **Michael Parsons** for all the continuous technical advices.

.. .references

_`References`
=============

-  **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3
-  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161
-  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5
-  **Richard Sewall Hunter**, *The Measurement of Appearance, 2nd Edition*, published August 25, 1987, ISBN-13: 978-0471830061
-  **Edward J. Giorgianni & Thomas E. Madden**, *Digital Colour Management: Encoding Solutions - Second Edition*, Wiley, published November 2008, ISBN-13: 978-0-470-99436-8
-  **Charles Poynton**, *Digital Video and HD: Algorithms and Interfaces*, The Morgan Kaufmann Series in Computer Graphics, published December 2, 2012, ISBN-13: 978-0123919267
-  **Charles Poynton**, `Color FAQ <http://www.poynton.com/ColourFAQ.html>`_
-  **Charles Poynton**, `Gamma FAQ <http://www.poynton.com/GammaFAQ.html>`_

Algebra
-------

-  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: 9.2.4 Method of interpolation for uniformly spaced independent variable <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_ (Last accessed 28 May 2014), **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Page 33.

Planckian Radiator
------------------

-  `CIE 015:2004 Colorimetry, 3rd edition: Appendix E. Information on the Use of Planck's Equation for Standard Air. <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_

Chromatic Adaptation Transforms
-------------------------------

-  **Bruce Lindbloom**, `XYZ Scaling Chromatic Adaptation Transform <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Bradford Chromatic Adaptation Transform <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Von Kries Chromatic Adaptation Transform <http://brucelindbloom.com/Eqn_ChromAdapt.html>`_ (Last accessed 24 February 2014)
-  **Mark D. Fairchild**, `Fairchild Chromatic Adaptation Transform <http://rit-mcsl.org/fairchild//files/FairchildYSh.zip>`_ (Last accessed 28 July 2014)
-  `CAT02 Chromatic Adaptation <http://en.wikipedia.org/wiki/CIECAM02#CAT0>`_ (Last accessed 24 February 2014)

Colour Appearance Models
------------------------

-  **CIECAM02**, **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277., **CIECAM02**, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92., `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_ (Last accessed 30 July 2014)

Colour Rendering Index
----------------------

-  **Yoshi Ohno**, `Colour Rendering Index <http://cie2.nist.gov/TC1-69/NIST%20CQS%20simulation%207.4.xls>`_ (Last accessed 10 June 2014)

Colour Rendition Charts
-----------------------

-  `BabelColor ColorChecker RGB & Spectral Data <http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls>`_ (Last accessed 24 February 2014)
-  **N. Ohta**, `Macbeth ColorChecker Spectral Data <http://www.rit-mcsl.org/UsefulData/MacbethColorChecker.xls>`_ (Last accessed 9 June 2014)

Colourspace Derivation
----------------------

-  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.2 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_ (Last accessed 24 February 2014)

Colour Difference
-----------------

-  **Bruce Lindbloom**, `ΔE CIE 1976 <http://brucelindbloom.com/Eqn_DeltaE_CIE76.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `ΔE CIE 1994 <http://brucelindbloom.com/Eqn_DeltaE_CIE94.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `ΔE CIE 2000 <http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `ΔE CMC <http://brucelindbloom.com/Eqn_DeltaE_CMC.html>`_ (Last accessed 24 February 2014)

Colour Matching Functions
-------------------------

-  `Wright & Guild 1931 2 Degree RGB CMFs <http://www.cis.rit.edu/mcsl/research/1931.php>`_ (Last accessed 12 June 2014)
-  `Stiles & Burch 1955 2 Degree RGB CMFs <http://www.cvrl.org/stilesburch2_ind.htm>`_ (Last accessed 24 February 2014)
-  `Stiles & Burch 1959 10 Degree RGB CMFs <http://www.cvrl.org/stilesburch10_ind.htm>`_ (Last accessed 24 February 2014)
-  `CIE 1931 2 Degree Standard Observer <http://cvrl.ioo.ucl.ac.uk/cie.htm>`_ (Last accessed 24 February 2014)
-  `CIE 1964 10 Degree Standard Observer <http://cvrl.ioo.ucl.ac.uk/cie.htm>`_ (Last accessed 24 February 2014)
-  `CIE 2012 2 Degree Standard Observer <http://cvrl.ioo.ucl.ac.uk/ciexyzpr.htm>`_ (Last accessed 24 February 2014)
-  `CIE 2012 10 Degree Standard Observer <http://cvrl.ioo.ucl.ac.uk/ciexyzpr.htm>`_ (Last accessed 24 February 2014)
-  **Wright & Guild 1931 2 Degree RGB CMFs to CIE 1931 2 Degree Standard Observer**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Pages 138, 139.
-  **Stiles & Burch 1959 10 Degree RGB CMFs to CIE 1964 10 Degree Standard Observer**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Page 141.
-  `Stiles & Burch 1959 10 Degree RGB CMFs to Stockman & Sharpe 10 Degree Cone Fundamentals <http://div1.cie.co.at/?i_ca_id=551&pubid=48>`_ (Last accessed 23 June 2014)
-  `Stockman & Sharpe 2 Degree Cone Fundamentals to CIE 2012 2 Degree Standard Observer <http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm>`_ (Last accessed 25 June 2014)
-  `Stockman & Sharpe 10 Degree Cone Fundamentals to CIE 2012 10 Degree Standard Observer <http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm>`_ (Last accessed 25 June 2014)

Cone Fundamentals
-----------------

-  `Stockman & Sharpe 2 Degree Cone Fundamentals <http://www.cvrl.org/cones.htm>`_ (Last accessed 23 June 2014)
-  `Stockman & Sharpe 10 Degree Cone Fundamentals <http://www.cvrl.org/cones.htm>`_ (Last accessed 23 June 2014)

Correlated Colour Temperature
-----------------------------

-  **Alan R. Roberston**, *Adobe DNG SDK 1.3.0.0*: *dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp* (Last accessed 2 December 2013), **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Page 227
-  **Yoshi Ohno**, `Practical Use and Calculation of CCT and Duv <http://dx.doi.org/10.1080/15502724.2014.839020>`_ (Last accessed 3 March 2014)
-  **C. S. McCamy**, `Correlated Colour Temperature Approximation <http://en.wikipedia.org/wiki/Color_temperature#Approximation>`_ (Last accessed 28 June 2014)
-  **Javier Hernandez-Andres, Raymond L. Lee, Jr., and Javier Romero**, `Calculating correlated color temperatures across the entire gamut of daylight and skylight chromaticities <http://www.ugr.es/~colorimg/pdfs/ao_1999_5703.pdf>`_ (Last accessed 28 June 2014)
-  **Bongsoon Kang Ohak Moon, Changhee Hong, Honam Lee, Bonghwan Cho and Youngsun Kim**, `Design of Advanced Color - Temperature Control System for HDTV Applications <http://icpr.snu.ac.kr/resource/wop.pdf/J01/2002/041/R06/J012002041R060865.pdf>`_ (Last accessed 29 June 2014)
-  **CIE Method of Calculating D-Illuminants**, **D. B. Judd, D. L. Macadam, G. Wyszecki, H. W. Budde, H. R. Condit, S. T. Henderson and J. L. Simonds**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Page 145

Deprecated Transformations
--------------------------

-  `RGB to HSV <http://www.easyrgb.com/index.php?X=MATH&H=20#text20>`_ (Last accessed 18 May 2014)
-  `HSV to RGB <http://www.easyrgb.com/index.php?X=MATH&H=21#text21>`_ (Last accessed 18 May 2014)
-  `RGB to HSL <http://www.easyrgb.com/index.php?X=MATH&H=18#text18>`_ (Last accessed 18 May 2014)
-  `HSL to RGB <http://www.easyrgb.com/index.php?X=MATH&H=21#text21>`_ (Last accessed 18 May 2014)
-  `RGB to CMY <http://www.easyrgb.com/index.php?X=MATH&H=11#text11>`_ (Last accessed 18 May 2014)
-  `CMY to RGB <http://www.easyrgb.com/index.php?X=MATH&H=12#text12>`_ (Last accessed 18 May 2014)
-  `CMY to CMYK <http://www.easyrgb.com/index.php?X=MATH&H=13#text13>`_ (Last accessed 18 May 2014)
-  `CMYK to CMY <http://www.easyrgb.com/index.php?X=MATH&H=14#text14>`_ (Last accessed 18 May 2014)

Illuminants Relative Spectral Power Distributions
-------------------------------------------------

-  `A <http://files.cie.co.at/204.xls>`_ (Last accessed 24 February 2014)
-  `B <http://onlinelibrary.wiley.com/store/10.1002/9781119975595.app5/asset/app5.pdf?v=1&t=hwc899dh&s=01d1e0b27764970185be52b69b4480f3305ddb6c>`_ (Last accessed 12 June 2014)
-  `C <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `D50 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `D55 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `D60 <http://www.brucelindbloom.com/Eqn_DIlluminant.html>`_ (Last accessed 5 April 2014)
-  `D65 <http://files.cie.co.at/204.xls>`_ (Last accessed 24 February 2014)
-  `D75 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F1 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F2 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F3 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F4 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F5 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F6 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F7 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F8 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F9 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F10 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F11 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `F12 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 24 February 2014)
-  `FL3.1 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.2 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.3 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.4 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.5 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.6 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.7 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.8 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.9 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.10 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.11 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.12 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.13 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.14 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `FL3.15 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `HP1 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `HP2 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `HP3 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `HP4 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)
-  `HP5 <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.tables.xls>`_ (Last accessed 12 June 2014)

Illuminants Chromaticity Coordinates
------------------------------------

-  `Illuminants Chromaticity Coordinates <http://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants>`_ (Last accessed 24 February 2014)

Lightness
---------

-  `Lightness Glasser 1958 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Lightness Wyszecki 1964 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  **Charles Poynton**, `Lightness 1976 <http://www.poynton.com/PDFs/GammaFAQ.pdf>`_ (Last accessed 12 April 2014)

Luminance
---------

-  `Luminance <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_ (Last accessed 24 February 2014)
-  `Luminance Newhall 1943 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Luminance 1976 <http://www.poynton.com/PDFs/GammaFAQ.pdf>`_ (Last accessed 12 April 2014)
-  `Luminance ASTM D1535-08 <http://www.scribd.com/doc/89648322/ASTM-D1535-08e1-Standard-Practice-for-Specifying-Color-by-the-Munsell-System>`_

Luminous Efficiency Functions
-----------------------------

-  `CIE 1924 Photopic Standard Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `Judd Modified CIE 1951 Photopic Standard Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `Judd-Vos Modified CIE 1978 Photopic Standard Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `CIE 1964 Photopic 10 Degree Standard Observer <http://cvrl.ioo.ucl.ac.uk/cie.htm>`_ (Last accessed 24 February 2014)
-  `CIE 2008 2 Degree Physiologically Relevant LEF <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `CIE 2008 10 Degree Physiologically Relevant LEF <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `CIE 1951 Scotopic Standard Observer <http://www.cvrl.org/lumindex.htm>`_ (Last accessed 19 April 2014)
-  `Mesopic Weighting Function <http://en.wikipedia.org/wiki/Mesopic#Mesopic_weighting_function>`_ (Last accessed 20 June 2014)

Munsell Renotation System
-------------------------

-  **Paul Centore**, `An Open-Source Inversion Algorithm for the Munsell Renotation <http://www.99main.com/~centore/ColourSciencePapers/OpenSourceInverseRenotationArticle.pdf>`_ (Last accessed 26 July 2014)
- `The Munsell and Kubelka-Munk Toolbox <http://www.99main.com/~centore/MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html>`_ (Last accessed 26 July 2014)

Munsell Value
-------------

-  `Munsell Value Priest 1920 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Munsell Value Munsell 1933 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Munsell Value Moon 1943 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Munsell Value Saunderson 1944 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Munsell Value Ladd 1955 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  `Munsell Value ASTM D1535-08 <http://en.wikipedia.org/wiki/Lightness>`_ (Last accessed 13 April 2014)
-  **Munsell Value McCamy 1987**, `Standard Test Method for Specifying Color by the Munsell System - ASTM-D1535-1989 <https://law.resource.org/pub/us/cfr/ibr/003/astm.d1535.1989.pdf>`_ (Last accessed 23 July 2014)

Optimal Colour Stimuli
-----------------------------

-  **A**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Pages 776, 777
-  **C**, **David MacAdam**. *Maximum Visual Efficiency of Colored Materials* JOSA, Vol. 25, Pages 361, 367
-  **D65**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Pages 778, 779

RGB Colourspaces
----------------

-  `ACES RGB <http://www.oscars.org/science-technology/council/projects/aces.html>`_ (Last accessed 24 February 2014)
-  `ACES RGB Log <http://www.dropbox.com/sh/iwd09buudm3lfod/AAA-X1nVs_XLjWlzNhfhqiIna/ACESlog_v1.0.pdf>`_ (Last accessed 17 May 2014)
-  `ACES RGB Proxy 10 <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_ (Last accessed 17 May 2014)
-  `ACES RGB Proxy 12 <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_ (Last accessed 17 May 2014)
-  `Adobe RGB 1998 <http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf>`_ (Last accessed 24 February 2014)
-  `Adobe Wide Gamut RGB <http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space>`_ (Last accessed 13 April 2014)
-  `ALEXA Wide Gamut RGB <http://www.arri.com/?eID=registration&file_uid=8026>`_ (Last accessed 13 April 2014)
-  `Apple RGB <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 11 April 2014)
-  `Best RGB <http://www.hutchcolor.com/profiles/BestRGB.zip>`_ (Last accessed 11 April 2014)
-  `Beta RGB <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 11 April 2014)
-  `CIE RGB <http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data>`_ (Last accessed 24 February 2014)
-  `C-Log <http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf>`_ (Last accessed 18 April 2014)
-  `ColorMatch Colorspace <http://www.brucelindbloom.com/WorkingSpaceInfo.html>`_ (Last accessed 12 April 2014)
-  `DCI-P3 <http://www.hp.com/united-states/campaigns/workstations/pdfs/lp2480zx-dci--p3-emulation.pdf>`_ (Last accessed 24 February 2014)
-  `Don RGB 4 <http://www.hutchcolor.com/profiles/DonRGB4.zip>`_ (Last accessed 12 April 2014)
-  `ECI RGB v2 <http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip>`_ (Last accessed 13 April 2014)
-  `Ekta Space PS 5 <http://www.josephholmes.com/Ekta_Space.zip>`_ (Last accessed 13 April 2014)
-  `Max RGB <http://www.hutchcolor.com/profiles/MaxRGB.zip>`_ (Last accessed 12 April 2014)
-  `NTSC RGB <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf>`_ (Last accessed 13 April 2014)
-  `Pal/Secam RGB <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf>`_ (Last accessed 13 April 2014)
-  `Pointer's Gamut <http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls>`_ (Last accessed 24 February 2014)
-  `ProPhoto RGB <http://www.color.org/ROMMRGB.pdf>`_ (Last accessed 24 February 2014)
-  `Rec. 709 <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf>`_ (Last accessed 24 February 2014)
-  `Rec. 2020 <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-0-201208-I!!PDF-E.pdf>`_ (Last accessed 13 April 2014)
-  `Russell RGB <http://www.russellcottrell.com/photo/RussellRGB.htm>`_ (Last accessed 11 April 2014)
-  `S-Log <http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf>`_ (Last accessed 13 April 2014)
-  `SMPTE-C RGB <http://standards.smpte.org/content/978-1-61482-164-9/rp-145-2004/SEC1.body.pdf>`_ (Last accessed 13 April 2014)
-  `sRGB <http://www.color.org/srgb.pdf>`_ (Last accessed 24 February 2014)
-  `Xtreme RGB <http://www.hutchcolor.com/profiles/MaxRGB.zip>`_ (Last accessed 12 April 2014)

Spectrum
--------

-  **Spectral to XYZ Tristimulus Values**, **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3, Page 158.
-  **Stearns Spectral Bandpass Dependence Correction**, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Page 38.
-  `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: 9. INTERPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_ (Last accessed 28 May 2014)
-  `CIE 015:2004 Colorimetry, 3rd edition: 7.2.2.1 Extrapolationn <https://law.resource.org/pub/us/cfr/ibr/003/cie.15.2004.pdf>`_, `CIE 167:2005 Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: 10. EXTRAPOLATION <http://div1.cie.co.at/?i_ca_id=551&pubid=47>`_ (Last accessed 28 May 2014)

Transformations
---------------

-  **Bruce Lindbloom**, `XYZ to xyY <http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `xyY to XYZ <http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html>`_ (Last accessed 24 February 2014)
-  `XYZ to UCS <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)
-  `UCS to XYZ <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)
-  `UCS to uv <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)
-  `UCS uv to xy <http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ>`_ (Last accessed 24 February 2014)
-  `XYZ to UVW <http://en.wikipedia.org/wiki/CIE_1964_color_space>`_ (Last accessed 10 June 2014)
-  **Bruce Lindbloom**, `XYZ to Luv <http://brucelindbloom.com/Eqn_XYZ_to_Luv.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Luv to XYZ <http://brucelindbloom.com/Eqn_Luv_to_XYZ.html>`_ (Last accessed 24 February 2014)
-  `Luv to uv <http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation>`_ (Last accessed 24 February 2014)
-  `Luv uv to xy <http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Luv to LCHuv <http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `LCHuv to Luv <http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `XYZ to Lab <http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Lab to XYZ <http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `Lab to LCHab <http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html>`_ (Last accessed 24 February 2014)
-  **Bruce Lindbloom**, `LCHab to Lab <http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html>`_ (Last accessed 24 February 2014)
-  **Paul Centore**, `xyY to Munsell Colour <http://www.99main.com/~centore/MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html>`_ (Last accessed 26 July 2014)
-  **Paul Centore**, `Munsell Colour to xyY <http://www.99main.com/~centore/MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html>`_ (Last accessed 26 July 2014)

.. raw:: html

    <br/>

.. .about

_`About`
========

| **Colour** by Thomas Mansencal - Michael Parsons - 2013 - 2014
| Copyright © 2013 - 2014 – Thomas Mansencal – `thomas.mansencal@gmail.com <mailto:thomas.mansencal@gmail.com>`_
| This software is released under terms of New BSD License: http://opensource.org/licenses/BSD-3-Clause
| `http://www.thomasmansencal.com/ <http://www.thomasmansencal.com/>`_