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
-  Correlated colour temperature calculation:

  -  Wyszecki & Roberston method implementation.
  -  Yoshi Ohno method implementation.
  -  CIE D-illuminant implementation.

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
  -  D-illuminant CCT to xy.

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
-  Colourspaces derivation.
-  Colour difference calculation with following methods:

  -  ΔE CIE 1976.
  -  ΔE CIE 1994.
  -  ΔE CIE 2000.
  -  ΔE CMC.

-  Colour rendering index calculation.
-  Colour rendition chart data.
-  Colourspaces visualisation within **Autodesk Maya**.
-  First order colour fit.
-  Comprehensive plotting capabilities.

.. raw:: html

    <br/>

