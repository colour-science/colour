_`Introduction`
===============

**Color** is a **Python** color science package implementing a comprehensive number of color theory transformations and algorithms.

_`History`
----------

**Color** started as a raw conversion building block for `The Moving Picture Company <http://www.moving-picture.com>`_ stills ingestion pipeline.

Generic objects have been extracted, reorganised and are now provided as a nice packaged API while keeping the undisclosable code private. The original `MPC <http://www.moving-picture.com>`_ *camelCase* naming convention and code style has been changed for *Pep8* compliance.

Matplotlib implementation idea is coming from the excellent *Mark Kness*'s `ColorPy <http://markkness.net/colorpy/ColorPy.html>`_ **Python** package.

_`Highlights`
-------------

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

- Color rendition charts spectral data.
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
  -  CIE D-illuminant implementation.

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
-  Colorspaces derivation.
-  Color difference calculation with following methods:

  -  ΔE CIE 1976.
  -  ΔE CIE 1994.
  -  ΔE CIE 2000.
  -  ΔE CMC.

-  Color rendering index calculation.
-  Color rendition chart data.
-  Colorspaces visualisation within **Autodesk Maya**.
-  First order color fit.
-  Comprehensive plotting capabilities.

.. raw:: html

    <br/>

