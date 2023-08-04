Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 897 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/spectrum.py

    -   Line 1189 : # TODO: Provide support for fractional interval like 0.1, etc...


-   colour/colorimetry/tristimulus_values.py

    -   Line 1064 : # TODO: Investigate code vectorisation.


-   colour/appearance/ciecam02.py

    -   Line 376 : # TODO: Compute hue composition.


-   colour/appearance/ciecam16.py

    -   Line 324 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 312 : # TODO: Compute hue composition.


-   colour/appearance/hellwig2022.py

    -   Line 354 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 486 : # TODO: Implement hue quadrature & composition computation.
    -   Line 517 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 286 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 310 : # TODO: Implement hue quadrature & composition computation.
    -   Line 323 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 395 : # TODO: Implement hue composition computation.


-   colour/recovery/tests/test_jiang2013.py

    -   Line 61 : # TODO: Last eigen value seems to be very sensitive and produce differences on ARM.


-   colour/io/ocio.py

    -   Line 30 : # TODO: Reinstate coverage and doctests when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/ctl.py

    -   Line 63 : # TODO: Reinstate coverage when "ctlrender" is trivially available cross-platform.


-   colour/io/tests/test_ocio.py

    -   Line 37 : # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/tests/test_ctl.py

    -   Line 39 : # TODO: Reinstate coverage when "ctlrender" is tivially available cross-platform.


-   colour/io/tests/test_image.py

    -   Line 307 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/graph/conversion.py

    -   Line 1416 : # TODO: Remove the following warning whenever the automatic colour conversion graph implementation is considered stable.


-   colour/models/rgb/derivation.py

    -   Line 239 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/tests/test_rgb_colourspace.py

    -   Line 346 : # TODO: Remove tests when dropping deprecated signature support.
    -   Line 549 : # TODO: Remove tests when dropping deprecated signature support.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 325 : # TODO: Simplify that monster.


-   colour/utilities/verbose.py

    -   Line 627 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library.


-   colour/utilities/array.py

    -   Line 574 : # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is addressed.
    -   Line 862 : # TODO: Investigate behaviour on Windows.
    -   Line 919 : # TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]" when Python 3.7 is dropped.


-   colour/plotting/models.py

    -   Line 1997 : # TODO: Filter appropriate colour models. NOTE: "dtype=object" is required for ragged array support in "Numpy" 1.24.0.


-   colour/plotting/graph.py

    -   Line 88 : # TODO: Investigate API to trigger the conversion graph build.


-   colour/plotting/common.py

    -   Line 788 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 907 : # TODO: Consider using "MutableMapping" here.


-   colour/characterisation/aces_it.py

    -   Line 409 : # TODO: Remove when removing the "colour.sd_blackbody" definition warning.


-   colour/characterisation/correction.py

    -   Line 461 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1247 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 423 : # TODO: Check for interpolator compatibility.
    -   Line 483 : # TODO: Check for extrapolator compatibility.


-   colour/hints/__init__.py

    -   Line 117 : # TODO: Revisit to use Protocol.


-   colour/algebra/tests/test_interpolation.py

    -   Line 1174 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers - `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
