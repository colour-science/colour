Colour - TODO
=============

TODO
----

-   colour/__init__.py

    -   Line 888 : # TODO: Remove legacy printing support when deemed appropriate.


-   colour/colorimetry/spectrum.py

    -   Line 1174 : # TODO: Provide support for fractional interval like 0.1, etc...


-   colour/colorimetry/tristimulus_values.py

    -   Line 1052 : # TODO: Investigate code vectorisation.


-   colour/appearance/ciecam02.py

    -   Line 382 : # TODO: Compute hue composition.


-   colour/appearance/ciecam16.py

    -   Line 329 : # TODO: Compute hue composition.


-   colour/appearance/cam16.py

    -   Line 317 : # TODO: Compute hue composition.


-   colour/appearance/hellwig2022.py

    -   Line 360 : # TODO: Compute hue composition.


-   colour/appearance/hunt.py

    -   Line 495 : # TODO: Implement hue quadrature & composition computation.
    -   Line 528 : # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.


-   colour/appearance/rlab.py

    -   Line 292 : # TODO: Implement hue composition computation.


-   colour/appearance/nayatani95.py

    -   Line 317 : # TODO: Implement hue quadrature & composition computation.
    -   Line 330 : # TODO: Investigate components usage. M_RG, M_YB = tsplit(colourfulness_components(C_RG, C_YB, brightness_ideal_white))


-   colour/appearance/llab.py

    -   Line 369 : # TODO: Implement hue composition computation.


-   colour/recovery/otsu2018.py

    -   Line 663 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.


-   colour/recovery/tests/test_jiang2013.py

    -   Line 61 : # TODO: Last eigen value seems to be very sensitive and produce differences on ARM.


-   colour/io/ocio.py

    -   Line 30 : # TODO: Reinstate coverage and doctests when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/ctl.py

    -   Line 66 : # TODO: Reinstate coverage when "ctlrender" is trivially available cross-platform.


-   colour/io/tests/test_ocio.py

    -   Line 37 : # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is released.


-   colour/io/tests/test_ctl.py

    -   Line 39 : # TODO: Reinstate coverage when "ctlrender" is tivially available cross-platform.


-   colour/io/tests/test_image.py

    -   Line 307 : # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0 image = read_image_OpenImageIO( os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16') self.assertIs(image.dtype, np.dtype('float16')) self.assertEqual(np.min(image), 0.0) self.assertEqual(np.max(image), 1.0)


-   colour/io/luts/sequence.py

    -   Line 115 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.


-   colour/io/luts/operator.py

    -   Line 80 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.
    -   Line 256 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.


-   colour/io/luts/lut.py

    -   Line 171 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.


-   colour/graph/conversion.py

    -   Line 1364 : # TODO: Remove the following warning whenever the automatic colour conversion graph implementation is considered stable.


-   colour/models/rgb/derivation.py

    -   Line 249 : # TODO: Investigate if we return an ndarray here with primaries and whitepoint stacked together.


-   colour/models/rgb/tests/test_derivation.py

    -   Line 325 : # TODO: Simplify that monster.


-   colour/utilities/verbose.py

    -   Line 631 : # TODO: Implement support for "pyproject.toml" file whenever "TOML" is supported in the standard library.


-   colour/utilities/array.py

    -   Line 559 : # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is addressed.
    -   Line 605 : # TODO: Reassess implementation when and if https://github.com/numpy/numpy/issues/11956 is addressed.
    -   Line 833 : # TODO: Investigate behaviour on Windows.
    -   Line 890 : # TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]" when Python 3.7 is dropped.


-   colour/plotting/models.py

    -   Line 2003 : # TODO: Filter appropriate colour models.


-   colour/plotting/graph.py

    -   Line 89 : # TODO: Investigate API to trigger the conversion graph build.


-   colour/plotting/common.py

    -   Line 797 : # TODO: Reassess according to https://github.com/matplotlib/matplotlib/issues/1077
    -   Line 920 : # TODO: Consider using "MutableMapping" here.
    -   Line 1526 : # TODO: Remove when "Matplotlib" minimum version can be set to 3.5.0.


-   colour/characterisation/aces_it.py

    -   Line 402 : # TODO: Remove when removing the "colour.sd_blackbody" definition warning.


-   colour/characterisation/correction.py

    -   Line 409 : # TODO: Generalise polynomial expansion.


-   colour/notation/munsell.py

    -   Line 1248 : # TODO: Consider refactoring implementation.


-   colour/continuous/signal.py

    -   Line 421 : # TODO: Check for interpolator compatibility.
    -   Line 481 : # TODO: Check for extrapolator compatibility.


-   colour/hints/__init__.py

    -   Line 46 : # TODO: Drop "typing_extensions" when "Google Colab" uses Python >= 3.8. Remove exclusion in ".pre-commit-config.yaml" file for "pyupgrade".
    -   Line 163 : # TODO: Use "typing.Literal" when minimal Python version is raised to 3.8.
    -   Line 166 : # TODO: Revisit to use Protocol.
    -   Line 169 : # TODO: Drop mocking when minimal "Numpy" version is 1.21.x.
    -   Line 263 : # TODO: Use "numpy.typing.NDArray" when minimal Numpy version is raised to 1.21.
    -   Line 270 : # TODO: Drop when minimal Python is raised to 3.9.


-   colour/algebra/interpolation.py

    -   Line 429 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.
    -   Line 825 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.
    -   Line 1050 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.
    -   Line 1425 : # TODO: Remove pragma when https://github.com/python/mypy/issues/3004 is resolved.


-   colour/algebra/tests/test_interpolation.py

    -   Line 1171 : # TODO: Revisit if the interpolator can be applied on non-uniform "x" independent variable.

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour <https://github.com/colour-science/colour>`__
