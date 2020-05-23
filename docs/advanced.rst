Advanced
========

Environment
-----------

Various environment variables can be used to modify **Colour** behaviour at
runtime:

-   `COLOUR_SCIENCE__FLOAT_PRECISION`: Sets the float precision for most of
    **Colour** computations. Possible values are `float16`, `float32` and
    `float64` (default). Changing float precision might result in various
    **Colour** `functionality breaking entirely <https://github.com/numpy/numpy/issues/6860>`__.
    *With great power comes great responsibility*.
-   `COLOUR_SCIENCE__INT_PRECISION`: Sets the integer precision for most of
    **Colour** computations. Possible values are `int8`, `int16`, `int32`,
    and `int64` (default). Changing integer precision
    *will almost certainly break* **Colour**!
    *With great power comes great responsibility*.
-   `COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK`: results in the
    :func:`warnings.showwarning` definition to be replaced with the
    :func:`colour.utilities.show_warning` definition and thus providing
    complete traceback from the point where the warning occurred.

Using Colour without Scipy
--------------------------

With the release of `Colour 0.3.8 <https://github.com/colour-science/colour/releases/tag/v0.3.8>`__,
`SciPy <http://www.scipy.org/>`__ became a requirement.

**Scipy** is notoriously hard to compile, especially
`on Windows <https://colour-science.slack.com/messages/C02KH93GT/>`__.
Some Digital Content Creation (DCC) applications are shipping Python interpreters
compiled with versions of
`Visual Studio <https://visualstudio.microsoft.com/>`__ such as 2011 or 2015.
Those are incompatible with the Python Wheels commonly built with
`Visual Studio 2008 (Python 2.7) or Visual Studio 2017 (Python 3.6) <https://devguide.python.org/setup/?highlight=windows#windows>`__.

It is however possible to use **Colour** in a partially broken and mock **Scipy**
by using the `mock_for_colour.py <https://github.com/colour-science/colour/tree/develop/utilities>`__
module.

Assuming it is available for import, a typical usage would be as follows:

.. code-block:: python

    import sys
    from mock_for_colour import MockModule

    for module in ('scipy', 'scipy.interpolate', 'scipy.spatial',
                   'scipy.spatial.distance', 'scipy.optimize'):
        sys.modules[str(module)] = MockModule(str(module))

    import colour

    xyY = (0.4316, 0.3777, 0.1008)
    colour.xyY_to_XYZ(xyY)

.. code-block:: text

    array([ 0.11518475,  0.1008    ,  0.05089373])

Or directly using the ``mock_scipy_for_colour`` definition:

.. code-block:: python

    from mock_for_colour import mock_scipy_for_colour

    mock_scipy_for_colour()

    import colour

    xyY = (0.4316, 0.3777, 0.1008)
    colour.xyY_to_XYZ(xyY)

.. code-block:: text

    array([ 0.11518475,  0.1008    ,  0.05089373])

Anything relying on the spectral code will be unusable, but a great amount of
useful functionality will still be available.
