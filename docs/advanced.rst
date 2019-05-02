Advanced
========

Using Colour without Scipy
--------------------------

With the release of `Colour 0.3.8 <https://github.com/colour-science/colour/releases/tag/v0.3.8>`_,
`SciPy <http://www.scipy.org/>`_ became a requirement.

**Scipy** is notoriously hard to compile, especially
`on Windows <https://colour-science.slack.com/messages/C02KH93GT/>`_.
Some Digital Content Creation (DCC) applications are shipping Python interpreters
compiled with versions of
`Visual Studio <https://visualstudio.microsoft.com/>`_ such as 2011 or 2015.
Those are incompatible with the Python Wheels commonly built with
`Visual Studio 2008 (Python 2.7) or Visual Studio 2017 (Python 3.6) <https://devguide.python.org/setup/?highlight=windows#windows>`_.

It is however possible to use **Colour** in a partially broken and mock **Scipy**
by using the `mock_for_colour.py <https://github.com/colour-science/colour/tree/develop/utilities>`_
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
