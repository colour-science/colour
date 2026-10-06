Advanced Usage
==============

This page describes some advanced usage scenarios of **Colour**.

Environment
-----------

Various environment variables can be used to modify **Colour** behaviour at
runtime:

-   ``COLOUR_SCIENCE__DEFAULT_INT_DTYPE``: Set the default integer dtype for
    most of **Colour** computations. Possible values are `int32` and `int64`
    (default). Changing the integer dtype *will almost certainly break*
    **Colour**! *With great power comes great responsibility*.
-   ``COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE``: Set the float dtype for most of
    **Colour** computations. Possible values are `float16`, `float32` and
    `float64` (default). Changing the float dtype might result in various
    **Colour** `functionality breaking entirely <https://github.com/numpy/numpy/issues/6860>`__.
    *With great power comes great responsibility*.
-   ``COLOUR_SCIENCE__DEFAULT_COMPLEX_DTYPE``: Set the complex dtype for most of
    **Colour** computations. Possible values are `complex64` and `complex128`
    (default). Changing the complex dtype might result in various **Colour**
    functionality breaking entirely. *With great power comes great responsibility*.
-   ``COLOUR_SCIENCE__DISABLE_CACHING``: Disable the caches that can be
    disabled, useful for debugging purposes.
-   ``COLOUR_SCIENCE__DOCUMENTATION_BUILD``: Signal that the documentation is
    being built, equivalent to the *READTHEDOCS* environment variable, as
    queried by :func:`colour.utilities.is_documentation_building`.
-   ``COLOUR_SCIENCE__COLOUR__IMPORT_VAAB_COLOUR``: Import
    `vaab/colour <https://github.com/vaab/colour>`__ injection into
    **Colour** namespace. This solves the clash with
    `vaab/colour <https://github.com/vaab/colour>`__ by loading a known subset
    of the objects given by vaab/colour-0.1.5 into our namespace.
-   ``COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK``: Result in the
    :func:`warnings.showwarning` definition to be replaced with the
    :func:`colour.utilities.show_warning` definition and thus providing
    complete traceback from the point where the warning occurred.
-   ``COLOUR_SCIENCE__FILTER_RUNTIME_WARNINGS``: Filter *Colour* runtime
    warnings.
-   ``COLOUR_SCIENCE__FILTER_USAGE_WARNINGS``: Filter *Colour* usage warnings.
-   ``COLOUR_SCIENCE__FILTER_COLOUR_WARNINGS``: Filter *Colour* warnings, this
    also filters *Colour* usage and runtime warnings.
-   ``COLOUR_SCIENCE__FILTER_PYTHON_WARNINGS``: Filter *Python* warnings.
-   ``COLOUR_SCIENCE__ARRAY_API``: Enable *Python Array API Standard* dispatch,
    allowing alternative backends such as *JAX*, *PyTorch*, and *CuPy*.
    See `Array API Support`_ for details.

JEnv File
---------

**Colour** will also read the ``~/.colour-science/colour-science.jenv`` JSON
file if it exists. The syntax is that of a mapping of environment variable and
values as follows:

.. code-block:: json

    {
      "COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK": "True"
    }

Caching
-------

**Colour** uses various internal caches to improve speed and prevent redundant
processes, notably for spectral related computations.

The internal caches are managed with the :attr:`colour.utilities.CACHE_REGISTRY`
cache registry object:

.. code-block:: python

    import colour

    print(colour.utilities.CACHE_REGISTRY)

.. code-block:: text

    {'colour.colorimetry.spectrum._CACHE_RESHAPED_SDS_AND_MSDS': '0 item(s)',
     'colour.colorimetry.tristimulus_values._CACHE_LAGRANGE_INTERPOLATING_COEFFICIENTS': '0 '
                                                                                         'item(s)',
     'colour.colorimetry.tristimulus_values._CACHE_SD_TO_XYZ': '0 item(s)',
     'colour.colorimetry.tristimulus_values._CACHE_TRISTIMULUS_WEIGHTING_FACTORS': '0 '
                                                                                   'item(s)',
     'colour.quality.cfi2017._CACHE_TCS_CIE2017': '0 item(s)',
     'colour.volume.macadam_limits._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ': '0 item(s)',
     'colour.volume.macadam_limits._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS': '0 '
                                                                                      'item(s)',
     'colour.volume.spectrum._CACHE_OUTER_SURFACE_XYZ': '0 item(s)',
     'colour.volume.spectrum._CACHE_OUTER_SURFACE_XYZ_POINTS': '0 item(s)'}

See :class:`colour.utilities.CacheRegistry` class documentation for more information
on how to manage the cache registry.

Array API Support
-----------------

**Colour** has opt-in support for the
`Python Array API Standard <https://data-apis.org/array-api/latest/>`__,
enabling alternative backends such as `JAX <https://jax.readthedocs.io>`__,
`PyTorch <https://pytorch.org>`__ (including MPS GPU), and
`CuPy <https://cupy.dev>`__ alongside the default *NumPy* backend.

Enabling Array API Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Array API dispatch is **disabled by default**. *NumPy* remains the sole
backend unless explicitly opted in:

.. code-block:: bash

    # Environment variable (before importing Colour):
    export COLOUR_SCIENCE__ARRAY_API=1

.. code-block:: python

    from colour.utilities import (
        array_api_enable,
        is_array_api_enabled,
        set_array_api_enabled,
    )

    # Programmatic toggle:
    set_array_api_enabled(True)

    # Context manager (recommended for isolated use):
    with array_api_enable(True):
        # All Colour functions dispatch to the caller's backend.
        ...

    # Check current state:
    is_array_api_enabled()

When dispatch is enabled, **Colour** inspects the input arrays to determine
which backend to use. Pass a *JAX* array and the computation runs in *JAX*;
pass a *PyTorch* tensor and it runs in *PyTorch*.

Writing Backend-Aware Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every function that operates on arrays should follow a similar pattern.
From :func:`colour.intermediate_lightness_function_CIE1976`:

.. code-block:: python

    def intermediate_lightness_function_CIE1976(Y, Y_n=100):
        Y = as_float_array(Y)

        xp = array_namespace(Y, Y_n)

        Y_n = xp_as_float_array(Y_n, xp=xp, like=Y)

        Y_Y_n = Y / Y_n

        f_Y_Y_n = xp.where(
            Y_Y_n > (24 / 116) ** 3,
            spow(Y_Y_n, 1 / 3),
            (841 / 108) * Y_Y_n + 16 / 116,
        )

        return as_float(f_Y_Y_n)

**Conventions**

1. **Convert primary inputs first.** Call ``to_domain_*`` or
   ``as_float_array`` on all primary inputs **before** ``array_namespace``.
   These conversions ensure the inputs are float arrays and preserve the
   caller's backend.

2. **One** ``array_namespace`` **per function, immediately after the
   conversions.** Call ``xp = array_namespace(...)`` exactly once, on the
   line after the primary input conversions and **before any** ``tsplit``
   call or secondary promotion, passing every input that a caller might
   provide as a backend tensor. Pass converted names only: never inline a
   call (``array_namespace(as_float_array(a))``), never pass a raw name
   when the converted one exists, and never call with zero arguments when
   the function has an array input. Module-level constants (matrices,
   lookup tables) and scalar curve parameters with library defaults (e.g.
   the ``r`` exponent of *ARIB STD-B67*) do not need to be included; the
   latter are promoted per convention 4 instead.

3. **Blank line before and after** ``xp = array_namespace(...)``.

4. **Promote secondary parameters with** ``xp_as_float_array``. After
   obtaining ``xp``, convert secondary parameters (scalars, optional
   arguments, module-level constants) using
   ``xp_as_float_array(param, xp=xp, like=primary)`` which enforces
   ``DTYPE_FLOAT_DEFAULT`` and places the array on the correct device. For
   integer data, use ``xp_as_int_array``. For data that should preserve its
   original dtype, use ``xp_as_array``.

5. **Do not promote the primary input after** ``to_domain_*``.
   ``to_domain_*`` functions are backend-aware and return arrays in the
   correct namespace and dtype. However, **secondary parameters** that went
   through ``to_domain_*`` but could be scalars (e.g. ``Y`` when the primary
   is ``xy``) still need ``xp_as_float_array`` promotion because
   ``to_domain_*(scalar)`` returns a *NumPy* array regardless of the target
   backend.

6. **Promote dataclass-extracted variables.** When a function receives a
   dataclass (e.g. ``CAM_Specification_CIECAM02``) and extracts fields via
   ``astuple`` or ``tsplit``, those fields may be *NumPy* arrays even when
   other arguments are backend tensors. After ``to_domain_*`` scaling, these
   variables still need ``xp_as_float_array`` promotion because the dataclass
   fields do not carry backend information.

7. **Use** ``xp.*`` **for standard operations:** ``xp.sqrt``, ``xp.exp``,
   ``xp.log``, ``xp.where``, ``xp.stack``, ``xp.zeros``, ``xp.ones``,
   ``xp.full``, ``xp.abs``, ``xp.sum``, ``xp.mean``, ``xp.clip``,
   ``xp.squeeze``, ``xp.expand_dims``, ``xp.broadcast_to``, etc.

8. **Use** ``float("nan")`` **and** ``float("inf")`` **instead of**
   ``np.nan`` **and** ``np.inf`` as array fill values inside backend-aware
   code (e.g. in ``xp.full``, ``xp.where``). Scalar constants like
   ``np.pi`` are plain Python floats and are fine to use anywhere.

9. **Return arrays in the caller's namespace.** Functions like ``tstack``,
   ``from_range_100``, and ``as_float_array`` are already namespace-aware and
   preserve the input backend.

Array Conversion Helpers
^^^^^^^^^^^^^^^^^^^^^^^^

Two families of conversion helpers cover the two distinct conversion roles
inside a backend-aware function:

-   ``as_float_array`` / ``as_int_array`` / ``as_complex_array`` : *Auto-
    detect* helpers used at function entry on the primary inputs. They are
    namespace-aware: when *Array API* dispatch is enabled and the input is
    a non-*NumPy* array, the result is returned in the input's native
    namespace and on its device. The function does not need to know which
    backend is in play; the conversion preserves it.
-   ``xp_as_float_array`` / ``xp_as_int_array`` / ``xp_as_array`` --
    *Explicit-namespace* helpers used to promote secondary parameters
    (scalars, lists, module-level constants) into a namespace that has
    already been resolved via :func:`array_namespace`. The ``like``
    parameter matches the device of an existing primary array, which is
    how a Python float or a *NumPy* matrix lands on the right *PyTorch
    MPS* / *CUDA* / *JAX* device.

The split mirrors a real semantic distinction: primary inputs *carry*
namespace information (so the helper can discover it), secondary inputs
typically *do not* (so the caller has to push it in along with a device
target). Mixing the two roles is the most common mis-use; if a value
comes in via the function signature as the load-bearing array argument,
it is a primary, use ``as_float_array``. Everything else is a secondary,
use ``xp_as_float_array`` with ``xp=xp, like=primary``.

The explicit-namespace helpers in detail:

-   ``xp_as_float_array(a, *, xp=None, like=None)`` : Converts to float using
    ``DTYPE_FLOAT_DEFAULT``. Use for colour values, scalars, matrices, and any
    data that should be floating-point. The ``like`` parameter matches the
    device of an existing array.
-   ``xp_as_int_array(a, *, xp=None, like=None)`` : Converts to integer using
    ``DTYPE_INT_DEFAULT``. Use for indices, counts, and integer data.
-   ``xp_as_array(a, *, dtype=None, xp=None, like=None, copy=None)`` : Generic
    conversion with optional explicit ``dtype`` and copy semantics. Use when
    the original dtype should be preserved (e.g. boolean masks) or when a
    specific dtype is needed.

When ``xp`` Is Not Needed
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many functions do not call ``array_namespace`` directly because the
infrastructure handles dispatch transparently:

-   ``tsplit`` / ``tstack``: Split and stack along the last axis.
-   ``as_float_array`` / ``as_int_array`` / ``as_complex_array``: Convert to
    typed arrays, namespace-aware when dispatch is enabled.
-   ``to_domain_*`` / ``from_range_*``: Domain and range scaling.
-   ``vecmul``: Matrix-vector multiplication.
-   ``spow``: Safe power function.

If a function only uses these utilities and standard arithmetic
(``+``, ``-``, ``*``, ``/``, ``**``), it is already backend-aware without
calling ``array_namespace`` explicitly.

Compatibility Helpers (``xp_*``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Array API Standard does not cover every *NumPy* function. **Colour**
provides ``xp_*`` compatibility helpers for operations that either are not
in the standard or differ across backends. Except for the testing helpers,
each helper takes ``xp`` as a keyword-only parameter defaulting to ``None``;
when ``xp`` is omitted the namespace is derived from the input array. The
exact signature is listed per helper below.

**Array Creation and Conversion**

The ``xp_as_array`` / ``xp_as_float_array`` / ``xp_as_int_array``
explicit-namespace helpers are described in the *Helpers in detail*
section above. The remaining creation helpers are:

-   ``xp_astype(a, dtype, *, xp=None)`` : Portable ``a.astype(dtype)``.
-   ``xp_ascontiguousarray(a, *, xp=None)`` : Materialise a C-contiguous array.
-   ``xp_linspace(start, stop, *, num=50, xp=None, like=None, **kwargs)`` :
    ``np.linspace``.

**Shape Manipulation**

-   ``xp_atleast_1d(a, *, xp=None)`` : ``np.atleast_1d``.
-   ``xp_atleast_2d(a, *, xp=None)`` : ``np.atleast_2d``.
-   ``xp_reshape(a, shape, *, xp=None)`` : ``np.reshape``.
-   ``xp_broadcast_to(a, shape, *, xp=None)`` : ``np.broadcast_to``.
-   ``xp_pad(a, pad_width, *args, xp=None, **kwargs)`` : ``np.pad``.
-   ``xp_insert(a, indices, values, *, axis=None, xp=None)`` : ``np.insert``.
-   ``xp_resize(a, new_shape, *, xp=None)`` : ``np.resize`` (NumPy fallback).
-   ``xp_squeeze(a, *, axis=None, xp=None)`` : ``np.squeeze``. When ``axis`` is
    ``None``, all size-1 dimensions are squeezed (the *Array API* standard
    requires an explicit ``axis``).

**Math and Statistics**

-   ``xp_average(a, *, axis=None, weights=None, xp=None)`` : ``np.average``.
-   ``xp_median(a, *, axis=None, xp=None)`` : ``np.median`` (NumPy fallback).
-   ``xp_nanmean(a, *, axis=None, xp=None)`` : ``np.nanmean``.
-   ``xp_gradient(f, *varargs, xp=None, axis=None)`` : ``np.gradient``
    (NumPy fallback).
-   ``xp_trapezoid(y, *, x=None, dx=1.0, axis=-1, xp=None)`` :
    ``np.trapezoid`` (NumPy fallback).
-   ``xp_radians(a, *, xp=None)`` / ``xp_degrees(a, *, xp=None)`` : Angle
    conversion (namespace derived from ``a``).
-   ``xp_round(a, *, decimals=0, xp=None)`` : ``np.round``.
-   ``xp_sinc(a, *, xp=None)`` : ``np.sinc``.

**Selection and Comparison**

-   ``xp_select(condlist, choicelist, *, default=0, xp=None)`` : ``np.select``
    (native Array API implementation using ``xp.where``).
-   ``xp_interp(x, x_data, fp, *, xp=None)`` : ``np.interp`` (NumPy fallback).
-   ``xp_nan_to_num(a, *, nan=0.0, posinf=None, neginf=None, xp=None)`` :
    ``np.nan_to_num``.
-   ``xp_isclose(a, b, *, rtol=1e-5, atol=1e-8, xp=None)`` : ``np.isclose``.
-   ``xp_isin(element, test_elements, *, xp=None, like=None)`` : ``np.isin``.

**Linear Algebra**

-   ``xp_lstsq(a, b, *, rcond=None, xp=None)`` : ``np.linalg.lstsq``.
-   ``xp_eig(a, *, xp=None)`` : ``np.linalg.eig`` (NumPy fallback for MPS).
-   ``xp_eigh(a, *, xp=None)`` : ``np.linalg.eigh`` (NumPy fallback for MPS).
-   ``xp_create_diagonal(a, *, xp=None)`` : ``np.diag``.
-   ``xp_matrix_transpose(a, *, xp=None)`` : Matrix transpose materialised as
    a C-contiguous array.

**Set Operations**

-   ``xp_unique(a, *, xp=None, **kwargs)`` : ``np.unique``.
-   ``xp_setxor1d(a, b, *, xp=None)`` : ``np.setxor1d``.

**Testing**

-   ``xp_assert_close(actual, desired, *, rtol=None, atol=None,
    err_msg="")`` :
    ``np.testing.assert_allclose`` (converts both arguments via
    ``as_ndarray`` internally).
-   ``xp_assert_equal(actual, desired, *, err_msg="")`` :
    ``np.testing.assert_array_equal``.

See :mod:`colour.utilities.array` for the full list.

**Usage example** from
:func:`colour.temperature.CCT_to_xy_Kang2002`:

.. code-block:: python

    def CCT_to_xy_Kang2002(CCT):
        CCT = as_float_array(CCT)

        xp = array_namespace(CCT)

        CCT_3 = CCT**3
        CCT_2 = CCT**2

        x = xp.where(
            CCT <= 4000,
            -0.2661239 * 10**9 / CCT_3
            - 0.2343589 * 10**6 / CCT_2
            + 0.8776956 * 10**3 / CCT
            + 0.179910,
            -3.0258469 * 10**9 / CCT_3
            + 2.1070379 * 10**6 / CCT_2
            + 0.2226347 * 10**3 / CCT
            + 0.24039,
        )

        x_3 = x**3
        x_2 = x**2

        cnd_l = [CCT <= 2222, xp.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
        i = -1.1063814 * x_3 - 1.34811020 * x_2 + 2.18555832 * x - 0.20219683
        j = -0.9549476 * x_3 - 1.37418593 * x_2 + 2.09137015 * x - 0.16748867
        k = 3.0817580 * x_3 - 5.8733867 * x_2 + 3.75112997 * x - 0.37001483
        y = xp_select(cnd_l, [i, j, k], xp=xp)

        return tstack([x, y])

Converting NumPy Arrays for Backend Use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*NumPy* constants (e.g. matrices defined at module level) must be promoted
into the caller's namespace inside backend-aware functions. Use
``xp_as_float_array`` with the ``like`` parameter to match the device and
enforce ``DTYPE_FLOAT_DEFAULT``.
From :func:`colour.adaptation.matrix_chromatic_adaptation_VonKries`:

.. code-block:: python

    def matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr, transform="CAT02"):
        XYZ_w = as_float_array(XYZ_w)
        XYZ_wr = as_float_array(XYZ_wr)

        xp = array_namespace(XYZ_w, XYZ_wr)

        M = xp_as_float_array(CHROMATIC_ADAPTATION_TRANSFORMS[transform], xp=xp, like=XYZ_w)

        RGB_w = vecmul(M, XYZ_w)
        RGB_wr = vecmul(M, XYZ_wr)

        with sdiv_mode():
            D = sdiv(RGB_wr, RGB_w)

        ...

The ``like`` parameter places the constant on the correct device (e.g. MPS
GPU). ``xp_as_float_array`` enforces ``DTYPE_FLOAT_DEFAULT``, ensuring that
a ``float64`` module-level constant is correctly narrowed to ``float32`` when
the global default dtype has been changed.

Domain/Range Functions and Promotion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters that go through ``to_domain_*`` (e.g. ``to_domain_100``,
``to_domain_1``) are already backend-aware; they preserve the input
namespace and return float arrays. These do **not** need a subsequent
``xp_as_float_array`` call. Only raw scalars and optional parameters that
did not go through ``to_domain_*`` need promotion.
From :func:`colour.adaptation.chromatic_adaptation_forward_CMCCAT2000`:

.. code-block:: python

    def chromatic_adaptation_forward_CMCCAT2000(
        XYZ, XYZ_w, XYZ_wr, L_A1, L_A2, surround=...
    ):
        # to_domain_* handles dtype and backend; no further promotion needed.
        XYZ = to_domain_100(XYZ)
        XYZ_w = to_domain_100(XYZ_w)
        XYZ_wr = to_domain_100(XYZ_wr)

        xp = array_namespace(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)

        # Only raw scalars need promotion.
        L_A1 = xp_as_float_array(L_A1, xp=xp, like=XYZ)
        L_A2 = xp_as_float_array(L_A2, xp=xp, like=XYZ)

        ...

Writing Backend-Aware Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests are parametrised across backends using the ``xp`` fixture defined in
``conftest.py``. Every test method that exercises a backend-aware function
should accept the ``xp`` parameter.
From ``colour/models/tests/test_igpgtg.py``:

.. code-block:: python

    class TestXYZ_to_IgPgTg:
        def test_XYZ_to_IgPgTg(self, xp: ModuleType) -> None:
            xp_assert_close(
                XYZ_to_IgPgTg(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
                [0.42421258, 0.18632491, 0.10689223],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        def test_n_dimensional_XYZ_to_IgPgTg(self, xp: ModuleType) -> None:
            XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
            IgPgTg = as_ndarray(XYZ_to_IgPgTg(XYZ))

            XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
            IgPgTg = xp.tile(xp_as_array(IgPgTg, xp=xp), (6, 1))
            xp_assert_close(XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS)

            XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
            IgPgTg = xp_reshape(xp_as_array(IgPgTg, xp=xp), (2, 3, 3), xp=xp)
            xp_assert_close(XYZ_to_IgPgTg(XYZ), IgPgTg, atol=TOLERANCE_ABSOLUTE_TESTS)

**Conventions**

-   **Inputs**: Use ``xp_as_array([...], xp=xp)`` for all data passed to the
    function under test.
-   **Expected values**: Use **plain Python lists** as the second argument to
    ``xp_assert_close`` / ``xp_assert_equal``. No ``np.array`` wrapping
    needed : the assertion helpers convert internally.
-   **Element-wise arithmetic on expected values**: For arithmetic
    like ``expected * 100``, use ``xp_as_array([...], xp=xp) * 100`` since
    ``[...] * 100`` is Python list repetition.
-   **n-dimensional tiling**: Use ``xp.tile`` and ``xp_reshape`` for shape
    tests.
-   **Scalar-input tests**: If a test only passes scalars (not arrays), add
    ``xp: ModuleType  # noqa: ARG002`` to ensure it runs across backends even
    though ``xp`` is not directly referenced.
-   **NumPy-only functions**: If the function under test is inherently
    *NumPy*-only (e.g. uses ``SpectralDistribution``, ``scipy.optimize``, or
    in-place assignment), add ``xp: ModuleType  # noqa: ARG002`` and keep
    inputs as ``np.array``.

.. note::

    In tests, ``xp_as_array`` is preferred over ``xp_as_float_array``
    because test inputs are explicit literals whose dtype is already correct.
    ``xp_as_float_array`` is for source code where ``DTYPE_FLOAT_DEFAULT``
    enforcement matters.

What Stays NumPy-Only
~~~~~~~~~~~~~~~~~~~~~

Not everything needs backend support. The following are intentionally *NumPy*-only:

-   **Spectral classes**: ``SpectralDistribution``, ``MultiSpectralDistributions``,
    ``Signal``, ``MultiSignals`` : these use *Pandas* internally.
-   **I/O operations**: File reading/writing, image loading.
-   **Scipy-dependent optimisation**: Functions using ``scipy.optimize``,
    ``scipy.interpolate`` : callbacks must receive *NumPy* arrays. Use
    ``as_ndarray()`` to convert before passing to *SciPy*.
-   **In-place operations**: *JAX* arrays are immutable
    (``a[i] = v`` raises ``TypeError``). Algorithms requiring in-place
    mutation stay *NumPy*-only.

Common Pitfalls
~~~~~~~~~~~~~~~

1.  **List repetition vs element-wise multiplication**: ``[1, 2, 3] * 10``
    produces ``[1, 2, 3, 1, 2, 3, ...]`` (30 elements).
    ``np.array([1, 2, 3]) * 10`` produces ``[10, 20, 30]``. Element-wise
    arithmetic requires wrapping with ``xp_as_array`` or
    ``xp_as_float_array`` first.

2.  **MPS GPU limitations**: Apple's MPS backend does not support
    ``complex128`` or ``float64``. The ``xp_as_array`` helper
    automatically falls back to ``float32`` / ``complex64`` and emits a
    :class:`colour.utilities.runtime_warning` reporting the downcast.
    The ``xp_eig`` / ``xp_eigh`` helpers fall back to *NumPy* when the
    backend lacks ``linalg.eig``. The ``xp_linspace`` helper falls back
    to ``float32`` on the same path, also with a warning.

3.  **Mixing namespaces**: Never multiply a *NumPy* array by a *PyTorch*
    tensor directly. Use ``xp_as_float_array`` to promote constants into
    the caller's namespace first. Mixing two non-*NumPy* backends in the
    same call (e.g. a *JAX* array and a *PyTorch* tensor) routes through
    :func:`colour.utilities.array_namespace` and raises ``TypeError``;
    there is no implicit cross-backend coercion.

4.  **Scalar-promotion cache**: ``xp_as_array`` memoises promotions of
    *Python* scalars and small (``<= 16`` element) *NumPy* constants
    keyed on ``(value, namespace, device, dtype)``. The cache lives in
    :attr:`colour.utilities.CACHE_REGISTRY` and is bypassed when the
    caching context is disabled
    (``with caching_enable(False): ...``). The cache eliminates repeated
    CPU-to-GPU transfers of module-level matrices and tolerance scalars;
    when debugging unexpected device placement, disable it first.

5.  **Module-level constants**: Matrices and lookup tables defined at module
    level with ``np.array`` are fine : they stay as *NumPy* arrays. Convert
    them inside functions using
    ``xp_as_float_array(CONSTANT, xp=xp, like=input)``.

6.  **Choosing the right conversion helper**: Use ``xp_as_float_array`` for
    colour data, scalars, and constants (enforces ``DTYPE_FLOAT_DEFAULT``).
    Use ``xp_as_int_array`` for indices and counts. Use ``xp_as_array`` only
    when the original dtype must be preserved (boolean masks, generic
    pass-through).

Debugging
~~~~~~~~~

Use :class:`colour.utilities.trace_array_namespace` to trace which array
namespace each function call resolves to:

.. code-block:: python

    import numpy as np
    import colour
    from colour.utilities import trace_array_namespace

    XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    with trace_array_namespace():
        Lab = colour.XYZ_to_Lab(XYZ)

This prints an indented call tree showing every function invocation, the types
and shapes of array arguments, and return values:

.. code-block:: text

    XYZ_to_Lab(XYZ: ndarray[3], illuminant: ndarray[2])
      to_domain_1(a: ndarray[3], scale_factor: int, dtype: None)
        as_float_array(a: ndarray[3], dtype: type)
          as_array(a: ndarray[3], dtype: type)
          -> ndarray[3]
        -> ndarray[3]
        ndarray_copy(a: ndarray[3])
          array_namespace()
          is_numpy_namespace(xp: module)
        -> ndarray[3]
        ...
      -> ndarray[3]
      tsplit(a: ndarray[3], dtype: None)
        ...
      -> ndarray[3]
      ...
    -> ndarray[3]

When using a non-*NumPy* backend, array arguments show their backend type
and calls where multiple backends coexist are flagged as ``MIXED``:

.. code-block:: python

    import torch
    import colour
    from colour.utilities import array_api_enable, trace_array_namespace

    XYZ = torch.tensor([0.20654008, 0.12197225, 0.05136952], dtype=torch.float64)
    with array_api_enable(True), trace_array_namespace():
        Lab = colour.XYZ_to_Lab(XYZ)

.. code-block:: text

    XYZ_to_Lab(XYZ: torch.Tensor[3], illuminant: ndarray[2]) [MIXED]
      to_domain_1(a: torch.Tensor[3], scale_factor: int, dtype: None)
        as_float_array(a: torch.Tensor[3], dtype: type)
          array_namespace()
          -> torch.Tensor[3]
        -> torch.Tensor[3]
        ndarray_copy(a: torch.Tensor[3])
          ...
        -> torch.Tensor[3]
        ...
      -> torch.Tensor[3]
      ...
    -> torch.Tensor[3]

The ``[MIXED]`` flag on the first line indicates that ``XYZ`` is a
*PyTorch* tensor while ``illuminant`` is a *NumPy* array (the default
*D65* illuminant). This is expected here : the function promotes the
illuminant internally.
