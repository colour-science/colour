"""
CTL Processing
==============

Defines the object for the *Color Transformation Language* (CTL) processing:

-   :func:`colour.io.ctl_render`
-   :func:`colour.io.process_image_ctl`
-   :func:`colour.io.template_ctl_transform_float`
-   :func:`colour.io.template_ctl_transform_float3`
"""

from __future__ import annotations

import os
import numpy as np
import subprocess  # nosec
import textwrap
import tempfile

from colour.hints import (
    Any,
    ArrayLike,
    Dict,
    NDArrayFloat,
    Sequence,
)
from colour.io import as_3_channels_image, read_image, write_image
from colour.utilities import (
    as_float_array,
    as_float,
    optional,
    required,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "EXECUTABLE_CTL_RENDER",
    "ARGUMENTS_CTL_RENDER_DEFAULTS",
    "ctl_render",
    "process_image_ctl",
    "template_ctl_transform_float",
    "template_ctl_transform_float3",
]


EXECUTABLE_CTL_RENDER: str = "ctlrender"
"""
*ctlrender* executable name.
"""

ARGUMENTS_CTL_RENDER_DEFAULTS: tuple = ("-verbose", "-force")
"""
*ctlrender* invocation default arguments.
"""

# TODO: Reinstate coverage when "ctlrender" is trivially available
# cross-platform.


@required("ctlrender")
def ctl_render(
    path_input: str,
    path_output: str,
    ctl_transforms: Sequence[str] | Dict[str, Sequence[str]],
    *args: Any,
    **kwargs: Any,
) -> subprocess.CompletedProcess:  # pragma: no cover
    """
    Call *ctlrender* on given input image using given *CTL* transforms.

    Parameters
    ----------
    path_input
        Input image path.
    path_output
        Output image path.
    ctl_transforms
        Sequence of *CTL* transforms to apply on the image, either paths to
        existing *CTL* transforms, multi-line *CTL* code transforms or a mix of
        both or dictionary of sequence of *CTL* transforms to apply on the
        image and their sequence of parameters.

    Other Parameters
    ----------------
    args
        Arguments passed to *ctlrender*, e.g. ``-verbose``, ``-force``.
    kwargs
        Keywords arguments passed to the sub-process calling *ctlrender*, e.g.
        to define the environment variables such as ``CTL_MODULE_PATH``.

    Notes
    -----
    -   The multi-line *CTL* code transforms are written to disk in a temporary
        location so that they can be used by *ctlrender*.

    Returns
    -------
    :class:`subprocess.CompletedProcess`
        *ctlrender* process completed output.

    Examples
    --------
    >>> ctl_adjust_exposure_float = template_ctl_transform_float(
    ...     "rIn * pow(2, exposure)",
    ...     description="Adjust Exposure",
    ...     parameters=["input float exposure = 0.0"],
    ... )
    >>> TESTS_ROOT_RESOURCES = os.path.join(
    ...     os.path.dirname(__file__), "tests", "resources"
    ... )
    >>> print(
    ...     ctl_render(
    ...         f"{TESTS_ROOT_RESOURCES}/CMS_Test_Pattern.exr",
    ...         f"{TESTS_ROOT_RESOURCES}/CMS_Test_Pattern_Float.exr",
    ...         {ctl_adjust_exposure_float: ["-param1 exposure 3.0"]},
    ...         "-verbose",
    ...         "-force",
    ...     ).stderr.decode("utf-8")
    ... )  # doctest: +SKIP
    global ctl parameters:
    <BLANKLINE>
    destination format: exr
           input scale: default
          output scale: default
    <BLANKLINE>
       ctl script file: \
/var/folders/xr/sf4r3m2s761fl25h8zsl3k4w0000gn/T/tmponm0kvu2.ctl
         function name: main
       input arguments:
                   rIn: float (varying)
                   gIn: float (varying)
                   bIn: float (varying)
                   aIn: float (varying)
              exposure: float (defaulted)
      output arguments:
                  rOut: float (varying)
                  gOut: float (varying)
                  bOut: float (varying)
                  aOut: float (varying)
    <BLANKLINE>
    <BLANKLINE>
    """

    if len(args) == 0:
        args = ARGUMENTS_CTL_RENDER_DEFAULTS

    kwargs["capture_output"] = kwargs.get("capture_output", True)

    command = [EXECUTABLE_CTL_RENDER]

    ctl_transforms_mapping: Dict[str, Sequence]
    if isinstance(ctl_transforms, Sequence):
        ctl_transforms_mapping = dict.fromkeys(ctl_transforms, [])
    else:
        ctl_transforms_mapping = ctl_transforms

    temp_filenames = []
    for ctl_transform, parameters in ctl_transforms_mapping.items():
        if "\n" in ctl_transform:
            _descriptor, temp_filename = tempfile.mkstemp(suffix=".ctl")
            with open(temp_filename, "w") as temp_file:
                temp_file.write(ctl_transform)
                ctl_transform = temp_filename
                temp_filenames.append(temp_filename)
        elif not os.path.exists(ctl_transform):
            raise FileNotFoundError(
                f'{ctl_transform} "CTL" transform does not exist!'
            )

        command.extend(["-ctl", ctl_transform])
        for parameter in parameters:
            command.extend(parameter.split())

    command += [path_input, path_output]

    for arg in args:
        command += arg.split()

    completed_process = subprocess.run(command, **kwargs)  # nosec

    for temp_filename in temp_filenames:
        os.remove(temp_filename)

    return completed_process


@required("ctlrender")
def process_image_ctl(
    a: ArrayLike,
    ctl_transforms: Sequence[str] | Dict[str, Sequence[str]],
    *args: Any,
    **kwargs: Any,
) -> NDArrayFloat:  # pragma: no cover
    """
    Process given image data with *ctlrender* using given *CTL* transforms.

    Parameters
    ----------
    a
        Image data to process with *ctlrender*.
    ctl_transforms
        Sequence of *CTL* transforms to apply on the image, either paths to
        existing *CTL* transforms, multi-line *CTL* code transforms or a mix of
        both or dictionary of sequence of *CTL* transforms to apply on the
        image and their sequence of parameters.

    Other Parameters
    ----------------
    args
        Arguments passed to *ctlrender*, e.g. ``-verbose``, ``-force``.
    kwargs
        Keywords arguments passed to the sub-process calling *ctlrender*, e.g.
        to define the environment variables such as ``CTL_MODULE_PATH``.

    Notes
    -----
    -   The multi-line *CTL* code transforms are written to disk in a temporary
        location so that they can be used by *ctlrender*.

    Returns
    -------
    :class`numpy.ndarray`
        Processed image data.

    Examples
    --------
    >>> from colour.utilities import full
    >>> ctl_transform = template_ctl_transform_float("rIn * 2")
    >>> a = 0.18
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    0.3601074...
    >>> a = [0.18]
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    array([ 0.3601074...])
    >>> a = [0.18, 0.18, 0.18]
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    array([ 0.3601074...,  0.3601074...,  0.3601074...])
    >>> a = [[0.18, 0.18, 0.18]]
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    array([[ 0.3601074...,  0.3601074...,  0.3601074...]])
    >>> a = [[[0.18, 0.18, 0.18]]]
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    array([[[ 0.3601074...,  0.3601074...,  0.3601074...]]])
    >>> a = full([4, 2, 3], 0.18)
    >>> process_image_ctl(a, [ctl_transform])  # doctest: +SKIP
    array([[[ 0.3601074...,  0.3601074...,  0.3601074...],
            [ 0.3601074...,  0.3601074...,  0.3601074...]],
    <BLANKLINE>
           [[ 0.3601074...,  0.3601074...,  0.3601074...],
            [ 0.3601074...,  0.3601074...,  0.3601074...]],
    <BLANKLINE>
           [[ 0.3601074...,  0.3601074...,  0.3601074...],
            [ 0.3601074...,  0.3601074...,  0.3601074...]],
    <BLANKLINE>
           [[ 0.3601074...,  0.3601074...,  0.3601074...],
            [ 0.3601074...,  0.3601074...,  0.3601074...]]])
    """

    a = as_float_array(a)
    shape, dtype = a.shape, a.dtype
    a = as_3_channels_image(a)

    _descriptor, temp_input_filename = tempfile.mkstemp(suffix="-input.exr")
    _descriptor, temp_output_filename = tempfile.mkstemp(suffix="-output.exr")

    write_image(a, temp_input_filename)

    ctl_render(
        temp_input_filename,
        temp_output_filename,
        ctl_transforms,
        *args,
        **kwargs,
    )

    b = read_image(temp_output_filename).astype(dtype)[..., 0:3]

    os.remove(temp_input_filename)
    os.remove(temp_output_filename)

    if len(shape) == 0:
        return as_float(np.squeeze(b)[0])
    elif shape[-1] == 1:
        return np.reshape(b[..., 0], shape)
    else:
        return np.reshape(b, shape)


def template_ctl_transform_float(
    R_function: str,
    G_function: str | None = None,
    B_function: str | None = None,
    description: str | None = None,
    parameters: Sequence[str] | None = None,
    imports: Sequence[str] | None = None,
    header: str | None = None,
) -> str:
    """
    Generate the code for a *CTL* transform to test a function processing
    per-float channel.

    Parameters
    ----------
    R_function
        Function call to process the *red* channel.
    G_function
        Function call to process the *green* channel.
    B_function
        Function call to process the *blue* channel.
    description
        Description of the *CTL* transform.
    parameters
        List of parameters to use with the *CTL* transform.
    imports
        List of imports to use with the *CTL* transform.
    header
        Header code that can be used to define various functions and globals.

    Returns
    -------
    :class:`str`
        *CTL* transform code.

    Examples
    --------
    >>> print(
    ...     template_ctl_transform_float(
    ...         "rIn * pow(2, exposure)",
    ...         description="Adjust Exposure",
    ...         parameters=["input float exposure = 0.0"],
    ...     )
    ... )
    // Adjust Exposure
    <BLANKLINE>
    void main
    (
        output varying float rOut,
        output varying float gOut,
        output varying float bOut,
        output varying float aOut,
        input varying float rIn,
        input varying float gIn,
        input varying float bIn,
        input varying float aIn = 1.0,
        input float exposure = 0.0
    )
    {
        rOut = rIn * pow(2, exposure);
        gOut = rIn * pow(2, exposure);
        bOut = rIn * pow(2, exposure);
        aOut = aIn;
    }
    >>> def format_imports(imports):
    ...     return [f'import "{i}";' for i in imports]
    ...
    >>> print(
    ...     template_ctl_transform_float(
    ...         "Y_2_linCV(rIn, CINEMA_WHITE, CINEMA_BLACK)",
    ...         "Y_2_linCV(gIn, CINEMA_WHITE, CINEMA_BLACK)",
    ...         "Y_2_linCV(bIn, CINEMA_WHITE, CINEMA_BLACK)",
    ...         imports=format_imports(
    ...             [
    ...                 "ACESlib.Utilities",
    ...                 "ACESlib.Transform_Common",
    ...             ]
    ...         ),
    ...     )
    ... )
    // "float" Processing Function
    <BLANKLINE>
    import "ACESlib.Utilities";
    import "ACESlib.Transform_Common";
    <BLANKLINE>
    void main
    (
        output varying float rOut,
        output varying float gOut,
        output varying float bOut,
        output varying float aOut,
        input varying float rIn,
        input varying float gIn,
        input varying float bIn,
        input varying float aIn = 1.0)
    {
        rOut = Y_2_linCV(rIn, CINEMA_WHITE, CINEMA_BLACK);
        gOut = Y_2_linCV(gIn, CINEMA_WHITE, CINEMA_BLACK);
        bOut = Y_2_linCV(bIn, CINEMA_WHITE, CINEMA_BLACK);
        aOut = aIn;
    }
    """  # noqa: D405, D407, D410, D411

    G_function = optional(G_function, R_function)
    B_function = optional(B_function, R_function)
    parameters = optional(parameters, "")
    imports = optional(imports, [])
    header = optional(header, "")

    ctl_file_content = ""

    if description:
        ctl_file_content += f"// {description}\n"
    else:
        ctl_file_content += '// "float" Processing Function\n'

    ctl_file_content += "\n"

    if imports:
        ctl_file_content += "\n".join(imports)
        ctl_file_content += "\n\n"

    if header:
        ctl_file_content += f"{header}\n"

    ctl_file_content += """
void main
(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut,
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn = 1.0
""".strip()

    if parameters:
        ctl_file_content += ",\n"
        ctl_file_content += textwrap.indent(",\n".join(parameters), " " * 4)
        ctl_file_content += "\n"

    ctl_file_content += f"""
)
{{
    rOut = {R_function};
    gOut = {G_function};
    bOut = {B_function};
    aOut = aIn;
}}
""".strip()

    return ctl_file_content


def template_ctl_transform_float3(
    RGB_function: str,
    description: str | None = None,
    parameters: Sequence[str] | None = None,
    imports: Sequence[str] | None = None,
    header: str | None = None,
) -> str:
    """
    Generate the code for a *CTL* transform to test a function processing
    RGB channels.

    Parameters
    ----------
    RGB_function
        Function call to process the *RGB* channels.
    description
        Description of the *CTL* transform.
    parameters
        List of parameters to use with the *CTL* transform.
    imports
        List of imports to use with the *CTL* transform.
    header
        Header code that can be used to define various functions and globals.

    Returns
    -------
    :class:`str`
        *CTL* transform code.

    Examples
    --------
    >>> def format_imports(imports):
    ...     return [f'import "{i}";' for i in imports]
    ...
    >>> print(
    ...     template_ctl_transform_float3(
    ...         "darkSurround_to_dimSurround(rgbIn)",
    ...         imports=format_imports(
    ...             [
    ...                 "ACESlib.Utilities",
    ...                 "ACESlib.Transform_Common",
    ...                 "ACESlib.ODT_Common",
    ...             ]
    ...         ),
    ...     )
    ... )
    // "float3" Processing Function
    <BLANKLINE>
    import "ACESlib.Utilities";
    import "ACESlib.Transform_Common";
    import "ACESlib.ODT_Common";
    <BLANKLINE>
    void main
    (
        output varying float rOut,
        output varying float gOut,
        output varying float bOut,
        output varying float aOut,
        input varying float rIn,
        input varying float gIn,
        input varying float bIn,
        input varying float aIn = 1.0)
    {
        float rgbIn[3] = {rIn, gIn, bIn};
    <BLANKLINE>
        float rgbOut[3] = darkSurround_to_dimSurround(rgbIn);
    <BLANKLINE>
        rOut = rgbOut[0];
        gOut = rgbOut[1];
        bOut = rgbOut[2];
        aOut = aIn;
    }
    """  # noqa: D405, D407, D410, D411

    parameters = optional(parameters, "")
    imports = optional(imports, [])
    header = optional(header, "")

    ctl_file_content = ""

    if description:
        ctl_file_content += f"// {description}\n"
    else:
        ctl_file_content += '// "float3" Processing Function\n'

    ctl_file_content += "\n"

    if imports:
        ctl_file_content += "\n".join(imports)
        ctl_file_content += "\n\n"

    if header:
        ctl_file_content += f"{header}\n"

    ctl_file_content += """
void main
(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut,
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn = 1.0
""".strip()

    if parameters:
        ctl_file_content += ",\n"
        ctl_file_content += textwrap.indent(",\n".join(parameters), " " * 4)
        ctl_file_content += "\n"

    ctl_file_content += """
)
{{
    float rgbIn[3] = {{rIn, gIn, bIn}};

    float rgbOut[3] = {RGB_function};

    rOut = rgbOut[0];
    gOut = rgbOut[1];
    bOut = rgbOut[2];
    aOut = aIn;
}}
""".strip().format(
        RGB_function=RGB_function
    )

    return ctl_file_content
