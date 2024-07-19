"""
Defines helper functionality for CLF tests.
"""
import os
import subprocess
import tempfile

import colour_clf_io as clf
import numpy as np

from colour.io.luts.clf import apply

__all__ = [
    "assert_ocio_consistency",
    "assert_ocio_consistency_for_file",
    "snippet_to_process_list",
]


EXAMPLE_WRAPPER = """<?xml version="1.0" ?>
<ProcessList id="Example Wrapper" compCLFversion="3.0">
{0}
</ProcessList>
"""


def wrap_snippet(snippet: str) -> str:
    """# noqa: D401
    Takes a string that should contain the text representation of a CLF node, and
    returns valid CLF document. Essentially the given string is pasted into the
    `ProcessList` if a CLF document.

    This is useful to quickly convert example snippets of Process Nodes into valid CLF
    documents for parsing.
    """
    return EXAMPLE_WRAPPER.format(snippet)


def snippet_to_process_list(snippet: str) -> clf.ProcessList:
    """# noqa: D401
    Takes a string that should contain a valid body for a XML Process List and
    returns the parsed `ProcessList`.
    """
    doc = wrap_snippet(snippet)
    return clf.parse_clf(doc)


def snippet_as_tmp_file(snippet):
    doc = wrap_snippet(snippet)
    tmp_folder = tempfile.gettempdir()
    file_name = os.path.join(tmp_folder, "colour_snippet.clf")
    with open(file_name, "w") as f:
        f.write(doc)
    return file_name


def ocio_outout_for_file(path, rgb=None):
    if rgb is None:
        return subprocess.check_output(["ociochecklut", f"{path}"])  # noqa: S603 S607
    else:
        return subprocess.check_output(
            ["ociochecklut", f"{path}", f"{rgb[0]}", f"{rgb[1]}", f"{rgb[2]}"]  # noqa: S603 S607
        )


def ocio_output_for_snippet(snippet, rgb=None):
    f = snippet_as_tmp_file(snippet)
    try:
        return result_as_array(ocio_outout_for_file(f, rgb))
    finally:
        os.remove(f)


def result_as_array(result_text):
    result_parts = result_text.decode("utf-8").strip().split()
    if len(result_parts) != 3:
        raise RuntimeError(f"Invalid OCIO result: {result_text}")
    result_values = list(map(float, result_parts))
    return np.array(result_values)


def assert_ocio_consistency(value, snippet):
    """Assert that the colour library calculates the same output os the `ociocheclut`
    tool for the given input.
    """
    process_list = snippet_to_process_list(snippet)
    process_list_output = apply(process_list, value, use_normalised_values=True)
    ocio_output = ocio_output_for_snippet(snippet, value)
    np.testing.assert_array_almost_equal(process_list_output, ocio_output)


def assert_ocio_consistency_for_file(value_rgb, clf_path):
    """Assert that the colour library calculates the same output os the `ociocheclut`
    tool for the given input.
    """
    from colour_clf_io import read_clf

    clf_data = read_clf(clf_path)
    process_list_output = apply(clf_data, value_rgb, use_normalised_values=True)
    ocio_output = result_as_array(ocio_outout_for_file(clf_path, value_rgb))
    np.testing.assert_array_almost_equal(process_list_output, ocio_output)
