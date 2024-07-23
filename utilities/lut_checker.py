import tempfile
import subprocess
import os
from colour.io.luts.tests.test_clf_common import wrap_snippet
import numpy as np

def snippet_as_tmp_file(snippet):
    doc = wrap_snippet(snippet)
    tmp_folder = tempfile.gettempdir()
    file_name = os.path.join(tmp_folder, 'colour_snippet.clf')
    with open(file_name, "w") as f:
        f.write(doc)
    return file_name


def snippet_in_lut_checker(snippet, rgb=None):
    f = snippet_as_tmp_file(snippet)
    try:
        if rgb is None:
            return subprocess.check_output("ociochecklut", f"{f}")
        else:
            return subprocess.check_output(["ociochecklut", f"{f}", f"{rgb[0]}", f"{rgb[1]}", f"{rgb[2]}"])
    finally:
        os.remove(f)

def result_as_array(result):
    result = result.decode("utf-8").strip().split()
    assert len(result) == 3
    result = list(map(float, result))
    return np.array(result)


def ocio_reference_output(snippet, rgb):
    result = snippet_in_lut_checker(snippet, rgb)
    return result_as_array(result)


if __name__ == "__main__":

    snippet = """
        <LUT1D id="lut-23" name="4 Value Lut" inBitDepth="12i" outBitDepth="12i">
            <Description>1D LUT - Turn 4 grey levels into 4 inverted codes</Description>
            <Array dim="4 1">
                3
                2
                1
                0
            </Array>
        </LUT1D>
        """

    print(ocio_reference_output(snippet, (0,0,0)))

