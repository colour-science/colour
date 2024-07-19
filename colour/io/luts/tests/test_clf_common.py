import colour.io.clf as clf

EXAMPLE_WRAPPER = """<?xml version="1.0" ?>
<ProcessList xmlns="urn:NATAS:AMPAS:LUT:v2.0" id="Example Wrapper" compCLFversion="2.0">
{0}
</ProcessList>
"""


def wrap_snippet(snippet: str) -> str:
    """
    Takes a string that should contain a the text representation of a CLF node, and
    returns valid CLF document. Essentially the given string is pasted into the
    `ProcessList` if a CLF document.

    This is useful to quickly convert example snippets of Process Nodes into valid CLF
    documents for parsing.
    """
    return EXAMPLE_WRAPPER.format(snippet)


def snippet_to_process_list(snippet: str) -> clf.ProcessList:
    """
    Takes a string that should contain a valid body for a XML Process List and
    returns the parsed `ProcessList`.
    """
    doc = wrap_snippet(snippet)
    return clf.parse_clf(doc)
