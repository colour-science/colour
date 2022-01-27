# -*- coding: utf-8 -*-

import sys
from colour.hints import Any
from colour.utilities.deprecation import (
    ModuleAPI,
    ObjectRenamed,
    ObjectRemoved,
)


class deprecated(ModuleAPI):
    def __getattr__(self, attribute):
        return super(deprecated, self).__getattr__(attribute)


NAME: Any = None
"""
An non-deprecated module attribute.

NAME
"""

NEW_NAME: Any = None
"""
A module attribute with a new name.

NAME
"""

try:
    sys.modules[
        "colour.utilities.tests.test_deprecated"
    ] = deprecated(  # type: ignore[assignment]
        sys.modules["colour.utilities.tests.test_deprecated"],
        {
            "OLD_NAME": ObjectRenamed(
                name=("colour.utilities.tests.test_deprecated.OLD_NAME"),
                new_name=("colour.utilities.tests.test_deprecated.NEW_NAME"),
            ),
            "REMOVED": ObjectRemoved(
                name="colour.utilities.tests.test_deprecated.REMOVED"
            ),
        },
    )
except KeyError:  # pragma: no cover
    pass

del ModuleAPI
del ObjectRenamed
del ObjectRemoved
del sys
