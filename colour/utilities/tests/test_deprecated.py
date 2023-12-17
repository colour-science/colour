"""Define the unit tests helper module for the deprecation management."""

import contextlib
import sys

from colour.hints import Any
from colour.utilities.deprecation import (
    ModuleAPI,
    ObjectRemoved,
    ObjectRenamed,
)


class deprecated(ModuleAPI):
    """Define a class acting like the *deprecated* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


NAME: Any = None
"""An non-deprecated module attribute."""

NEW_NAME: Any = None
"""A module attribute with a new name."""

with contextlib.suppress(KeyError):
    sys.modules[
        "colour.utilities.tests.test_deprecated"
    ] = deprecated(  # pyright: ignore
        sys.modules["colour.utilities.tests.test_deprecated"],
        {
            "OLD_NAME": ObjectRenamed(
                name="colour.utilities.tests.test_deprecated.OLD_NAME",
                new_name="colour.utilities.tests.test_deprecated.NEW_NAME",
            ),
            "REMOVED": ObjectRemoved(
                name="colour.utilities.tests.test_deprecated.REMOVED"
            ),
        },
    )

del ModuleAPI
del ObjectRenamed
del ObjectRemoved
del sys
