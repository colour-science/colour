# noqa: D100
import sys

from colour.hints import Any
from colour.utilities.deprecation import (
    ModuleAPI,
    ObjectRenamed,
    ObjectRemoved,
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
