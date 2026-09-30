"""Google Colab helpers shared by the interactive-map functions."""

from __future__ import annotations

import sys


def enable_widgets() -> bool:
    """Let Colab render third-party widgets (``ipyleaflet``, ``leafmap``).

    Colab shows only its own widgets by default; a map built with ``ipyleaflet`` comes out
    as an empty cell with a text representation. Enabling the custom widget manager fixes
    that. Call it before the map is displayed. Outside Colab it does nothing.

    Returns True if the custom widget manager was enabled.
    """
    if "google.colab" not in sys.modules:
        return False
    from google.colab import output

    output.enable_custom_widget_manager()
    return True
