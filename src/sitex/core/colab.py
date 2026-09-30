"""Google Colab helpers shared by the interactive-map functions."""

from __future__ import annotations

import sys


def enable_widgets() -> bool:
    """Let Colab render third-party widgets (``ipyleaflet``, ``leafmap``).

    Colab shows only its own widgets by default; a map built with ``ipyleaflet`` comes out
    as an empty cell with a text representation. Enabling the custom widget manager fixes
    that. It takes effect for the cells that run after it, so a notebook must call it in an
    earlier cell than the one that displays the map (the setup cell is the place); calling
    it inside the cell that shows the map is too late. Outside Colab it does nothing.

    Returns True if the custom widget manager was enabled.
    """
    if "google.colab" not in sys.modules:
        return False
    from google.colab import output

    output.enable_custom_widget_manager()
    return True
