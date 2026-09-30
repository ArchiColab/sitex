"""``CityConfig`` — the single source of truth for one case-study site: one
place to derive the local UTM zone, the CAD/UCS origin, and a consistent
filename slug.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .geo import compute_cad_origin, utm_epsg_from_lonlat


def slugify(text: str) -> str:
    """ASCII, lowercase, underscore-separated slug for filenames.

    Strips diacritics (e.g. "Phường" -> "Phuong") so every module derives
    the same filename for the same city.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    ascii_text = ascii_text.lower().strip()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    return ascii_text.strip("_")


@dataclass
class CityConfig:
    """Everything downstream modules need to know about one case-study site.

    Create with ``place_name`` (a geocodable string, e.g.
    ``"Phường Pleiku, Gia Lai, Vietnam"``) plus a ``local_lat``/``local_lon``
    site-reference point, and the UTM zone, filename slug, and CAD/UCS origin
    are derived once, consistently.
    """

    place_name: str
    local_lat: float
    local_lon: float
    slug: str = ""
    local_epsg: int | None = None
    origin_round_m: int = 1000
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        if not self.slug:
            # Default slug is derived from the locality only (the text
            # before the first comma): a short city slug rather than the
            # full "city, province, country" geocode string.
            self.slug = slugify(self.place_name.split(",")[0])
        if self.local_epsg is None:
            self.local_epsg = utm_epsg_from_lonlat(self.local_lon, self.local_lat)
        self.origin_easting, self.origin_northing = compute_cad_origin(
            self.local_lon, self.local_lat, self.local_epsg, self.origin_round_m
        )
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def origin(self) -> tuple[float, float]:
        """``(easting, northing)`` CAD/UCS base point for this site."""
        return (self.origin_easting, self.origin_northing)
