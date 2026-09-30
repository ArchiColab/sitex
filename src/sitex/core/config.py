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

    Create with ``place_name`` (a geocodable string, an OSM ID such as
    ``"R13512464"``, or an openstreetmap.org link) and the UTM zone, filename
    slug, and CAD/UCS origin are derived once, consistently. The site-reference
    point is the centre of the place's OSM boundary; pass ``local_lat`` and
    ``local_lon`` together to override it with a fixed point.
    """

    place_name: str
    local_lat: float | None = None
    local_lon: float | None = None
    slug: str = ""
    local_epsg: int | None = None
    origin_round_m: int = 1000
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        if (self.local_lat is None) != (self.local_lon is None):
            raise ValueError("Give both local_lat and local_lon, or neither.")
        geocoded_slug = ""
        if self.local_lat is None:
            # Imported here: sitex.data.aoi pulls in osmnx, which a config with
            # explicit coordinates does not need.
            from sitex.data.aoi import resolve_aoi_by_place

            aoi = resolve_aoi_by_place(self.place_name)
            self.local_lat, self.local_lon = aoi.center_lat, aoi.center_lon
            geocoded_slug = aoi.slug
        if not self.slug and geocoded_slug:
            self.slug = geocoded_slug
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
