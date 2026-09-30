"""Global Building Atlas (GBA, TUM/zhu-xlab) acquisition -- bbox-filtered DuckDB fetch
against the ``source.coop`` GeoParquet mirror, no auth needed.

GBA's official distribution is EPSG:3857 with a documented history of files
mislabeled as EPSG:4326 elsewhere in the dataset family, and its own README
discourages bulk use of the WFS -- this uses the verified-WGS84
``data.source.coop/tge-labs/globalbuildingatlas-lod1`` mirror instead, queried via
DuckDB httpfs+spatial so only the AOI's bbox is pulled from each 5x5-degree tile
(never a full ~2.25GB tile download), and asserts the returned coordinates actually
fall inside the requested bbox rather than trusting the declared CRS.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkb as shapely_wkb

from sitex.data.aoi import AOI

GBA_BASE_URL = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1"
GBA_TILE_SIZE = 5


def gba_tiles_for_bbox(bbox: tuple[float, float, float, float], tile_size: int = GBA_TILE_SIZE) -> list[str]:
    """5x5-degree GBA tile names covering a WGS84 bbox (west_lon, north_lat, east_lon,
    south_lat; e/w 3-digit, n/s 2-digit padding)."""
    west, south, east, north = bbox

    def lon_tag(deg):
        return ("e" if deg >= 0 else "w") + f"{abs(int(deg)):03d}"

    def lat_tag(deg):
        return ("n" if deg >= 0 else "s") + f"{abs(int(deg)):02d}"

    w = int(math.floor(west / tile_size) * tile_size)
    e = int(math.ceil(east / tile_size) * tile_size)
    s = int(math.floor(south / tile_size) * tile_size)
    n = int(math.ceil(north / tile_size) * tile_size)

    tiles = []
    for lon in range(w, e, tile_size):
        for lat in range(s, n, tile_size):
            tiles.append(f"{lon_tag(lon)}_{lat_tag(lat + tile_size)}_{lon_tag(lon + tile_size)}_{lat_tag(lat)}")
    return tiles


def fetch_gba_buildings(aoi: AOI, output_path: Path, tile_size: int = GBA_TILE_SIZE) -> gpd.GeoDataFrame:
    """Bbox-filtered fetch of GBA buildings (``source``/``id``/``height``/``var``/
    geometry) via DuckDB httpfs+spatial against the remote GeoParquet tile(s).
    Caches to a GeoPackage; skips the DuckDB scan on re-runs if that file already
    exists."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Reusing already-fetched {output_path}")
        return gpd.read_file(output_path)

    west, south, east, north = aoi.bbox
    tiles = gba_tiles_for_bbox(aoi.bbox, tile_size)

    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET threads=4;")

    frames = []
    for tile in tiles:
        url = f"{GBA_BASE_URL}/{tile}.parquet"
        sql = f"""
            SELECT source, id, height, var, ST_AsWKB(geometry) AS geom_wkb
            FROM read_parquet('{url}')
            WHERE bbox.xmin <= {east} AND bbox.xmax >= {west}
              AND bbox.ymin <= {north} AND bbox.ymax >= {south}
        """
        try:
            df = con.execute(sql).df()
        except duckdb.Error as exc:
            print(f"  [WARN] tile {tile} not reachable ({exc}) -- skipping")
            continue
        print(f"  tile {tile}: {len(df):,} buildings")
        frames.append(df)
    con.close()

    if not frames:
        raise RuntimeError(f"No GBA tiles reachable for bbox={aoi.bbox}")

    df = pd.concat(frames, ignore_index=True)
    df["geometry"] = df["geom_wkb"].apply(lambda b: shapely_wkb.loads(bytes(b)))
    gdf = gpd.GeoDataFrame(df.drop(columns=["geom_wkb"]), geometry="geometry", crs="EPSG:4326")

    # Confirm real-world coordinates, not just trust the declared CRS -- GBA's own
    # FAQ documents mislabeled-CRS files elsewhere in this dataset family.
    gxmin, gymin, gxmax, gymax = gdf.total_bounds
    print(f"Fetched {len(gdf):,} buildings, coordinate range: "
          f"lon [{gxmin:.4f}, {gxmax:.4f}]  lat [{gymin:.4f}, {gymax:.4f}]")
    assert west - 0.1 <= gxmin and gxmax <= east + 0.1, "GBA coordinates fall outside the requested bbox -- CRS mismatch?"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved -> {output_path}")
    return gdf
