"""Overture Maps acquisition for one :class:`~sitex.data.aoi.AOI`, via DuckDB range
requests against Overture's S3 GeoParquet files (no full-dataset download needed).

Extracted from ``00-Data-Overture_Extractor.ipynb``. ``extract_by_aoi()`` replaces the
original notebook's separate ``extract_by_place`` / ``extract_by_point`` — both did the
same bbox-query-then-clip work and differed only in how the bbox/clip mask was built,
which is now ``sitex.data.aoi``'s job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
from shapely import wkt

from sitex.data.aoi import AOI

# Each entry: Overture theme/type, fields to select, output layer name.
LAYERS: dict[str, dict[str, str]] = {
    "buildings": {
        "theme": "buildings",
        "type": "building",
        "fields": 'id, names."primary" AS name, height, num_floors, class, ST_AsText(geometry) AS geom_wkt',
        "output_name": "BLDG",
    },
    "roads": {
        "theme": "transportation",
        "type": "segment",
        "fields": 'id, names."primary" AS name, class, subtype, ST_AsText(geometry) AS geom_wkt',
        "filter": "subtype = 'road'",
        "output_name": "ROADS",
    },
    "places": {
        "theme": "places",
        "type": "place",
        "fields": 'id, names."primary" AS name, categories."primary" AS category, confidence, ST_AsText(geometry) AS geom_wkt',
        "output_name": "POIS",
    },
    "addresses": {
        "theme": "addresses",
        "type": "address",
        "fields": "id, street, postcode, city, ST_AsText(geometry) AS geom_wkt",
        "output_name": "ADDRESSES",
    },
    "land_use": {
        "theme": "base",
        "type": "land_use",
        "fields": "id, subtype, class, ST_AsText(geometry) AS geom_wkt",
        "output_name": "LULC",
    },
    "land": {
        "theme": "base",
        "type": "land",
        "fields": "id, subtype, class, ST_AsText(geometry) AS geom_wkt",
        "output_name": "LAND",
    },
    "water": {
        "theme": "base",
        "type": "water",
        "fields": "id, subtype, class, ST_AsText(geometry) AS geom_wkt",
        "output_name": "WATER",
    },
    "divisions": {
        "theme": "divisions",
        "type": "division_area",
        "fields": 'id, names."primary" AS name, subtype, ST_AsText(geometry) AS geom_wkt',
        "output_name": "DIVISIONS",
    },
}

DEFAULT_LAYERS = ["buildings", "roads", "places", "land_use", "water"]


def get_latest_release() -> str:
    """Current Overture release string, found by listing what's actually on S3.

    Never hardcode — releases roll every ~4-8 weeks. Deliberately doesn't use the
    ``overturemaps`` package's own release lookup: its API isn't stable across
    versions (``overturemaps.releases`` doesn't exist in 0.14.0, the version this
    project has installed — confirmed by actually importing it, not assumed), so this
    queries the S3 bucket directly via the same DuckDB/httpfs stack used for the data
    itself.
    """
    con = init_duckdb()
    try:
        rows = con.execute(
            "SELECT DISTINCT regexp_extract(file, 'release/([^/]+)/', 1) AS release "
            "FROM glob('s3://overturemaps-us-west-2/release/*/theme=buildings/type=building/*')"
        ).fetchall()
        releases = sorted(r[0] for r in rows if r[0])
        if not releases:
            raise RuntimeError("No releases found under s3://overturemaps-us-west-2/release/")
        return releases[-1]
    finally:
        con.close()


def init_duckdb(memory_limit: str = "3GB", threads: int = 4, timeout_ms: int = 120_000):
    """In-memory DuckDB connection configured for Overture S3 queries."""
    con = duckdb.connect(":memory:")
    for stmt in [
        "INSTALL httpfs; LOAD httpfs;",
        "INSTALL spatial; LOAD spatial;",
        "SET s3_region='us-west-2';",
        f"SET memory_limit='{memory_limit}';",
        f"SET threads={threads};",
        f"SET http_timeout={timeout_ms};",
    ]:
        con.execute(stmt)
    return con


def check_release(release: str) -> bool:
    """Quick check that a release path is accessible on S3.

    Reads one row, not a row count: ``count(*)`` forces DuckDB to scan the entire
    buildings theme (2.5B+ rows globally) before it can return anything, even with an
    outer ``LIMIT 1`` — that only limits the aggregated result (always 1 row for a bare
    count), not the scan. Measured this directly: the count(*) version took 709s for a
    plain reachability check. ``SELECT 1 ... LIMIT 1`` lets DuckDB push the limit down
    to the scan and stop after the first row.
    """
    con = init_duckdb()
    s3 = f"s3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*"
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{s3}', hive_partitioning=1) LIMIT 1").fetchone()
        print(f"Release '{release}' is accessible.")
        return True
    except Exception as exc:
        print(f"Release '{release}' not accessible: {exc}")
        return False
    finally:
        con.close()


def extract_layer(con, layer_id: str, bbox: tuple[float, float, float, float], release: str) -> gpd.GeoDataFrame | None:
    """Query one Overture layer within a bounding box."""
    cfg = LAYERS[layer_id]
    xmin, ymin, xmax, ymax = bbox
    s3 = (
        f"s3://overturemaps-us-west-2/release/{release}"
        f"/theme={cfg['theme']}/type={cfg['type']}/*"
    )
    extra = f"AND {cfg['filter']}" if cfg.get("filter") else ""
    sql = f"""
        SELECT {cfg['fields']}
        FROM read_parquet('{s3}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin <= {xmax}
          AND bbox.xmax >= {xmin}
          AND bbox.ymin <= {ymax}
          AND bbox.ymax >= {ymin}
          {extra}
    """
    try:
        df = con.execute(sql).df()
    except Exception as exc:
        print(f"  [ERROR] {layer_id}: {exc}")
        return None

    if df.empty:
        return None

    df["geometry"] = df["geom_wkt"].apply(wkt.loads)
    df = df.drop(columns=["geom_wkt"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def clip_to_mask(gdf: gpd.GeoDataFrame, mask_geom) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to a Shapely geometry; falls back to unclipped on error."""
    try:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.make_valid()
        return gpd.clip(gdf, mask_geom)
    except Exception as exc:
        print(f"  [WARN] clip failed ({exc}), keeping bbox result")
        return gdf


def _download_layers(layers: list[str], bbox, clip_mask, release: str) -> dict[str, gpd.GeoDataFrame]:
    con = init_duckdb()
    gdfs = {}
    for layer_id in layers:
        print(f"\nExtracting '{layer_id}'...")
        gdf = extract_layer(con, layer_id, bbox, release)
        if gdf is None or gdf.empty:
            print("  -> No features found")
            continue
        gdf = clip_to_mask(gdf, clip_mask)
        if gdf is None or gdf.empty:
            print("  -> Empty after clipping")
            continue
        print(f"  -> {len(gdf):,} features")
        gdfs[layer_id] = gdf
    con.close()
    return gdfs


def export_layers(gdfs: dict[str, gpd.GeoDataFrame], slug: str, output_dir: Path) -> dict[str, Any]:
    """Save extracted layers to GeoPackage files: ``{slug}_overture.gpkg`` (all layers)
    plus one single-layer file per of buildings/places/roads/land_use, for notebooks
    that only need one theme."""
    output_dir = Path(output_dir)
    main_path = output_dir / f"{slug}_overture.gpkg"
    streets_path = output_dir / f"{slug}_streets.gpkg"
    landuse_path = output_dir / f"{slug}_landuse.gpkg"
    buildings_path = output_dir / f"{slug}_buildings.gpkg"
    places_path = output_dir / f"{slug}_places.gpkg"

    for p in [main_path, streets_path, landuse_path, buildings_path, places_path]:
        if p.exists():
            p.unlink()

    written = []
    for layer_id, gdf in gdfs.items():
        layer_name = LAYERS[layer_id]["output_name"]
        gdf.to_file(main_path, layer=layer_name, driver="GPKG")
        written.append(layer_name)
        if layer_id == "roads":
            gdf.to_file(streets_path, layer=layer_name, driver="GPKG")
        if layer_id == "land_use":
            gdf.to_file(landuse_path, layer=layer_name, driver="GPKG")
        if layer_id == "buildings":
            gdf.to_file(buildings_path, layer=layer_name, driver="GPKG")
        if layer_id == "places":
            gdf.to_file(places_path, layer=layer_name, driver="GPKG")

    print(f"\nMain       -> {main_path.resolve()}")
    if "roads" in gdfs:
        print(f"Streets    -> {streets_path.resolve()}")
    if "land_use" in gdfs:
        print(f"Landuse    -> {landuse_path.resolve()}")
    if "buildings" in gdfs:
        print(f"Buildings  -> {buildings_path.resolve()}")
    if "places" in gdfs:
        print(f"Places     -> {places_path.resolve()}")
    print(f"Layers written: {written}")

    return {
        "main": main_path,
        "buildings": buildings_path,
        "places": places_path,
        "streets": streets_path,
        "landuse": landuse_path,
        "written": written,
    }


def extract_by_aoi(
    aoi: AOI,
    output_dir: Path,
    layers: list[str] | None = None,
    release: str | None = None,
) -> dict[str, Any]:
    """Download Overture layers for ``aoi.bbox``, clip each to ``aoi.boundary``, and
    export to GeoPackage. Works the same whether ``aoi`` came from a place name or a
    drawn map rectangle — the AOI method is already resolved by this point."""
    if layers is None:
        layers = DEFAULT_LAYERS
    if release is None:
        release = get_latest_release()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_gdf = gpd.GeoDataFrame(geometry=[aoi.boundary], crs="EPSG:4326")
    boundary_gdf.to_crs(epsg=aoi.local_epsg).to_file(
        output_dir / f"{aoi.slug}_boundary.gpkg", driver="GPKG"
    )

    gdfs = _download_layers(layers, aoi.bbox, aoi.boundary, release)
    return export_layers(gdfs, aoi.slug, output_dir)
