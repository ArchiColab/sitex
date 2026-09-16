"""DEM, Sentinel-2, and Landsat acquisition for one :class:`~sitex.data.aoi.AOI`.

Extracted from ``00-Data-Acquisition_RemoteSensing.ipynb``. Each function takes the
same resolved ``AOI`` (see ``sitex.data.aoi``) so all three sources — and the student
notebook that calls them — agree on one bounding box regardless of which AOI method
produced it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import shape

from sitex.data.aoi import AOI

# ── DEM (OpenTopography) ───────────────────────────────────────────────────────

def download_dem(aoi: AOI, dem_dir: Path, api_key: str | None = None) -> tuple[Path, Path]:
    """Download NASADEM (DTM, bare-earth) + AW3D30 (DSM, top surface) for ``aoi``.

    Reads ``OPENTOPOGRAPHY_API_KEY`` from the environment if ``api_key`` isn't passed —
    get a free key at https://portal.opentopography.org/. Raises ``ValueError`` rather
    than silently hitting OpenTopography with a placeholder key.
    """
    from bmi_topography import Topography

    api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not api_key or api_key == "YOUR API KEY HERE":
        raise ValueError(
            "No OpenTopography API key. Pass api_key=, or set the "
            "OPENTOPOGRAPHY_API_KEY environment variable — free key at "
            "https://portal.opentopography.org/"
        )
    os.environ["OPENTOPOGRAPHY_API_KEY"] = api_key

    dem_dir = Path(dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = aoi.bbox

    nasadem = Topography(
        dem_type="NASADEM", south=south, north=north, west=west, east=east,
        output_format="GTiff", cache_dir="cache",
    )
    dtm_out = dem_dir / "dtm_nasadem.tif"
    shutil.copy2(nasadem.fetch(), dtm_out)

    aw3d30 = Topography(
        dem_type="AW3D30", south=south, north=north, west=west, east=east,
        output_format="GTiff", cache_dir="cache",
    )
    dsm_out = dem_dir / "dsm_aw3d30.tif"
    shutil.copy2(aw3d30.fetch(), dsm_out)

    return dtm_out, dsm_out


# ── Sentinel-2 (OpenEO / CDSE) ───────────────────────────────────────────────────

S2_COLLECTION = "SENTINEL2_L2A"
S2_BANDS = ["B04", "B08", "B11"]  # Red, NIR, SWIR — NDVI/NDBI (change detection, UHI)
S2_WATER_BANDS = ["B03", "B08"]  # Green, NIR — NDWI (flood risk)
S2_MAX_CLOUD = 10


def connect_cdse():
    """Authenticate to the Copernicus Data Space Ecosystem (OIDC — opens a browser tab
    on first login, then reuses the cached refresh token)."""
    import openeo

    connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    return connection


def download_sentinel2_composite(
    conn, aoi: AOI, start_date: str, end_date: str, bands: list[str], output_path: Path,
    max_cloud: int = S2_MAX_CLOUD,
) -> Path:
    """Server-side median composite over ``[start_date, end_date)``. Band order in the
    output GeoTIFF matches ``bands`` (band 1 = ``bands[0]``, etc).

    Forces the output CRS to ``aoi.local_epsg`` via ``resample_spatial`` instead of the
    backend's default, which returns whichever UTM tile the AOI happens to fall in —
    not guaranteed stable if the AOI sits near a UTM zone boundary.
    """
    west, south, east, north = aoi.bbox
    cube = conn.load_collection(
        S2_COLLECTION,
        spatial_extent={"west": west, "south": south, "east": east, "north": north},
        temporal_extent=[start_date, end_date],
        bands=bands,
        max_cloud_cover=max_cloud,
    )
    cube = cube.resample_spatial(resolution=10, projection=aoi.local_epsg)
    output_path = Path(output_path)
    cube.reduce_temporal("median").download(str(output_path), format="GTiff")
    return output_path


def download_sentinel2_annual_composite(conn, aoi: AOI, year: int, output_path: Path) -> Path:
    """Server-side median composite for one year. 3-band GeoTIFF: 1=B04, 2=B08, 3=B11."""
    return download_sentinel2_composite(
        conn, aoi, f"{year}-01-01", f"{year}-12-31", S2_BANDS, output_path,
    )


# ── Landsat 8/9 (STAC, Level-1 -> Level-2 fallback) ──────────────────────────────

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BAND_SPECS = {"B4": "red", "B5": "nir08", "B10": "lwir11"}


def search_landsat(aoi: AOI, begin_date: str, end_date: str, max_cloud: int = 20):
    """Search Landsat Collection 2, Level-1 first, falling back to Level-2 (Level-1
    coverage has gaps for scenes downlinked through international ground stations,
    common for Southeast Asia). Returns ``(items, product_level, collection_id)``."""
    from pystac_client import Client

    catalog = Client.open(STAC_URL)

    def _search(collection: str):
        search = catalog.search(
            collections=[collection],
            bbox=list(aoi.bbox),
            datetime=f"{begin_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8", "landsat-9"]}},
        )
        return list(search.items())

    product_level, collection = "L1", "landsat-c2-l1"
    items = _search(collection)
    if not items:
        product_level, collection = "L2", "landsat-c2-l2"
        items = _search(collection)
    return items, product_level, collection


def _find_asset(item, common_name: str, fallback_keys: list[str]) -> str:
    for _key, asset in item.assets.items():
        bands = asset.extra_fields.get("eo:bands", [])
        if any(b.get("common_name") == common_name for b in bands):
            return asset.href
    for key in fallback_keys:
        if key in item.assets:
            return item.assets[key].href
    raise KeyError(f"Could not find an asset for '{common_name}' among {sorted(item.assets.keys())}")


def download_landsat_bands(item, aoi: AOI, output_dir: Path, band_specs: dict = BAND_SPECS) -> dict[str, str]:
    """Stream, clip to ``aoi``, and save each band in ``band_specs`` as its own GeoTIFF."""
    import planetary_computer

    item_signed = planetary_computer.sign(item)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aoi_gdf = gpd.GeoDataFrame(geometry=[shape(aoi.geojson["geometry"])], crs="EPSG:4326")

    def _download_and_clip(href: str, target_path: Path) -> Path:
        with rasterio.open(href) as src:
            aoi_local = aoi_gdf.to_crs(src.crs)
            out_image, out_transform = rasterio.mask.mask(src, aoi_local.geometry, crop=True)
            out_meta = src.meta.copy()
        out_meta.update(height=out_image.shape[1], width=out_image.shape[2], transform=out_transform)
        with rasterio.open(target_path, "w", **out_meta) as dst:
            dst.write(out_image)
        return target_path

    band_paths = {}
    for band, common_name in band_specs.items():
        href = _find_asset(item_signed, common_name, fallback_keys=[band, band.lower()])
        target = output_dir / f"{item.id}_{band}_clipped.tif"
        band_paths[band] = str(_download_and_clip(href, target))
    return band_paths


# ── Handoff file ─────────────────────────────────────────────────────────────────

def save_scene_info(
    aoi: AOI,
    output_path: Path,
    dem_paths: tuple[Path, Path] | None = None,
    sentinel2_paths: dict[int, Path] | None = None,
    landsat_item: Any = None,
    product_level: str | None = None,
    band_paths: dict[str, str] | None = None,
) -> Path:
    """Write the unified ``scene_info.json`` handoff file downstream analysis
    notebooks (UHI, urban change detection, DEM terrain/contour) read from."""
    scene_info: dict[str, Any] = {
        "location_name": aoi.slug,
        "bbox": list(aoi.bbox),
        "aoi_geojson": aoi.geojson,
        "local_lat": aoi.center_lat,
        "local_lon": aoi.center_lon,
    }
    if dem_paths is not None:
        scene_info["dem_paths"] = {"dtm": str(dem_paths[0]), "dsm": str(dem_paths[1])}
    if sentinel2_paths is not None:
        scene_info["sentinel2_paths"] = {str(year): str(path) for year, path in sentinel2_paths.items()}
    if landsat_item is not None:
        scene_info.update({
            "landsat_id": landsat_item.id,
            "product_level": product_level,
            "acquisition_date": landsat_item.properties.get("datetime"),
            "cloud_cover": landsat_item.properties.get("eo:cloud_cover"),
            "platform": landsat_item.properties.get("platform"),
            "band_paths": band_paths or {},
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=2)
    return output_path
