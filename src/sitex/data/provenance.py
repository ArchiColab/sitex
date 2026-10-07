"""Record of where the data of one site comes from.

``build_provenance`` returns one table with a row per data layer the notebooks downloaded:
what it is, who owns it, its date, its resolution, its licence, which hexagon columns it
feeds, and what the file on disk looks like (CRS, resolution, date saved). The first rows
state the coordinate systems: the working CRS (the global UTM zone, in metres) and, for
information only, the VN-2000 equivalent.

Facts about a dataset (provider, licence, attribution, native resolution) are written below
and must be kept in step with the table "Data sources and licences" in the ``sitex``
README, which was read from the providers' pages. Facts about the site (CRS, file
resolution, dates, scene) are read from the files. What cannot be found stays ``[TO FILL]``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

TO_FILL = "[TO FILL]"

COLUMNS = [
    "layer", "kind", "dataset", "provider", "data_date", "native_resolution", "file", "file_crs",
    "file_resolution", "licence", "attribution", "used_for", "retrieved", "notes",
]

# kind: raster | vector | derived | crs
_DATASETS = {
    "gedtm30": dict(
        layer="Terrain model (DTM)", kind="raster", dataset="GEDTM30",
        provider="OpenGeoHub", native_resolution="30 m",
        licence="CC BY 4.0 (as listed on the Zenodo record)",
        attribution="Ho et al. (2025), PeerJ 13, e19673; doi 10.5281/zenodo.18887460. Built in part from the Copernicus DEM "
                    "and ALOS World 3D, whose own terms can also apply; read them at the providers. Machine-learning product: "
                    "check it before critical use",
        used_for="elevation_m of the buildings (01-ARCH-Building_Morphology); contour, terrain and 3D-model notebooks",
        notes="Estimate of the bare ground: removes tall tree cover strongly (mangrove) and building height only partly. "
              "Replaced NASADEM (negative heights in mangrove areas of the Mekong delta, Cycle 2 test); GLO-30 was tried first.",
        files=["dem/dtm_gedtm30.tif"]),
    "sentinel2": dict(
        layer="Sentinel-2 annual composites (bands B04, B08, B11)", kind="raster",
        dataset="Sentinel-2, median of all images of a year, made on the Copernicus Data Space server",
        provider="European Union / ESA, Copernicus Data Space Ecosystem",
        native_resolution="10 m (B04, B08); B11 20 m resampled to 10 m",
        licence="Copernicus open data terms", attribution="Contains modified Copernicus Sentinel data",
        used_for="ndvi_s2_mean, ndbi_s2_mean (hex_uhi); 02-ENV-Urban_Change_Detection",
        notes="One composite per year, so it is not the date of the Landsat scene.",
        files=["sentinel2/s2_*_B04_B08_B11.tif"]),
    "landsat": dict(
        layer="Landsat 8/9 surface temperature and bands 4, 5", kind="raster",
        dataset="Landsat Collection 2 Level-2 (bands 4, 5, 10 and the quality band)",
        provider="USGS / NASA, served by Microsoft Planetary Computer",
        native_resolution="30 m (bands 4, 5); the thermal band is measured at 100 m and resampled to 30 m",
        licence="USGS open data policy", attribution="Credit USGS/NASA Landsat",
        used_for="lst_mean_c, hei_mean, ndvi_mean (hex_uhi); 02-ENV-UHI_Analysis",
        notes="One scene, one moment (about 10:00 local time); surface, not air temperature. Clouds removed with the quality band.",
        files=[]),
    "worldcover": dict(
        layer="Land cover", kind="raster", dataset="ESA WorldCover 10 m 2021", provider="ESA WorldCover project",
        data_date="2021", native_resolution="10 m", licence="CC BY 4.0",
        attribution="© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by "
                    "ESA WorldCover consortium",
        used_for="green environment patches (access_environment_min, gap_class in hex_green); land-cover tables in the UHI notebook",
        notes="Shows 2021: green built over since then still appears.",
        files=["landcover/esa_worldcover_2021_{slug}.tif"]),
    "canopy": dict(
        layer="Tree canopy height", kind="raster", dataset="High resolution canopy height (Meta, WRI)",
        provider="Meta and WRI", native_resolution="about 1 m",
        licence="CC BY 4.0 (as listed on the AWS registry for the 2024 release; confirm for the version you use)",
        attribution="Meta and WRI; source imagery © Maxar",
        used_for="canopy-height map in 01-ARCH-Building_Morphology (no hexagon column)",
        notes="Each pixel is the canopy height in metres; 0 means no vegetation.",
        files=["canopy_height/canopy_height_chmv2_{slug}.tif"]),
    "open_buildings": dict(
        layer="Building heights (Open Buildings)", kind="raster", dataset="Open Buildings 2.5D Temporal (Google)",
        provider="Google Research", native_resolution=TO_FILL,
        licence="CC BY 4.0 or ODbL (your choice)", attribution="Sirko et al. (2023), arXiv:2310.11622",
        used_for="height_final, height_source (buildings); mean_height, pct_height_measured (hex_morphology)",
        notes="Google caps predicted heights at 100 m: treat a value at that limit as 'tall', not as a measurement.",
        files=["buildings/open_buildings_height_*_{slug}.tif"]),
    "gba": dict(
        layer="Building footprints with heights (GBA)", kind="vector", dataset="Global Building Atlas",
        provider="TUM / zhu-xlab", data_date=TO_FILL, native_resolution="n/a (vector)",
        licence="Split: ODbL polygons; CC BY-NC 4.0 for LoD1 models, height maps and some polygons",
        attribution="Zhu et al. (2025), Earth System Science Data 17(12), 6647–6668. Non-commercial terms apply",
        used_for="height_final, height_source (buildings); mean_height, pct_height_measured (hex_morphology)",
        notes="CC BY-NC terms can apply to the combined result: check before you publish or reuse it.",
        files=["buildings/gba_{slug}.gpkg"]),
    "overture_buildings": dict(
        layer="Building footprints", kind="vector", dataset="Overture Maps: buildings", provider="Overture Maps Foundation",
        data_date=TO_FILL, native_resolution="n/a (vector)",
        licence="ODbL; buildings also include CC BY 4.0 sources",
        attribution="© OpenStreetMap contributors; Overture Maps Foundation",
        used_for="buildings, n_buildings, built_area_m2, coverage_ratio (hex_morphology)",
        notes="Missing or merged footprints are not corrected.",
        files=["overture/{slug}_buildings.gpkg"]),
    "overture_landuse": dict(
        layer="Land use", kind="vector", dataset="Overture Maps: base (land use)", provider="Overture Maps Foundation",
        data_date=TO_FILL, native_resolution="n/a (vector)",
        licence="ODbL; buildings also include CC BY 4.0 sources (as listed for Overture buildings, base and transportation)",
        attribution="© OpenStreetMap contributors; Overture Maps Foundation",
        used_for="building function class (bldg_function, lu_class in buildings and hex_morphology)",
        notes="", files=["overture/{slug}_landuse.gpkg"]),
    "overture_places": dict(
        layer="Places", kind="vector", dataset="Overture Maps: places", provider="Overture Maps Foundation",
        native_resolution="n/a (vector)", licence="CDLA Permissive 2.0 (Foursquare part: Apache 2.0)",
        attribution="Overture Maps Foundation",
        used_for="n_amenities_15min (hex_amenities), after the name rules of 03-NA03; places and Gehl notebook",
        notes="The category of a place is often wrong; the notebooks type places by name.",
        files=["overture/{slug}_places*.geojson"]),
    "osm": dict(
        layer="Streets (walking and driving networks)", kind="vector",
        dataset="OpenStreetMap extract of the country from Geofabrik", provider="OpenStreetMap contributors; extracts by Geofabrik GmbH",
        native_resolution="n/a (vector)", licence="ODbL",
        attribution="© OpenStreetMap contributors; extracts by Geofabrik GmbH",
        used_for="walking network: hex_spacesyntax, hex_amenities, hex_green; street_length_m",
        notes="The extract is up to a day older than live OpenStreetMap.",
        files=["osm/{slug}_walk.gpkg", "osm/{slug}_drive.gpkg"]),
}


def _crs_label(crs) -> str:
    if crs is None:
        return "none"
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    return f"EPSG:{epsg}" if epsg else crs.to_string()[:60]


def _unit_resolution(res: float, crs) -> str:
    from pyproj import CRS

    try:
        pyproj_crs = CRS.from_wkt(crs.to_wkt()) if hasattr(crs, "to_wkt") else CRS.from_user_input(crs)
        metres = pyproj_crs.is_projected and abs(pyproj_crs.axis_info[0].unit_conversion_factor - 1.0) < 1e-9
        degrees = pyproj_crs.is_geographic
    except Exception:
        metres = degrees = False
    if metres:
        return f"{res:.4g} m"
    if degrees:
        return f"{res:.4g} degrees (about {res * 111_320:.2g} m)"
    return f"{res:.4g} (CRS units)"


def _file_facts(path: Path) -> tuple[str, str]:
    """(CRS label, resolution text) of a raster or vector file; '' when it cannot be read."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            import rasterio

            with rasterio.open(path) as src:
                return _crs_label(src.crs), _unit_resolution(abs(src.res[0]), src.crs)
        from pyproj import CRS

        try:
            import pyogrio

            crs_text = pyogrio.read_info(path).get("crs")
        except ImportError:
            import fiona

            with fiona.open(path) as src:
                crs_text = src.crs_wkt
        return (_crs_label(CRS.from_user_input(crs_text)) if crs_text else "none"), "n/a (vector)"
    except Exception:
        return "", ""


def _saved(paths: list[Path]) -> str:
    dates = sorted({dt.date.fromtimestamp(p.stat().st_mtime).isoformat() for p in paths})
    return (dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}") + " (file date: last save, not necessarily the download)"


def _find(data_dir: Path, patterns: list[str], slug: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found += sorted(data_dir.glob(pattern.format(slug=slug)))
    return found


def _landsat_scene(data_dir: Path) -> tuple[dict | None, Path | None, int]:
    candidates = sorted((data_dir / "landsat" / "output").glob("*/scene_info.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None, None, 0
    info = json.loads(candidates[-1].read_text(encoding="utf8"))
    return info, candidates[-1], len(candidates)


def _crs_rows(local_epsg: int, lon: float | None, lat: float | None, hex_resolution: int | None) -> list[dict]:
    from pyproj import CRS

    working = CRS.from_epsg(local_epsg)
    rows = [dict(
        layer="Working CRS", kind="crs", dataset=f"EPSG:{local_epsg}, {working.name}", native_resolution="metre",
        used_for="all files of the analysis",
        notes="The global UTM zone of the site, in metres: lengths and areas can be calculated directly. The hexagon tables and "
              "the analysis of vector data use it. No file of the analysis is stored in VN-2000.")]
    if hex_resolution is not None:
        rows.append(dict(
            layer="Hex grid CRS", kind="crs", dataset=f"H3 resolution {hex_resolution}", native_resolution="about 0.1 km2 per hexagon",
            used_for="all hex_* tables (key h3_id)",
            notes=f"Hexagon polygons in EPSG:{local_epsg}; the lat and lon columns of the grid are the hexagon centres in EPSG:4326."))
    if lon is not None and lat is not None:
        try:
            from sitex.core import geo

            new, old = geo.vn2000_epsg_from_lonlat(lon, lat, scheme="2025"), geo.vn2000_epsg_from_lonlat(lon, lat, scheme="epsg")
        except Exception:
            new = old = None
        if new:
            note = (f"Information only. VN-2000 is the Vietnamese national CRS (TM-3, one central meridian per province). "
                    f"For this site: EPSG:{new}, {CRS.from_epsg(new).name} (2025 provincial scheme)")
            if old and old != new:
                note += f"; before 2025: EPSG:{old}, {CRS.from_epsg(old).name}"
            note += (". To combine these results with official VN-2000 data in GIS, CAD or BIM, reproject the vector data to "
                     f"EPSG:{local_epsg} (see 00-Data-Acquisition). The table of provincial meridians is not yet checked against the legal text.")
            rows.append(dict(layer="VN-2000 equivalent", kind="crs", dataset=f"EPSG:{new}, {CRS.from_epsg(new).name}",
                             used_for="information only", notes=note))
    return rows


def build_provenance(data_dir, slug: str, local_epsg: int, lon: float | None = None, lat: float | None = None,
                     hex_resolution: int | None = None) -> pd.DataFrame:
    """The data sources record of one site (see the module docstring).

    ``data_dir`` is the folder that holds ``dem/``, ``landsat/``, ``overture/`` and so on;
    ``lon`` and ``lat`` (the centre of the area) are used to name the VN-2000 equivalent.
    Layers whose files are not found are left out.
    """
    data_dir = Path(data_dir)
    rows = _crs_rows(local_epsg, lon, lat, hex_resolution)

    for key, spec in _DATASETS.items():
        spec = dict(spec)
        files = _find(data_dir, spec.pop("files"), slug)
        row_files = files

        if key == "landsat":
            info, info_path, n_scenes = _landsat_scene(data_dir)
            if info is None:
                continue
            scene_dir = info_path.parent
            row_files = sorted(scene_dir.glob("*_B*_clipped.tif"))
            spec["dataset"] += f", scene {info.get('landsat_id', scene_dir.name)}"
            spec["data_date"] = str(info.get("acquisition_date", TO_FILL))[:10]
            if n_scenes > 1:
                spec["notes"] += f" The folder holds {n_scenes} scenes; this is the most recently written one."
        elif key == "sentinel2":
            years = sorted({m.group(1) for p in files if (m := re.search(r"s2_(\d{4})_", p.name))})
            spec["data_date"] = ", ".join(years) if years else TO_FILL
        elif key == "open_buildings":
            years = sorted({m.group(1) for p in files if (m := re.search(r"_(\d{4})_", p.name))})
            spec["data_date"] = ", ".join(years) if years else TO_FILL
        elif key == "overture_places":
            releases = sorted({m.group(1) for p in files if (m := re.search(r"_places_(\d{4}-\d{2}-\d{2}(?:\.\d+)?)", p.name))})
            spec["data_date"] = f"release {releases[-1]}" if releases else TO_FILL
        elif key in ("overture_buildings", "overture_landuse"):
            places = _find(data_dir, _DATASETS["overture_places"]["files"], slug)
            hint = next((m.group(1) for p in places if (m := re.search(r"_places_(\d{4}-\d{2}-\d{2}(?:\.\d+)?)", p.name))), None)
            if hint:
                spec["data_date"] = f"{TO_FILL} (release not saved in the file; the places file of the same run is release {hint})"
        elif key == "osm":
            spec["data_date"] = TO_FILL
            spec["notes"] += " The date of the Geofabrik extract is not saved in the file."

        if not row_files:
            continue
        crs, res = _file_facts(row_files[0])
        rel = [str(p.relative_to(data_dir)).replace("\\", "/") for p in row_files]
        rows.append({**spec, "file": "; ".join(rel[:4]) + (f"; … ({len(rel)} files)" if len(rel) > 4 else ""),
                     "file_crs": crs, "file_resolution": res if spec["kind"] == "raster" else "n/a (vector)",
                     "retrieved": _saved(row_files)})

    hex_path = data_dir / f"hex_grid_{slug}.gpkg"
    if hex_path.exists():
        rows.append(dict(
            layer="Hex grid", kind="derived", dataset=f"H3 hexagons, resolution {hex_resolution or TO_FILL}",
            provider="built by sitex from the OpenStreetMap boundary of the area", native_resolution="about 0.1 km2 per hexagon",
            file=hex_path.name, file_crs=_file_facts(hex_path)[0], file_resolution="n/a (vector)",
            licence="derived from the OpenStreetMap boundary (ODbL)", attribution="© OpenStreetMap contributors",
            used_for="the shared unit of all hex_* tables (key h3_id)", retrieved=_saved([hex_path]),
            notes="A hexagon is kept when its centre lies inside the boundary."))

    other = [r["layer"] for r in rows if r.get("kind") in ("raster", "vector", "derived")
             and r.get("file_crs") and r["file_crs"] != f"EPSG:{local_epsg}"]
    if other:
        rows[0]["notes"] += (" Stored in another CRS (see file_crs): " + "; ".join(other) +
                             ". EPSG:4326 means degrees. The notebooks reproject such vector layers, and sample or resample such rasters, when they use them; the files on disk are not changed.")
    for r in rows:
        if r.get("kind") in ("raster", "vector") and not r.get("data_date"):
            r["data_date"] = TO_FILL      # an empty date would read as "not applicable"
    return pd.DataFrame(rows).reindex(columns=COLUMNS).fillna("")
