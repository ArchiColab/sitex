"""Mass — building footprints to enriched morphology, type, and 3D massing.

Extracted from ``01-ARCH-Building_Morphology.ipynb``. Land use is joined and
buildings are classified *before* height/floor estimation runs: a landuse-
polygon match overrides footprint-morphology classification, because in a
low-rise, semi-rural district a hospital or school footprint is not reliably
bigger or more regular than an ordinary house — an area/shape threshold alone
would silently misclassify it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from shapely import constrained_delaunay_triangles

from ..core.config import CityConfig

CIVIC_LANDUSE = {
    "civic", "education", "medical", "healthcare",
    "hospital", "clinic", "school", "college", "university",
    "government", "stadium", "sport", "pitch", "playground",
    "recreation", "park", "plaza", "cemetery", "religious", "place_of_worship",
}
INDUSTRIAL_LANDUSE = {"industrial", "logistics", "storage", "commercial_services"}
COMMERCIAL_LANDUSE = {"commercial", "retail", "mixed_use", "shopping"}
RESIDENTIAL_LANDUSE = {"residential", "living", "housing", "farmland"}


# ── 1. Load & clean ─────────────────────────────────────────────────────────

def load_buildings(path: Path, local_epsg: int) -> gpd.GeoDataFrame:
    """Load an Overture buildings GeoPackage, clean, and project to ``local_epsg``.

    Keeps only valid Polygon/MultiPolygon geometry, explodes MultiPolygons
    (momepy requires simple Polygons), and assigns the ``uID`` integer column
    momepy's functional API and every join below key on.
    """
    bldg = gpd.read_file(path)
    bldg = bldg[
        bldg.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        & bldg.geometry.notna()
        & bldg.geometry.is_valid
    ].copy()
    bldg = bldg.explode(index_parts=False).reset_index(drop=True)
    if bldg.crs is None or bldg.crs.to_epsg() != local_epsg:
        bldg = bldg.to_crs(epsg=local_epsg)
    bldg["uID"] = bldg.index.astype(int)
    return bldg


def load_buildings_for_city(city: CityConfig, filename: str | None = None) -> gpd.GeoDataFrame:
    """``load_buildings()`` reading ``{city.data_dir}/overture/{city.slug}_buildings.gpkg``."""
    path = city.data_dir / "overture" / (filename or f"{city.slug}_buildings.gpkg")
    return load_buildings(path, city.local_epsg)


# ── 2. Morphological characters (momepy) ────────────────────────────────────

def compute_morphology(bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add momepy footprint-morphology columns: area, perimeter, shape descriptors."""
    bldg = bldg.copy()
    bldg["area"] = bldg.geometry.area
    bldg["perimeter"] = bldg.geometry.length
    bldg["lal"] = momepy.longest_axis_length(bldg)
    bldg["convexity"] = momepy.convexity(bldg)
    bldg["elongation"] = momepy.elongation(bldg)
    bldg["rectangularity"] = momepy.rectangularity(bldg)
    bldg["eri"] = momepy.equivalent_rectangular_index(bldg)
    bldg["corners"] = momepy.corners(bldg)
    bldg["squareness"] = momepy.squareness(bldg)
    return bldg


# ── 3. Tessellation ──────────────────────────────────────────────────────────

def _union_all(gdf: gpd.GeoDataFrame):
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union


def load_boundary_for_city(city: CityConfig, filename: str | None = None) -> gpd.GeoDataFrame | None:
    """Load the Overture study-area boundary for ``city``, or ``None`` if absent.

    Passing this to ``compute_tessellation()`` matters: without it, the
    tessellation/block-detection limit falls back to the buildings' own
    convex hull, which can merge real distinct blocks near the AOI edge into
    one oversized enclosure once streets are clipped against a looser limit.
    """
    path = city.data_dir / "overture" / (filename or f"{city.slug}_boundary.gpkg")
    if not Path(path).exists():
        return None
    boundary = gpd.read_file(path)
    if boundary.crs is None or boundary.crs.to_epsg() != city.local_epsg:
        boundary = boundary.to_crs(epsg=city.local_epsg)
    return boundary


def compute_tessellation(bldg: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame | None = None):
    """Morphological (Voronoi-based) tessellation cell per building.

    Falls back to the buildings' own convex hull + 50 m buffer as the
    study-area limit when no explicit ``boundary`` layer is given. Returns
    ``(tessellation, limit)`` — ``limit`` is reused by ``detect_blocks()``.
    """
    limit = _union_all(boundary) if boundary is not None else _union_all(bldg).convex_hull.buffer(50)
    tessellation = momepy.morphological_tessellation(bldg.geometry, clip=limit)
    tessellation["uID"] = bldg.loc[tessellation.index, "uID"]
    return tessellation, limit


# ── 4. Streets, blocks, intersection density ────────────────────────────────

def load_streets_main_component(path: Path, local_epsg: int, snap_decimals: int = 2):
    """Load a streets layer, keep LineStrings, and filter to the main connected component.

    ``snap_decimals`` rounds endpoint coordinates (metres) before building the
    connectivity graph — 2 decimal places is cm precision in a metric CRS —
    so near-duplicate nodes from slightly misaligned digitising still merge
    into one graph node. Returns ``(streets, graph, main_nodes)``.
    """
    streets_raw = gpd.read_file(path)
    streets = streets_raw[
        streets_raw.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()
    if streets.crs is None or streets.crs.to_epsg() != local_epsg:
        streets = streets.to_crs(epsg=local_epsg)

    def snap_pt(coord):
        return (round(coord[0], snap_decimals), round(coord[1], snap_decimals))

    graph = nx.Graph()
    for idx, row in streets.iterrows():
        lines = (
            list(row.geometry.geoms)
            if row.geometry.geom_type == "MultiLineString"
            else [row.geometry]
        )
        for line in lines:
            coords = list(line.coords)
            graph.add_edge(snap_pt(coords[0]), snap_pt(coords[-1]), idx=idx)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    main_nodes = components[0] if components else set()

    def in_main(row):
        lines = (
            list(row.geometry.geoms)
            if row.geometry.geom_type == "MultiLineString"
            else [row.geometry]
        )
        return snap_pt(list(lines[0].coords)[0]) in main_nodes

    streets = streets[streets.apply(in_main, axis=1)].reset_index(drop=True)
    return streets, graph, main_nodes


def load_streets_for_city(city: CityConfig, filename: str | None = None, snap_decimals: int = 2):
    path = city.data_dir / "overture" / (filename or f"{city.slug}_streets.gpkg")
    return load_streets_main_component(path, city.local_epsg, snap_decimals)


def detect_blocks(streets: gpd.GeoDataFrame, limit, local_epsg: int) -> gpd.GeoDataFrame:
    """Polygonise ``streets`` into street-enclosed blocks (``momepy.enclosures``)."""
    limit_gdf = gpd.GeoDataFrame(geometry=[limit], crs=f"EPSG:{local_epsg}")
    blocks = momepy.enclosures(streets, limit=limit_gdf)
    blocks["block_id"] = blocks.index.astype(int)
    return blocks


def compute_intersection_density(
    graph: nx.Graph,
    main_nodes: set,
    blocks: gpd.GeoDataFrame,
    local_epsg: int,
    min_degree: int = 3,
) -> gpd.GeoDataFrame:
    """Add ``n_intersections``/``int_density`` (per km²) to ``blocks``.

    An intersection is a graph node of degree >= ``min_degree`` — an endpoint
    shared by more than one street segment.
    """
    intersections = [n for n, deg in graph.degree() if deg >= min_degree and n in main_nodes]
    int_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([n[0] for n in intersections], [n[1] for n in intersections]),
        crs=f"EPSG:{local_epsg}",
    )
    joined = gpd.sjoin(int_gdf, blocks[["block_id", "geometry"]], how="left", predicate="within")
    int_per_block = joined.groupby("block_id").size().rename("n_intersections")

    blocks = blocks.join(int_per_block, on="block_id")
    blocks["n_intersections"] = blocks["n_intersections"].fillna(0).astype(int)
    blocks["block_area_km2"] = blocks.geometry.area / 1e6
    blocks["int_density"] = blocks["n_intersections"] / blocks["block_area_km2"].replace(0, np.nan)
    return blocks


def join_block_to_buildings(bldg: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial-join each building (by centroid) to its enclosing block.

    Adds ``block_id``/``int_density``.
    """
    bldg_centroid = bldg.copy()
    bldg_centroid["geometry"] = bldg.geometry.centroid

    bldg_block = gpd.sjoin(
        bldg_centroid[["uID", "geometry"]],
        blocks[["block_id", "int_density", "n_intersections", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="geometry")

    return bldg.merge(bldg_block[["uID", "block_id", "int_density"]], on="uID", how="left")


# ── 5. Land use join & building-type classification ─────────────────────────

def join_landuse(bldg: gpd.GeoDataFrame, path: Path, local_epsg: int) -> gpd.GeoDataFrame:
    """Add ``lu_class`` from an Overture landuse polygon layer.

    Sets ``lu_class`` to ``None`` for every row if ``path`` doesn't exist,
    rather than raising — landuse coverage is optional, morphology-only
    classification still works without it. Must run before height/floor
    estimation; see the module docstring.
    """
    bldg = bldg.copy()
    if not Path(path).exists():
        bldg["lu_class"] = None
        return bldg

    landuse = gpd.read_file(path)
    if landuse.crs is None or landuse.crs.to_epsg() != local_epsg:
        landuse = landuse.to_crs(epsg=local_epsg)

    bldg_c = bldg.copy()
    bldg_c["geometry"] = bldg.geometry.centroid
    joined_lu = (
        gpd.sjoin(bldg_c[["uID", "geometry"]], landuse[["class", "geometry"]], how="left", predicate="within")
        .drop(columns="geometry")
        .rename(columns={"class": "lu_class"})
        .drop_duplicates(subset="uID")
    )
    return bldg.merge(joined_lu[["uID", "lu_class"]], on="uID", how="left")


def join_landuse_for_city(bldg: gpd.GeoDataFrame, city: CityConfig, filename: str | None = None) -> gpd.GeoDataFrame:
    path = city.data_dir / "overture" / (filename or f"{city.slug}_landuse.gpkg")
    return join_landuse(bldg, path, city.local_epsg)


def _classify_row(row: pd.Series) -> str:
    lu = str(row.get("lu_class", "")).lower() if pd.notna(row.get("lu_class")) else ""
    area = row["area"]
    eri = row["eri"]
    int_density = row["int_density"] if pd.notna(row.get("int_density")) else 0

    if lu in CIVIC_LANDUSE:
        return "civic"
    if lu in INDUSTRIAL_LANDUSE:
        return "industrial"
    if lu in COMMERCIAL_LANDUSE:
        return "mixed"
    if lu in RESIDENTIAL_LANDUSE:
        return "residential"

    if area > 500 and eri > 0.65 and int_density < 200:
        return "industrial"
    if 80 < area < 400 and int_density > 200:
        return "mixed"
    return "residential"


def classify_building_type(bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``bldg_type``: Overture landuse polygon overrides footprint morphology.

    Requires ``lu_class`` (``join_landuse``), ``area``/``eri`` (``compute_morphology``),
    and ``int_density`` (``join_block_to_buildings``) already present.
    """
    bldg = bldg.copy()
    bldg["bldg_type"] = bldg.apply(_classify_row, axis=1)
    return bldg


BUILDING_TYPE_COLORS = {
    "residential": "#f0a500",
    "industrial": "#5b6abf",
    "civic": "#2ca25f",
    "mixed": "#e03531",
    "other": "#aaaaaa",
}


def plot_building_types(bldg: gpd.GeoDataFrame, center_lat: float, center_lon: float, zoom_start: int = 14):
    """Interactive building-type map, coloured per ``BUILDING_TYPE_COLORS``.

    Uses folium rather than ``GeoDataFrame.plot()`` — the latter (any
    matplotlib rendering of a GeoDataFrame, not just this function) has been
    observed to crash the Python interpreter outright on this project's
    Windows/conda GDAL build, independent of the data plotted. This is the
    same category of environment fragility ``POI_Accessibility.ipynb``
    documents for its own summary chart (hand-built as raw SVG for the same
    reason) — folium sidesteps it entirely since it never touches matplotlib.
    """
    import folium

    bldg_wgs = bldg.to_crs(epsg=4326)

    def _style(feature):
        color = BUILDING_TYPE_COLORS.get(feature["properties"].get("bldg_type"), BUILDING_TYPE_COLORS["other"])
        return {"fillColor": color, "color": color, "weight": 0.3, "fillOpacity": 0.85}

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB positron")
    folium.GeoJson(
        bldg_wgs[["bldg_type", "geometry"]],
        name="Building type",
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(fields=["bldg_type"], aliases=["Type"]),
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── 6. Height / floor estimation ─────────────────────────────────────────────

@dataclass
class HeightRules:
    """Thresholds for the rule-based floor-count/height estimator below.

    The defaults are **placeholder assumptions tuned for one semi-rural
    Vietnamese context (Pleiku)** — not a measurement, and not universal. A
    typical footprint area, aspect ratio, or storey height that holds for
    Pleiku's low-rise row houses will not hold for a dense inner-city district,
    a suburban context, or a different building culture. Anyone reusing this
    for a different site **must inspect these numbers against real local
    reference buildings first** (satellite/street-view spot checks, local
    building codes, or a field survey) and edit the fields here — do not trust
    the defaults for a place they weren't tuned on. See the ARCH overview
    §5.1/§5.6: ``height_est``/``num_floors_est`` are modelled, not measured,
    and only as good as these thresholds.
    """

    # Below this footprint area: a small single-storey shed/hut.
    small_shed_area_max: float = 40.0
    small_shed_height: float = 3.0

    # Narrow "tube house" (typical Vietnamese row-house form): area below
    # `row_house_area_max` AND elongated (`eri` above `row_house_eri_min`).
    row_house_area_max: float = 120.0
    row_house_eri_min: float = 0.6
    row_house_floors_range: tuple[int, int] = (1, 3)
    row_house_height_range: tuple[float, float] = (3.0, 9.0)

    # Same area bracket, but not elongated: an ordinary small detached house.
    small_house_floors_range: tuple[int, int] = (2, 3)
    small_house_height_range: tuple[float, float] = (6.0, 9.0)

    # Mid-size footprint (below `midsize_area_max`), regular in plan
    # (`convexity` above `midsize_convexity_min`): shophouse or larger house.
    midsize_area_max: float = 300.0
    midsize_convexity_min: float = 0.85
    shophouse_floors_range: tuple[int, int] = (2, 4)
    shophouse_height_range: tuple[float, float] = (6.0, 12.0)

    # Same area bracket, irregular in plan: treated more conservatively.
    irregular_floors_range: tuple[int, int] = (2, 3)
    irregular_height_range: tuple[float, float] = (6.0, 9.0)

    # Larger footprint (below `public_area_max`): school/clinic/public
    # building — fixed rather than randomised, since these are civic types.
    public_area_max: float = 800.0
    public_floors: int = 2
    public_height: float = 8.0

    # Above `public_area_max`: warehouse/large shed — single storey, tall.
    warehouse_floors: int = 1
    warehouse_height: float = 6.0


def _estimate_floors_height_row(row: pd.Series, rules: HeightRules) -> pd.Series:
    area = row["area"]
    eri = row["eri"]
    conv = row["convexity"]

    if area < rules.small_shed_area_max:
        floors, height = 1, rules.small_shed_height
    elif area < rules.row_house_area_max and eri > rules.row_house_eri_min:
        floors = random.randint(*rules.row_house_floors_range)
        height = round(random.uniform(*rules.row_house_height_range), 1)
    elif area < rules.row_house_area_max:
        floors = random.randint(*rules.small_house_floors_range)
        height = round(random.uniform(*rules.small_house_height_range), 1)
    elif area < rules.midsize_area_max and conv > rules.midsize_convexity_min:
        floors = random.randint(*rules.shophouse_floors_range)
        height = round(random.uniform(*rules.shophouse_height_range), 1)
    elif area < rules.midsize_area_max:
        floors = random.randint(*rules.irregular_floors_range)
        height = round(random.uniform(*rules.irregular_height_range), 1)
    elif area < rules.public_area_max:
        floors, height = rules.public_floors, rules.public_height
    else:
        floors, height = rules.warehouse_floors, rules.warehouse_height

    return pd.Series({"num_floors_est": floors, "height_est": height})


def estimate_floors_height(
    bldg: gpd.GeoDataFrame, seed: int = 42, rules: HeightRules | None = None
) -> gpd.GeoDataFrame:
    """Fill missing ``height``/``num_floors`` with a seeded, rule-based estimate.

    Known Overture values are kept as-is; only missing values are estimated.
    The result is a plausible, reproducible estimate, not a measurement — flag
    it as modelled wherever it's used downstream, per the ARCH overview §5.1/§5.6.

    ``rules`` controls every threshold/range the estimate uses (see
    ``HeightRules``) and defaults to the Pleiku-tuned values if omitted —
    **check and pass your own** ``HeightRules(...)`` for a different site
    rather than relying on the default.
    """
    rules = rules or HeightRules()
    random.seed(seed)
    bldg = bldg.copy()

    bldg["num_floors_est"] = bldg["num_floors"].copy() if "num_floors" in bldg.columns else np.nan
    bldg["height_est"] = bldg["height"].copy() if "height" in bldg.columns else np.nan

    missing = bldg["num_floors_est"].isna() | bldg["height_est"].isna()
    est = bldg.loc[missing].apply(lambda row: _estimate_floors_height_row(row, rules), axis=1)
    bldg.loc[missing, ["num_floors_est", "height_est"]] = est

    bldg["volume_est"] = bldg["area"] * bldg["height_est"]
    return bldg


# ── 7. Export ─────────────────────────────────────────────────────────────

_EXPORT_EXTRA_COLS = [
    "uID", "geometry",
    "area", "perimeter", "lal",
    "convexity", "elongation", "rectangularity", "eri", "corners", "squareness",
    "block_id", "int_density",
    "num_floors_est", "height_est", "volume_est",
    "bldg_type",
]
_EXPORT_ORIG_COLS = ["id", "name", "height", "num_floors", "class", "subtype", "lu_class"]


def select_export_columns(bldg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Order columns for export: original Overture attributes first, then
    derived morphology/classification columns — whichever of each are present."""
    orig_keep = [c for c in _EXPORT_ORIG_COLS if c in bldg.columns]
    ordered = orig_keep + [c for c in _EXPORT_EXTRA_COLS if c not in orig_keep]
    ordered = [c for c in ordered if c in bldg.columns]
    return bldg[ordered].copy()


def shift_to_local_origin(gdf: gpd.GeoDataFrame, origin_easting: float, origin_northing: float) -> gpd.GeoDataFrame:
    """Translate ``gdf`` by ``-(origin_easting, origin_northing)`` and clear its CRS.

    The CAD-import copy every export in this project pairs with its
    true-coordinate GIS version, so geometry lands near ``(0, 0)`` instead of
    hundreds of km away — the precision Revit/Rhino need near their origin.
    """
    shifted = gdf.copy()
    shifted["geometry"] = shifted.geometry.translate(-origin_easting, -origin_northing)
    shifted = shifted.set_crs(None, allow_override=True)
    return shifted


def export_enriched_buildings(bldg: gpd.GeoDataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    select_export_columns(bldg).to_file(out_path, driver="GPKG")
    return out_path


def export_enriched_buildings_for_city(bldg: gpd.GeoDataFrame, city: CityConfig, filename: str | None = None) -> Path:
    path = city.data_dir / "gdf" / (filename or f"{city.slug}_buildings_enriched.gpkg")
    return export_enriched_buildings(bldg, path)


def export_enriched_buildings_local_for_city(
    bldg: gpd.GeoDataFrame, city: CityConfig, filename: str | None = None
) -> Path:
    path = city.data_dir / "gdf" / (filename or f"{city.slug}_buildings_enriched_local.gpkg")
    path.parent.mkdir(parents=True, exist_ok=True)
    local = shift_to_local_origin(select_export_columns(bldg), *city.origin)
    local.to_file(path, driver="GPKG")
    return path


# ── 8. 3D massing ────────────────────────────────────────────────────────────

def sample_dem_at_centroids(bldg: gpd.GeoDataFrame, dtm_path: Path) -> np.ndarray:
    """Sample a DEM raster at each building centroid — one point per call.

    A batched multi-point ``.sample()`` call has been observed to crash the
    interpreter on some Windows/conda GDAL builds (reproduced with as few as
    2 points); sampling one point at a time is slower but stable, and still
    only takes a few seconds for tens of thousands of buildings.
    """
    import rasterio

    centroids_wgs = bldg.geometry.centroid.to_crs(epsg=4326)
    lon = centroids_wgs.x.values
    lat = centroids_wgs.y.values

    with rasterio.open(dtm_path) as dtm:
        nodata = dtm.nodata
        base_z = np.array(
            [next(dtm.sample([(lon[i], lat[i])]))[0] for i in range(len(lon))],
            dtype=float,
        )

    bad = (base_z == nodata) | np.isnan(base_z)
    if bad.any():
        base_z[bad] = np.nanmedian(base_z[~bad])
    return base_z


def extrude_footprint(poly, base_z: float, height: float):
    """Extrude one polygon's exterior ring into a closed solid from ``base_z``
    to ``base_z + height``.

    Walls are simple ring-edge quads, valid for any ring shape. Caps are
    triangulated with shapely's own constrained Delaunay (GEOS) rather than a
    fan from vertex 0 — a fan only tiles correctly for a convex footprint; on a
    concave one (L/U-shapes, common in this dataset) some fan triangles fall
    outside the polygon entirely. GEOS triangulation is already part of the
    ``arch`` extra's hard dependencies (unlike ``trimesh``'s mapbox_earcut
    backend used in ``build_city_mesh`` below, which has been observed to
    crash intermittently over thousands of calls on some environments), so it
    doesn't add a new crash-prone library.

    Triangulating against ``poly`` itself (not just its exterior ring) matters
    for a footprint with a courtyard (interior ring/hole): a hole-boundary
    vertex is absent from the exterior-ring vertex index below, so any
    triangle touching it is dropped rather than wrongly bridging across the
    hole — the roof cap just leaves a gap there instead of filling it in
    solid. This still doesn't wall or cap the hole itself.

    The triangulator's own winding isn't guaranteed, so each cap triangle's
    orientation is checked (via its 2D signed area) and flipped as needed to
    keep normals facing outward — down for the bottom cap, up for the top.

    Returns ``(vertices[N,3], faces[M,3])``, or ``None`` for a degenerate ring.
    """
    ring = np.asarray(poly.exterior.coords)
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    n = len(ring)
    if n < 3:
        return None

    bottom = np.column_stack([ring, np.full(n, base_z)])
    top = np.column_stack([ring, np.full(n, base_z + height)])
    verts = np.vstack([bottom, top])  # bottom: 0..n-1 ; top: n..2n-1

    # ── Cap triangulation (respects concave boundaries, unlike a vertex-0 fan) ──
    index_of = {tuple(np.round(p, 6)): i for i, p in enumerate(ring)}
    faces = []
    for tri in constrained_delaunay_triangles(poly).geoms:
        pts = np.asarray(tri.exterior.coords)[:3]
        idx = [index_of.get(tuple(np.round(p, 6))) for p in pts]
        if None in idx:
            continue  # triangle touches a hole boundary (or a Steiner point) — skip it
        a, b, c = idx
        ax, ay = ring[a]; bx, by = ring[b]; cx, cy = ring[c]
        signed_area2 = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if signed_area2 < 0:                     # (a, b, c) already clockwise -> normal points down
            faces.append((a, b, c))               # bottom cap
            faces.append((n + a, n + c, n + b))   # top cap (reversed)
        else:                                     # (a, b, c) already counter-clockwise -> normal points up
            faces.append((a, c, b))               # bottom cap (reversed)
            faces.append((n + a, n + b, n + c))   # top cap

    for i in range(n):  # walls
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))

    return verts, np.array(faces, dtype=np.int64)


def build_city_mesh(bldg_local: gpd.GeoDataFrame, base_z: np.ndarray, heights: np.ndarray):
    """Extrude every footprint from ``base_z`` to ``base_z + height`` into one merged mesh."""
    import trimesh

    all_verts, all_faces = [], []
    v_offset = 0
    for geom, h, bz in zip(bldg_local.geometry.values, heights, base_z):
        if geom is None or geom.is_empty or h is None or not np.isfinite(h) or h <= 0:
            continue
        result = extrude_footprint(geom, float(bz), float(h))
        if result is None:
            continue
        verts, faces = result
        all_verts.append(verts)
        all_faces.append(faces + v_offset)
        v_offset += len(verts)

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_city_mesh_for_city(
    bldg: gpd.GeoDataFrame,
    bldg_local: gpd.GeoDataFrame,
    city: CityConfig,
    dtm_filename: str = "dtm_nasadem.tif",
    filename: str | None = None,
):
    """Sample the DEM from ``bldg`` (true CRS) and write the mesh from ``bldg_local``
    (shifted to the CAD origin) to ``{city.output_dir}/{city.slug}_city.obj``.

    ``bldg`` and ``bldg_local`` must be the same rows in the same order — as
    they are when ``bldg_local`` comes from ``export_enriched_buildings_local_for_city``'s
    ``shift_to_local_origin(select_export_columns(bldg), ...)``.
    """
    dtm_path = city.data_dir / "dem" / dtm_filename
    base_z = sample_dem_at_centroids(bldg, dtm_path)
    mesh = build_city_mesh(bldg_local, base_z, bldg_local["height_est"].values)
    out_path = city.output_dir / (filename or f"{city.slug}_city.obj")
    mesh.export(out_path)
    return out_path, base_z


def export_buildings_3d_geojson(bldg: gpd.GeoDataFrame, base_z: np.ndarray, out_path: Path) -> Path:
    """3D-vertex GeoJSON for deck.gl/geolibre: every footprint vertex gets
    ``z = base_z`` (from the DEM, not Overture's near-empty ``height`` field),
    and ``height_est`` goes in as a feature property for client-side extrusion.
    """
    import json

    from shapely.geometry import Polygon, mapping

    bldg_wgs = bldg.to_crs(epsg=4326)

    features = []
    for geom, h, bz, uid in zip(bldg_wgs.geometry.values, bldg["height_est"].values, base_z, bldg["uID"].values):
        if geom is None or geom.is_empty or h is None or not np.isfinite(h):
            continue
        ring_3d = [(x, y, float(bz)) for x, y in geom.exterior.coords]
        features.append({
            "type": "Feature",
            "geometry": mapping(Polygon(ring_3d)),
            "properties": {"uID": int(uid), "height": float(h), "base_z": float(bz)},
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    return out_path


def export_buildings_3d_geojson_for_city(
    bldg: gpd.GeoDataFrame, base_z: np.ndarray, city: CityConfig, filename: str | None = None
) -> Path:
    out_path = city.output_dir / (filename or f"{city.slug}_buildings_3d.geojson")
    return export_buildings_3d_geojson(bldg, base_z, out_path)


def _polygon_rings(geom):
    """Yield exterior + interior rings of a Polygon as coordinate lists."""
    yield list(geom.exterior.coords)
    for interior in geom.interiors:
        yield list(interior.coords)


def export_buildings_flat_dxf(bldg_local: gpd.GeoDataFrame, out_path: Path) -> Path:
    """Flat (Z = 0) building-footprint DXF — plain 2D polylines, for site plans /
    figure-ground massing studies in plan view.

    One layer per ``bldg_type`` (``BLDG_RESIDENTIAL``, ``BLDG_INDUSTRIAL``, ...) so
    a CAD user can isolate/filter by use. ``bldg_local`` must already be shifted to
    the CAD/UCS origin (``shift_to_local_origin``) — DXF carries no CRS metadata.
    """
    import ezdxf

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for geom, btype in zip(bldg_local.geometry.values, bldg_local["bldg_type"].values):
        if geom is None or geom.is_empty:
            continue
        layer = f"BLDG_{str(btype).upper()}"
        for ring in _polygon_rings(geom):
            msp.add_lwpolyline([(x, y) for x, y in ring], dxfattribs={"layer": layer})
    doc.saveas(out_path)
    return out_path


def export_buildings_flat_dxf_for_city(
    bldg_local: gpd.GeoDataFrame, city: CityConfig, filename: str | None = None
) -> Path:
    out_path = city.output_dir / (filename or f"{city.slug}_buildings_flat.dxf")
    return export_buildings_flat_dxf(bldg_local, out_path)


def export_buildings_dem_dxf(bldg_local: gpd.GeoDataFrame, base_z: np.ndarray, out_path: Path) -> Path:
    """Building-footprint DXF draped on the DEM — each footprint's ring placed at
    its own ground elevation (``base_z``, e.g. from ``sample_dem_at_centroids``) as
    a flat 3D polyline, not extruded to roof height (see ``build_city_mesh`` for
    that). For checking footprints against sloped terrain.

    Same per-``bldg_type`` layering as ``export_buildings_flat_dxf``. ``bldg_local``
    and ``base_z`` must be the same rows in the same order.
    """
    import ezdxf

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for geom, btype, bz in zip(bldg_local.geometry.values, bldg_local["bldg_type"].values, base_z):
        if geom is None or geom.is_empty:
            continue
        layer = f"BLDG_{str(btype).upper()}"
        for ring in _polygon_rings(geom):
            pts_3d = [(x, y, float(bz)) for x, y in ring]
            msp.add_polyline3d(pts_3d, dxfattribs={"layer": layer})
    doc.saveas(out_path)
    return out_path


def export_buildings_dem_dxf_for_city(
    bldg_local: gpd.GeoDataFrame, base_z: np.ndarray, city: CityConfig, filename: str | None = None
) -> Path:
    out_path = city.output_dir / (filename or f"{city.slug}_buildings_dem.dxf")
    return export_buildings_dem_dxf(bldg_local, base_z, out_path)
