"""H3 hex grid shared by every notebook that reports per hexagon.

``00-Data-Acquisition.ipynb`` builds the grid once and saves it as ``data/hex_grid.gpkg``;
the accessibility notebooks build the same grid on the fly with :func:`build_h3_grid`.

The three ``*_to_hex`` functions move a notebook's results onto that grid, one row per
hexagon, keyed by ``h3_id``: :func:`points_to_hex`, :func:`polygons_to_hex` (vectors) and
:func:`rasters_to_hex` (rasters), and :func:`lines_to_hex` (streets); :func:`assign_h3_id` adds
``h3_id`` to a feature table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

DEFAULT_HEX_RESOLUTION = 9  # about 0.105 km2 per hexagon, edge about 174 m


def build_h3_grid(boundary_poly, resolution: int) -> gpd.GeoDataFrame:
    """H3 hex grid covering ``boundary_poly`` (Polygon or MultiPolygon, EPSG:4326).

    Columns: h3_id, geometry, lat, lon. A hexagon is kept when its centre lies inside
    the boundary, so hexagons on the edge may reach slightly outside it.
    """
    import geopandas as gpd
    import h3
    from shapely.geometry import Polygon

    parts = list(boundary_poly.geoms) if hasattr(boundary_poly, "geoms") else [boundary_poly]
    hex_ids = set()
    for part in parts:
        exterior_coords = [(lat, lng) for lng, lat in part.exterior.coords]
        hex_ids.update(h3.h3shape_to_cells(h3.LatLngPoly(exterior_coords), resolution))

    hex_data = []
    for h_id in sorted(hex_ids):
        boundary_pts = h3.cell_to_boundary(h_id)
        poly = Polygon([(lng, lat) for lat, lng in boundary_pts])
        centroid = poly.centroid
        hex_data.append({"h3_id": h_id, "geometry": poly, "lat": centroid.y, "lon": centroid.x})

    return gpd.GeoDataFrame(hex_data, crs="EPSG:4326")


def make_hex_grid(boundary_poly, local_epsg: int, resolution: int = DEFAULT_HEX_RESOLUTION) -> gpd.GeoDataFrame:
    """The shared hex grid, ready to save: polygons in the site's local metric CRS.

    Columns: h3_id, lat, lon (hex centroid, EPSG:4326), area_m2 (polygon area in
    ``local_epsg``), geometry (hex polygon in ``local_epsg``).
    """
    local = build_h3_grid(boundary_poly, resolution).to_crs(epsg=local_epsg)
    local["area_m2"] = local.area
    return local[["h3_id", "lat", "lon", "area_m2", "geometry"]]


def _cell_ids(gdf, resolution: int) -> list[str]:
    """H3 cell of every feature's centroid (a point is its own centroid)."""
    import warnings

    import h3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # centroid of a geographic CRS: fine at this scale
        centres = gdf.geometry.centroid.to_crs(epsg=4326)
    return [h3.latlng_to_cell(p.y, p.x, resolution) for p in centres]


def _most_frequent(values):
    """Most frequent value (ties: the first in sorted order); NaN if all are missing."""
    import numpy as np

    counts = values.dropna().value_counts()
    return np.nan if counts.empty else counts.sort_index().idxmax()


def assign_h3_id(features: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame) -> pd.Series:
    """``h3_id`` of the hexagon holding each feature's centroid; NaN outside the grid.

    Same assignment as :func:`points_to_hex` and :func:`polygons_to_hex`; use it to add
    ``h3_id`` to a feature table you export.
    """
    import h3
    import pandas as pd

    ids = pd.Series(_cell_ids(features, h3.get_resolution(hexes["h3_id"].iloc[0])), index=features.index)
    return ids.where(ids.isin(hexes["h3_id"]))


def _features_to_hex(features, hexes, aggs: dict) -> pd.DataFrame:
    table = features.drop(columns="geometry").assign(h3_id=assign_h3_id(features, hexes))
    spec = {out: (src, _most_frequent if how == "mode" else how) for out, (src, how) in aggs.items()}
    result = table.groupby("h3_id").agg(**spec).reindex(hexes["h3_id"]).reset_index()
    for out, (_src, how) in aggs.items():
        if how == "count":  # a hexagon without features has 0 of them ...
            result[out] = result[out].fillna(0).astype("int64")
        elif how == "sum":
            result[out] = result[out].fillna(0)
        # ... but no mean, median or most frequent value: those stay NaN
    return result


def points_to_hex(points: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, **aggs) -> pd.DataFrame:
    """Summarise point features per hexagon.

    Each point goes to the hexagon that contains it; points outside the grid are dropped.
    ``aggs`` name the output columns: ``n_pois=("poi_id", "count")`` counts, and
    ``mean_rating=("rating", "mean")`` averages. The summary can be ``"count"``, ``"sum"``,
    ``"mean"``, ``"median"``, ``"min"``, ``"max"`` or ``"mode"`` (most frequent value).

    Returns one row per hexagon of ``hexes``, in its order: ``h3_id`` and the named columns.
    Hexagons without points get 0 for counts and sums, NaN for the other summaries.
    """
    return _features_to_hex(points, hexes, aggs)


def polygons_to_hex(polygons: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, **aggs) -> pd.DataFrame:
    """Summarise polygon features (buildings, parcels) per hexagon.

    Each polygon goes to the hexagon that contains its centroid, whole: a building across
    a hexagon edge is not split, so sums add up to the total of the polygons inside the
    grid. Same ``aggs`` and return value as :func:`points_to_hex`; for example
    ``built_area_m2=("area_m2", "sum")``, ``dominant_function=("bldg_function", "mode")``.
    """
    return _features_to_hex(polygons, hexes, aggs)


def lines_to_hex(lines: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, length_col: str = "street_length_m", **means) -> pd.DataFrame:
    """Length-weighted mean of street metrics per hexagon.

    Lines are cut at the hexagon edges, because a street can be much longer than a
    hexagon (an axial line crosses many). Each piece counts with its length, so
    ``integration_mean="integration"`` is the mean of the ``integration`` column over
    the street inside the hexagon.

    Returns one row per hexagon of ``hexes``: ``h3_id``, ``length_col`` (metres of street
    inside the hexagon, 0 if none) and one column per keyword. The mean is NaN for a
    hexagon without street, or when the metric is missing on all of it.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    sources = list(means.values())
    pieces = gpd.overlay(lines[sources + ["geometry"]], hexes[["h3_id", "geometry"]].to_crs(lines.crs),
                         how="intersection", keep_geom_type=True)
    pieces["_len"] = pieces.length
    rows = []
    for h_id, grp in pieces.groupby("h3_id"):
        row = {"h3_id": h_id, length_col: grp["_len"].sum()}
        for out, src in means.items():
            ok = grp[src].notna()
            row[out] = np.average(grp.loc[ok, src], weights=grp.loc[ok, "_len"]) if ok.any() else np.nan
        rows.append(row)
    result = pd.DataFrame(rows, columns=["h3_id", length_col, *means])
    result = result.set_index("h3_id").reindex(hexes["h3_id"]).reset_index()
    result[length_col] = result[length_col].fillna(0.0)
    return result


def rasters_to_hex(
    hexes: gpd.GeoDataFrame,
    layers: dict,
    transform,
    crs,
    min_valid_frac: float = 0.5,
    supersample: int = 4,
) -> pd.DataFrame:
    """Area-weighted mean of one or more rasters per hexagon.

    ``layers`` maps the output column name to a 2D array; all arrays share one grid
    (``transform``, ``crs``) and use NaN for no data. Every pixel is weighted by the
    share of it that lies inside the hexagon (each pixel is cut into ``supersample`` x
    ``supersample`` parts to measure that share). Only pixels valid in *all* layers count,
    so the columns of one row describe the same ground.

    Returns one row per hexagon, in the order of ``hexes``: ``h3_id``, one column per layer,
    ``n_valid_px`` (valid pixels in the hexagon, area-weighted, so fractional) and
    ``valid_frac`` (their share of the hexagon area, 0 to 1). Below ``min_valid_frac`` the
    layer columns are NaN; a hexagon outside the raster has ``valid_frac`` 0.
    """
    import numpy as np
    import pandas as pd
    import rasterio.features
    import rasterio.windows
    from affine import Affine
    from pyproj import CRS

    names = list(layers)
    stack = np.stack([np.asarray(layers[name], dtype="float64") for name in names])
    valid = np.isfinite(stack).all(axis=0)
    stack = np.where(valid, stack, 0.0)
    height, width = valid.shape
    pixel_area = abs(transform.a * transform.e)
    k = int(supersample)

    geoms = hexes.geometry if hexes.crs == CRS.from_user_input(crs) else hexes.to_crs(crs).geometry
    rows = []
    for h_id, geom in zip(hexes["h3_id"], geoms):
        win = rasterio.windows.from_bounds(*geom.bounds, transform=transform)
        r0, c0 = max(int(np.floor(win.row_off)), 0), max(int(np.floor(win.col_off)), 0)
        r1 = min(int(np.ceil(win.row_off + win.height)), height)
        c1 = min(int(np.ceil(win.col_off + win.width)), width)
        row = {"h3_id": h_id, **{name: np.nan for name in names}, "n_valid_px": 0.0, "valid_frac": 0.0}
        if r1 > r0 and c1 > c0:
            fine = rasterio.features.rasterize(
                [(geom, 1)],
                out_shape=((r1 - r0) * k, (c1 - c0) * k),
                transform=transform * Affine.translation(c0, r0) * Affine.scale(1 / k),
                fill=0,
                dtype="uint8",
            )
            share = fine.reshape(r1 - r0, k, c1 - c0, k).mean(axis=(1, 3))
            weight = share * valid[r0:r1, c0:c1]
            n_valid = float(weight.sum())
            row["n_valid_px"] = n_valid
            row["valid_frac"] = min(n_valid * pixel_area / geom.area, 1.0)
            if row["valid_frac"] >= min_valid_frac and n_valid > 0:
                means = (stack[:, r0:r1, c0:c1] * weight).sum(axis=(1, 2)) / n_valid
                row.update(zip(names, means))
        rows.append(row)

    return pd.DataFrame(rows)
