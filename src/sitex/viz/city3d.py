"""3D city viewer — the extruded building mesh, optionally draped with a raster or
vector accessibility variable.

Extracted from ``04-VIZ-3D_City_Viewer.ipynb``. Rendering is Plotly ``Mesh3d`` (pure
JS/WebGL, confirmed safe in this project's environment) — never matplotlib. Neither
drape section colours the OBJ file itself: a merged OBJ has no per-building grouping
once flattened, so each rebuilds the same city geometry from the enriched buildings
GPKG (``sitex.arch.morphology.extrude_footprint``) and colours each building's own
faces by a sampled/joined value instead.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from ..arch.morphology import extrude_footprint


def load_city_mesh(obj_path: Path):
    """Load ``city.obj`` (``01-ARCH-Building_Morphology.ipynb``, Section 13) for inline viewing."""
    import trimesh

    return trimesh.load(obj_path, process=False)


def plot_mesh_elevation(mesh, title: str = "City mesh"):
    """Elevation-coloured Mesh3d — the terrain slope reads even with no separate ground layer."""
    import plotly.graph_objects as go

    v, f = mesh.vertices, mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=v[:, 2], colorscale="Viridis", colorbar=dict(title="Elevation (m)"),
        flatshading=True, lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
    )])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="X (m, local origin)", yaxis_title="Y (m, local origin)", zaxis_title="Elevation (m)"),
        width=950, height=750, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def sample_raster_at_centroids(bldg_utm: gpd.GeoDataFrame, raster_path: Path):
    """Sample a raster (already in the buildings' own CRS) at each building centroid —
    one point per call (a batched multi-point ``.sample()`` has been observed to crash
    the interpreter on some Windows/conda GDAL builds). Returns ``(values, outside_mask)``."""
    import rasterio

    centroids = bldg_utm.geometry.centroid
    cx, cy = centroids.x.values, centroids.y.values
    with rasterio.open(raster_path) as ds:
        nodata = ds.nodata
        vals = np.array([next(ds.sample([(cx[i], cy[i])]))[0] for i in range(len(cx))], dtype=float)
    outside = np.isnan(vals) if nodata is None else ((vals == nodata) | np.isnan(vals))
    return vals, outside


def sample_vector_at_centroids(bldg_utm: gpd.GeoDataFrame, gpkg_path: Path, value_col: str):
    """Spatial-join a polygon grid (e.g. an H3 accessibility grid) onto building
    centroids. Returns ``(values, outside_mask)`` — ``outside_mask`` covers buildings
    with no matching polygon."""
    grid = gpd.read_file(gpkg_path)
    centroids = bldg_utm[["uID"]].copy()
    centroids["geometry"] = bldg_utm.geometry.centroid
    centroids = gpd.GeoDataFrame(centroids, geometry="geometry", crs=bldg_utm.crs)
    joined = (
        gpd.sjoin(centroids, grid[[value_col, "geometry"]], how="left", predicate="within")
        .drop(columns="geometry")
        .drop_duplicates(subset="uID")
    )
    vals = joined.set_index("uID")[value_col].reindex(bldg_utm["uID"].values).values.astype(float)
    return vals, np.isnan(vals)


def robust_range(values: np.ndarray, mask: np.ndarray | None = None, pct: tuple = (2, 98)) -> tuple:
    """Percentile-based colour range instead of exact min/max, so a handful of outlier
    pixels/cells don't wash out the rest of the scale."""
    valid = values if mask is None else values[~mask]
    valid = valid[np.isfinite(valid)]
    lo, hi = np.percentile(valid, pct)
    if lo == hi:
        lo, hi = float(np.min(valid)), float(np.max(valid))
    return float(lo), float(hi)


def build_colored_mesh(bldg_local: gpd.GeoDataFrame, base_z: np.ndarray, values: np.ndarray, outside_mask: np.ndarray, cmap_name: str, vmin: float, vmax: float, reversed_cmap: bool = False, no_data_color: tuple = (0.55, 0.55, 0.55)):
    """Extrude every footprint and colour its faces by ``values`` (normalized to
    ``[vmin, vmax]``, typically ``robust_range``'s percentile range). Buildings flagged
    in ``outside_mask`` get a flat no-data grey instead of being blended into the
    scale. Returns ``(vertices, faces, face_colors_uint8)``."""
    import matplotlib
    import matplotlib.colors as mcolors

    cmap = matplotlib.colormaps[cmap_name + ("_r" if reversed_cmap else "")]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    all_verts, all_faces, all_face_colors = [], [], []
    v_offset = 0
    for geom, h, bz, val, nd in zip(bldg_local.geometry.values, bldg_local["height_est"].values, base_z, values, outside_mask):
        if geom is None or geom.is_empty or h is None or not np.isfinite(h) or h <= 0:
            continue
        result = extrude_footprint(geom, float(bz), float(h))
        if result is None:
            continue
        verts, faces = result
        all_verts.append(verts)
        all_faces.append(faces + v_offset)
        rgb = no_data_color if nd else cmap(norm(val))[:3]
        all_face_colors.extend([rgb] * len(faces))
        v_offset += len(verts)

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    face_colors = (np.array(all_face_colors) * 255).astype(np.uint8)
    return verts, faces, face_colors


def plot_colored_mesh(verts: np.ndarray, faces: np.ndarray, face_colors: np.ndarray, title: str):
    """Plotly Mesh3d with explicit per-face RGB colours (from ``build_colored_mesh``)."""
    import plotly.graph_objects as go

    facecolor_strings = [f"rgb({r},{g},{b})" for r, g, b in face_colors[:, :3]]
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=facecolor_strings, flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.5, specular=0.05),
    )])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="X (m, local origin)", yaxis_title="Y (m, local origin)", zaxis_title="Elevation (m)"),
        width=950, height=750, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
