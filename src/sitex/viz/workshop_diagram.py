"""Two-panel workshop diagrams — an exploded data stack beside the 3D site model.

Two stacks, same right panel:

- ``stack="environment"`` — buildings footprint, DEM hillshade, land cover
  (ESA WorldCover), canopy height (Meta/WRI CHM), NDVI, UHI.
- ``stack="network"`` — street network, raw Overture places, the name-typed public
  amenities (``03-NA03-Amenity_Isochrones.ipynb``), green accessibility (walk to public
  green, ``03-NA06-Green_Accessibility.ipynb``) and space-syntax reach
  (``03-NA01-SpaceSyntax.ipynb``).

The right panel is always the focused site's ``{site_slug}_site.obj``, as exported by
``01-SITE-3D_Model.ipynb`` — the only 3D model made, since 3D is produced for the
focused site alone, never for the whole ward.

Paths are derived from the caller's own ``CityConfig`` (``city.data_dir``,
``city.output_dir``, ``city.slug``), the same convention every other ``sitex.viz``
module uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from rasterio.mask import mask as rio_mask

from ..core.config import CityConfig
from ..data.landcover import WC_CLASSES

# All vector/raster layers in the exploded stack are read into this CRS for plotting.
TARGET_CRS = "EPSG:4326"
STACKS = ("environment", "network")


@dataclass
class LayerStyle:
    label: str
    color: str | None = None
    alpha: float = 1.0
    cmap: str | None = None


@dataclass
class WorkshopDiagramConfig:
    """Style/geometry knobs for the diagrams -- *not* file locations. Which files to
    read comes from the caller's own ``CityConfig`` instead (see module docstring),
    except ``ndvi_path``/``lst_path`` (which NDVI date or Landsat scene to show is a
    per-run choice) and ``site_slug`` (which focused site's model to show).
    """

    ndvi_path: Path
    lst_path: Path
    site_slug: str = "site_1"
    streets_layer: str = "edges"

    # Vertical gap between exploded layers, as a fraction of the site's N-S extent.
    layer_gap_frac: float = 0.42

    elev: float = 30
    azim: float = -62
    figsize: tuple = (20, 13)
    dpi: int = 220

    # Right panel: the site model. Heights are small next to a ~2 km site, so they are
    # exaggerated vertically to stay readable.
    site_elev: float = 32
    site_azim: float = -55
    site_z_exaggeration: float = 3.0
    site_zoom: float = 0.85              # model size within its panel (1 = fill it)

    # Largest raster side (pixels) drawn per plate: the 1 m canopy raster is decimated.
    raster_max_px: int = 900

    # ── Layout: titles, labels, legends (figure fractions: 0 = left/bottom, 1 = right/top)
    suptitle_y: float = 0.90
    suptitle_fontsize: float = 15
    stack_titles: dict = field(default_factory=lambda: {
        "environment": "Environmental data stack",
        "network": "Network & accessibility stack",
    })
    stack_panel_title: str = "Data stack"
    stack_title_gap: float = 0.45        # height above the top plate, in plate gaps
    site_panel_title: str = "Site model"
    site_title_dy: float = 0.03          # distance above the model's top edge
    panel_title_fontsize: float = 12
    label_offset: float = 0.12           # layer label distance right of the plate, in plate widths
    label_fontsize: float = 9.5
    legend_fontsize: float = 8.5
    amenity_legend_dy: float = -0.03     # amenity legend: distance below its plate label
    site_legend_dy: float = -0.02        # site legend: distance below the model's bottom edge

    # ── Site model context (other exports of 01-SITE-3D_Model, same coordinates)
    site_show_terrain: bool = True       # {site_slug}_terrain.obj
    site_show_landcover: bool = True     # {site_slug}_landcover.obj, coloured by its .mtl
    site_terrain_color: str = "#e6dfd1"

    # environment stack
    buildings: LayerStyle = field(default_factory=lambda: LayerStyle("Buildings footprint\n(Overture)", "#3a4a5c", 0.9))
    dem: LayerStyle = field(default_factory=lambda: LayerStyle("DEM — terrain hillshade\n(GEDTM30)", alpha=0.95, cmap="gray"))
    landcover: LayerStyle = field(default_factory=lambda: LayerStyle("Land cover\n(ESA WorldCover 2021)", alpha=0.95))
    canopy: LayerStyle = field(default_factory=lambda: LayerStyle("Canopy height\n(Meta/WRI CHM, 1 m)", alpha=0.95, cmap="Greens"))
    ndvi: LayerStyle = field(default_factory=lambda: LayerStyle("NDVI — vegetation index\n(Sentinel-2)", alpha=0.95, cmap="RdYlGn"))
    uhi: LayerStyle = field(default_factory=lambda: LayerStyle(
        "UHI — land surface temperature\n(Landsat, remote sensing)", alpha=0.95, cmap="inferno"
    ))

    # network stack
    streets: LayerStyle = field(default_factory=lambda: LayerStyle("Street network\n(OSM)", "#e8823c", 1.0))
    places: LayerStyle = field(default_factory=lambda: LayerStyle("Places, raw\n(Overture)", "#1f9d7c", 0.85))
    amenities: LayerStyle = field(default_factory=lambda: LayerStyle("Public amenities, typed by name\n(Amenity Isochrones)", alpha=0.95))
    green: LayerStyle = field(default_factory=lambda: LayerStyle(
        "Green accessibility — walk to public green\n(Green Accessibility)", alpha=0.85, cmap="YlGn_r"
    ))
    reach: LayerStyle = field(default_factory=lambda: LayerStyle("Reach — street length within 800 m\n(Space Syntax)", cmap="plasma"))

    # Amenity groups shown on the amenities plate, with their colours.
    amenity_colors: dict = field(default_factory=lambda: {
        "Healthcare": "#e41a1c", "Education": "#377eb8",
        "Daily Needs & Grocery": "#4daf4a", "Civic & Government": "#984ea3",
    })
    green_max_min: float = 20.0          # walk times beyond this are drawn grey
    reach_column: str = "reach_R800"

    # Site model colours, matched by keyword in the OBJ group name.
    site_colors: dict = field(default_factory=lambda: {
        "RESIDENTIAL": "#d9d2c3", "MIXED": "#e0a458", "COMMERCIAL": "#e76f51",
        "INDUSTRIAL": "#8d99ae", "PUBLIC": "#2a9d8f", "CIVIC": "#2a9d8f", "EDUCATION": "#2a9d8f",
    })
    site_default_color: str = "#c6cbd1"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _boundary_path(city: CityConfig) -> Path:
    return city.data_dir / "overture" / f"{city.slug}_boundary.gpkg"


def _buildings_path(city: CityConfig) -> Path:
    return city.data_dir / "overture" / f"{city.slug}_buildings.gpkg"


def _streets_path(city: CityConfig) -> Path:
    return city.data_dir / "osm" / f"{city.slug}_streets.gpkg"


def _places_path(city: CityConfig) -> Path:
    return city.data_dir / "overture" / f"{city.slug}_places.gpkg"


def _dem_path(city: CityConfig) -> Path:
    return city.data_dir / "dem" / "dtm_gedtm30.tif"


def _landcover_path(city: CityConfig) -> Path:
    return city.data_dir / "landcover" / f"esa_worldcover_2021_{city.slug}.tif"


def _canopy_path(city: CityConfig) -> Path:
    return city.data_dir / "canopy_height" / f"canopy_height_chmv2_{city.slug}.tif"


def _amenities_path(city: CityConfig) -> Path:
    return city.data_dir / "gdf" / f"pois_amenities_{city.slug}.gpkg"


def _green_path(city: CityConfig) -> Path:
    return city.data_dir / "gdf" / "green_accessibility_grid.gpkg"


def _reach_path(city: CityConfig) -> Path:
    return city.data_dir / "gdf" / "segment_results.gpkg"


def site_obj_path(city: CityConfig, style: WorkshopDiagramConfig, part: str = "site") -> Path:
    """``part`` = "site" (buildings), "terrain" or "landcover"."""
    return city.output_dir / f"{style.site_slug}_{part}.obj"


def input_paths(city: CityConfig, style: WorkshopDiagramConfig, stack: str) -> dict[str, Path]:
    """Every file one diagram reads, for a quick existence check before rendering."""
    common = {"boundary": _boundary_path(city), "site model": site_obj_path(city, style)}
    if style.site_show_terrain:
        common["site terrain"] = site_obj_path(city, style, "terrain")
    if style.site_show_landcover:
        common["site land cover"] = site_obj_path(city, style, "landcover")
    if stack == "environment":
        layers = {
            "buildings": _buildings_path(city), "DEM": _dem_path(city), "land cover": _landcover_path(city),
            "canopy height": _canopy_path(city), "NDVI": style.ndvi_path, "UHI": style.lst_path,
        }
    elif stack == "network":
        layers = {
            "streets": _streets_path(city), "places": _places_path(city), "amenities": _amenities_path(city),
            "green accessibility": _green_path(city), "space syntax": _reach_path(city),
        }
    else:
        raise ValueError(f"stack must be one of {STACKS}, got {stack!r}")
    return {**common, **layers}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read(path: Path, preferred_layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector file; in a multi-layer GeoPackage take ``preferred_layer`` when
    present (a stale layer from an old slug can otherwise be the default)."""
    layers = fiona.listlayers(path)
    layer = preferred_layer if preferred_layer in layers else layers[0]
    return gpd.read_file(path, layer=layer)


def load_boundary(city: CityConfig) -> gpd.GeoDataFrame:
    return _read(_boundary_path(city), f"{city.slug}_boundary").to_crs(TARGET_CRS)


def _clipped(path: Path, boundary: gpd.GeoDataFrame, layer: str | None = None) -> gpd.GeoDataFrame:
    return gpd.clip(_read(path, layer).to_crs(TARGET_CRS), boundary)


def load_buildings(city: CityConfig, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _clipped(_buildings_path(city), boundary, f"{city.slug}_buildings")


def load_streets(city: CityConfig, style: WorkshopDiagramConfig, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _clipped(_streets_path(city), boundary, style.streets_layer)


def load_places(city: CityConfig, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _clipped(_places_path(city), boundary, f"{city.slug}_places")


def load_amenities(city: CityConfig, style: WorkshopDiagramConfig, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = _clipped(_amenities_path(city), boundary)
    return gdf[(gdf["keep"] == 1) & gdf["amenity_group"].isin(style.amenity_colors)]


def load_raster(path: Path, boundary: gpd.GeoDataFrame, target_crs: str = TARGET_CRS,
                max_px: int | None = None, categorical: bool = False):
    """Read a raster cropped to the boundary and return an (X, Y, Z) grid in target_crs.

    Works for rasters in any source CRS: the pixel-center grid is built in the raster's
    native CRS, then transformed into target_crs. Rasters larger than ``max_px`` on a
    side are reduced first — block mean for continuous values, every n-th pixel for
    ``categorical`` classes (a mean of class codes would be meaningless).
    """
    with rasterio.open(path) as src:
        geom = [boundary.to_crs(src.crs).geometry.union_all()]
        fill = src.nodata if src.nodata is not None else 0
        arr, transform = rio_mask(src, geom, crop=True, nodata=fill, filled=True)
        arr = arr[0].astype("float64")
        arr[arr == fill] = np.nan
        src_crs = src.crs

    step = 1
    if max_px and max(arr.shape) > max_px:
        step = math.ceil(max(arr.shape) / max_px)
        if categorical:
            arr = arr[::step, ::step]
        else:
            h, w = (arr.shape[0] // step) * step, (arr.shape[1] // step) * step
            blocks = arr[:h, :w].reshape(h // step, step, w // step, step)
            with np.errstate(all="ignore"):
                arr = np.nanmean(blocks, axis=(1, 3))

    h, w = arr.shape
    xs = transform.c + transform.a * step * (np.arange(w) + 0.5)
    ys = transform.f + transform.e * step * (np.arange(h) + 0.5)
    x, y = np.meshgrid(xs, ys)

    if str(src_crs) != target_crs:
        transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
        x, y = transformer.transform(x, y)

    return x, y, arr


def hillshade(elevation: np.ndarray, cellsize: float = 30.0, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Standard analytical hillshade from an elevation grid."""
    az_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elevation, cellsize)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    return np.clip(shaded, 0, 1)


def site_extent_m(boundary: gpd.GeoDataFrame, local_epsg: int) -> tuple:
    """Real-world width/height of the site in meters, used for the 3D box aspect."""
    minx, miny, maxx, maxy = boundary.to_crs(epsg=local_epsg).total_bounds
    return maxx - minx, maxy - miny


def load_obj(path: Path) -> dict[str, np.ndarray]:
    """Faces of a grouped OBJ as ``{group name: (n_faces, 3, 3) array}``. Only
    ``v``/``f``/``o``/``g`` lines are read; polygons are fanned into triangles."""
    verts, groups, current = [], {}, "default"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith(("o ", "g ")):
                current = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                tris = groups.setdefault(current, [])
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts)
    return {name: v[np.asarray(tris)] for name, tris in groups.items() if tris}


def load_mtl_colors(obj_path: Path) -> dict[str, tuple]:
    """``{material name: (r, g, b)}`` from the ``.mtl`` next to an OBJ (empty if none)."""
    mtl = Path(obj_path).with_suffix(".mtl")
    colors, name = {}, None
    if mtl.exists():
        for line in mtl.read_text(encoding="utf-8").splitlines():
            if line.startswith("newmtl "):
                name = line.split(maxsplit=1)[1].strip()
            elif line.startswith("Kd ") and name:
                colors[name] = tuple(float(v) for v in line.split()[1:4])
    return colors


# ---------------------------------------------------------------------------
# Layer plotting
# ---------------------------------------------------------------------------

def add_polygon_layer(ax, gdf: gpd.GeoDataFrame, z: float, color, alpha: float) -> None:
    """``color`` is one colour, or one per row of ``gdf``."""
    polys, colors = [], []
    per_row = not isinstance(color, str)
    for geom, c in zip(gdf.geometry, color if per_row else [color] * len(gdf)):
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            xy = np.asarray(part.exterior.coords)
            polys.append(np.column_stack([xy, np.full(len(xy), z)]))
            colors.append(c)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolor="none", alpha=alpha), autolim=False)


def add_line_layer(ax, gdf: gpd.GeoDataFrame, z: float, color, linewidth: float = 0.7, alpha: float = 1.0) -> None:
    """``color`` is one colour, or one per row of ``gdf``."""
    segments, colors = [], []
    per_row = not isinstance(color, str)
    for geom, c in zip(gdf.geometry, color if per_row else [color] * len(gdf)):
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            xy = np.asarray(part.coords)
            segments.append(np.column_stack([xy, np.full(len(xy), z)]))
            colors.append(c)
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha), autolim=False)


def add_point_layer(ax, gdf: gpd.GeoDataFrame, z: float, color, size: float = 5.0, alpha: float = 0.85) -> None:
    pts = gdf.geometry.representative_point()
    ax.scatter(pts.x.to_numpy(), pts.y.to_numpy(), zs=z, zdir="z", s=size, c=color, alpha=alpha,
               depthshade=False, linewidths=0)


def add_raster_layer(ax, x, y, arr, z: float, cmap: str, levels: int = 30, alpha: float = 0.95):
    vmin, vmax = np.nanpercentile(arr, [2, 98])
    return ax.contourf(x, y, arr, zdir="z", offset=z, levels=np.linspace(vmin, vmax, levels),
                       cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)


def add_categorical_raster_layer(ax, x, y, arr, z: float, classes: dict, alpha: float = 0.95):
    """Class-coded raster with fixed colours: ``classes`` is ``{code: (label, colour)}``."""
    codes = sorted(classes)
    edges = [codes[0] - 5] + [(a + b) / 2 for a, b in zip(codes[:-1], codes[1:])] + [codes[-1] + 5]
    return ax.contourf(x, y, arr, zdir="z", offset=z, levels=edges,
                       colors=[classes[c][1] for c in codes], alpha=alpha)


def plate_corners(boundary: gpd.GeoDataFrame) -> np.ndarray:
    minx, miny, maxx, maxy = boundary.total_bounds
    return np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])


def add_plate_frame(ax, boundary: gpd.GeoDataFrame, z: float, color: str = "black", linewidth: float = 0.8) -> None:
    ring = np.asarray(boundary.geometry.iloc[0].exterior.coords)
    ax.plot(ring[:, 0], ring[:, 1], zs=z, zdir="z", color=color, linewidth=linewidth, alpha=0.6)


def add_vertical_connectors(ax, corners: np.ndarray, z_levels: list, color: str = "gray") -> None:
    for cx, cy in corners:
        ax.plot([cx, cx], [cy, cy], [min(z_levels), max(z_levels)],
                color=color, linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.35)


def _to_figure(fig, ax, x, y, z) -> tuple[float, float]:
    """Figure-fraction position of the 3D point (x, y, z) on ``ax``. The axes are first
    shrunk to their box aspect (normally done only at draw time), so the position
    matches the saved image."""
    ax.apply_aspect()
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return tuple(fig.transFigure.inverted().transform(ax.transData.transform((x2, y2))))


def add_layer_label(fig, ax, boundary: gpd.GeoDataFrame, z: float, text: str,
                    offset: float = 0.12, fontsize: float = 9.5) -> tuple[float, float]:
    """Place a label at the screen position corresponding to 3D point (x, y, z).

    Drawn with ``fig.text`` (figure space, not axes/data space) so it always renders on
    top of every plate -- a plain ``ax.text(x, y, z, ...)`` call can end up visually
    behind or crowded against a neighboring plate once the 3D-to-2D projection is
    applied to a very elongated footprint like a ward.
    """
    minx, miny, maxx, maxy = boundary.total_bounds
    fx, fy = _to_figure(fig, ax, maxx + (maxx - minx) * offset, miny + (maxy - miny) * 0.5, z)
    fig.text(fx, fy, text, fontsize=fontsize, va="center", ha="left", fontweight="medium")
    return fx, fy


def _values_to_colors(values, cmap: str, vmin: float, vmax: float, over_color: str = "#d0d0d0"):
    """Colours for ``values``; NaN/inf/above-``vmax`` get ``over_color``."""
    values = np.asarray(values, dtype="float64")
    rgba = plt.get_cmap(cmap)(Normalize(vmin, vmax, clip=True)(values))
    bad = ~np.isfinite(values) | (values > vmax)
    rgba[bad] = plt.matplotlib.colors.to_rgba(over_color)
    return rgba


# ---------------------------------------------------------------------------
# Stack layers
# ---------------------------------------------------------------------------

@dataclass
class StackLayer:
    """One plate: its label, a ``draw(ax, z)`` callable, and whether it gets a boundary
    frame (area layers do; sparse footprints/points read better without one)."""
    label: str
    draw: Callable
    frame: bool = True


def environment_layers(city: CityConfig, style: WorkshopDiagramConfig, boundary: gpd.GeoDataFrame) -> list[StackLayer]:
    buildings = load_buildings(city, boundary)
    dem = load_raster(_dem_path(city), boundary)
    lc = load_raster(_landcover_path(city), boundary, max_px=style.raster_max_px, categorical=True)
    chm_x, chm_y, chm = load_raster(_canopy_path(city), boundary, max_px=style.raster_max_px)
    chm[chm <= 0] = np.nan  # no canopy: leave the plate empty there
    ndvi = load_raster(style.ndvi_path, boundary, max_px=style.raster_max_px)
    lst = load_raster(style.lst_path, boundary, max_px=style.raster_max_px)
    s = style
    return [
        StackLayer(s.buildings.label, lambda ax, z: add_polygon_layer(ax, buildings, z, s.buildings.color, s.buildings.alpha), frame=False),
        StackLayer(s.dem.label, lambda ax, z: add_raster_layer(ax, dem[0], dem[1], hillshade(dem[2]), z, s.dem.cmap, alpha=s.dem.alpha)),
        StackLayer(s.landcover.label, lambda ax, z: add_categorical_raster_layer(ax, *lc, z, WC_CLASSES, alpha=s.landcover.alpha)),
        StackLayer(s.canopy.label, lambda ax, z: add_raster_layer(ax, chm_x, chm_y, chm, z, s.canopy.cmap, alpha=s.canopy.alpha)),
        StackLayer(s.ndvi.label, lambda ax, z: add_raster_layer(ax, *ndvi, z, s.ndvi.cmap, alpha=s.ndvi.alpha)),
        StackLayer(s.uhi.label, lambda ax, z: add_raster_layer(ax, *lst, z, s.uhi.cmap, alpha=s.uhi.alpha)),
    ]


def network_layers(city: CityConfig, style: WorkshopDiagramConfig, boundary: gpd.GeoDataFrame) -> list[StackLayer]:
    s = style
    streets = load_streets(city, style, boundary)
    places = load_places(city, boundary)
    amenities = load_amenities(city, style, boundary)
    amenity_colors = amenities["amenity_group"].map(s.amenity_colors).tolist()

    green = _clipped(_green_path(city), boundary)
    green_colors = _values_to_colors(green["access_public_min"], s.green.cmap, 0, s.green_max_min)

    reach = _clipped(_reach_path(city), boundary)
    rv = reach[s.reach_column].to_numpy(dtype="float64")
    lo, hi = np.nanpercentile(rv, [2, 98])
    reach_colors = _values_to_colors(rv, s.reach.cmap, lo, hi)

    return [
        StackLayer(s.streets.label, lambda ax, z: add_line_layer(ax, streets, z, s.streets.color)),
        StackLayer(s.places.label, lambda ax, z: add_point_layer(ax, places, z, s.places.color, alpha=s.places.alpha), frame=False),
        StackLayer(s.amenities.label, lambda ax, z: add_point_layer(ax, amenities, z, amenity_colors, size=14, alpha=s.amenities.alpha)),
        StackLayer(s.green.label, lambda ax, z: add_polygon_layer(ax, green, z, green_colors, s.green.alpha)),
        StackLayer(s.reach.label, lambda ax, z: add_line_layer(ax, reach, z, reach_colors, linewidth=1.0)),
    ]


def build_stack_panel(fig, ax, city: CityConfig, style: WorkshopDiagramConfig, stack: str = "environment") -> None:
    boundary = load_boundary(city)
    if stack == "environment":
        layers = environment_layers(city, style, boundary)
    elif stack == "network":
        layers = network_layers(city, style, boundary)
    else:
        raise ValueError(f"stack must be one of {STACKS}, got {stack!r}")

    dx, dy = site_extent_m(boundary, city.local_epsg)
    gap = dy * style.layer_gap_frac
    z_levels = [i * gap for i in range(len(layers))]
    ax.set_box_aspect((dx, dy, (len(layers) - 1) * gap * 1.15))

    for layer, z in zip(layers, z_levels):
        layer.draw(ax, z)
        if layer.frame:
            add_plate_frame(ax, boundary, z)
    add_vertical_connectors(ax, plate_corners(boundary), z_levels)

    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(z_levels[0] - gap * 0.2, z_levels[-1] + gap * 0.35)
    ax.set_axis_off()
    ax.view_init(elev=style.elev, azim=style.azim)

    # Titles, labels and legends last: their screen position depends on the final
    # camera and limits. The title sits just above the top plate, not at the top of the
    # 3D box (mostly empty space).
    fx, fy = _to_figure(fig, ax, (minx + maxx) / 2, (miny + maxy) / 2, z_levels[-1] + gap * style.stack_title_gap)
    fig.text(fx, fy, style.stack_panel_title, fontsize=style.panel_title_fontsize, fontweight="bold",
             ha="center", va="bottom")

    label_pos = {}
    for layer, z in zip(layers, z_levels):
        label_pos[layer.label] = add_layer_label(fig, ax, boundary, z, layer.label,
                                                 style.label_offset, style.label_fontsize)

    if stack == "network":
        lx, ly = label_pos[style.amenities.label]
        handles = [Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in style.amenity_colors.items()]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(lx, ly + style.amenity_legend_dy),
                   fontsize=style.legend_fontsize, frameon=False, borderaxespad=0, handletextpad=0.3)


# ---------------------------------------------------------------------------
# Right panel: the focused site's 3D model
# ---------------------------------------------------------------------------

def _site_color(style: WorkshopDiagramConfig, group: str) -> str:
    name = group.upper()
    return next((c for key, c in style.site_colors.items() if key in name), style.site_default_color)


def _shaded_face_colors(tris: np.ndarray, color: str, light_dir=(-0.4, -0.6, 0.7)) -> np.ndarray:
    """One RGBA per triangle: ``color`` darkened by how far the face turns away from
    the light, so roofs, lit walls and shaded walls read as separate planes."""
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.asarray(light_dir) / np.linalg.norm(light_dir)
    intensity = 0.55 + 0.45 * np.abs(normals @ light)  # abs: winding-independent
    rgba = np.tile(plt.matplotlib.colors.to_rgba(color), (len(tris), 1))
    rgba[:, :3] *= intensity[:, None]
    return rgba


def build_site_panel(fig, ax, city: CityConfig, style: WorkshopDiagramConfig) -> None:
    """Terrain, land cover and buildings of the focused site. Ground and buildings are
    two collections drawn in a fixed order (ground first): matplotlib sorts faces by
    their centre, and the long, thin land-cover triangles would otherwise be drawn over
    small buildings. From an above-ground camera the buildings are always in front."""
    tris_all, colors_all = [], []

    if style.site_show_terrain and site_obj_path(city, style, "terrain").exists():
        for tris in load_obj(site_obj_path(city, style, "terrain")).values():
            tris_all.append(tris)
            colors_all.append(_shaded_face_colors(tris, style.site_terrain_color))

    if style.site_show_landcover and site_obj_path(city, style, "landcover").exists():
        lc_path = site_obj_path(city, style, "landcover")
        mtl = load_mtl_colors(lc_path)
        for name, tris in load_obj(lc_path).items():
            tris_all.append(tris)
            # flat colour: nearly horizontal, and shading its thin triangles shows streaks
            colors_all.append(np.tile((*mtl.get(name, (0.7, 0.7, 0.7)), 1.0), (len(tris), 1)))

    buildings = load_obj(site_obj_path(city, style))
    b_tris = np.concatenate(list(buildings.values()))
    b_colors = np.concatenate([_shaded_face_colors(t, _site_color(style, n)) for n, t in buildings.items()])

    ax.computed_zorder = False
    if tris_all:
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris_all), facecolors=np.concatenate(colors_all),
                                             edgecolors="none", zorder=1), autolim=False)
    ax.add_collection3d(Poly3DCollection(b_tris, facecolors=b_colors, edgecolors="none", zorder=2), autolim=False)

    allv = np.concatenate([b_tris.reshape(-1, 3)] + [t.reshape(-1, 3) for t in tris_all])
    (minx, miny, minz), (maxx, maxy, maxz) = allv.min(axis=0), allv.max(axis=0)
    ax.set_box_aspect((maxx - minx, maxy - miny, (maxz - minz) * style.site_z_exaggeration), zoom=style.site_zoom)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(minz, maxz)
    ax.set_axis_off()
    ax.view_init(elev=style.site_elev, azim=style.site_azim)

    # Title just above, legend just below the model's projected outline.
    corners = [(x, y, z) for x in (minx, maxx) for y in (miny, maxy) for z in (minz, maxz)]
    screen = np.array([_to_figure(fig, ax, *c) for c in corners])
    cx = (screen[:, 0].min() + screen[:, 0].max()) / 2
    fig.text(cx, screen[:, 1].max() + style.site_title_dy, style.site_panel_title,
             fontsize=style.panel_title_fontsize, fontweight="bold", ha="center", va="bottom")

    handles = [Line2D([], [], marker="s", linestyle="", color=_site_color(style, n),
                      label=n.replace("BLDG_", "").title()) for n in buildings]
    if style.site_show_landcover and site_obj_path(city, style, "landcover").exists():
        handles += [Line2D([], [], marker="s", linestyle="", color=c, label=n.replace("LC_", "").replace("_", " ").title())
                    for n, c in load_mtl_colors(site_obj_path(city, style, "landcover")).items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(cx, screen[:, 1].min() + style.site_legend_dy),
               ncol=min(len(handles), 4), fontsize=style.legend_fontsize, frameon=False, borderaxespad=0)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(city: CityConfig, style: WorkshopDiagramConfig, stack: str = "environment", place_label: str | None = None):
    """Build one two-panel figure: the ``stack`` on the left, the site model on the
    right. ``place_label`` defaults to the locality part of ``city.place_name``."""
    label = place_label or city.place_name.split(",")[0]

    fig = plt.figure(figsize=style.figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.5, 1), wspace=0.02)
    ax_stack = fig.add_subplot(gs[0, 0], projection="3d")
    ax_site = fig.add_subplot(gs[0, 1], projection="3d")

    build_stack_panel(fig, ax_stack, city, style, stack)
    build_site_panel(fig, ax_site, city, style)

    fig.suptitle(f"{style.stack_titles[stack]} — {label}", fontsize=style.suptitle_fontsize,
                 fontweight="bold", y=style.suptitle_y)
    return fig, (ax_stack, ax_site)
