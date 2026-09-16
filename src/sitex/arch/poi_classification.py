"""Program — points of interest to functional group, Gehl activity, and public-amenity flags.

Pure table lookups extracted from ``01-ARCH-POIsClassification.ipynb`` — no live LLM call.
The classification lookup (category -> functional group / Gehl activity) is a
place-independent table, prepared once by the background
``00- Overture - Activities Classifier.ipynb`` notebook from a handful of sample cities
just to get broad Overture category coverage. It is not tied to those cities: it is
applied here to whatever AOI's places ``00-Data-Acquisition.ipynb`` (Section 8,
``sitex.data.overture.extract_by_aoi``) already extracted, by category slug alone — any
OSM place, not only the sample cities used to build the lookup.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..core.config import CityConfig

# Straightforward functional groups -> public-amenity yes/no.
FUNCTIONAL_GROUP_TO_PUBLIC_AMENITY = {
    "Healthcare": True,
    "Education": True,
    "Civic & Government": False,
    "Community & Religion": False,
    "Culture & Arts": False,
    "Daily Needs & Grocery": False,
    "Food & Drink": False,
    "Retail & Shopping": False,
    "Accommodation & Travel": False,
    "Beauty & Personal Care": False,
    "Pet Services": False,
    "Home & Garden Services": False,
    "Business & Professional Services": False,
    "Utilities & Infrastructure": False,  # background infrastructure, not a place people visit
    "Agriculture & Industry": False,
}

# Recreation & Sport (public parks/trails vs. private gyms/clubs) and Automotive &
# Transport (public transit vs. private car dealers/repair) are genuinely mixed and
# need a finer split by Overture's own taxonomy branch, not a single yes/no.
MIXED_FUNCTIONAL_GROUPS = ("Recreation & Sport", "Automotive & Transport")
PUBLIC_L1_BRANCHES = {
    "park", "recreational_trail_or_path", "land_feature", "scenic_viewpoint", "water_feature", "rural_attraction",
    "ground_transport_facility_or_service", "transport_interchange",
    "air_transport_facility_or_service", "water_transport_facility_or_service", "parking",
}

GEHL_ACTIVITIES = ("Necessary", "Optional", "Social")

GEHL_COLORS = {
    "Necessary": "#4a90d9",
    "Optional": "#7bc67e",
    "Social": "#f4a261",
    "Unclassified": "#cccccc",
}

FINAL_COLUMNS = [
    "id", "name", "overture_category", "functional_group",
    "gehl_activity", "gehl_activities",
    "is_necessary", "is_optional", "is_social",
    "necessary_score", "social_score", "optional_score",
    "is_public_amenity",
    "thirdspace_score", "is_thirdspace",
    "geometry",
]


# ── 1. Classification lookup ────────────────────────────────────────────────

def load_classification_lookup(path: Path) -> pd.DataFrame:
    """Load the category -> functional group / Gehl activity lookup (JSON -> table)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run '00- Overture - Activities Classifier.ipynb' once "
            "in the background to generate it."
        )
    with open(path, encoding="utf-8") as f:
        classification = json.load(f)
    lookup = pd.DataFrame.from_dict(classification, orient="index").reset_index()
    return lookup.rename(columns={"index": "overture_category"})


def load_classification_lookup_for_city(
    city: CityConfig, filename: str = "overture_place_classification.json"
) -> pd.DataFrame:
    return load_classification_lookup(city.data_dir / "reference" / filename)


def _l1_branch(hierarchy) -> str | None:
    """Second segment of a 'group > branch > leaf > ...' taxonomy path, if present."""
    if not isinstance(hierarchy, str) or not hierarchy.strip():
        return None
    parts = [p.strip() for p in hierarchy.split(">")]
    return parts[1] if len(parts) > 1 else None


def load_taxonomy_l1_lookup(path: Path) -> dict:
    """Category slug -> L1 taxonomy branch, indexed by both current and legacy category names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run '00- Overture - Activities Classifier.ipynb' once "
            "(it downloads and caches Overture's taxonomy) before this, no BART needed."
        )
    taxonomy_df = pd.read_csv(path, encoding="utf-8-sig")
    category_to_l1: dict = {}
    for row in taxonomy_df.to_dict("records"):
        for cat_col, hier_col in [
            ("New Primary Category", "New Primary Hierarchy"),
            ("Old Primary Category", "Old Primary Hierarchy"),
        ]:
            cat = str(row.get(cat_col) or "").strip()
            if cat and cat not in category_to_l1:
                category_to_l1[cat] = _l1_branch(row.get(hier_col))
    return category_to_l1


def load_taxonomy_l1_lookup_for_city(city: CityConfig, filename: str = "overture_taxonomy.csv") -> dict:
    return load_taxonomy_l1_lookup(city.data_dir / "reference" / filename)


def _is_public_amenity(functional_group: str, overture_category: str, category_to_l1: dict) -> bool:
    if functional_group in FUNCTIONAL_GROUP_TO_PUBLIC_AMENITY:
        return FUNCTIONAL_GROUP_TO_PUBLIC_AMENITY[functional_group]
    if functional_group in MIXED_FUNCTIONAL_GROUPS:
        return category_to_l1.get(overture_category) in PUBLIC_L1_BRANCHES
    return False  # unmapped / "Other / Unclassified" defaults to not-public


def add_public_amenity_flag(lookup: pd.DataFrame, category_to_l1: dict) -> pd.DataFrame:
    """Add ``is_public_amenity``: whether an accessibility study (isochrones to
    healthcare, education, ...) should treat this category as an essential
    public/civic service — a different question from Gehl activity (why
    someone visits), so a hospital is both Necessary and a public amenity,
    while a private gym is Optional and not one.
    """
    lookup = lookup.copy()
    lookup["is_public_amenity"] = [
        _is_public_amenity(fg, cat, category_to_l1)
        for fg, cat in zip(lookup["functional_group"], lookup["overture_category"])
    ]
    return lookup


# ── 2. AOI places ────────────────────────────────────────────────────────────

def load_places_for_aoi(city: CityConfig, filename: str | None = None) -> gpd.GeoDataFrame:
    """Load one AOI's Overture places, extracted by ``00-Data-Acquisition.ipynb``
    (Section 8, ``sitex.data.overture.extract_by_aoi``) for ``city.slug``.

    Any OSM-geocodable place works here — the site is whatever ``CityConfig`` was
    built with, not one of a fixed set of pre-downloaded cities. Column names are
    normalized (``category`` -> ``primary_category``, ``name`` -> ``names``) to match
    the schema ``classify_places`` expects, since ``extract_by_aoi``'s live DuckDB
    query and the classifier notebook's cached backup name these differently.
    """
    filename = filename or f"{city.slug}_places.gpkg"
    path = city.data_dir / "overture" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run '00-Data-Acquisition.ipynb' (Section 8, Overture Maps) "
            f"for this AOI first, so it can extract the 'places' layer for '{city.slug}'."
        )
    places = gpd.read_file(path)
    return places.rename(columns={"category": "primary_category", "name": "names"})


# ── 3. Classification join ──────────────────────────────────────────────────

_LOOKUP_JOIN_COLS = [
    "functional_group", "gehl_activity", "gehl_activities", "is_public_amenity",
    "necessary_score", "social_score", "optional_score", "thirdspace_score",
]


def classify_places(places: gpd.GeoDataFrame, lookup: pd.DataFrame):
    """Join places to the classification lookup on ``primary_category``.

    Categories present in this city's data but missing from the reference
    table (new/rare categories) default to ``"Other / Unclassified"`` /
    ``"Unclassified"`` / not-a-public-amenity rather than dropping the row.
    Returns ``(pois, missing_categories)`` — the second is a value-count
    ``Series`` of categories that hit that default, for a coverage check.
    """
    lookup_indexed = lookup.set_index("overture_category")

    pois = places.dropna(subset=["primary_category"]).reset_index(drop=True).copy()
    join_cols = [c for c in _LOOKUP_JOIN_COLS if c in lookup_indexed.columns]
    pois = pois.join(lookup_indexed[join_cols], on="primary_category")

    missing = pois.loc[pois["functional_group"].isna(), "primary_category"].value_counts()

    pois["functional_group"] = pois["functional_group"].fillna("Other / Unclassified")
    pois["gehl_activity"] = pois["gehl_activity"].fillna("Unclassified")
    pois["is_public_amenity"] = pois["is_public_amenity"].fillna(False).astype(bool)
    pois["gehl_activities"] = pois["gehl_activities"].apply(lambda x: x if isinstance(x, list) else ["Unclassified"])

    pois = pois.rename(columns={"names": "name", "primary_category": "overture_category"})
    return pois, missing


# ── 4. Gehl activity flags ───────────────────────────────────────────────────

def add_gehl_activity_flags(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``is_necessary``/``is_optional``/``is_social`` — a POI can support more
    than one — plus ``is_thirdspace``, then flatten ``gehl_activities`` to a
    pipe-separated string (GeoJSON/GPKG can't hold list-typed columns).
    """
    pois = pois.copy()
    for activity in GEHL_ACTIVITIES:
        pois[f"is_{activity.lower()}"] = pois["gehl_activities"].apply(lambda activities, a=activity: a in activities)

    # thirdspace_score is a first-pass rule: 1.0 for candidate third places, 0.0 for
    # excluded functional groups, NaN for categories missing from the reference table.
    # Anything below 1.0 (including NaN) is treated as not-a-third-place.
    pois["is_thirdspace"] = pois["thirdspace_score"] == 1.0

    pois["gehl_activities"] = pois["gehl_activities"].apply(lambda xs: "|".join(xs))
    return pois


def select_final_columns(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return pois[[c for c in FINAL_COLUMNS if c in pois.columns]]


# ── 5. Map ────────────────────────────────────────────────────────────────

def plot_pois_gehl_map(pois: gpd.GeoDataFrame, zoom_start: int = 15):
    """Interactive Gehl-activity map: one togglable layer per activity (a POI
    can appear in more than one), plus independent public-amenity and
    candidate-third-place overlays, off by default.
    """
    import folium

    center = [pois.geometry.y.mean(), pois.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron", control_scale=True)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
    ).add_to(m)

    activity_flag_cols = {"Necessary": "is_necessary", "Optional": "is_optional", "Social": "is_social"}
    activity_groups = {}
    for activity, flag_col in activity_flag_cols.items():
        count = pois[flag_col].sum()
        fg = folium.FeatureGroup(name=f"{activity} ({count:,})")
        activity_groups[activity] = fg
        fg.add_to(m)

    unclassified_mask = ~(pois["is_necessary"] | pois["is_optional"] | pois["is_social"])
    if unclassified_mask.any():
        fg = folium.FeatureGroup(name=f"Unclassified ({unclassified_mask.sum():,})")
        activity_groups["Unclassified"] = fg
        fg.add_to(m)

    for _, row in pois.iterrows():
        qualifying = [activity for activity, col in activity_flag_cols.items() if row[col]]
        if not qualifying:
            qualifying = ["Unclassified"]
        popup = f"{row.get('name', '')} — {row['functional_group']} ({row['gehl_activities']})"
        for activity in qualifying:
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color=GEHL_COLORS.get(row["gehl_activity"], "#999999"),
                fill=True,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(activity_groups[activity])

    public_pois = pois[pois["is_public_amenity"]]
    public_amenity_group = folium.FeatureGroup(name=f"Public amenities ({len(public_pois):,})", show=False)
    for _, row in public_pois.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color="#000000", weight=1, fill=True,
            fill_color="#e63946", fill_opacity=0.9,
            popup=f"{row.get('name', '')} — {row['functional_group']}",
        ).add_to(public_amenity_group)
    public_amenity_group.add_to(m)

    thirdspace_pois = pois[pois["is_thirdspace"]]
    thirdspace_group = folium.FeatureGroup(name=f"Third places, draft ({len(thirdspace_pois):,})", show=False)
    for _, row in thirdspace_pois.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color="#000000", weight=1, fill=True,
            fill_color="#6a4c93", fill_opacity=0.9,
            popup=f"{row.get('name', '')} — {row['functional_group']}",
        ).add_to(thirdspace_group)
    thirdspace_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_rows = "".join(
        f'<div style="margin:2px 0;"><span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:50%;margin-right:6px;"></span>{activity}</div>'
        for activity, color in GEHL_COLORS.items()
        if activity in activity_groups
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>Gehl activity</b>
        {legend_rows}
        <div style="margin-top:6px;"><span style="display:inline-block;width:12px;height:12px;
            background:#e63946;border:1px solid #000;border-radius:50%;margin-right:6px;"></span>
            Public amenity (layer)</div>
        <div style="margin-top:4px;"><span style="display:inline-block;width:12px;height:12px;
            background:#6a4c93;border:1px solid #000;border-radius:50%;margin-right:6px;"></span>
            Third place, draft (layer)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── 6. Export ─────────────────────────────────────────────────────────────

def export_classified_pois_geojson(pois: gpd.GeoDataFrame, out_path: Path) -> Path:
    """Export in WGS84 — GeoJSON convention, and what the folium map needs."""
    pois.to_file(out_path, driver="GeoJSON")
    return out_path


def export_classified_pois_geojson_for_city(pois: gpd.GeoDataFrame, city: CityConfig) -> Path:
    path = city.data_dir / "overture" / f"pois_classified_{city.slug}.geojson"
    return export_classified_pois_geojson(pois, path)


def export_classified_pois_gpkg(pois: gpd.GeoDataFrame, out_path: Path, local_epsg: int) -> Path:
    """Reproject to a metric UTM CRS and export.

    Required for any spatial join against buildings (``sitex.arch.morphology``)
    or street-network outputs, which are already in a metric CRS — joining
    the WGS84 GeoJSON version instead silently distorts distances.
    """
    pois.to_crs(epsg=local_epsg).to_file(out_path, driver="GPKG")
    return out_path


def export_classified_pois_gpkg_for_city(
    pois: gpd.GeoDataFrame, city: CityConfig, local_epsg: int | None = None
) -> Path:
    """Reprojects to ``local_epsg`` (default ``city.local_epsg``) — supersedes the
    original notebook's hardcoded ``CITY_LOCAL_EPSG`` per-city dict, since
    ``CityConfig`` already derives the correct UTM zone from the site's own
    coordinates instead of a lookup table someone has to remember to extend.
    """
    epsg = local_epsg if local_epsg is not None else city.local_epsg
    path = city.data_dir / "overture" / f"pois_classified_{city.slug}.gpkg"
    return export_classified_pois_gpkg(pois, path, epsg)
