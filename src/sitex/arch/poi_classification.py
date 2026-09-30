"""Program — points of interest to functional group and Gehl activity.

Whether a place is a *public* amenity is not decided here: category-level typing is
too coarse for that (a veterinary clinic tagged ``hospital``, a private clinic next to
a commune health station). ``03-NA03-Amenity_Isochrones.ipynb`` decides it per place,
from the place's name (:mod:`sitex.arch.poi_amenities`).

Pure table lookups — no live LLM call.
The classification lookup (category -> functional group / Gehl activity) is a
place-independent table, prepared once by the background
``00- Overture - Activities Classifier.ipynb`` notebook from a handful of sample cities
just to get broad Overture category coverage. It is not tied to those cities: it is
applied here to whatever AOI's places ``00-Data-Acquisition.ipynb`` (Section 8,
``sitex.data.overture.extract_by_aoi``) already extracted, by category slug alone — any
OSM place, not only the sample cities used to build the lookup.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..core.config import CityConfig

# Every functional group a reviewed lookup row may use.
FUNCTIONAL_GROUPS = (
    "Healthcare", "Education", "Civic & Government", "Community & Religion", "Culture & Arts",
    "Daily Needs & Grocery", "Food & Drink", "Retail & Shopping", "Accommodation & Travel",
    "Beauty & Personal Care", "Pet Services", "Home & Garden Services",
    "Business & Professional Services", "Utilities & Infrastructure", "Agriculture & Industry",
    "Recreation & Sport", "Automotive & Transport",
)

# Overture's second-level ("L1") taxonomy branches, hand-mapped once to this project's
# functional groups. Used by '00- Overture - Activities Classifier.ipynb' to build the
# lookup, and by ``classify_places`` as a fallback for categories the lookup has not
# seen (every place carries its own ``hierarchy`` since the 2026-09-23 schema).
L1_TO_FUNCTIONAL_GROUP = {
    # arts_and_entertainment
    "amusement_attraction": "Recreation & Sport",
    "animal_attraction": "Recreation & Sport",
    "arts_and_crafts_space": "Culture & Arts",
    "event_venue": "Culture & Arts",
    "festival_venue": "Culture & Arts",
    "gaming_venue": "Recreation & Sport",
    "movie_theater": "Culture & Arts",
    "museum": "Culture & Arts",
    "nightlife_venue": "Food & Drink",
    "performing_arts_venue": "Culture & Arts",
    "rural_attraction": "Recreation & Sport",
    "science_attraction": "Culture & Arts",
    "social_club": "Community & Religion",
    "spiritual_advising": "Community & Religion",
    "stadium_arena": "Recreation & Sport",
    "ticket_office_or_booth": "Culture & Arts",
    # community_and_government
    "armed_forces_branch": "Civic & Government",
    "children_hall": "Community & Religion",
    "civic_organization": "Civic & Government",
    "community_service": "Community & Religion",
    "government_office": "Civic & Government",
    "military_site": "Civic & Government",
    "public_and_government_association": "Civic & Government",
    "public_facility": "Civic & Government",
    "public_safety_service": "Civic & Government",
    "public_utility": "Utilities & Infrastructure",
    "social_or_community_service": "Community & Religion",
    # cultural_and_historic
    "cultural_center": "Culture & Arts",
    "historic_site": "Culture & Arts",
    "memorial_site": "Culture & Arts",
    "place_of_worship": "Community & Religion",
    "religious_landmark": "Community & Religion",
    "religious_organization": "Community & Religion",
    "religious_retreat_or_center": "Community & Religion",
    # education
    "education_office": "Education",
    "educational_facility": "Education",
    "educational_service": "Education",
    "library": "Community & Religion",
    "place_of_learning": "Education",
    "research_institute": "Education",
    # food_and_drink
    "alcoholic_beverage_venue": "Food & Drink",
    "casual_eatery": "Food & Drink",
    "non_alcoholic_beverage_venue": "Food & Drink",
    "restaurant": "Food & Drink",
    # geographic_entities
    "built_feature": "Civic & Government",
    "land_feature": "Recreation & Sport",
    "scenic_viewpoint": "Recreation & Sport",
    "water_feature": "Recreation & Sport",
    # health_care
    "doctor": "Healthcare",
    "emergency_or_urgent_care_facility": "Healthcare",
    "hospital": "Healthcare",
    "medical_center": "Healthcare",
    "medical_service": "Healthcare",
    "outpatient_care_facility": "Healthcare",
    "specialized_medical_facility": "Healthcare",
    # lifestyle_services
    "animal_or_pet_service": "Pet Services",
    "food_service": "Food & Drink",
    "personal_care_service": "Beauty & Personal Care",
    "personal_or_beauty_service": "Beauty & Personal Care",
    "wellness_service": "Beauty & Personal Care",
    # lodging
    "bed_and_breakfast": "Accommodation & Travel",
    "cabin": "Accommodation & Travel",
    "campground": "Accommodation & Travel",
    "cottage": "Accommodation & Travel",
    "country_house": "Accommodation & Travel",
    "holiday_park": "Accommodation & Travel",
    "hostel": "Accommodation & Travel",
    "hotel": "Accommodation & Travel",
    "houseboat": "Accommodation & Travel",
    "inn": "Accommodation & Travel",
    "lodge": "Accommodation & Travel",
    "mountain_hut": "Accommodation & Travel",
    "private_lodging": "Accommodation & Travel",
    "resort": "Accommodation & Travel",
    "retreat": "Accommodation & Travel",
    "rv_park": "Accommodation & Travel",
    "ryokan": "Accommodation & Travel",
    "self_catering_accommodation": "Accommodation & Travel",
    "service_apartment": "Accommodation & Travel",
    # services_and_business
    "agricultural_service": "Agriculture & Industry",
    "b2b_service": "Business & Professional Services",
    "building_or_construction_service": "Home & Garden Services",
    "business_to_business": "Business & Professional Services",
    "corporate_or_business_office": "Business & Professional Services",
    "design_service": "Business & Professional Services",
    "environmental_or_ecological_service": "Agriculture & Industry",
    "event_or_party_service": "Business & Professional Services",
    "family_service": "Community & Religion",
    "financial_service": "Business & Professional Services",
    "home_service": "Home & Garden Services",
    "housing_or_property_service": "Business & Professional Services",
    "industrial_facility_or_service": "Agriculture & Industry",
    "laundry_service": "Home & Garden Services",
    "legal_service": "Business & Professional Services",
    "media_service": "Business & Professional Services",
    "printing_service": "Business & Professional Services",
    "private_establishments_and_corporates": "Business & Professional Services",
    "professional_service": "Business & Professional Services",
    "public_utility_provider": "Utilities & Infrastructure",
    "real_estate": "Business & Professional Services",
    "real_estate_service": "Business & Professional Services",
    "rental_service": "Business & Professional Services",
    "security_service": "Business & Professional Services",
    "shipping_or_delivery_service": "Automotive & Transport",
    "storage_facility": "Business & Professional Services",
    "technical_service": "Business & Professional Services",
    "telecommunications_service": "Utilities & Infrastructure",
    "utility_energy_infrastructure": "Utilities & Infrastructure",
    # shopping
    "convenience_store": "Daily Needs & Grocery",
    "department_store": "Retail & Shopping",
    "discount_store": "Retail & Shopping",
    "fashion_and_apparel_store": "Retail & Shopping",
    "food_and_beverage_store": "Daily Needs & Grocery",
    "kiosk": "Retail & Shopping",
    "market": "Daily Needs & Grocery",
    "online_shop": "Retail & Shopping",
    "second_hand_store": "Retail & Shopping",
    "shopping_mall": "Retail & Shopping",
    "shopping_service": "Retail & Shopping",
    "specialty_store": "Retail & Shopping",
    "superstore": "Daily Needs & Grocery",
    "vehicle_dealer": "Automotive & Transport",
    "warehouse_club_store": "Daily Needs & Grocery",
    # sports_and_recreation
    "park": "Recreation & Sport",
    "recreational_equipment_rental": "Recreation & Sport",
    "recreational_trail_or_path": "Recreation & Sport",
    "sport_league": "Recreation & Sport",
    "sport_or_fitness_facility": "Recreation & Sport",
    "sport_or_recreation_club": "Recreation & Sport",
    "sport_team": "Recreation & Sport",
    "sports_and_fitness_instruction": "Recreation & Sport",
    # travel_and_transportation
    "air_transport_facility_or_service": "Automotive & Transport",
    "aircraft_and_air_transport": "Automotive & Transport",
    "fueling_station": "Automotive & Transport",
    "ground_transport_facility_or_service": "Automotive & Transport",
    "parking": "Automotive & Transport",
    "transport_interchange": "Automotive & Transport",
    "travel_service": "Accommodation & Travel",
    "vehicle_service": "Automotive & Transport",
    "water_transport_facility_or_service": "Automotive & Transport",
}

# Fallback for a category that only matches at the top level (no L1 branch).
L0_TO_FUNCTIONAL_GROUP = {
    "arts_and_entertainment": "Culture & Arts",
    "community_and_government": "Civic & Government",
    "cultural_and_historic": "Culture & Arts",
    "education": "Education",
    "food_and_drink": "Food & Drink",
    "geographic_entities": "Civic & Government",
    "health_care": "Healthcare",
    "lifestyle_services": "Beauty & Personal Care",
    "lodging": "Accommodation & Travel",
    "services_and_business": "Business & Professional Services",
    "shopping": "Retail & Shopping",
    "sports_and_recreation": "Recreation & Sport",
    "travel_and_transportation": "Automotive & Transport",
}

GEHL_ACTIVITIES = ("Necessary", "Optional", "Social")

# "Third place" (Oldenburg) first-pass rule: every functional group is a candidate third
# place except these task-driven / institutional ones.
THIRDSPACE_EXCLUDED_GROUPS = frozenset({"Education", "Healthcare", "Civic & Government"})

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


def classification_csv_to_dict(csv_path: Path, key: str = "overture_category") -> dict:
    """Turn the hand-edited lookup CSV into the dict written as the classification JSON.

    Keeps only what ``load_classification_lookup`` / ``classify_places`` read. The three
    Gehl scores come from their own columns (the ones edited by hand); ``gehl_scores`` is
    rebuilt from them and only fills in an empty score column. ``gehl_activities`` is
    pipe-separated in the CSV (``Optional|Social``). ``thirdspace_score`` is read from its
    column too (it is reviewed by hand); only an empty cell falls back to the first-pass
    rule (``THIRDSPACE_EXCLUDED_GROUPS``).
    """
    lookup = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if key not in (reader.fieldnames or []):
            raise ValueError(f"Key column '{key}' not found in {csv_path}. Columns: {reader.fieldnames}")

        for row in reader:
            entry = {
                field: row[field]
                for field in ("functional_group", "functional_group_source", "gehl_activity")
                if field in row
            }
            entry["gehl_activities"] = [a.strip() for a in (row.get("gehl_activities") or "").split("|") if a.strip()]

            fallback = json.loads(row["gehl_scores"]) if (row.get("gehl_scores") or "").strip() else {}
            scores = {}
            for activity in GEHL_ACTIVITIES:
                value = (row.get(f"{activity.lower()}_score") or "").strip()
                scores[activity] = float(value) if value else fallback.get(activity)
            entry["gehl_scores"] = scores
            entry["necessary_score"] = scores["Necessary"]
            entry["social_score"] = scores["Social"]
            entry["optional_score"] = scores["Optional"]
            thirdspace = (row.get("thirdspace_score") or "").strip()
            if thirdspace:
                entry["thirdspace_score"] = float(thirdspace)
            else:
                entry["thirdspace_score"] = 0.0 if entry["functional_group"] in THIRDSPACE_EXCLUDED_GROUPS else 1.0

            lookup[row[key]] = entry
    return lookup


def validate_classification_csv(csv_path: Path, key: str = "overture_category") -> list[str]:
    """Problems a hand edit can introduce, one message per problem (empty list = OK):
    duplicate categories, unknown functional groups, empty or misspelled Gehl labels,
    ``gehl_activity`` missing from ``gehl_activities``, ``thirdspace_score`` not 0/1, and
    Gehl scores outside 0-1."""
    groups = set(FUNCTIONAL_GROUPS)
    problems = []
    seen = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            cat = row.get(key, "")
            where = f"line {line} ({cat})"
            if cat in seen:
                problems.append(f"{where}: duplicate category")
            seen.add(cat)
            if row.get("functional_group") not in groups:
                problems.append(f"{where}: unknown functional_group {row.get('functional_group')!r}")
            activities = [a.strip() for a in (row.get("gehl_activities") or "").split("|") if a.strip()]
            if not activities:
                problems.append(f"{where}: gehl_activities is empty")
            elif any(a not in GEHL_ACTIVITIES for a in activities):
                problems.append(f"{where}: gehl_activities {row.get('gehl_activities')!r} "
                                f"(use {'|'.join(GEHL_ACTIVITIES)}, pipe-separated)")
            if activities and row.get("gehl_activity") not in activities:
                problems.append(f"{where}: gehl_activity {row.get('gehl_activity')!r} is not in gehl_activities")
            if (row.get("thirdspace_score") or "").strip() not in ("", "0", "1", "0.0", "1.0"):
                problems.append(f"{where}: thirdspace_score {row.get('thirdspace_score')!r} (use 0 or 1)")
            for col in ("necessary_score", "social_score", "optional_score"):
                value = (row.get(col) or "").strip()
                try:
                    if value and not 0 <= float(value) <= 1:
                        problems.append(f"{where}: {col} {value} outside 0-1")
                except ValueError:
                    problems.append(f"{where}: {col} {value!r} is not a number")
    return problems


def export_classification_json(csv_path: Path, json_path: Path, backup: bool = True) -> tuple[Path, Path | None]:
    """Rebuild the classification JSON from the hand-edited CSV.

    The previous JSON is copied to ``{stem}_backup_{timestamp}.json`` first unless
    ``backup=False``. Returns ``(json_path, backup_path_or_None)``.
    """
    json_path = Path(json_path)
    lookup = classification_csv_to_dict(csv_path)

    backup_path = None
    if backup and json_path.exists():
        backup_path = json_path.with_name(f"{json_path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{json_path.suffix}")
        shutil.copy2(json_path, backup_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    return json_path, backup_path


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


def functional_group_from_hierarchy(hierarchy) -> str | None:
    """Functional group from an Overture ``taxonomy.hierarchy`` path ('group > branch >
    leaf', or the list form): the L1 branch if it is mapped, else the L0 group."""
    if isinstance(hierarchy, (list, tuple)):
        hierarchy = " > ".join(hierarchy)
    if not isinstance(hierarchy, str) or not hierarchy.strip():
        return None
    l0 = hierarchy.split(">")[0].strip()
    l1 = _l1_branch(hierarchy)
    return L1_TO_FUNCTIONAL_GROUP.get(l1) or L0_TO_FUNCTIONAL_GROUP.get(l0)


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
    "functional_group", "gehl_activity", "gehl_activities",
    "necessary_score", "social_score", "optional_score", "thirdspace_score",
]


def classify_places(places: gpd.GeoDataFrame, lookup: pd.DataFrame):
    """Join places to the classification lookup on ``primary_category``.

    Categories present in this city's data but missing from the reference
    table (new/rare categories) take their functional group from the place's own
    Overture ``hierarchy`` (L1 branch, see ``L1_TO_FUNCTIONAL_GROUP``) when the places
    carry one, else ``"Other / Unclassified"``. Their Gehl activity stays
    ``"Unclassified"``: that label comes only from the reviewed lookup.
    Returns ``(pois, missing_categories)`` — the second is a value-count
    ``Series`` of categories not in the lookup, for a coverage check.
    """
    lookup_indexed = lookup.set_index("overture_category")

    pois = places.dropna(subset=["primary_category"]).reset_index(drop=True).copy()
    join_cols = [c for c in _LOOKUP_JOIN_COLS if c in lookup_indexed.columns]
    pois = pois.join(lookup_indexed[join_cols], on="primary_category")

    not_in_lookup = pois["functional_group"].isna()
    missing = pois.loc[not_in_lookup, "primary_category"].value_counts()

    if "hierarchy" in pois.columns:
        pois.loc[not_in_lookup, "functional_group"] = pois.loc[not_in_lookup, "hierarchy"].apply(
            functional_group_from_hierarchy
        )
    pois["functional_group"] = pois["functional_group"].fillna("Other / Unclassified")
    pois["gehl_activity"] = pois["gehl_activity"].fillna("Unclassified")
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
    can appear in more than one), plus a candidate-third-place overlay, off by
    default.
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

    thirdspace_pois = pois[pois["is_thirdspace"]]
    thirdspace_group = folium.FeatureGroup(name=f"Third places ({len(thirdspace_pois):,})", show=False)
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
            background:#6a4c93;border:1px solid #000;border-radius:50%;margin-right:6px;"></span>
            Third place</div>
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
    # Replace the file, don't add to it: a stale layer left from an earlier run (e.g.
    # under an old slug) is what gpd.read_file(path) without layer= would return.
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    pois.to_crs(epsg=local_epsg).to_file(out_path, driver="GPKG")
    return out_path


def export_classified_pois_gpkg_for_city(
    pois: gpd.GeoDataFrame, city: CityConfig, local_epsg: int | None = None
) -> Path:
    """Reprojects to ``local_epsg`` (default ``city.local_epsg``, the UTM zone
    ``CityConfig`` derives from the site's own coordinates) and exports.
    """
    epsg = local_epsg if local_epsg is not None else city.local_epsg
    path = city.data_dir / "overture" / f"pois_classified_{city.slug}.gpkg"
    return export_classified_pois_gpkg(pois, path, epsg)
