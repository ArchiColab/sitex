import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sitex.arch.poi_classification import (
    add_gehl_activity_flags,
    classification_csv_to_dict,
    classify_places,
    export_classification_json,
    validate_classification_csv,
    export_classified_pois_geojson,
    export_classified_pois_gpkg,
    load_classification_lookup,
    functional_group_from_hierarchy,
    load_places_for_aoi,
    plot_pois_gehl_map,
    select_final_columns,
)
from sitex.core.config import CityConfig

CLASSIFICATION = {
    "restaurant": {
        "functional_group": "Food & Drink",
        "gehl_activity": "Necessary",
        "gehl_activities": ["Necessary"],
        "necessary_score": 0.8, "social_score": 0.1, "optional_score": 0.1,
        "thirdspace_score": 0.0,
    },
    "hospital": {
        "functional_group": "Healthcare",
        "gehl_activity": "Necessary",
        "gehl_activities": ["Necessary"],
        "necessary_score": 0.9, "social_score": 0.0, "optional_score": 0.0,
        "thirdspace_score": 0.0,
    },
    "cafe": {
        "functional_group": "Food & Drink",
        "gehl_activity": "Social",
        "gehl_activities": ["Optional", "Social"],
        "necessary_score": 0.1, "social_score": 0.6, "optional_score": 0.3,
        "thirdspace_score": 1.0,
    },
    "park": {
        "functional_group": "Recreation & Sport",
        "gehl_activity": "Optional",
        "gehl_activities": ["Optional", "Social"],
        "necessary_score": 0.0, "social_score": 0.3, "optional_score": 0.7,
        "thirdspace_score": 1.0,
    },
    "gym": {
        "functional_group": "Recreation & Sport",
        "gehl_activity": "Optional",
        "gehl_activities": ["Optional"],
        "necessary_score": 0.0, "social_score": 0.1, "optional_score": 0.9,
        "thirdspace_score": 0.0,
    },
}

@pytest.fixture
def classification_json(tmp_path):
    path = tmp_path / "overture_place_classification.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CLASSIFICATION, f)
    return path


@pytest.fixture
def places_gpkg(tmp_path):
    """One AOI's places as ``extract_by_aoi`` writes them (2026-09-23 schema: ``category``
    is ``taxonomy.primary``, plus the ``hierarchy`` path)."""
    gdf = gpd.GeoDataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "name": ["Joe's Diner", "City Hospital", "Corner Cafe", "Central Park", "Gold Gym", "Mystery Kiosk"],
            "category": ["restaurant", "hospital", "cafe", "park", "gym", "noodle_stall"],
            "hierarchy": [
                "food_and_drink > restaurant",
                "health_care > hospital",
                "food_and_drink > non_alcoholic_beverage_venue > cafe",
                "sports_and_recreation > park",
                "sports_and_recreation > sport_or_fitness_facility > gym",
                "food_and_drink > casual_eatery > noodle_stall",
            ],
            "confidence": [0.9, 0.95, 0.7, 0.8, 0.6, 0.4],
        },
        geometry=[
            Point(108.00, 13.98), Point(108.001, 13.981), Point(108.002, 13.982),
            Point(108.003, 13.983), Point(108.004, 13.984), Point(108.005, 13.985),
        ],
        crs="EPSG:4326",
    )
    overture_dir = tmp_path / "overture"
    overture_dir.mkdir(parents=True, exist_ok=True)
    path = overture_dir / "pleiku_places.gpkg"
    gdf.to_file(path, layer="POIS", driver="GPKG")
    return path


@pytest.fixture
def city(tmp_path):
    return CityConfig(
        place_name="Pleiku, Gia Lai, Vietnam",
        local_lat=13.98,
        local_lon=108.00,
        slug="pleiku",
        data_dir=tmp_path,
        output_dir=tmp_path / "outputs",
    )


def test_load_classification_lookup(classification_json):
    lookup = load_classification_lookup(classification_json)
    assert len(lookup) == 5
    assert "overture_category" in lookup.columns
    assert set(lookup["overture_category"]) == set(CLASSIFICATION.keys())


def test_load_classification_lookup_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classification_lookup(tmp_path / "does_not_exist.json")


def test_load_places_for_aoi_normalizes_columns(places_gpkg, city):
    places = load_places_for_aoi(city)
    assert len(places) == 6
    assert {"primary_category", "names", "hierarchy"} <= set(places.columns)


def test_functional_group_from_hierarchy():
    assert functional_group_from_hierarchy("food_and_drink > casual_eatery > noodle_stall") == "Food & Drink"
    assert functional_group_from_hierarchy(["health_care", "hospital"]) == "Healthcare"
    assert functional_group_from_hierarchy("lodging") == "Accommodation & Travel"  # L0 only
    assert functional_group_from_hierarchy(None) is None


def test_classify_places_joins_and_flags_missing(places_gpkg, classification_json, city):
    places = load_places_for_aoi(city)
    lookup = load_classification_lookup(classification_json)

    # A category missing from the lookup and without a hierarchy -> Unclassified.
    places.loc[0, "primary_category"] = "food_truck"
    places.loc[0, "hierarchy"] = None

    pois, missing = classify_places(places, lookup)
    assert "food_truck" in missing.index
    assert pois.loc[pois["overture_category"] == "food_truck", "functional_group"].iloc[0] == "Other / Unclassified"
    assert pois.loc[pois["overture_category"] == "food_truck", "gehl_activities"].iloc[0] == ["Unclassified"]
    # Missing from the lookup but with a hierarchy -> functional group from its L1 branch;
    # Gehl activity still Unclassified (only the reviewed lookup sets it).
    stall = pois[pois["overture_category"] == "noodle_stall"].iloc[0]
    assert stall["functional_group"] == "Food & Drink"
    assert stall["gehl_activities"] == ["Unclassified"]
    assert "noodle_stall" in missing.index
    # A genuinely-matched category keeps its real functional group.
    assert pois.loc[pois["overture_category"] == "hospital", "functional_group"].iloc[0] == "Healthcare"


def test_add_gehl_activity_flags_multi_label(places_gpkg, classification_json, city):
    places = load_places_for_aoi(city)
    lookup = load_classification_lookup(classification_json)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)

    cafe_row = pois[pois["overture_category"] == "cafe"].iloc[0]
    assert cafe_row["is_optional"] == True
    assert cafe_row["is_social"] == True
    assert cafe_row["is_necessary"] == False
    assert cafe_row["gehl_activities"] == "Optional|Social"
    assert cafe_row["is_thirdspace"] == True

    restaurant_row = pois[pois["overture_category"] == "restaurant"].iloc[0]
    assert restaurant_row["is_thirdspace"] == False


def test_select_final_columns_drops_extras(places_gpkg, classification_json, city):
    places = load_places_for_aoi(city)
    lookup = load_classification_lookup(classification_json)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)

    final = select_final_columns(pois)
    assert "confidence" not in final.columns  # present in the raw backup, not in FINAL_COLUMNS
    assert "is_necessary" in final.columns
    assert "geometry" in final.columns


def test_export_classified_pois_geojson_and_gpkg(tmp_path, places_gpkg, classification_json, city):
    places = load_places_for_aoi(city)
    lookup = load_classification_lookup(classification_json)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)
    final = select_final_columns(pois)

    geojson_path = tmp_path / "pois_classified_pleiku.geojson"
    export_classified_pois_geojson(final, geojson_path)
    assert geojson_path.exists()
    reloaded = gpd.read_file(geojson_path)
    assert reloaded.crs.to_epsg() == 4326

    gpkg_path = tmp_path / "pois_classified_pleiku.gpkg"
    export_classified_pois_gpkg(final, gpkg_path, local_epsg=32649)
    assert gpkg_path.exists()
    reloaded_gpkg = gpd.read_file(gpkg_path)
    assert reloaded_gpkg.crs.to_epsg() == 32649


def test_plot_pois_gehl_map_returns_folium_map(places_gpkg, classification_json, city):
    import folium

    places = load_places_for_aoi(city)
    lookup = load_classification_lookup(classification_json)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)
    final = select_final_columns(pois)

    m = plot_pois_gehl_map(final)
    assert isinstance(m, folium.Map)


CSV_ROWS = [
    # Scores edited in their columns; the gehl_scores cell still has the old values.
    {"overture_category": "cafe", "functional_group": "Food & Drink", "functional_group_source": "taxonomy",
     "gehl_activity": "Social", "gehl_activities": "Optional|Social",
     "gehl_scores": '{"Necessary": 0.9, "Social": 0.0, "Optional": 0.0}',
     "necessary_score": "0.1", "social_score": "0.6", "optional_score": "0.3",
     "thirdspace_score": "0", "count": "12"},  # reviewed: not a third place
    # Empty score column -> falls back to the gehl_scores cell.
    {"overture_category": "hospital", "functional_group": "Healthcare", "functional_group_source": "taxonomy",
     "gehl_activity": "Necessary", "gehl_activities": "Necessary",
     "gehl_scores": '{"Necessary": 0.9, "Social": 0.05, "Optional": 0.0}',
     "necessary_score": "", "social_score": "0.05", "optional_score": "0",
     "thirdspace_score": "", "count": "3"},  # empty -> first-pass rule
]


@pytest.fixture
def classification_csv(tmp_path):
    path = tmp_path / "overture_place_classification.csv"
    pd.DataFrame(CSV_ROWS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_classification_csv_to_dict_reads_edited_score_columns(classification_csv):
    lookup = classification_csv_to_dict(classification_csv)
    cafe = lookup["cafe"]
    assert cafe["gehl_activities"] == ["Optional", "Social"]
    assert cafe["necessary_score"] == 0.1  # the edited column, not the stale gehl_scores cell
    assert cafe["gehl_scores"] == {"Necessary": 0.1, "Optional": 0.3, "Social": 0.6}
    assert cafe["thirdspace_score"] == 0.0  # the reviewed column, not the rule (Food & Drink -> 1.0)
    assert "count" not in cafe  # bookkeeping columns stay in the CSV

    hospital = lookup["hospital"]
    assert hospital["necessary_score"] == 0.9  # empty column -> gehl_scores fallback
    assert hospital["thirdspace_score"] == 0.0  # empty column -> rule: Healthcare is excluded


def test_export_classification_json_round_trip_and_backup(classification_csv, tmp_path):
    json_path = tmp_path / "overture_place_classification.json"
    json_path.write_text("{}", encoding="utf-8")

    written, backup = export_classification_json(classification_csv, json_path)
    assert written == json_path
    assert backup is not None and backup.exists() and backup.read_text(encoding="utf-8") == "{}"

    lookup = load_classification_lookup(json_path).set_index("overture_category")
    assert lookup.loc["cafe", "functional_group"] == "Food & Drink"
    assert lookup.loc["cafe", "social_score"] == 0.6


def test_validate_classification_csv(classification_csv, tmp_path):
    assert validate_classification_csv(classification_csv) == []

    rows = [dict(r) for r in CSV_ROWS]
    rows[0]["gehl_activities"] = ""                   # emptied by accident
    rows[1]["gehl_activities"] = "Necessary|Social "  # trailing space is fine
    rows[1]["functional_group"] = "Health care"       # typo
    rows[1]["thirdspace_score"] = "yes"
    path = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    problems = validate_classification_csv(path)
    assert any("cafe" in p and "gehl_activities is empty" in p for p in problems)
    assert any("hospital" in p and "unknown functional_group" in p for p in problems)
    assert any("hospital" in p and "thirdspace_score" in p for p in problems)
    assert len(problems) == 3
