import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sitex.arch.poi_classification import (
    add_gehl_activity_flags,
    add_public_amenity_flag,
    classify_places,
    export_classified_pois_geojson,
    export_classified_pois_gpkg,
    load_classification_lookup,
    load_places_for_city,
    load_taxonomy_l1_lookup,
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

TAXONOMY_ROWS = [
    {"New Primary Category": "park", "New Primary Hierarchy": "recreation > park > urban_park",
     "Old Primary Category": "", "Old Primary Hierarchy": ""},
    {"New Primary Category": "gym", "New Primary Hierarchy": "recreation > fitness_center > gym",
     "Old Primary Category": "", "Old Primary Hierarchy": ""},
]


@pytest.fixture
def classification_json(tmp_path):
    path = tmp_path / "overture_place_classification.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CLASSIFICATION, f)
    return path


@pytest.fixture
def taxonomy_csv(tmp_path):
    path = tmp_path / "overture_taxonomy.csv"
    pd.DataFrame(TAXONOMY_ROWS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def places_backup(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "names": ["Joe's Diner", "City Hospital", "Corner Cafe", "Central Park", "Gold Gym", "Mystery Kiosk"],
            "primary_category": ["restaurant", "hospital", "cafe", "park", "gym", "unknown_category"],
            "confidence": [0.9, 0.95, 0.7, 0.8, 0.6, 0.4],
            "city": ["pleiku", "pleiku", "pleiku", "pleiku", "pleiku", "hanoi"],
        },
        geometry=[
            Point(108.00, 13.98), Point(108.001, 13.981), Point(108.002, 13.982),
            Point(108.003, 13.983), Point(108.004, 13.984), Point(105.8, 21.0),
        ],
        crs="EPSG:4326",
    )
    overture_dir = tmp_path / "overture"
    overture_dir.mkdir(parents=True, exist_ok=True)
    path = overture_dir / "places_all_cities_backup.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def city(tmp_path):
    return CityConfig(
        place_name="Pleiku, Gia Lai, Vietnam",
        local_lat=13.98,
        local_lon=108.00,
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


def test_load_taxonomy_l1_lookup(taxonomy_csv):
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    assert category_to_l1["park"] == "park"
    assert category_to_l1["gym"] == "fitness_center"


def test_add_public_amenity_flag_healthcare_always_true(classification_json, taxonomy_csv):
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)

    row = lookup.set_index("overture_category")
    assert row.loc["hospital", "is_public_amenity"] is True or row.loc["hospital", "is_public_amenity"] == True
    assert row.loc["restaurant", "is_public_amenity"] == False


def test_add_public_amenity_flag_recreation_split_by_l1_branch(classification_json, taxonomy_csv):
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)

    row = lookup.set_index("overture_category")
    # Both "park" and "gym" are Recreation & Sport, but only "park" maps to a public L1 branch.
    assert row.loc["park", "is_public_amenity"] == True
    assert row.loc["gym", "is_public_amenity"] == False


def test_load_places_for_city_filters_by_city_key(places_backup, city):
    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    assert len(places) == 5  # excludes the hanoi row
    assert (places["city"] == "pleiku").all()


def test_load_places_for_city_unknown_key_raises(places_backup, city):
    with pytest.raises(ValueError, match="not in the backup"):
        load_places_for_city(city, city_key="da_nang", filename=places_backup.name)


def test_classify_places_joins_and_flags_missing(places_backup, classification_json, taxonomy_csv, city):
    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)

    # "unknown_category" isn't in pleiku's subset (it belongs to the hanoi row), so no
    # missing categories are expected here; add a genuinely-missing one to test the path.
    places.loc[0, "primary_category"] = "food_truck"

    pois, missing = classify_places(places, lookup)
    assert "food_truck" in missing.index
    assert pois.loc[pois["overture_category"] == "food_truck", "functional_group"].iloc[0] == "Other / Unclassified"
    assert pois.loc[pois["overture_category"] == "food_truck", "is_public_amenity"].iloc[0] == False
    assert pois.loc[pois["overture_category"] == "food_truck", "gehl_activities"].iloc[0] == ["Unclassified"]
    # A genuinely-matched category keeps its real functional group.
    assert pois.loc[pois["overture_category"] == "hospital", "functional_group"].iloc[0] == "Healthcare"


def test_add_gehl_activity_flags_multi_label(places_backup, classification_json, taxonomy_csv, city):
    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)
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


def test_select_final_columns_drops_extras(places_backup, classification_json, taxonomy_csv, city):
    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)

    final = select_final_columns(pois)
    assert "confidence" not in final.columns  # present in the raw backup, not in FINAL_COLUMNS
    assert "is_necessary" in final.columns
    assert "geometry" in final.columns


def test_export_classified_pois_geojson_and_gpkg(tmp_path, places_backup, classification_json, taxonomy_csv, city):
    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)
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


def test_plot_pois_gehl_map_returns_folium_map(places_backup, classification_json, taxonomy_csv, city):
    import folium

    places = load_places_for_city(city, city_key="pleiku", filename=places_backup.name)
    lookup = load_classification_lookup(classification_json)
    category_to_l1 = load_taxonomy_l1_lookup(taxonomy_csv)
    lookup = add_public_amenity_flag(lookup, category_to_l1)
    pois, _ = classify_places(places, lookup)
    pois = add_gehl_activity_flags(pois)
    final = select_final_columns(pois)

    m = plot_pois_gehl_map(final)
    assert isinstance(m, folium.Map)
