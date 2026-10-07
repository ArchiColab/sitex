import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sitex.data import provenance as prov


def _raster(path, res=30.0, epsg=32649):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=from_origin(180000, 1550000, res, res),
    ) as dst:
        dst.write(np.ones((4, 4), dtype="float32"), 1)


def _site(tmp_path):
    _raster(tmp_path / "dem" / "dtm_gedtm30.tif")
    scene = tmp_path / "landsat" / "output" / "LC09_X_20260328"
    _raster(scene / "LC09_X_20260328_B10_clipped.tif")
    (scene / "scene_info.json").write_text(
        json.dumps({"landsat_id": "LC09_X_20260328", "acquisition_date": "2026-03-28T03:07:10Z"}), encoding="utf8")
    places = tmp_path / "overture"
    places.mkdir()
    (places / "demo_places_2026-09-23.1.geojson").write_text("{}", encoding="utf8")
    return tmp_path


def test_columns_and_crs_rows(tmp_path):
    df = prov.build_provenance(_site(tmp_path), "demo", 32649, lon=108.0, lat=13.98, hex_resolution=9)
    assert list(df.columns) == prov.COLUMNS
    crs = df[df["kind"] == "crs"]
    assert list(crs["layer"]) == ["Working CRS", "Hex grid CRS", "VN-2000 equivalent"]
    assert "EPSG:32649" in crs.iloc[0]["dataset"] and "metres" in crs.iloc[0]["notes"]
    assert "EPSG:9217" in crs.iloc[2]["dataset"]          # Pleiku: VN-2000 / TM-3 108-15 (2025 scheme)
    assert "No file of the analysis is stored in VN-2000" in crs.iloc[0]["notes"]


def test_layers_read_from_files(tmp_path):
    df = prov.build_provenance(_site(tmp_path), "demo", 32649).set_index("layer")
    dem = df.loc["Terrain model (DTM)"]
    assert dem["file_crs"] == "EPSG:32649" and dem["file_resolution"] == "30 m" and dem["licence"].startswith("CC BY 4.0")
    assert dem["data_date"] == prov.TO_FILL
    land = df.loc["Landsat 8/9 surface temperature and bands 4, 5"]
    assert land["data_date"] == "2026-03-28" and "LC09_X_20260328" in land["dataset"]
    assert "last save" in land["retrieved"]
    assert df.loc["Places"]["data_date"] == "release 2026-09-23.1"


def test_missing_layers_are_left_out_and_unknown_stays_to_fill(tmp_path):
    df = prov.build_provenance(_site(tmp_path), "demo", 32649)
    layers = set(df["layer"])
    assert "Land cover" not in layers and "Streets (walking and driving networks)" not in layers
    assert "Hex grid" not in layers                        # no hex_grid_demo.gpkg in the folder
    # no VN-2000 row without a position, and no crash
    assert "VN-2000 equivalent" not in set(df["layer"])


def test_empty_folder_gives_only_the_crs_row(tmp_path):
    df = prov.build_provenance(tmp_path, "demo", 32649)
    assert list(df["layer"]) == ["Working CRS"]


def test_raster_in_degrees_shows_metres_and_the_working_crs_note_names_it(tmp_path):
    _raster(tmp_path / "landcover" / "esa_worldcover_2021_demo.tif", res=0.0001, epsg=4326)
    df = prov.build_provenance(tmp_path, "demo", 32649).set_index("layer")
    land = df.loc["Land cover"]
    assert land["file_crs"] == "EPSG:4326" and land["file_resolution"] == "0.0001 degrees (about 11 m)"
    assert "Land cover" in df.loc["Working CRS"]["notes"] and "the files on disk are not changed" in df.loc["Working CRS"]["notes"]


def test_vector_file_crs_is_read(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    (tmp_path / "osm").mkdir()
    gpd.GeoDataFrame({"a": [1]}, geometry=[Point(181000, 1548000)], crs="EPSG:32649").to_file(tmp_path / "osm" / "demo_walk.gpkg")
    df = prov.build_provenance(tmp_path, "demo", 32649).set_index("layer")
    streets = df.loc["Streets (walking and driving networks)"]
    assert streets["file_crs"] == "EPSG:32649" and streets["licence"] == "ODbL" and streets["data_date"] == prov.TO_FILL
