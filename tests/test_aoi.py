import pytest

from sitex.data import aoi as aoi_lib


def test_slugify_strips_punctuation_and_lowercases():
    assert aoi_lib._slugify("Chau Doc, An Giang, Vietnam") == "chau_doc"
    assert aoi_lib._slugify("Phường Pleiku") == "phường_pleiku"


def test_bbox_to_geojson_shape():
    geojson = aoi_lib._bbox_to_geojson((105.0, 10.6, 105.2, 10.8))
    coords = geojson["geometry"]["coordinates"][0]
    assert coords[0] == coords[-1]  # closed ring
    assert geojson["geometry"]["type"] == "Polygon"


def test_bbox_from_point_is_centred_and_roughly_sized():
    west, south, east, north = aoi_lib._bbox_from_point(105.0880, 10.6881, dist_m=1000)
    assert west < 105.0880 < east
    assert south < 10.6881 < north
    # ~1km buffer each way -> ~2km box, generous tolerance for lon/lat degree distortion
    assert 0.01 < (east - west) < 0.05
    assert 0.01 < (north - south) < 0.05


def test_build_aoi_map_returns_widget_and_default_bbox():
    ui, default_bbox = aoi_lib.build_aoi_map(10.6881, 105.0880, dist_m=1000, zoom=13)
    assert hasattr(ui, "_sitex_drawn_bbox")
    assert ui._sitex_drawn_bbox["value"] is None  # nothing drawn yet
    assert default_bbox[0] < 105.0880 < default_bbox[2]


def test_resolve_aoi_from_map_falls_back_to_default_bbox_when_undrawn():
    ui, default_bbox = aoi_lib.build_aoi_map(10.6881, 105.0880, dist_m=1000, zoom=13)
    aoi = aoi_lib.resolve_aoi_from_map(ui, default_bbox, slug="undrawn_test")
    assert tuple(aoi.bbox) == pytest.approx(tuple(default_bbox))
    assert aoi.slug == "undrawn_test"


def test_resolve_aoi_from_map_uses_the_drawn_rectangle():
    ui, default_bbox = aoi_lib.build_aoi_map(10.6881, 105.0880, dist_m=1000, zoom=13)
    handler = ui.children[1]._interaction_callbacks.callbacks[0]

    # Two clicks on opposite corners simulate a student drawing a rectangle.
    handler(type="click", coordinates=[10.68, 105.07])
    handler(type="click", coordinates=[10.70, 105.10])

    aoi = aoi_lib.resolve_aoi_from_map(ui, default_bbox, slug="drawn_test")
    assert aoi.bbox != pytest.approx(tuple(default_bbox))
    assert aoi.bbox == pytest.approx((105.07, 10.68, 105.10, 10.70))


def test_resolve_aoi_from_map_single_click_does_not_produce_a_box():
    ui, default_bbox = aoi_lib.build_aoi_map(10.6881, 105.0880, dist_m=1000, zoom=13)
    handler = ui.children[1]._interaction_callbacks.callbacks[0]

    handler(type="click", coordinates=[10.68, 105.07])  # only one corner clicked

    aoi = aoi_lib.resolve_aoi_from_map(ui, default_bbox, slug="single_click_test")
    assert tuple(aoi.bbox) == pytest.approx(tuple(default_bbox))  # falls back, not a degenerate box
