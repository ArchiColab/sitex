import pytest

from sitex.data import aoi as aoi_lib


def test_slugify_strips_punctuation_and_lowercases():
    assert aoi_lib._slugify("Chau Doc, An Giang, Vietnam") == "chau_doc"
    assert aoi_lib._slugify("Phường Pleiku") == "phường_pleiku"


def test_parse_osm_id_accepts_ids_and_links():
    assert aoi_lib.parse_osm_id("R7147125") == "R7147125"
    assert aoi_lib.parse_osm_id(" r7147125 ") == "R7147125"
    assert aoi_lib.parse_osm_id(
        "https://www.openstreetmap.org/relation/7147125#map=12/10.3714/105.4283"
    ) == "R7147125"
    assert aoi_lib.parse_osm_id("https://www.openstreetmap.org/way/42") == "W42"
    assert aoi_lib.parse_osm_id("13566853") == "R13566853"
    assert aoi_lib.parse_osm_id(
        "https://www.openstreetmap.org/relation/13566853#map=13/10.36641/105.41856&layers=S"
    ) == "R13566853"
    assert aoi_lib.parse_osm_id("Phường Pleiku, Gia Lai, Vietnam") is None


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


def _fake_colab(monkeypatch):
    import sys
    import types

    registered = {}
    output = types.SimpleNamespace(register_callback=lambda name, fn: registered.update({name: fn}))
    fake_colab = types.ModuleType("google.colab")
    fake_colab.output = output
    fake_google = types.ModuleType("google")
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    return registered


def test_build_aoi_map_in_colab_reports_clicks_through_the_callback(monkeypatch):
    from sitex.data import aoi

    registered = _fake_colab(monkeypatch)
    site_map, default_bbox = aoi.build_aoi_map(13.98, 108.0, dist_m=1000, zoom=16)

    (callback,) = registered
    assert callback in site_map._repr_html_()
    assert "__ID__" not in site_map._repr_html_() and "__CONFIG__" not in site_map._repr_html_()

    # nothing drawn: the default box is kept
    assert aoi.resolve_aoi_from_map(site_map, default_bbox, slug="t").bbox == tuple(default_bbox)

    # two clicks in the browser reach Python as (west, south, east, north)
    registered[callback](108.01, 13.98, 108.02, 13.99)
    assert aoi.resolve_aoi_from_map(site_map, default_bbox, slug="t").bbox == (108.01, 13.98, 108.02, 13.99)

    # a new first click clears the drawn box again
    registered[callback]()
    assert aoi.resolve_aoi_from_map(site_map, default_bbox, slug="t").bbox == tuple(default_bbox)
