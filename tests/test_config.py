from sitex.core.config import CityConfig, slugify


def test_slugify_strips_diacritics():
    assert slugify("Phường Pleiku") == "phuong_pleiku"


def test_cityconfig_default_slug_uses_locality_only():
    cfg = CityConfig(
        place_name="Phường Pleiku, Gia Lai, Vietnam",
        local_lat=13.98,
        local_lon=108.00,
    )
    assert cfg.slug == "phuong_pleiku"


def test_cityconfig_derives_epsg_and_origin():
    cfg = CityConfig(
        place_name="Phường Pleiku, Gia Lai, Vietnam",
        local_lat=13.98,
        local_lon=108.00,
    )
    assert cfg.local_epsg == 32649
    easting, northing = cfg.origin
    assert easting % 1000 == 0
    assert northing % 1000 == 0


def test_cityconfig_explicit_slug_and_epsg_are_respected():
    cfg = CityConfig(
        place_name="Da Nang, Vietnam",
        local_lat=16.05,
        local_lon=108.20,
        slug="danang",
        local_epsg=32648,
    )
    assert cfg.slug == "danang"
    assert cfg.local_epsg == 32648
