# SiteX

Open-data site analysis toolkit for urban design students working in **data-scarce
contexts**, for e.g. secondary cities in a developing country with no municipal GIS portal, no cadastral layer, no
official DTM. Extracted from a set of case-study notebooks built for Pleiku, Gia Lai,
Vietnam, and written to generalize to any geocodable place (Vietnam alone spans UTM
zones 48N and 49N, split at 108°E — a built-in stress test for anything hardcoding a
CRS).

## Install

```bash
pip install git+https://github.com/ArchiColab/sitex.git
```

## Status

Early extraction in progress. Current modules:

- `sitex.core.geo` — `utm_epsg_from_lonlat()`, `compute_cad_origin()`
- `sitex.core.config` — `CityConfig`, `slugify()`

Planned (see the source project's thesis notes): `acquisition`, `arch`, `env`,
`network`, `viz` — one submodule per original notebook family.

## Quickstart

```python
from sitex.core.config import CityConfig

pleiku = CityConfig(
    place_name="Phường Pleiku, Gia Lai, Vietnam",
    local_lat=13.98,
    local_lon=108.00,
)

pleiku.slug          # "phuong_pleiku"
pleiku.local_epsg    # 32649
pleiku.origin        # (easting, northing) CAD/UCS base point, floored to 1000 m
```

## Develop

```bash
pip install -e ".[dev]"
pytest
```
