# SiteX

![SiteX banner](images/banner.png)

Open-data site analysis toolkit for urban design students working in **data-scarce
contexts**, for e.g. secondary cities in a developing country with no municipal GIS portal, no cadastral layer, no
official DTM. Extracted from a set of case-study notebooks built for Pleiku, Gia Lai,
Vietnam, and written to generalize to any geocodable place (Vietnam alone spans UTM
zones 48N and 49N, split at 108°E — a built-in stress test for anything hardcoding a
CRS).

The teaching notebooks that use this library are in a separate repository,
[ArchiColab/sitex-course](https://github.com/ArchiColab/sitex-course).

## Context of this work

SiteX was developed by Chau Nguyen as part of a master's thesis at Metropolia University
of Applied Sciences, in the Computing in Construction programme. It is also the
experimental case of the thesis: an **agentic system for spatial analysis**.

### Development approach: an agentic system with a human in the loop

In this thesis, an *agentic system* is a software system in which a large language model
is given a goal and a set of tools (here: reading and editing files, running Python and
GIS code) and decides its own next steps in a loop of acting, observing the result and
adjusting, instead of answering one prompt at a time.

The agent used here was Claude Code (Anthropic). It worked with a human in the loop: the
author defined the research questions, the workshop design, the analytical methods and
the data sources, reviewed and ran the agent's output, corrected it, and decided what
entered the library. Claude was a tool in this process and is not an author of the work.
Responsibility for the code, methods and results rests with the author. The method, its
limits and its evaluation are reported in the thesis.

## Install

SiteX needs Python 3.10 or newer and a geospatial stack (GDAL, PROJ) that is hard to
build with `pip` alone, especially on Windows. The recommended route is a conda
environment from [conda-forge](https://conda-forge.org/), using
[Miniforge](https://github.com/conda-forge/miniforge) or Anaconda.

```bash
git clone https://github.com/ArchiColab/sitex.git
cd sitex
conda env create -f environment.yml
conda activate sitex
```

This installs the compiled geospatial packages with conda and then SiteX with all its
extras with pip. Check that it works:

```bash
python -c "import sitex; print('ok')"
```

**Already have a working geo environment?** Install SiteX into it with pip. Each
analysis family has its own optional dependencies, so install only what you use:

```bash
pip install "sitex[network] @ git+https://github.com/ArchiColab/sitex.git"
```

Available extras: `data`, `arch`, `env`, `network`, `viz`, `dev`. The dependencies are
declared in `pyproject.toml`, so there is no separate `requirements.txt`.

## Architecture

SiteX is a set of small modules in four layers. One place description feeds all of them,
and the layers pass files (GeoPackage, GeoTIFF, GeoJSON) to each other rather than
objects, so a student can run any step on its own and open the result in QGIS or CAD.

```mermaid
flowchart TD
    CFG["core<br/>CityConfig · local UTM zone · CAD origin"]
    DATA["data<br/>AOI · OSM · Overture · DEM · Sentinel-2 · Landsat<br/>WorldCover · canopy height · building datasets"]
    ARCH["arch<br/>morphology · terrain · POI classification"]
    ENV["env<br/>UHI · change detection · flood risk · terrain"]
    NET["network<br/>space syntax · isochrones · POI and green accessibility"]
    VIZ["viz<br/>webmaps · 3D city · heatmaps · overlays"]
    OUT["Outputs<br/>GeoPackage · GeoTIFF · DXF · QML · HTML maps"]

    CFG --> DATA
    DATA --> ARCH
    DATA --> ENV
    DATA --> NET
    ARCH --> NET
    ARCH --> VIZ
    ENV --> VIZ
    NET --> VIZ
    ARCH --> OUT
    ENV --> OUT
    NET --> OUT
    VIZ --> OUT
```

| Package | Role | Main modules |
|---|---|---|
| `sitex.core` | Shared foundation | `config` (`CityConfig`), `geo` (UTM zone, CAD origin), `raster_viz`, `svg_charts`, `pipeline_diagram` |
| `sitex.data` | Acquisition of open data for one area of interest | `aoi`, `street_network`, `overture`, `remote_sensing` (DEM, Sentinel-2, Landsat), `landcover`, `canopy_height`, `gba`, `open_buildings` |
| `sitex.arch` | Buildings, ground and programme | `morphology`, `terrain` (contours, DXF), `poi_classification`, `poi_amenities` |
| `sitex.env` | Environmental analysis | `uhi`, `change_detection`, `flood_risk`, `terrain_env` |
| `sitex.network` | Street-network analysis and accessibility | `spacesyntax`, `isochrones`, `poi_accessibility`, `green_accessibility`, `street_edits`, `accessibility_common` |
| `sitex.viz` | Maps and scenes for discussion | `interactive_webmap`, `city3d`, `heatmaps`, `building_overlays`, `city_morphology_maps`, `raster_scenes`, `workshop_diagram` |

Design choices:

- **`CityConfig` is the single source of truth for a site.** It derives the local UTM
  zone, the CAD/UCS origin and a filename slug from a place name and a point, so no
  module hardcodes a CRS. Most functions have a `..._for_city()` wrapper that takes it.
- **No accounts where avoidable.** Global datasets are streamed by bounding box (DuckDB
  range requests on GeoParquet, GDAL `/vsicurl/` on cloud GeoTIFFs), so nothing large is
  downloaded first. Sentinel-2 through the Copernicus Data Space Ecosystem needs a free
  login.
- **Analysis and acquisition are separate.** `sitex.data` only downloads and saves;
  the other packages read the saved files.
- **Outputs are for design tools.** Results are written as files that open in QGIS
  and CAD (GeoPackage, GeoTIFF, DXF, QGIS style files), next to interactive HTML maps.

## Status

Extraction from the case-study notebooks is complete: all six packages above are in
place and covered by a test suite (`pytest`).

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

## Data sources and licences

The MIT licence of this repository covers the **code only**. SiteX does not ship the
datasets; it downloads them at run time from their providers, and each dataset keeps its
own licence. Check the terms before you publish or reuse a result, and give the
attribution the provider asks for.

| Data | Module | Licence | Attribution / note |
|---|---|---|---|
| OpenStreetMap (streets, boundaries) | `data.aoi`, `data.street_network` | ODbL | © OpenStreetMap contributors |
| Overture Maps: buildings, base, transportation | `data.overture` | ODbL; buildings also include CC BY 4.0 sources | © OpenStreetMap contributors; Overture Maps Foundation |
| Overture Maps: places | `data.overture` | CDLA Permissive 2.0 (Foursquare part: Apache 2.0) | Overture Maps Foundation |
| Global Building Atlas | `data.gba` | Split: ODbL polygons; **CC BY-NC 4.0** for LoD1 models, height maps and some polygons | Zhu et al. (2025), *Earth System Science Data* 17(12), 6647–6668. Non-commercial terms apply |
| Open Buildings 2.5D Temporal (Google) | `data.open_buildings` | CC BY 4.0 or ODbL (your choice) | Sirko et al. (2023), arXiv:2310.11622 |
| High resolution canopy height (Meta, WRI) | `data.canopy_height` | CC BY 4.0 (as listed on the AWS registry for the 2024 release; confirm for the version you use) | Meta and WRI; source imagery © Maxar |
| ESA WorldCover 10 m 2021 | `data.landcover` | CC BY 4.0 | © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium |
| Sentinel-2 (Copernicus Data Space) | `data.remote_sensing` | Copernicus open data terms | Contains modified Copernicus Sentinel data |
| Landsat 8/9 Collection 2 Level-2 (Planetary Computer) | `data.remote_sensing` | USGS open data policy | Credit USGS/NASA Landsat |
| NASADEM (DTM) and AW3D30 (DSM) | `data.remote_sensing` | NASA open data; JAXA terms of use for AW3D30 | Read the JAXA terms before redistributing tiles |

Licence details for these datasets were read from the providers' own pages, and
providers change them. The provider's page is the reference, not this table. If you
combine layers with different licences (for example ODbL and CC BY-NC), the combination
can carry the stricter terms, and that is your responsibility to check.

Space syntax functions in `sitex.network.spacesyntax` are adapted from Nabil Mohareb
(2024), The American University in Cairo (space-syntaxNM).

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## Licence

Code: MIT, see [LICENSE](LICENSE).

## Citation

If you use SiteX, please cite the thesis it belongs to:

> Nguyen, C. (2026). *[Thesis title]*. Master's thesis, Metropolia University of Applied
> Sciences, Computing in Construction.
