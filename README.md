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

Available extras: `data`, `arch`, `env`, `network`, `viz`, `hex` (DuckDB and Parquet, for the hexagon database), `dev`. The dependencies are
declared in `pyproject.toml`, so there is no separate `requirements.txt`.

**For the street network** (the Geofabrik route, see
[Data source decisions](#data-source-decisions)):
[`osmium-tool`](https://osmcode.org/osmium-tool/) is a command-line program, not a Python
package, so `pip` cannot install it and it is not in `pyproject.toml`. The conda
environment above includes it. In an existing environment, install it with
`conda install -c conda-forge osmium-tool`; in Google Colab, the notebook runs
`apt-get install osmium-tool` for you. Without it only the street download fails; the
other datasets are not affected.

## Architecture

SiteX is a set of small modules in four layers. One place description feeds all of them,
and the layers pass files (GeoPackage, GeoTIFF, GeoJSON) to each other rather than
objects, so a student can run any step on its own and open the result in QGIS or CAD.

```mermaid
flowchart TD
    CFG["core<br/>CityConfig · place slug · local CRS · CAD origin · hex grid"]
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
| `sitex.core` | Shared foundation | `config` (`CityConfig`, `slugify`), `geo` (local CRS, VN-2000, CAD origin), `hexgrid` (H3 grid and `*_to_hex` functions), `raster_viz`, `svg_charts`, `pipeline_diagram` |
| `sitex.data` | Acquisition of open data for one area of interest | `aoi`, `street_network`, `overture`, `remote_sensing` (DEM, Sentinel-2, Landsat), `landcover`, `canopy_height`, `gba`, `open_buildings`, `provenance` (record of data sources and CRS) |
| `sitex.arch` | Buildings, ground and programme | `morphology`, `terrain` (contours, DXF), `poi_classification`, `poi_amenities` |
| `sitex.env` | Environmental analysis | `uhi`, `change_detection`, `flood_risk`, `terrain_env` |
| `sitex.network` | Street-network analysis and accessibility | `spacesyntax`, `isochrones`, `poi_accessibility`, `green_accessibility`, `street_edits`, `accessibility_common` |
| `sitex.viz` | Maps and scenes for discussion | `interactive_webmap`, `city3d`, `heatmaps`, `building_overlays`, `city_morphology_maps`, `raster_scenes`, `workshop_diagram` |

Design choices:

- **`CityConfig` is the single source of truth for a site.** It derives the local CRS,
  the CAD/UCS origin and a filename slug from a place name and a point, so no module
  hardcodes a CRS. Most functions have a `..._for_city()` wrapper that takes it.
- **The slug comes from the OSM place name.** When the place is geocoded, the slug is
  made from the `name` tag of its OSM boundary with `slugify()`: accents are stripped,
  the text is lower-cased and every other run of characters becomes `_` (so "Phường
  Pleiku" gives `phuong_pleiku`). All modules and notebooks therefore name the files of
  one place the same way. If you give coordinates and no geocoding is done, the slug
  is made from the text before the first comma of `place_name`. You can also set `slug`
  yourself.
- **The local CRS is the global UTM zone.** `local_epsg` is the UTM zone of the site
  (for example EPSG:32649 for Pleiku), and every analysis layer is kept in it, in metres.
  Vietnam spans two zones, 48N and 49N. **VN-2000 is opt-in** and is not used for
  analysis. It is the national grid, with a different central meridian for each province,
  which the authorities require for submitted documents. `local_epsg_from_lonlat(...,
  scheme="2025")` gives the VN-2000 zone of the province (new provinces from 1 July 2025;
  `scheme="epsg"` gives the older assignment of the EPSG registry), from a bundled province
  layer. For DXF and OBJ files, which carry no CRS, `cad_frame(..., vn2000=True)` writes
  the model in the VN-2000 zone instead (`CAD_IN_VN2000` in the CAD notebooks, `False`
  by default).
- **No accounts where avoidable.** Global datasets are streamed by bounding box (DuckDB
  range requests on GeoParquet, GDAL `/vsicurl/` on cloud GeoTIFFs), so nothing large is
  downloaded first. Sentinel-2 through the Copernicus Data Space Ecosystem needs a free
  login.
- **Analysis and acquisition are separate.** `sitex.data` only downloads and saves;
  the other packages read the saved files.
- **Outputs are for design tools.** Results are written as files that open in QGIS
  and CAD (GeoPackage, GeoTIFF, DXF, QGIS style files), next to interactive HTML maps.

## Pipeline

The workshop notebooks run in this order. Each notebook reads the files of the earlier ones
and saves its own, and the hexagon grid is the common frame that joins them.

```mermaid
flowchart LR
    A["00 Data acquisition<br/>boundary · streets · buildings · DEM · imagery<br/>+ basic hex grid (H3, res. 9)"]
    B["01-03 Analysis<br/>morphology · terrain · UHI · space syntax<br/>POIs · isochrones · green access"]
    C["Export per hexagon<br/>one Parquet file and legend per notebook<br/>(h3_id)"]
    D["05 Hex integration<br/>all tables in one DuckDB file<br/>SQL screening"]
    E["06 Ask the hexagons<br/>(optional) question in words → SQL → answer"]
    A --> B --> C --> D -.-> E
```

- **Basic hex grid.** `00-Data-Acquisition.ipynb` builds one H3 grid for the place
  (`sitex.core.hexgrid`, resolution 9: about 0.1 km² and 330 m across) and saves it as
  `hex_grid_{slug}.gpkg`. No analysis value is stored in this file; it only holds the
  hexagon ids and their outlines in the local CRS.
- **Hex integration.** Each analysis notebook ends with an *Export for interpretation*
  section. It moves its results onto the grid with `points_to_hex`, `polygons_to_hex`,
  `lines_to_hex` or `rasters_to_hex`, one row per hexagon keyed by `h3_id`, and writes a
  legend for every column. `05-INT-Hex_Integration.ipynb` reads these files into one
  DuckDB database file and answers questions by SQL, for example *which hexagons are hot
  and have no public green within a short walk?* The numbers are per hexagon, not per
  resident.
- **Exploring the indicators with a language model (optional).** `06-INT-Ask_the_Hexagons.ipynb`
  adds a language model on top of the database. The model never calculates: it reads the
  question, the column names and the legends, and writes one SQL query. DuckDB runs the
  query (read-only), and the model explains the first rows. The SQL is always shown, so
  it can be checked. Choose the model with `PROVIDER`: a free Gemini key, an open model on
  the Colab GPU, or a chat window where you copy and paste the prompt. Nothing else in the
  workshop depends on this notebook. A static web page in the course repository (`hex-demo`)
  does the same in the browser, in English and Vietnamese, with the user's own key.

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

pleiku.slug          # "phuong_pleiku" (from the text before the first comma)
pleiku.local_epsg    # 32649 (UTM zone 49N; VN-2000 only with vn2000_scheme="2025")
pleiku.origin        # (easting, northing) CAD/UCS base point, floored to 1000 m
```

## Data source decisions

Two data sources were changed after the evaluation, and the reasons are recorded here so
that a reader of the notebooks knows why they differ from the first version.

**Elevation: GEDTM30 as the terrain model.** The initial version used NASADEM as the
terrain model (DTM). A check on ten Vietnamese towns found negative heights in flat delta
land. In NASADEM up to 11.1% of the pixels of one place (Long Xuyên) were below 0 m, which is
implausible in flat delta land. The current version uses GEDTM30 (Ho et al., 2025), a 30 m
terrain model estimated with machine learning from the Copernicus DEM, ALOS World 3D and about
30 billion lidar ground points (ICESat-2, GEDI). GEDTM30 is a machine-learning product, and
OpenTopography advises checking it before use in critical applications. GEDTM30 is read for the
bounding box from a public file at `s3.opengeohub.org`, so it needs no account or key, and no
OpenTopography key is used anywhere in the course. The AW3D30 surface model was dropped: no
notebook used it, since the canopy height comes from the Meta/WRI map. The file is now
`dem/dtm_gedtm30.tif` (before `dem/dtm_nasadem.tif` in the first version).

**Streets: a Geofabrik extract instead of Overpass.** OSM streets used to be downloaded
from a public Overpass server through OSMnx. Those servers are shared services with rate
limits, and they can become unreachable: during testing, connections to `overpass-api.de`
timed out and its mirror `overpass.private.coffee` accepted connections but did not answer
within 180 s. Very large boundaries can also exceed what an Overpass server accepts in one
query. The course notebook therefore downloads the OSM extract of the whole country from
[Geofabrik](https://download.geofabrik.de/) once (for Vietnam about 330 MB, updated daily),
crops it to the boundary or, for a drawn rectangle, to its box with `osmium-tool`, and
builds the `drive`, `walk` and `drive_service` street graphs from the cropped file. After
the download no query depends on a server, and the size of the site does not matter.
The cost is the one large download, data that can be up to a day older than the live map,
and the extra program to install (see [Install](#install)). For an initial site analysis
this is sufficient. The Overpass function `download_street_networks` is still in
`sitex.data.street_network` next to `street_networks_from_geofabrik`. The Geofabrik route
was tested on one ward in Hà Nội with the full Vietnam file (download about 4.5 minutes,
crop and graphs a few seconds); it has not yet been compared with an Overpass result or
run in Google Colab.

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
| GEDTM30 terrain model (OpenGeoHub) | `data.remote_sensing` | CC BY 4.0 (as listed on the Zenodo record) | Ho et al. (2025), *PeerJ* 13, e19673; doi 10.5281/zenodo.18887460. Built in part from the Copernicus DEM and ALOS World 3D, whose own terms can also apply; read them at the providers. Machine-learning product: check it before critical use |
| OpenStreetMap extract from Geofabrik (streets) | `data.street_network` | ODbL | © OpenStreetMap contributors; extracts by Geofabrik GmbH |

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

## Reference
> Ho, Y.-F., Grohmann, C. H., Lindsay, J., Reuter, H. I., Parente, L., Witjes, M., & Hengl, T. (2025). 
> GEDTM30: Global ensemble digital terrain model at 30 m and derived multiscale terrain variables. 
>*PeerJ*, *13*, e19673. https://doi.org/10.7717/peerj.19673


## Citation

If you use SiteX, please cite the thesis it belongs to:

> Nguyen, C. (2026). *[SITEX: An AI-Assisted Notebook Library for Computational Site Analysis in Data-Scarce Contexts]*. 
> Master's thesis, Metropolia University of Applied Sciences, Computing in Construction.
