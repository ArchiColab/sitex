import numpy as np
import rasterio
from rasterio.transform import from_origin

from sitex.data import landcover as lc


def test_worldcover_tiles_for_bbox_single_tile():
    # Entirely inside the N12E108 3-degree cell
    tiles = lc.worldcover_tiles_for_bbox((108.0, 12.5, 108.5, 13.0))
    assert tiles == ["N12E108"]


def test_worldcover_tiles_for_bbox_spans_two_tiles():
    # Straddles the 108E boundary, like the Pleiku AOI in the source notebook
    tiles = lc.worldcover_tiles_for_bbox((107.9, 12.5, 108.1, 13.0))
    assert set(tiles) == {"N12E105", "N12E108"}


def test_worldcover_tiles_for_bbox_negative_lat_lon():
    tiles = lc.worldcover_tiles_for_bbox((-70.5, -12.5, -70.1, -12.1))
    assert tiles == ["S15W072"]


def test_align_categorical_to_grid_nearest_neighbour(tmp_path):
    # Source: a 4x4 raster, class 80 in the top-left quadrant, 10 elsewhere.
    src_data = np.full((4, 4), 10, dtype=np.uint8)
    src_data[:2, :2] = 80
    src_transform = from_origin(0, 4, 1, 1)
    src_path = tmp_path / "wc.tif"
    with rasterio.open(
        src_path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="uint8", crs="EPSG:4326", transform=src_transform,
    ) as dst:
        dst.write(src_data, 1)

    # Destination: same extent/CRS, coarser 2x2 grid -> nearest neighbour should still
    # read class 80 in the top-left cell.
    dst_transform = from_origin(0, 4, 2, 2)
    aligned = lc.align_categorical_to_grid(src_path, dst_transform, "EPSG:4326", (2, 2))

    assert aligned.shape == (2, 2)
    assert aligned[0, 0] == 80
    assert aligned[1, 1] == 10
