import numpy as np
import pytest

from sitex.core.raster_viz import (
    add_caption,
    alpha_composite,
    colorize_array,
    colorize_categorical,
    hstack_images,
    mask_to_rgba,
    plot_raster_folium,
    render_map_panel,
    save_array_png,
)


def test_colorize_array_shape_and_dtype():
    arr = np.linspace(0, 100, 25).reshape(5, 5)
    rgba = colorize_array(arr, cmap_name="viridis")
    assert rgba.shape == (5, 5, 4)
    assert rgba.dtype == np.uint8


def test_colorize_array_respects_vmin_vmax():
    arr = np.array([[0.0, 55.0], [100.0, 55.0]])
    wide = colorize_array(arr, vmin=0, vmax=100)
    narrow = colorize_array(arr, vmin=40, vmax=60)
    # Same raw value (55) normalizes to 0.55 in the wide window but 0.75 in the
    # narrow one, so the two colorizations must disagree on that pixel.
    assert not np.array_equal(wide[0, 1], narrow[0, 1])


def test_colorize_array_nan_is_transparent():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    rgba = colorize_array(arr, nan_transparent=True)
    assert rgba[0, 1, 3] == 0
    assert rgba[0, 0, 3] != 0


def test_colorize_categorical_maps_fixed_colors():
    arr = np.array([[-1, 0], [1, -9999]])
    colors = {-1: (215, 48, 39), 0: (255, 255, 191), 1: (26, 152, 80)}
    rgba = colorize_categorical(arr, colors)
    assert tuple(rgba[0, 0]) == (215, 48, 39, 255)
    assert tuple(rgba[0, 1]) == (255, 255, 191, 255)
    assert tuple(rgba[1, 0]) == (26, 152, 80, 255)
    assert tuple(rgba[1, 1]) == (0, 0, 0, 0)  # -9999 not in the mapping -> transparent


def test_colorize_categorical_accepts_rgba_tuples():
    arr = np.array([[2]])
    rgba = colorize_categorical(arr, {2: (10, 20, 30, 128)})
    assert tuple(rgba[0, 0]) == (10, 20, 30, 128)


def test_mask_to_rgba_colors_only_true_pixels():
    mask = np.array([[True, False], [False, True]])
    rgba = mask_to_rgba(mask, color=(10, 20, 30), alpha=200)
    assert tuple(rgba[0, 0]) == (10, 20, 30, 200)
    assert tuple(rgba[0, 1]) == (0, 0, 0, 0)
    assert tuple(rgba[1, 1]) == (10, 20, 30, 200)


def test_alpha_composite_fully_opaque_overlay_replaces_base():
    base = np.full((2, 2, 4), 255, dtype=np.uint8)
    base[..., :3] = [255, 0, 0]  # red base
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[..., :3] = [0, 0, 255]  # blue overlay
    overlay[..., 3] = 255  # fully opaque

    result = alpha_composite(base, overlay)
    assert tuple(result[0, 0][:3]) == (0, 0, 255)
    assert result[0, 0, 3] == 255


def test_alpha_composite_transparent_overlay_keeps_base():
    base = np.full((2, 2, 4), 255, dtype=np.uint8)
    base[..., :3] = [255, 0, 0]
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)  # alpha = 0 everywhere

    result = alpha_composite(base, overlay)
    assert tuple(result[0, 0][:3]) == (255, 0, 0)


def test_save_array_png_writes_readable_file(tmp_path):
    from PIL import Image

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    out_path = tmp_path / "test.png"
    save_array_png(rgba, out_path)

    assert out_path.exists()
    reloaded = np.array(Image.open(out_path))
    assert reloaded.shape == (4, 4, 4)


def test_hstack_images_combines_widths(tmp_path):
    from PIL import Image

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGBA", (10, 20), (255, 0, 0, 255)).save(p1)
    Image.new("RGBA", (15, 8), (0, 255, 0, 255)).save(p2)

    out_path = tmp_path / "combined.png"
    hstack_images([p1, p2], out_path, gap=5)

    combined = Image.open(out_path)
    assert combined.width == 10 + 15 + 5
    assert combined.height == 20  # tallest input


def test_add_caption_adds_band_above_image(tmp_path):
    from PIL import Image

    p1 = tmp_path / "a.png"
    Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(p1)

    out_path = tmp_path / "captioned.png"
    add_caption(p1, "Test title", out_path)
    captioned = Image.open(out_path)

    assert captioned.width == 100
    assert captioned.height > 50


def test_add_caption_defaults_to_overwriting_input(tmp_path):
    from PIL import Image

    p1 = tmp_path / "a.png"
    Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(p1)

    returned = add_caption(p1, "Test title")

    assert returned == p1
    assert Image.open(p1).height > 50


def test_render_map_panel_has_axis_and_colorbar_margins(tmp_path):
    from PIL import Image

    values = np.linspace(0, 100, 25).reshape(5, 5)
    x_coords = np.linspace(108.0, 108.1, 5)
    y_coords = np.linspace(14.1, 14.0, 5)

    out_path = tmp_path / "panel.png"
    render_map_panel(
        values, x_coords, y_coords,
        title="Test panel", cbar_label="m", plot_size=(50, 50), out_path=out_path,
    )
    panel = Image.open(out_path)

    # The plot area itself is only 50x50 — extra width/height means axis ticks,
    # labels, and the colorbar were actually drawn, not just the raw raster.
    assert panel.width > 50
    assert panel.height > 50


def test_render_map_panel_handles_non_square_grid(tmp_path):
    # Regression: x/y coordinate arrays of different lengths (a non-square raster,
    # the normal case — e.g. a 245x246 DTM) must not be indexed with the same tick
    # index array, or the shorter axis raises IndexError.
    values = np.linspace(0, 100, 5 * 7).reshape(5, 7)
    x_coords = np.linspace(108.0, 108.1, 7)
    y_coords = np.linspace(14.1, 14.0, 5)

    out_path = tmp_path / "panel.png"
    render_map_panel(values, x_coords, y_coords, out_path=out_path)

    assert out_path.exists()


def test_plot_raster_folium_adds_satellite_layer_and_scale_bar(tmp_path):
    from PIL import Image

    png_path = tmp_path / "flood.png"
    Image.new("RGBA", (10, 10), (0, 102, 255, 160)).save(png_path)

    m = plot_raster_folium(
        png_path,
        bounds=[[13.98, 108.0], [14.02, 108.05]],
        center=(14.0, 108.02),
        name="Flood extent",
    )

    html = m.get_root().render()
    assert "World_Imagery" in html  # satellite basemap tile URL
    assert "Satellite (Esri World Imagery)" in html  # its entry in the layer control
    assert m.control_scale is True
    assert "L.control.scale" in html  # the scale-bar control actually rendered


def test_plot_raster_folium_satellite_can_be_disabled(tmp_path):
    from PIL import Image

    png_path = tmp_path / "flood.png"
    Image.new("RGBA", (10, 10), (0, 102, 255, 160)).save(png_path)

    m = plot_raster_folium(
        png_path,
        bounds=[[13.98, 108.0], [14.02, 108.05]],
        center=(14.0, 108.02),
        satellite=False,
    )

    assert "World_Imagery" not in m.get_root().render()


def test_hstack_images_vertically_centers_shorter_image(tmp_path):
    from PIL import Image

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGBA", (10, 20), (255, 0, 0, 255)).save(p1)
    Image.new("RGBA", (10, 10), (0, 255, 0, 255)).save(p2)

    out_path = tmp_path / "combined.png"
    hstack_images([p1, p2], out_path, gap=0)
    combined = np.array(Image.open(out_path))

    # The shorter (10px) green image should be vertically centered within the 20px canvas:
    # rows 0-4 above it must still be the gap/background fill, not green.
    assert tuple(combined[0, 12]) != (0, 255, 0, 255)
    assert tuple(combined[10, 12]) == (0, 255, 0, 255)
