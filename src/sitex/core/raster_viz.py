"""Safe raster-to-image rendering.

This project's ``gis`` conda environment has a broken matplotlib rendering
pipeline: creating a ``Figure`` and never calling ``draw()``/``show()``/
``savefig()`` on it is fine, but any actual render — ``plt.show()``,
``fig.savefig()`` (PNG *or even SVG*) — crashes the Python process outright,
independent of what's plotted (reproduced with a bare ``imshow`` of random
noise). This is the same category of environment fragility
``POI_Accessibility.ipynb`` already worked around (hand-built SVG instead of
matplotlib) and the ``GeoDataFrame.plot()`` crash found while migrating
``01-ARCH-Building_Morphology.ipynb`` — this module is the fix for the raster
case: colorize a 2D array to RGBA using a matplotlib colormap *lookup only*
(pure NumPy indexing — no ``Figure``, no ``draw()``), then hand the bytes to
PIL (a separate rendering stack) for PNG encoding, or to folium as an
``ImageOverlay`` for an interactive map. Confirmed safe by direct testing in
this project's kernel; a ``Figure`` is only ever safe here if it is built and
then closed (``plt.close()``) without ever being rendered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def colorize_array(
    arr: np.ndarray,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    nan_transparent: bool = True,
) -> np.ndarray:
    """2D array -> ``uint8`` RGBA array via a matplotlib colormap *lookup* only.

    Never touches matplotlib's Figure/Axes/draw pipeline — see the module
    docstring for why that matters in this environment.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanmin(finite)) if vmin is None else vmin
    vmax = float(np.nanmax(finite)) if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps[cmap_name]

    rgba = (cmap(norm(arr)) * 255).astype(np.uint8)
    if nan_transparent:
        rgba[~np.isfinite(arr), 3] = 0
    return rgba


def colorize_categorical(arr: np.ndarray, category_colors: dict) -> np.ndarray:
    """Map integer category codes to fixed RGBA colors.

    For classified/clustered rasters (a fixed set of discrete classes) where a
    continuous colormap wouldn't make sense — e.g. a -1/0/+1 change-classification
    raster with specific colors per class, or K-Means cluster IDs. ``category_colors``
    maps each category value to an ``(R, G, B)`` or ``(R, G, B, A)`` tuple; any value
    not present comes out fully transparent (useful for a nodata sentinel like
    ``-9999``).
    """
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    for value, color in category_colors.items():
        if len(color) == 3:
            color = (*color, 255)
        rgba[arr == value] = color
    return rgba


def mask_to_rgba(mask: np.ndarray, color: tuple[int, int, int] = (0, 102, 255), alpha: int = 153) -> np.ndarray:
    """Boolean mask -> ``uint8`` RGBA overlay: ``color`` where True, transparent elsewhere."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return rgba


def alpha_composite(base_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite ``overlay_rgba`` over ``base_rgba`` (both ``uint8`` RGBA), opaque result."""
    base = base_rgba.astype(np.float64) / 255
    over = overlay_rgba.astype(np.float64) / 255
    a = over[..., 3:4]
    rgb = over[..., :3] * a + base[..., :3] * (1 - a)
    rgba = np.dstack([rgb, np.ones_like(a)])
    return (rgba * 255).astype(np.uint8)


def save_array_png(rgba: np.ndarray, out_path: Path) -> Path:
    """Save a ``uint8`` RGBA array as PNG via PIL — no matplotlib involved."""
    from PIL import Image

    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return Path(out_path)


def add_caption(
    image_path: Path,
    text: str,
    out_path: Path | None = None,
    font_size: int = 16,
    padding: int = 8,
    bg_color: tuple = (255, 255, 255, 255),
    text_color: tuple = (20, 20, 20, 255),
) -> Path:
    """Stamp a title caption in a band above a PNG — pure PIL (``ImageDraw``/
    ``ImageFont``), no matplotlib ``ax.set_title()``/``plt.suptitle()`` involved,
    so labelling a panel is safe where matplotlib's own render pipeline crashes
    (see module docstring). Wraps onto multiple lines when ``text`` is wider than
    the image (e.g. a long title over a narrow panel). ``out_path`` defaults to
    overwriting ``image_path``.
    """
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGBA")
    font = _load_font(font_size)

    measurer = ImageDraw.Draw(im)
    max_width = max(im.width - padding * 2, 1)

    def width_of(s: str) -> int:
        l, _, r, _ = measurer.textbbox((0, 0), s, font=font)
        return r - l

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or width_of(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    bboxes = [measurer.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max(bottom - top for _, top, _, bottom in bboxes)
    band_h = line_h * len(lines) + padding * (len(lines) + 1)

    canvas = Image.new("RGBA", (im.width, im.height + band_h), bg_color)
    draw = ImageDraw.Draw(canvas)
    y = padding
    for line, (left, top, right, _bottom) in zip(lines, bboxes):
        draw.text(((im.width - (right - left)) // 2, y - top), line, fill=text_color, font=font)
        y += line_h + padding
    canvas.paste(im, (0, band_h))

    out_path = Path(out_path) if out_path is not None else Path(image_path)
    canvas.save(out_path)
    return out_path


def add_legend(
    image_path: Path,
    entries: list[tuple[tuple[int, int, int], str]],
    out_path: Path | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    font_size: int = 14,
    padding: int = 10,
    swatch_size: int = 14,
    entry_gap: int = 24,
    bg_color: tuple = (255, 255, 255, 255),
    text_color: tuple = (20, 20, 20, 255),
) -> Path:
    """Stamp a color-swatch legend band below a PNG — pure PIL (``ImageDraw``),
    no matplotlib ``ax.legend()``/colorbar involved (see module docstring for
    why that matters here). ``entries`` is a list of ``(rgb_color, label)``
    pairs, each drawn as a small swatch followed by its label, wrapped onto
    multiple rows if wider than the image. ``title``/``subtitle`` are optional
    text lines drawn above the swatches — e.g. the compared time period and
    the thresholds that produced the classification. ``out_path`` defaults to
    overwriting ``image_path``.
    """
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGBA")
    font = _load_font(font_size)
    title_font = _load_font(font_size + 2)

    measurer = ImageDraw.Draw(im)
    max_width = max(im.width - padding * 2, 1)

    def wrap(text: str, use_font) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            left, _, right, _ = measurer.textbbox((0, 0), candidate, font=use_font)
            if not current or (right - left) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    text_lines: list[tuple[str, object]] = []
    if title:
        text_lines += [(line, title_font) for line in wrap(title, title_font)]
    if subtitle:
        text_lines += [(line, font) for line in wrap(subtitle, font)]

    text_bboxes = [measurer.textbbox((0, 0), line, font=f) for line, f in text_lines]
    text_line_h = max((b[3] - b[1] for b in text_bboxes), default=font_size)

    # Lay out swatch entries into rows that fit `max_width`.
    entry_bboxes = [measurer.textbbox((0, 0), label, font=font) for _, label in entries]
    entry_widths = [swatch_size + 6 + (b[2] - b[0]) for b in entry_bboxes]
    rows: list[list[int]] = []
    row: list[int] = []
    row_w = 0
    for i, w in enumerate(entry_widths):
        add_w = w + (entry_gap if row else 0)
        if row and row_w + add_w > max_width:
            rows.append(row)
            row, row_w = [], 0
            add_w = w
        row.append(i)
        row_w += add_w
    if row:
        rows.append(row)

    swatch_row_h = max(swatch_size, entry_bboxes[0][3] - entry_bboxes[0][1] if entry_bboxes else font_size)
    band_h = (
        padding
        + len(text_lines) * (text_line_h + padding // 2 if text_lines else 0)
        + len(rows) * (swatch_row_h + padding // 2)
        + padding // 2
    )

    canvas = Image.new("RGBA", (im.width, im.height + band_h), bg_color)
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = im.height + padding
    for line, f in text_lines:
        left, top, right, _bottom = measurer.textbbox((0, 0), line, font=f)
        draw.text((padding, y - top), line, fill=text_color, font=f)
        y += text_line_h + padding // 2

    for row in rows:
        x = padding
        row_top = y
        for i in row:
            color, label = entries[i]
            fill = (*color, 255) if len(color) == 3 else tuple(color)
            draw.rectangle(
                [x, row_top, x + swatch_size - 1, row_top + swatch_size - 1],
                fill=fill, outline=(120, 120, 120, 255),
            )
            _left, top, _right, _bottom = measurer.textbbox((0, 0), label, font=font)
            draw.text((x + swatch_size + 6, row_top - top), label, fill=text_color, font=font)
            x += entry_widths[i] + entry_gap
        y += swatch_row_h + padding // 2

    out_path = Path(out_path) if out_path is not None else Path(image_path)
    canvas.save(out_path)
    return out_path


def _load_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _vertical_text(text: str, font, fill: tuple):
    """Render ``text`` on a transparent strip and rotate it 90° — PIL has no
    native rotated-text draw call, so a y-axis label is built this way: draw
    horizontally on scratch space, crop to the glyphs, then rotate."""
    from PIL import Image, ImageDraw

    scratch = Image.new("RGBA", (400, 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(scratch)
    d.text((0, 0), text, fill=fill, font=font)
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    scratch = scratch.crop((left, top, right + 1, bottom + 1))
    return scratch.rotate(90, expand=True)


def render_map_panel(
    values: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cmap_name: str = "terrain",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    xlabel: str = "Longitude",
    ylabel: str = "Latitude",
    plot_size: tuple = (280, 280),
    n_ticks: int = 4,
    out_path: Path | None = None,
) -> Path:
    """One raster panel with a title, x/y coordinate tick labels, and a
    labelled colorbar — the hand-drawn PIL equivalent of xarray's
    ``DataArray.plot()`` on a matplotlib ``Axes`` (which draws exactly these
    three things automatically). Needed because that automatic version is a
    matplotlib render call, which crashes this kernel — see the module
    docstring. ``x_coords``/``y_coords`` are the raster's 1D coordinate arrays
    (e.g. ``dtm.x.values`` / ``dtm.y.values``), used only to label the ticks.
    """
    from PIL import Image, ImageDraw

    text_color = (20, 20, 20, 255)
    line_color = (80, 80, 80, 255)
    font = _load_font(12)
    title_font = _load_font(15)

    finite = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite)) if vmin is None else vmin
    vmax = float(np.nanmax(finite)) if vmax is None else vmax

    plot_w, plot_h = plot_size
    rgba = colorize_array(values, cmap_name, vmin=vmin, vmax=vmax)
    plot_img = Image.fromarray(rgba, mode="RGBA").resize((plot_w, plot_h), Image.NEAREST)

    # ── Layout: title band, plot, x/y tick margins, colorbar + its label ────
    left_m, bottom_m, top_m = 55, 40, (28 if title else 6)
    cbar_w, cbar_gap, cbar_label_gap = 18, 6, 45
    right_m = cbar_w + cbar_gap + cbar_label_gap

    canvas_w = left_m + plot_w + right_m
    canvas_h = top_m + plot_h + bottom_m
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    plot_x0, plot_y0 = left_m, top_m
    canvas.paste(plot_img, (plot_x0, plot_y0))
    draw.rectangle([plot_x0, plot_y0, plot_x0 + plot_w - 1, plot_y0 + plot_h - 1], outline=line_color)

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            (plot_x0 + (plot_w - (bbox[2] - bbox[0])) // 2, 4 - bbox[1]),
            title, fill=text_color, font=title_font,
        )

    # ── X ticks (longitude) along the bottom ────────────────────────────────
    x_tick_idx = np.linspace(0, len(x_coords) - 1, n_ticks).astype(int)
    for i in x_tick_idx:
        px = plot_x0 + int(i / max(len(x_coords) - 1, 1) * (plot_w - 1))
        draw.line([(px, plot_y0 + plot_h), (px, plot_y0 + plot_h + 4)], fill=line_color)
        label = f"{x_coords[i]:.4f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((px - (bbox[2] - bbox[0]) // 2, plot_y0 + plot_h + 6 - bbox[1]), label, fill=text_color, font=font)
    xl_bbox = draw.textbbox((0, 0), xlabel, font=font)
    draw.text(
        (plot_x0 + (plot_w - (xl_bbox[2] - xl_bbox[0])) // 2, canvas_h - 16 - xl_bbox[1]),
        xlabel, fill=text_color, font=font,
    )

    # ── Y ticks (latitude) along the left ───────────────────────────────────
    y_tick_idx = np.linspace(0, len(y_coords) - 1, n_ticks).astype(int)
    for i in y_tick_idx:
        py = plot_y0 + int(i / max(len(y_coords) - 1, 1) * (plot_h - 1))
        draw.line([(plot_x0 - 4, py), (plot_x0, py)], fill=line_color)
        label = f"{y_coords[i]:.4f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((plot_x0 - 8 - tw, py - th // 2 - bbox[1]), label, fill=text_color, font=font)
    ylabel_img = _vertical_text(ylabel, font, text_color)
    canvas.paste(ylabel_img, (2, plot_y0 + (plot_h - ylabel_img.height) // 2), ylabel_img)

    # ── Colorbar + its label, to the right ──────────────────────────────────
    cbar_x0 = plot_x0 + plot_w + cbar_gap
    gradient = np.linspace(vmax, vmin, plot_h).reshape(-1, 1)
    cbar_rgba = colorize_array(gradient, cmap_name, vmin=vmin, vmax=vmax)
    cbar_img = Image.fromarray(cbar_rgba, mode="RGBA").resize((cbar_w, plot_h), Image.NEAREST)
    canvas.paste(cbar_img, (cbar_x0, plot_y0))
    draw.rectangle([cbar_x0, plot_y0, cbar_x0 + cbar_w - 1, plot_y0 + plot_h - 1], outline=line_color)

    for frac, val in ((0.0, vmax), (0.5, (vmin + vmax) / 2), (1.0, vmin)):
        py = plot_y0 + int(frac * (plot_h - 1))
        draw.line([(cbar_x0 + cbar_w, py), (cbar_x0 + cbar_w + 4, py)], fill=line_color)
        label = f"{val:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((cbar_x0 + cbar_w + 6, py - (bbox[3] - bbox[1]) // 2 - bbox[1]), label, fill=text_color, font=font)

    if cbar_label:
        cbar_label_img = _vertical_text(cbar_label, font, text_color)
        canvas.paste(
            cbar_label_img,
            (canvas_w - cbar_label_img.width - 2, plot_y0 + (plot_h - cbar_label_img.height) // 2),
            cbar_label_img,
        )

    out_path = Path(out_path) if out_path is not None else Path("panel.png")
    canvas.save(out_path)
    return out_path


def hstack_images(paths: list, out_path: Path, gap: int = 10, gap_color: tuple = (255, 255, 255, 255)) -> Path:
    """Paste PNGs side by side into one wider comparison image — pure PIL, no
    matplotlib subplot/draw call involved, so it's safe to build a multi-panel
    comparison figure even where matplotlib's own render pipeline crashes."""
    from PIL import Image

    images = [Image.open(p).convert("RGBA") for p in paths]
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGBA", (total_w, max_h), gap_color)

    x = 0
    for im in images:
        canvas.paste(im, (x, (max_h - im.height) // 2))
        x += im.width + gap
    canvas.save(out_path)
    return Path(out_path)


def wgs84_bounds_for_folium(da) -> list:
    """A rioxarray ``DataArray``'s bounds (assumed already EPSG:4326) as folium's
    ``[[south, west], [north, east]]`` ``ImageOverlay`` convention."""
    minx, miny, maxx, maxy = da.rio.bounds()
    return [[miny, minx], [maxy, maxx]]


SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SATELLITE_ATTR = "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"


def plot_raster_folium(
    png_path: Path,
    bounds: list,
    center: tuple,
    name: str = "Raster",
    zoom_start: int = 14,
    tiles: str = "CartoDB positron",
    satellite: bool = True,
    control_scale: bool = True,
):
    """A single-layer interactive raster map (folium ``ImageOverlay`` reading a PNG file).

    ``bounds`` is ``[[south, west], [north, east]]`` — see ``wgs84_bounds_for_folium``.
    ``satellite`` adds Esri World Imagery as a second, switchable basemap (via the layer
    control already added below) — no API key needed. ``control_scale`` draws a metric +
    imperial distance scale bar in the map's bottom-left corner, so map distances can be
    read off directly instead of only eyeballing them against the lon/lat grid.
    """
    import folium

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles=tiles, control_scale=control_scale)
    if satellite:
        folium.TileLayer(
            tiles=SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite (Esri World Imagery)",
            overlay=False, control=True,
        ).add_to(m)
    folium.raster_layers.ImageOverlay(image=str(png_path), bounds=bounds, name=name).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
