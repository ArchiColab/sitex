"""Hand-built SVG charts — no matplotlib.

This project's ``gis`` conda environment crashes the Python process on any actual
matplotlib figure render (see ``sitex.core.raster_viz``'s module docstring) — a bar
chart or scatter plot can't go through matplotlib here at all, raster or not.
This module generalizes ``POI_Accessibility.ipynb``'s own hand-built-SVG workaround
into a small, reusable utility instead of a one-off per notebook.
"""

from __future__ import annotations

from xml.sax.saxutils import escape as _esc


def render_bar_chart_svg(
    labels: list,
    values: list,
    colors: list | None = None,
    title: str = "",
    value_fmt: str = "{:.1f}",
    width: int = 480,
    height: int = 320,
) -> str:
    """A minimal single-series bar chart, drawn directly as SVG markup.

    Display with ``IPython.display.SVG(render_bar_chart_svg(...))``.
    """
    margin_l, margin_r, margin_t, margin_b = 55, 20, 40, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    n = len(labels)
    colors = colors or (["#4a90d9"] * n)

    vmax = max(values) * 1.15 if values else 1.0
    slot_w = plot_w / n
    bar_w = slot_w * 0.6

    def y_of(v):
        return margin_t + plot_h * (1 - v / vmax)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif" font-size="11">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]
    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" font-size="13" '
            f'fill="#111">{_esc(title)}</text>'
        )

    baseline = margin_t + plot_h
    parts.append(f'<line x1="{margin_l}" y1="{baseline}" x2="{width - margin_r}" y2="{baseline}" stroke="#999"/>')

    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = margin_l + i * slot_w + slot_w / 2
        by = y_of(value)
        bh = baseline - by
        bx = cx - bar_w / 2
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{by - 6:.1f}" text-anchor="middle" fill="#222">'
            f'{value_fmt.format(value)}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{baseline + 16:.1f}" text-anchor="middle" fill="#333">'
            f'{_esc(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_dual_bar_chart_svg(
    labels: list,
    series: dict,
    colors: dict | None = None,
    title: str = "",
    max_val: float = 100,
    value_fmt: str = "{:.0f}%",
    width: int = 640,
    height: int = 360,
) -> str:
    """Two side-by-side bars per category — e.g. "% buildings" vs "% floor area"
    reached within a time budget, per essential-service category.

    ``series`` is ``{series_name: (values, opacity)}``, one entry per bar within a
    group, all sharing the same per-category ``colors``. ``labels`` and every
    ``values`` list must be the same length (one entry per category).
    """
    margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    n = len(labels)
    group_w = plot_w / n
    series_names = list(series.keys())
    bar_w = group_w * 0.32
    colors = colors or {label: "#4a90d9" for label in labels}

    def y_of(pct):
        return margin_t + plot_h * (1 - pct / max_val)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="sans-serif" font-size="11">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    for pct in (0, 25, 50, 75, 100):
        y = y_of(pct * max_val / 100)
        parts.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>'
        )
        parts.append(f'<text x="{margin_l - 8}" y="{y + 3:.1f}" text-anchor="end" fill="#666">{pct}%</text>')

    n_series = len(series_names)
    offsets = [i - (n_series - 1) / 2 for i in range(n_series)]

    for i, label in enumerate(labels):
        color = colors.get(label, "#4a90d9")
        gx = margin_l + i * group_w + group_w / 2

        for offset, series_name in zip(offsets, series_names):
            values, opacity = series[series_name]
            val = float(values[i])
            bx = gx + offset * (bar_w + 2) - bar_w / 2
            by = y_of(val)
            bh = margin_t + plot_h - by
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'fill="{color}" fill-opacity="{opacity}"/>'
            )
            parts.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" '
                f'fill="#333" font-size="9">{value_fmt.format(val)}</text>'
            )

        words = str(label).split(" ")
        line1 = _esc(" ".join(words[: len(words) // 2 + 1]))
        line2 = _esc(" ".join(words[len(words) // 2 + 1:]))
        ly = margin_t + plot_h + 16
        parts.append(f'<text x="{gx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#222">{line1}</text>')
        if line2:
            parts.append(f'<text x="{gx:.1f}" y="{ly + 13:.1f}" text-anchor="middle" fill="#222">{line2}</text>')

    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" font-size="13" '
            f'fill="#111">{_esc(title)}</text>'
        )

    ly = height - 14
    lx = margin_l
    for series_name, (_, opacity) in series.items():
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" fill="#555" fill-opacity="{opacity}"/>')
        parts.append(f'<text x="{lx + 14}" y="{ly}" fill="#333">{_esc(series_name)}</text>')
        lx += 20 + 8 * len(series_name)

    parts.append("</svg>")
    return "".join(parts)
