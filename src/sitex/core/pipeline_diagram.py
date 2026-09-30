"""Hand-built SVG pipeline diagrams for notebook section headers.

One small diagram sits under each ``## N. Section`` markdown header and shows
where that section sits in the whole pipeline. Each diagram is a row of flat,
rounded phase boxes on a thin connecting line. The active phase is a solid dark
card with a bold header, and every other phase is a plain, pale box.

- :func:`render_pipeline_overview` draws every phase as a solid, bold box,
  without sub-steps. Use one per notebook, in the title cell.
- :func:`render_pipeline_phase` draws the same row, but only ``highlight`` is
  the solid card and the other phases are pale. If the highlighted phase has
  ``substeps``, its card gets a second, lighter band that lists them.
  Sub-steps are never drawn for the other phases, so the reader sees only the
  detail of the current section.

Both functions return SVG markup as a string. Display it in a notebook with
``IPython.display.SVG(...)``, or write it to disk (``Path.write_text(...)``)
and embed it in a markdown cell with
``<img src="resources/charts/chart1.svg" width="1000"/>``.

**Use 4 to 6 phases per pipeline.** ``width`` is the
size at which the image will be displayed, the same number as in
``<img width="...">``. It is not a minimum: box width shrinks to fit the number
of phases, so every added phase makes the boxes and their text smaller on
screen. For a long workflow, group related steps into fewer phases and list
the detail in ``substeps`` instead of adding boxes to the row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape as _esc

DEFAULT_ACTIVE_COLOR = "#1B7192"  # matches this project's own #@title font colour
DEFAULT_LINE_COLOR = "#b7c3cc"


@dataclass
class PipelineStage:
    """One ``## N. Section`` of a notebook, for diagramming.

    ``substeps`` are only ever drawn when this stage is the ``highlight`` of a
    :func:`render_pipeline_phase` call — they're deliberately not shown for
    every other (non-active) phase.
    """

    id: str
    label: str
    substeps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- colour ----

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_color(color: str, toward: str, t: float) -> str:
    """Blend ``color`` a fraction ``t`` of the way toward ``toward`` (0..1)."""
    c1, c2 = _hex_to_rgb(color), _hex_to_rgb(toward)
    return _rgb_to_hex(tuple(a + (b - a) * t for a, b in zip(c1, c2)))


@dataclass
class _Palette:
    """Derived from a single ``active_color`` so overriding one colour keeps
    the header/body/dim/text relationships consistent, same idea as the
    reference diagram's own single dark-teal brand colour with tints."""

    header_fill: str
    body_fill: str
    header_text: str
    body_text: str
    dim_fill: str
    dim_text: str
    line: str


def _build_palette(active_color: str, line_color: str) -> _Palette:
    return _Palette(
        header_fill=active_color,
        body_fill=_lerp_color(active_color, "#ffffff", 0.35),
        header_text="#ffffff",
        body_text="#ffffff",
        dim_fill=_lerp_color(active_color, "#ffffff", 0.86),
        dim_text=_lerp_color(active_color, "#ffffff", 0.35),
        line=line_color,
    )


# ------------------------------------------------------------- geometry ----

def _wrap_text(label: str, max_chars: int) -> list[str]:
    """Greedy word-wrap — same approach as svg_charts's own two-line labels,
    generalized to any number of lines."""
    words = label.split()
    lines: list[str] = []
    current = ""
    for w in words:
        trial = f"{current} {w}".strip()
        if len(trial) > max_chars and current:
            lines.append(current)
            current = w
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _chars_per_box(w: float, font_size: float) -> int:
    """Average sans-serif character width is ~0.6x font-size -- deriving the
    wrap width from ``font_size`` (not a fixed constant) keeps labels wrapping
    sensibly however big the text is drawn at."""
    return max(6, int(w / (font_size * 0.62)))


def _plain_box(
    x: float, y: float, w: float, h: float, label: str, fill: str, text_color: str,
    *, rx: float = 10, font_size: float = 19, font_weight: str = "600",
) -> str:
    """A single-zone flat box — used for every non-active phase, and for an
    active phase with no sub-steps."""
    lines = _wrap_text(label, max_chars=_chars_per_box(w, font_size))
    line_h = font_size + 5
    ty0 = y + h / 2 - (len(lines) - 1) * line_h / 2 + font_size * 0.35
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>']
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{ty0 + i * line_h:.1f}" text-anchor="middle" '
            f'font-size="{font_size}" font-weight="{font_weight}" fill="{text_color}">{_esc(line)}</text>'
        )
    return "".join(parts)


def _card_box(
    x: float, y: float, w: float, header_h: float, body_h: float,
    label: str, substeps: list[str], palette: _Palette,
    *, rx: float = 10, font_size: float = 19, sub_font_size: float = 15,
) -> str:
    """The active phase's card: a bold header band + (if ``substeps``) a
    lighter body band listing them, clipped to one rounded-corner shape so it
    reads as a single attached component."""
    total_h = header_h + body_h
    clip_id = f"clip-{abs(hash((x, y, label))) % 10_000_000}"
    parts = [
        f"<defs><clipPath id=\"{clip_id}\">"
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{total_h:.1f}" rx="{rx}"/>'
        f"</clipPath></defs>",
        f'<g clip-path="url(#{clip_id})">',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{header_h:.1f}" fill="{palette.header_fill}"/>',
    ]
    if body_h > 0:
        parts.append(
            f'<rect x="{x:.1f}" y="{y + header_h:.1f}" width="{w:.1f}" height="{body_h:.1f}" '
            f'fill="{palette.body_fill}"/>'
        )
    parts.append("</g>")

    lines = _wrap_text(label, max_chars=_chars_per_box(w, font_size))
    line_h = font_size + 5
    ty0 = y + header_h / 2 - (len(lines) - 1) * line_h / 2 + font_size * 0.35
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{ty0 + i * line_h:.1f}" text-anchor="middle" '
            f'font-size="{font_size}" font-weight="700" fill="{palette.header_text}">{_esc(line)}</text>'
        )

    if substeps:
        pad_x, pad_y = 18, 14
        row_h = sub_font_size + 10
        ty = y + header_h + pad_y + sub_font_size * 0.8
        max_chars = _chars_per_box(w - 2 * pad_x - 14, sub_font_size)
        for step in substeps:
            step_lines = _wrap_text(step, max_chars=max_chars)
            parts.append(
                f'<circle cx="{x + pad_x + 3:.1f}" cy="{ty - sub_font_size * 0.32:.1f}" r="2.4" '
                f'fill="{palette.body_text}"/>'
            )
            for j, sl in enumerate(step_lines):
                parts.append(
                    f'<text x="{x + pad_x + 15:.1f}" y="{ty + j * row_h:.1f}" '
                    f'font-size="{sub_font_size}" fill="{palette.body_text}">{_esc(sl)}</text>'
                )
            ty += row_h * len(step_lines)

    return "".join(parts)


def _substeps_body_height(substeps: list[str], w: float, sub_font_size: float = 15) -> float:
    if not substeps:
        return 0.0
    pad_x, pad_y = 18, 14
    row_h = sub_font_size + 10
    max_chars = _chars_per_box(w - 2 * pad_x - 14, sub_font_size)
    n_lines = sum(len(_wrap_text(s, max_chars=max_chars)) for s in substeps)
    return pad_y * 2 + n_lines * row_h


def _connector(x1: float, x2: float, y: float, color: str, width: float = 1.6) -> str:
    """A plain thin line, no arrowhead — the boxes alone carry the "where am I"
    signal, not directional cues."""
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"/>'


def _svg_wrap(width: float, height: float, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="sans-serif">'
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="white"/>'
        f"{body}</svg>"
    )


def _row_width(n: int, box_w: float, gap: float) -> float:
    return n * box_w + (n - 1) * gap


def _resolve_canvas_width(min_width: float, row_w: float, side_margin: float) -> float:
    """The canvas must fit the row of boxes -- a caller-supplied ``width`` is
    only ever a *minimum*, never a hard clip, since a fixed default (sized for
    a handful of stages) would otherwise silently overflow/clip a longer
    pipeline. Only used when ``box_w`` is explicitly overridden -- see
    :func:`_fit_box_w` for the default (auto-sized) path."""
    return max(min_width, row_w + 2 * side_margin)


def _fit_box_w(n: int, target_width: float, gap: float, side_margin: float, min_box_w: float) -> tuple[float, float]:
    """Default sizing: ``target_width`` is the actual size the diagram will be
    *displayed* at (e.g. the notebook's own ``<img width="1000">``), not a
    floor -- so box width shrinks to fit ``n`` stages inside it, keeping text
    the same true size on screen regardless of stage count. Only if that would
    push a box below ``min_box_w`` does the canvas grow past ``target_width``
    instead (a fallback for an unusually long pipeline)."""
    available = target_width - 2 * side_margin - (n - 1) * gap
    box_w = available / n if n else target_width
    if box_w < min_box_w:
        box_w = min_box_w
        canvas_width = n * box_w + (n - 1) * gap + 2 * side_margin
    else:
        canvas_width = target_width
    return box_w, canvas_width


# ---------------------------------------------------------------- public ----

def render_pipeline_overview(
    stages: list[PipelineStage],
    out_path: Path | str | None = None,
    *,
    width: int = 1000,
    box_w: float | None = None,
    min_box_w: float = 130,
    box_h: float = 76,
    gap: float = 26,
    top_y: float = 34,
    bottom_pad: float = 28,
    side_margin: float = 36,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    line_color: str = DEFAULT_LINE_COLOR,
    font_size: float = 19,
) -> str:
    """The whole-notebook overview: every phase a solid, bold box, no
    sub-steps.

    Meant for the notebook's own title cell — "here is the whole pipeline"
    before any individual section dives into one part of it.

    ``width`` is the size you intend to *display* this at (e.g. the same
    number you'll pass as ``<img width="...">`` in the notebook) — box width
    auto-shrinks to fit however many ``stages`` you pass so text stays the
    same true size on screen. Designed for **4-6 stages**; many more than
    that will still render (falling back to a wider canvas once boxes would
    drop below ``min_box_w``) but reads better split across fewer, bigger
    phases than crammed into one row. Pass ``box_w`` explicitly to opt out of
    the auto-sizing and pick your own fixed box width instead.
    """
    palette = _build_palette(active_color, line_color)
    n = len(stages)
    if box_w is None:
        box_w, canvas_width = _fit_box_w(n, width, gap, side_margin, min_box_w)
    else:
        row_w = _row_width(n, box_w, gap)
        canvas_width = _resolve_canvas_width(width, row_w, side_margin)
    row_w = _row_width(n, box_w, gap)
    start_x = (canvas_width - row_w) / 2
    mid_y = top_y + box_h / 2

    xs = [start_x + i * (box_w + gap) for i in range(n)]
    parts = []
    for i in range(n - 1):
        parts.append(_connector(xs[i] + box_w, xs[i + 1], mid_y, palette.line))
    for x, stage in zip(xs, stages):
        parts.append(_plain_box(x, top_y, box_w, box_h, stage.label, palette.header_fill, palette.header_text, font_size=font_size))

    height = top_y + box_h + bottom_pad
    svg = _svg_wrap(canvas_width, height, "".join(parts))
    if out_path is not None:
        Path(out_path).write_text(svg, encoding="utf-8")
    return svg


def render_pipeline_phase(
    stages: list[PipelineStage],
    highlight: str,
    out_path: Path | str | None = None,
    *,
    width: int = 1000,
    box_w: float | None = None,
    min_box_w: float = 130,
    box_h: float = 76,
    gap: float = 26,
    top_y: float = 34,
    sub_box_w: float | None = None,
    bottom_pad: float = 28,
    side_margin: float = 36,
    active_color: str = DEFAULT_ACTIVE_COLOR,
    line_color: str = DEFAULT_LINE_COLOR,
    font_size: float = 19,
    sub_font_size: float = 15,
) -> str:
    """One phase's diagram: ``highlight`` is the solid/bold card, every other
    phase is flattened to a pale, low-contrast box.

    If the highlighted :class:`PipelineStage` has ``substeps``, its card
    grows a second, lighter band listing them (``sub_box_w`` defaults to the
    same width as ``box_w``, so header and body read as one attached card —
    pass a wider value if your sub-step text needs more room). Sub-steps are
    never drawn for a non-active phase, on purpose.

    ``width`` is the size you intend to *display* this at (e.g. the same
    number you'll pass as ``<img width="...">`` in the notebook) — box width
    auto-shrinks to fit however many ``stages`` you pass so text stays the
    same true size on screen. Designed for **4-6 stages**. Pass ``box_w``
    explicitly to opt out of the auto-sizing and pick your own fixed box
    width instead.

    Raises ``ValueError`` if ``highlight`` doesn't match any stage's ``id``.
    """
    ids = [s.id for s in stages]
    if highlight not in ids:
        raise ValueError(f"highlight={highlight!r} not found in stage ids {ids}")

    palette = _build_palette(active_color, line_color)
    n = len(stages)
    if box_w is None:
        box_w, canvas_width = _fit_box_w(n, width, gap, side_margin, min_box_w)
    else:
        row_w = _row_width(n, box_w, gap)
        canvas_width = _resolve_canvas_width(width, row_w, side_margin)
    row_w = _row_width(n, box_w, gap)
    start_x = (canvas_width - row_w) / 2
    mid_y = top_y + box_h / 2

    active = next(s for s in stages if s.id == highlight)
    card_w = sub_box_w if sub_box_w is not None else box_w
    body_h = _substeps_body_height(active.substeps, card_w, sub_font_size)

    xs = [start_x + i * (box_w + gap) for i in range(n)]
    parts = []
    for i in range(n - 1):
        parts.append(_connector(xs[i] + box_w, xs[i + 1], mid_y, palette.line))

    for x, stage in zip(xs, stages):
        if stage.id == highlight:
            card_x = x - (card_w - box_w) / 2
            card_x = min(max(card_x, side_margin), canvas_width - side_margin - card_w)
            parts.append(_card_box(card_x, top_y, card_w, box_h, body_h, stage.label, stage.substeps, palette, font_size=font_size, sub_font_size=sub_font_size))
        else:
            parts.append(_plain_box(x, top_y, box_w, box_h, stage.label, palette.dim_fill, palette.dim_text, font_weight="500", font_size=font_size))

    height = top_y + box_h + body_h + bottom_pad
    svg = _svg_wrap(canvas_width, height, "".join(parts))
    if out_path is not None:
        Path(out_path).write_text(svg, encoding="utf-8")
    return svg
