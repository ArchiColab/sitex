from sitex.core.svg_charts import render_bar_chart_svg, render_dual_bar_chart_svg


def test_render_bar_chart_svg_produces_valid_svg_tag():
    svg = render_bar_chart_svg(["A", "B"], [10.0, 20.0], colors=["#111111", "#222222"], title="Test")
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "Test" in svg


def test_render_bar_chart_svg_escapes_labels():
    svg = render_bar_chart_svg(["<script>"], [5.0])
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_render_bar_chart_svg_taller_bar_for_larger_value():
    import re

    svg = render_bar_chart_svg(["Low", "High"], [1.0, 100.0], colors=["#000", "#000"])
    heights = [float(h) for h in re.findall(r'height="([\d.]+)"', svg) if float(h) > 5]
    # Exactly two bar rects (plus the background rect, filtered out by the >5 threshold
    # coincidentally not applying) — just check the larger value produced a taller bar.
    bar_heights = re.findall(r'<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="([\d.]+)" fill="#000"', svg)
    bar_heights = [float(h) for h in bar_heights]
    assert len(bar_heights) == 2
    assert bar_heights[1] > bar_heights[0]


def test_render_dual_bar_chart_svg_two_bars_per_category():
    svg = render_dual_bar_chart_svg(
        labels=["Healthcare", "Education"],
        series={
            "Buildings": ([80.0, 60.0], 0.9),
            "Floor area": ([70.0, 50.0], 0.45),
        },
        colors={"Healthcare": "#e41a1c", "Education": "#377eb8"},
        title="Test",
    )
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    # One data bar per category per series (plus one legend swatch per series,
    # which shares the same opacity — hence 2 data bars + 1 legend swatch = 3).
    assert svg.count("fill-opacity=\"0.9\"") == 3
    assert svg.count("fill-opacity=\"0.45\"") == 3
    assert svg.count('fill="#e41a1c"') == 2  # Healthcare's two bars (Buildings + Floor area)
    assert svg.count('fill="#377eb8"') == 2  # Education's two bars


def test_render_dual_bar_chart_svg_escapes_category_labels():
    svg = render_dual_bar_chart_svg(
        labels=["<bad>"],
        series={"A": ([10.0], 0.9)},
    )
    assert "<bad>" not in svg
    assert "&lt;bad&gt;" in svg
