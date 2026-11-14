import json
from pathlib import Path

import plotly.graph_objects as go

# Fidelity layout: step -> series name -> fidelity.
Fidelity = dict[str, dict[str, float]]

# An entry named "<series>-stderr" holds the errors of that series, and is not a series itself.
STDERR_SUFFIX = "-stderr"

# Each series takes the next style, in order of first appearance in the results (an optimized series shares the style
# of its compiled counterpart, see series_styles).
COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]
# Half-widths of the error bar caps, narrowing so that series lying on top of each other all stay visible.
CAP_WIDTHS = [10, 8, 6, 5, 4, 4, 4, 4]

# The runner stores the results of the manually optimized circuits under keys with this prefix.
OPTIMIZED_PREFIX = "optimized-"
# An optimized series gets the color of its compiled counterpart, scaled by this factor.
OPTIMIZED_DARKENING = 0.6

# Figure layout, in pixels.
PLOT_HEIGHT = 380
MARGIN_TOP = 110
MARGIN_BOTTOM = 65
LEGEND_OFFSET = 12
TITLE_OFFSET = 18


def load_fidelity(source: Path) -> Fidelity:
    with source.open(encoding="utf-8") as f:
        return json.load(f)


def find_series_names(data: Fidelity, steps: list[str]) -> list[str]:
    names = list(
        dict.fromkeys(
            name for step in steps for name in data[step] if not name.endswith(STDERR_SUFFIX)
        )
    )

    n_bases = len(base_series_names(names))

    if n_bases > len(COLORS):
        raise ValueError(
            f"Found {n_bases} distinct series, but only {len(COLORS)} styles are defined."
        )

    return names


def base_series_names(series_names: list[str]) -> list[str]:
    """Returns the distinct series names with the ``optimized-`` prefix stripped, in order of first appearance."""
    return list(dict.fromkeys(name.removeprefix(OPTIMIZED_PREFIX) for name in series_names))


def darken(color: str, factor: float = OPTIMIZED_DARKENING) -> str:
    """Returns the ``#rrggbb`` ``color`` with each channel scaled by ``factor``."""
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))

    return "#" + "".join(f"{int(channel * factor):02x}" for channel in channels)


def series_styles(series_names: list[str]) -> list[tuple[str, int, str]]:
    """
    Returns the ``(color, cap width, legend symbol)`` of every series. An optimized series shares the hue and the cap
    width index of its compiled counterpart, but is darker, narrower and has a different legend symbol.
    """
    bases = base_series_names(series_names)
    styles: list[tuple[str, int, str]] = []

    for name in series_names:
        index = bases.index(name.removeprefix(OPTIMIZED_PREFIX))

        if name.startswith(OPTIMIZED_PREFIX):
            styles.append(
                (darken(COLORS[index]), max(round(CAP_WIDTHS[index] * 0.6), 2), "line-ew")
            )
        else:
            styles.append((COLORS[index], CAP_WIDTHS[index], "line-ns"))

    return styles


def series_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").capitalize()


def series_stderrs(data: Fidelity, steps: list[str], name: str) -> list[float]:
    # A series without stderrs shows up as a flat, zero-height bar.
    return [data[step].get(f"{name}{STDERR_SUFFIX}", 0.0) for step in steps]


def series_fidelities(data: Fidelity, steps: list[str], name: str) -> list[float | None]:
    # A series absent from a step has no data.
    return [data[step].get(name) for step in steps]


def add_series(fig: go.Figure, data: Fidelity, steps: list[str], series_names: list[str]) -> None:
    for name, (color, cap_width, _) in zip(series_names, series_styles(series_names)):
        fig.add_trace(
            go.Scatter(
                x=list(map(int, steps)),
                y=series_fidelities(data, steps, name),
                error_y=dict(
                    type="data",
                    array=series_stderrs(data, steps, name),
                    color=color,
                    thickness=1.2,
                    width=cap_width,
                ),
                name=series_label(name),
                legendgroup=name,
                showlegend=False,
                mode="none",
            )
        )


def add_legend_entries(fig: go.Figure, series_names: list[str]) -> None:
    # Plotly draws no legend symbol for a trace that only has error bars, so these empty traces provide one.
    for name, (color, _, symbol) in zip(series_names, series_styles(series_names)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=series_label(name),
                legendgroup=name,
                mode="markers",
                marker=dict(color=color, size=10, symbol=symbol, line=dict(color=color, width=2)),
            )
        )


def apply_layout(fig: go.Figure) -> None:
    height = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM

    fig.update_layout(
        title=dict(
            text="Tilted beam splitter: fidelity with the theoretical state",
            y=1 - TITLE_OFFSET / height,
            yanchor="top",
        ),
        template="plotly_white",
        height=height,
        width=980,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1 + LEGEND_OFFSET / PLOT_HEIGHT,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=MARGIN_TOP, b=MARGIN_BOTTOM, l=75, r=35),
    )


def build_figure(data: Fidelity) -> go.Figure:
    steps = sorted(data, key=int)
    series_names = find_series_names(data, steps)

    fig = go.Figure()

    add_series(fig, data, steps, series_names)
    add_legend_entries(fig, series_names)
    fig.update_xaxes(title_text="n (θ = n π / 32)", dtick=2)
    fig.update_yaxes(title_text="Fidelity", range=[-0.045, 1.045])
    apply_layout(fig)

    return fig


def save_figure(fig: go.Figure, directory: Path) -> None:
    fig.write_html(directory / "tilted_bs_fidelity_plot.html", include_plotlyjs=True)
    fig.write_image(directory / "tilted_bs_fidelity_plot.pdf")
    # The bars are only a few pixels tall at 10k shots, so the raster needs extra resolution.
    fig.write_image(directory / "tilted_bs_fidelity_plot.png", scale=3)


def main() -> None:
    directory = Path(__file__).parent
    data = load_fidelity(directory / "tilted_bs_fidelity.json")
    save_figure(build_figure(data), directory)


if __name__ == "__main__":
    main()
