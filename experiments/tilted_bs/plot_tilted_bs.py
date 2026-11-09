import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Results layout: step -> series name -> state -> probability.
Results = dict[str, dict[str, dict[str, float]]]

# An entry named "<series>-stdev" holds the errors of that series, and is not a series itself.
STDEV_SUFFIX = "-stdev"

# Each series takes the next style, in order of first appearance in the results.
COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]
# Half-widths of the error bar caps, narrowing so that series lying on top of each other all stay visible.
CAP_WIDTHS = [10, 8, 6, 5, 4, 4, 4, 4]

# Figure layout, in pixels.
PANEL_HEIGHT = 182
PANEL_GAP = 52
MARGIN_TOP = 140
MARGIN_BOTTOM = 65
LEGEND_OFFSET = 30
TITLE_OFFSET = 18


def load_results(source: Path) -> Results:
    with source.open(encoding="utf-8") as f:
        return json.load(f)


def find_series_names(data: Results, steps: list[str]) -> list[str]:
    names = list(dict.fromkeys(name for step in steps for name in data[step] if not name.endswith(STDEV_SUFFIX)))

    if len(names) > len(COLORS):
        raise ValueError(f"Found {len(names)} series, but only {len(COLORS)} styles are defined.")

    return names


def series_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").capitalize()


def series_stdevs(data: Results, steps: list[str], name: str, state: str) -> list[float]:
    # A series without stdevs, like the theoretical one, shows up as a flat, zero-height bar.
    return [data[step].get(f"{name}{STDEV_SUFFIX}", {}).get(state, 0.0) for step in steps]


def series_probabilities(data: Results, steps: list[str], name: str, state: str) -> list[float | None]:
    # A state absent from a series has probability 0; a series absent from a step has no data.
    return [data[step][name].get(state, 0.0) if name in data[step] else None for step in steps]


def add_state_panel(
    fig: go.Figure, data: Results, steps: list[str], series_names: list[str], state: str, row: int
) -> None:
    for name, color, cap_width in zip(series_names, COLORS, CAP_WIDTHS):
        fig.add_trace(
            go.Scatter(
                x=list(map(int, steps)),
                y=series_probabilities(data, steps, name, state),
                error_y=dict(
                    type="data",
                    array=series_stdevs(data, steps, name, state),
                    color=color,
                    thickness=1.2,
                    width=cap_width,
                ),
                name=series_label(name),
                legendgroup=name,
                showlegend=False,
                mode="none",
            ),
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="Probability", range=[-0.045, 1.045], row=row, col=1)


def add_legend_entries(fig: go.Figure, series_names: list[str]) -> None:
    # Plotly draws no legend symbol for a trace that only has error bars, so these empty traces provide one.
    for name, color in zip(series_names, COLORS):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=series_label(name),
                legendgroup=name,
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns", line=dict(color=color, width=2)),
            ),
            row=1,
            col=1,
        )


def apply_layout(fig: go.Figure, plot_height: int) -> None:
    height = MARGIN_TOP + plot_height + MARGIN_BOTTOM

    fig.update_layout(
        title=dict(
            text="Tilted beam splitter: state probabilities",
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
            y=1 + LEGEND_OFFSET / plot_height,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=MARGIN_TOP, b=MARGIN_BOTTOM, l=75, r=35),
    )


def build_figure(data: Results) -> go.Figure:
    steps = sorted(data, key=int)
    series_names = find_series_names(data, steps)
    states = ["1000", "0101", "0010"]

    rows = len(states)
    plot_height = rows * PANEL_HEIGHT + (rows - 1) * PANEL_GAP

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=PANEL_GAP / plot_height,
        subplot_titles=[f"State |{state}⟩" for state in states],
    )

    for row, state in enumerate(states, start=1):
        add_state_panel(fig, data, steps, series_names, state, row)

    add_legend_entries(fig, series_names)
    fig.update_xaxes(title_text="n (θ = n π / 32)", dtick=2, row=rows, col=1)
    apply_layout(fig, plot_height)

    return fig


def save_figure(fig: go.Figure, directory: Path) -> None:
    fig.write_html(directory / "tilted_bs_plot.html", include_plotlyjs=True)
    fig.write_image(directory / "tilted_bs_plot.pdf")
    # The bars are only about a pixel tall at 10k shots, so the raster needs extra resolution.
    fig.write_image(directory / "tilted_bs_plot.png", scale=3)


def main() -> None:
    directory = Path(__file__).parent
    data = load_results(directory / "tilted_bs_results.json")
    save_figure(build_figure(data), directory)


if __name__ == "__main__":
    main()
