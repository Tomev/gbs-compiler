import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main() -> None:
    source = Path(__file__).with_name("tilted_bs_results.json")

    with source.open(encoding="utf-8") as f:
        data = json.load(f)

    steps = sorted(data, key=int)
    states = ["0101", "1000", "0010"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"State |{state}⟩" for state in states],
    )

    for row, state in enumerate(states, start=1):
        theoretical = [data[step]["theoretical"].get(state, 0.0) for step in steps]
        noiseless = [data[step]["noiseless"].get(state, 0.0) for step in steps]

        fig.add_trace(
            go.Scatter(
                x=list(map(int, steps)),
                y=theoretical,
                name="Theoretical",
                mode="lines",
                line=dict(color="#2563EB", width=2.7),
                legendgroup="theoretical",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(map(int, steps)),
                y=noiseless,
                name="Noiseless",
                mode="markers",
                marker=dict(color="#EA580C", size=8, symbol="circle-open", line=dict(width=2.2)),
                legendgroup="noiseless",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="Probability", range=[-0.045, 1.045], row=row, col=1)

    fig.update_xaxes(title_text="Index", dtick=2, row=3, col=1)
    fig.update_layout(
        title="Tilted beam splitter: theoretical vs noiseless gate-based",
        template="plotly_white",
        height=830,
        width=980,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.075, xanchor="right", x=1),
        margin=dict(t=115, b=65, l=75, r=35),
    )

    output = Path(__file__).with_name("tilted_bs_plot.html")
    fig.write_html(output, include_plotlyjs=True)
    fig.write_image(Path(__file__).with_name("tilted_bs_plot.pdf"))
    fig.write_image(Path(__file__).with_name("tilted_bs_plot.png"))


if __name__ == "__main__":
    main()
