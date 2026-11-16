"""
Article figures of the tilted beam-splitter experiment (matplotlib): the probabilities of the Fock outcomes and the
quality metrics, both against the beam-splitter angle ``theta = n pi / 32``, n = 0..16.

Figures (saved next to this script, as PNG and PDF, never shown unless ``--show`` is given):

* ``vlq_tilted_bs_article_probabilities_<circuit>_plot.*``: two panels (raw and post-selected probabilities of the
  outcomes of the photon-number-conserving sector); lines are the exact curves, points the measured values.
* ``vlq_tilted_bs_article_metrics_<circuit>_plot.*``: the squared Bhattacharyya coefficient (SBC) with the exact
  distribution, raw and post-selected (top), and the kept fraction of the post-selection (bottom).

Data and conventions
--------------------
* Bitstrings are the ``q0q1q2q3`` keys of the runner. Mode 1 is encoded by the qubits ``q0q1`` and mode 2 by ``q2q3``,
  each as a binary number with the first qubit being the most significant bit (Fock cutoff 4, i.e. ``n = 0..3``).
  So ``1000`` is (n1, n2) = (2, 0), ``0101`` is (1, 1) and ``0010`` is (0, 2): the three outcomes of
  ``runner.compute_theoretical_probs``, all with ``n1 + n2 = 2``. All the 16 bitstrings are valid outcomes.
* The sector is ``n1 + n2 = 2``. Raw probability = share of all shots; post-selected probability = share of the shots
  in the sector; kept fraction = share of all shots in the sector (the sum of the raw sector probabilities).
* Exact curves: the closed form of ``runner.compute_theoretical_probs`` on a fine grid of theta.
* SBC = (sum_k sqrt(p_k q_k))^2, with p measured and q exact: over all 16 outcomes for the raw one, within the sector
  (both distributions renormalised there) for the post-selected one. The raw one is the classical fidelity the runner
  stores as ``noisy`` (``mitigated`` for the mitigated series) in ``tilted_bs_fidelity.json``.
* Traces, both from the noisy simulator of the VLQ device and read from ``tilted_bs_results.json`` for the chosen
  circuit version (keys ``optimized-`` prefixed with ``--optimized``); only those found are drawn:

  - raw (large hollow marker): ``noisy-vlq-<N>-samples``;
  - noise-mitigated (small filled marker drawn on top): ``noise_mitigated_noisy_vql``.

  A step without a key is left out of that trace. The mitigated series is used as stored. It is the nearest
  probability distribution to the M3 quasi-probabilities, *not* measured frequencies. Mitigated points mostly sit on
  their raw ones, hence the small marker on top of a larger hollow one (no point is shifted).

Left out
--------
* Error bars: the fragment had none, and the results hold only the multinomial standard deviations.
* Hardware results: the runner only simulates the VLQ device.
"""

import argparse
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# Number of shots of the runner results that are plotted, unless stated otherwise.
DEFAULT_N_SAMPLES: int = 3_000

DIRECTORY: Path = Path(__file__).parent
RESULTS_FILE: str = "tilted_bs_results.json"
SIMULATOR_KEY = "noisy-vlq-{n_samples}-samples"
MITIGATED_SIMULATOR_KEY = "noise_mitigated_noisy_vql"  # (sic) the key the runner writes
# Only the keys of the optimized circuits are prefixed (see runner.CircuitVersion).
OPTIMIZED_PREFIX: str = "optimized-"

# The experiment: the state |N1>|N2> goes through the beam splitter with theta = n pi / 32, n = 0..16.
N1: int = 1
N2: int = 1
TOTAL: int = N1 + N2
N_STEPS: int = 17
CUTOFF: int = 4
# Qubits per mode; mode 1 is q0q1, mode 2 is q2q3, the first qubit being the most significant bit.
QUBITS_PER_MODE: int = 2

Fock = tuple[int, int]
# All the (n1, n2) outcomes and those of the photon-number-conserving sector n1 + n2 = TOTAL.
FOCK: list[Fock] = [(n1, n2) for n1 in range(CUTOFF) for n2 in range(CUTOFF)]
SECTOR: list[Fock] = [outcome for outcome in FOCK if sum(outcome) == TOTAL]

SIMULATOR = "simulator"
SIMULATOR_MITIGATED = "simulator, mitigated"
BACKEND_LABELS: dict[str, str] = {
    SIMULATOR: "VLQ (noisy simulator)",
    SIMULATOR_MITIGATED: "VLQ (noisy simulator, noise-mitigated)",
}
# The raw trace is large and hollow, the mitigated one small and filled.
MARKER: str = "o"
RAW_MARKER_SIZE: float = 9.5
MITIGATED_MARKER_SIZE: float = 4.5
LINE_STYLES: list[str] = ["--", "-", ":", "-."]  # so that equal exact curves stay visible
SBC_COLORS: dict[str, str] = {"SBC_raw": "tab:blue", "SBC_post": "tab:red"}
X_LABEL: str = r"beam-splitter angle $\theta$ (rad)"
EXACT_GRID_POINTS: int = 300
# Legend layout: the height of a row and the margin below it (inches), the gap between the columns (figure width).
LEGEND_ROW_HEIGHT: float = 0.3
LEGEND_PAD: float = 0.2
LEGEND_GAP: float = 0.03
LEGEND_MAX_WIDTH: float = 0.97
LEGEND_FONT_SIZE: float = 10
LEGEND_MIN_FONT_SIZE: float = 6

# Metrics of a backend: one dict per theta with the keys "raw" and "post" (outcome -> probability), "SBC_raw",
# "SBC_post" and "kept".
Row = dict
Metrics = dict[str, list[Row]]
Thetas = dict[str, list[float]]


def bitstring_to_fock(bitstring: str) -> Fock:
    """Maps a ``q0q1q2q3`` bitstring to the photon numbers (n1, n2); each mode is a binary number, q0 and q2 first."""
    return (
        int(bitstring[:QUBITS_PER_MODE], 2),
        int(bitstring[QUBITS_PER_MODE :], 2),
    )


def exact_fock_probs(theta: float) -> dict[Fock, float]:
    """Returns the exact probabilities of the (n1, n2) outcomes (the missing ones are 0) at the angle ``theta``."""
    from runner import compute_theoretical_probs

    return {bitstring_to_fock(state): p for state, p in compute_theoretical_probs(theta).items()}


def fock_probs(probs: dict[str, float]) -> dict[Fock, float]:
    """Turns a bitstring -> probability dictionary into a (n1, n2) -> probability one over all the outcomes."""
    result: dict[Fock, float] = dict.fromkeys(FOCK, 0.0)

    for state, p in probs.items():
        result[bitstring_to_fock(state)] += p

    return result


def sbc(p: dict[Fock, float], q: dict[Fock, float], outcomes: list[Fock]) -> float:
    """Returns the squared Bhattacharyya coefficient of ``p`` and ``q`` renormalised over the ``outcomes``."""
    p_norm: float = sum(p[k] for k in outcomes)
    q_norm: float = sum(q.get(k, 0.0) for k in outcomes)

    if p_norm == 0 or q_norm == 0:
        return 0.0

    return float(sum(np.sqrt(p[k] / p_norm * q.get(k, 0.0) / q_norm) for k in outcomes) ** 2)


def compute_row(probs: dict[str, float], theta: float) -> Row:
    """Derives the raw and post-selected probabilities, the kept fraction and the SBCs at the angle ``theta``."""
    p: dict[Fock, float] = fock_probs(probs)
    q: dict[Fock, float] = exact_fock_probs(theta)
    kept: float = sum(p[k] for k in SECTOR)

    return {
        "raw": {str(k): p[k] for k in SECTOR},
        "post": {str(k): p[k] / kept if kept > 0 else 0.0 for k in SECTOR},
        "SBC_raw": sbc(p, q, FOCK),
        "SBC_post": sbc(p, q, SECTOR),
        "kept": kept,
    }


def circuit_prefix(optimized: bool) -> str:
    """Returns the prefix of the results keys of the chosen circuit version."""
    return OPTIMIZED_PREFIX if optimized else ""


def load_results(directory: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Reads the runner results (step -> series -> bitstring -> probability) from the ``directory``."""
    with (directory / RESULTS_FILE).open(encoding="utf-8") as f:
        return json.load(f)


def series_rows(results: dict[str, dict[str, dict[str, float]]], key: str) -> tuple[list[float], list[Row]] | None:
    """
    Derives the rows of the series ``key``, ordered by theta, from the steps that have it; returns ``None`` if no
    step has it.
    """
    steps: list[int] = sorted(int(n) for n, step in results.items() if key in step)
    thetas: list[float] = [n * np.pi / 32 for n in steps]

    if not steps:
        return None

    return thetas, [compute_row(results[str(n)][key], theta) for n, theta in zip(steps, thetas)]


def pi_ticks(ax: Axes, max_theta: float) -> None:
    """Sets the x ticks at multiples of pi/8, labelled as fractions of pi."""
    ticks = np.arange(0, max_theta + 1e-9, np.pi / 8)

    if len(ticks) < 3:
        return

    labels: list[str] = []

    for i in range(len(ticks)):
        f = Fraction(i, 8)
        labels.append(
            "0"
            if i == 0
            else f"{'' if f.numerator == 1 else f.numerator}π" + ("" if f.denominator == 1 else f"/{f.denominator}")
        )

    ax.set_xticks(ticks, labels)


def point_style(name: str, color: str) -> dict:
    """Returns the marker style of the trace ``name``: large and hollow if raw, small and filled if mitigated."""
    mitigated: bool = name.endswith("mitigated")

    return dict(
        marker=MARKER,
        ls="none",
        ms=MITIGATED_MARKER_SIZE if mitigated else RAW_MARKER_SIZE,
        mew=1 if mitigated else 1.5,
        color=color,
        mfc=color if mitigated else "none",
        zorder=4 if mitigated else 3,
    )


def backend_legend(names: Sequence[str]) -> list[Line2D]:
    return [Line2D([], [], **point_style(name, "0.3"), label=BACKEND_LABELS[name]) for name in names]


def legend_groups(color_handles: list[Line2D], names: Sequence[str]) -> list[list[Line2D]]:
    """Groups the legend entries into the columns: the colours and the trace markers."""
    return [group for group in (color_handles, backend_legend(names)) if group]


def legend_bottom(fig: Figure, groups: list[list[Line2D]]) -> float:
    """Returns the fraction of the figure height to keep free below the panels for the tallest legend column."""
    return (LEGEND_ROW_HEIGHT * max(map(len, groups)) + LEGEND_PAD) / fig.get_figheight()


def add_legends(fig: Figure, groups: list[list[Line2D]]) -> None:
    """
    Draws every group as its own legend, side by side and centred below the panels, so that a group never takes
    entries from another one, however many entries each has. The font shrinks until the legends fit the figure width.
    """
    renderer = fig.canvas.get_renderer()
    font_size = LEGEND_FONT_SIZE

    while True:
        legends = [fig.legend(handles=group, loc="lower left", frameon=False, fontsize=font_size) for group in groups]
        widths = [legend.get_window_extent(renderer).width / fig.bbox.width for legend in legends]
        total = sum(widths) + LEGEND_GAP * (len(widths) - 1)

        if total <= LEGEND_MAX_WIDTH or font_size <= LEGEND_MIN_FONT_SIZE:
            break

        for legend in legends:
            legend.remove()

        font_size -= 0.5

    x = (1 - total) / 2

    for legend, width in zip(legends, widths):
        legend.set_bbox_to_anchor((x, LEGEND_PAD / 2 / fig.get_figheight()), transform=fig.transFigure)
        x += width + LEGEND_GAP


def make_title(circuit: str, n_samples: int) -> str:
    return rf"$|{N1}\rangle|{N2}\rangle$ through the tilted beam splitter ({circuit} circuit, {n_samples} shots)"


def plot_probabilities(
    metrics: Metrics, thetas: Thetas, title: str, n_samples: int, outcome_colors: dict[Fock, str]
) -> Figure:
    """Draws the raw and post-selected probabilities against theta: lines are exact, points are measured."""
    import matplotlib.pyplot as plt

    tt = np.linspace(0, max(max(t) for t in thetas.values()), EXACT_GRID_POINTS)
    exact_tt = [exact_fock_probs(t) for t in tt]
    styles = dict(zip(SECTOR, LINE_STYLES))
    panel = {
        "raw": (
            "raw: share of all shots\n"
            rf"(the {len(FOCK) - len(SECTOR)} outcomes with $n_1+n_2\neq{TOTAL}$ are not shown, so points sum to < 1)",
            rf"$P(n_1, n_2)$ = share of all {n_samples} shots"
            + "\n(mitigated: probability of the mitigated distribution)",
        ),
        "post": (
            rf"post-selected: only shots with $n_1+n_2={TOTAL}$ kept, renormalised to sum 1",
            rf"$P(n_1, n_2 \mid n_1+n_2={TOTAL})$" + "\n" + rf"= share of the shots with $n_1+n_2={TOTAL}$",
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, kind in zip(axes, ["raw", "post"]):
        for k in SECTOR:
            ax.plot(tt, [e.get(k, 0.0) for e in exact_tt], color=outcome_colors[k], ls=styles[k], lw=2, alpha=0.6)

            for name, rows in metrics.items():
                ax.plot(thetas[name], [r[kind][str(k)] for r in rows], **point_style(name, outcome_colors[k]))

        ax.set_xlabel(X_LABEL)
        ax.set_title(panel[kind][0], fontsize=10)
        ax.set_ylabel(panel[kind][1])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        pi_ticks(ax, tt[-1])

    outcome_legend = [
        Line2D(
            [],
            [],
            color=outcome_colors[k],
            ls=styles[k],
            lw=2,
            label=rf"$(n_1, n_2) = ({k[0]}, {k[1]})$: exact (line), measured (points)",
        )
        for k in SECTOR
    ]
    groups = legend_groups(outcome_legend, list(metrics))
    fig.suptitle(title + r"; $n_1, n_2$ = photons measured in mode 1, 2")
    fig.tight_layout(rect=(0, legend_bottom(fig, groups), 1, 1))
    add_legends(fig, groups)

    return fig


def plot_metrics(metrics: Metrics, thetas: Thetas, title: str) -> Figure:
    """Draws the SBCs with the exact distribution (top) and the kept fraction (bottom) against theta."""
    import matplotlib.pyplot as plt

    sbc_labels = {
        "SBC_raw": "raw: all 16 outcomes",
        "SBC_post": rf"post-selected: $n_1+n_2={TOTAL}$ outcomes, renormalised",
    }
    fig, (ax_sbc, ax_kept) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 2])

    for name, rows in metrics.items():
        for m, color in SBC_COLORS.items():
            ax_sbc.plot(thetas[name], [r[m] for r in rows], **point_style(name, color))
            ax_sbc.plot(thetas[name], [r[m] for r in rows], color=color, lw=1, alpha=0.4)

        ax_kept.plot(thetas[name], [r["kept"] for r in rows], **point_style(name, "tab:gray"))
        ax_kept.plot(thetas[name], [r["kept"] for r in rows], color="tab:gray", lw=1, alpha=0.4)

    ax_sbc.set_ylabel("fidelity with exact distribution\n" + r"SBC $= (\Sigma_k \sqrt{p_k\, q_k})^2$  (1 = ideal)")
    ax_sbc.set_title("measured $p$ vs exact $q$ (squared Bhattacharyya coefficient)", fontsize=10)
    ax_kept.set_ylabel("kept fraction\n" + rf"share of shots with $n_1+n_2={TOTAL}$ (ideal 1)")
    ax_kept.set_title(r"photon-number conservation: shots surviving post-selection", fontsize=10)
    ax_kept.set_xlabel(X_LABEL)

    for ax in (ax_sbc, ax_kept):
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        pi_ticks(ax, max(max(t) for t in thetas.values()))

    sbc_legend = [Line2D([], [], color=c, lw=2, label=sbc_labels[m]) for m, c in SBC_COLORS.items()]
    groups = legend_groups(sbc_legend, list(metrics))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, legend_bottom(fig, groups), 1, 1))
    add_legends(fig, groups)

    return fig


def save_figure(fig: Figure, directory: Path, name: str, circuit: str) -> None:
    for extension, options in (("png", dict(dpi=200)), ("pdf", {})):
        path: Path = directory / f"vlq_tilted_bs_article_{name}_{circuit}_plot.{extension}"
        fig.savefig(path, **options)
        print(f"Saved {path}")


def _positive_int(value: str) -> int:
    """Parses a command line ``value`` into a positive integer (``argparse`` reports the errors raised here)."""
    try:
        number: int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got '{value}'") from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")

    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plots the article figures (probabilities and metrics against theta) of the tilted beam splitter."
    )
    parser.add_argument(
        "--n-samples",
        type=_positive_int,
        default=DEFAULT_N_SAMPLES,
        help="number of shots of the runner results to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="plot the manually optimized circuits instead of the compiled ones (the default)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DIRECTORY,
        help="folder with the results JSON, where the figures are saved too (default: next to this script)",
    )
    parser.add_argument("--show", action="store_true", help="also open the figures in windows")

    return parser.parse_args(argv)


def collect_traces(directory: Path, n_samples: int, optimized: bool) -> tuple[Metrics, Thetas]:
    """Gathers the traces found for the chosen circuit version, in the order of the legend."""
    prefix: str = circuit_prefix(optimized)
    results = load_results(directory)
    keys: dict[str, str] = {
        SIMULATOR: SIMULATOR_KEY.format(n_samples=n_samples),
        SIMULATOR_MITIGATED: MITIGATED_SIMULATOR_KEY,
    }
    metrics: Metrics = {}
    thetas: Thetas = {}

    for name, key in keys.items():
        found = series_rows(results, f"{prefix}{key}")

        if found is not None:
            thetas[name], metrics[name] = found

    if not metrics:
        raise ValueError(f"No {prefix}{keys[SIMULATOR]} series in {directory / RESULTS_FILE}.")

    return metrics, thetas


def main(
    n_samples: int = DEFAULT_N_SAMPLES, optimized: bool = False, show: bool = False, directory: Path = DIRECTORY
) -> None:
    """Plots both figures of the chosen circuit version and saves them to the ``directory``."""
    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    circuit: str = "optimized" if optimized else "compiled"
    metrics, thetas = collect_traces(directory, n_samples, optimized)

    outcome_colors = dict(zip(SECTOR, plt.get_cmap("tab10").colors))
    title = make_title(circuit, n_samples)

    fig = plot_probabilities(metrics, thetas, title, n_samples, outcome_colors)
    save_figure(fig, directory, "probabilities", circuit)
    save_figure(plot_metrics(metrics, thetas, title), directory, "metrics", circuit)

    if show:
        plt.show()


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    main(args.n_samples, args.optimized, args.show, args.directory)
