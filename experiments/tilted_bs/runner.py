import json
from pathlib import Path

import numpy as np
import pennylane as qml

from gbs_compiler import CompilerGBS


def create_bb_circuit(theta: float) -> str:
    """
    Crates a ``blackbird`` script with a |1>|1> state, a single tilted beam-splitter with parameter ``theta`` and a
    particle number resolving measurement.
    """
    return f"name TwoPhotonTiltedBS\nversion 1.0\n\nFock(1) | 0\nFock(1) | 1\n\nBSgate({theta}, 0) | (0, 1)"


def create_openqasm_circuit(bb_circuit: str, fock_cutoff: int = 4) -> str:
    compiler: CompilerGBS = CompilerGBS(fock_cutoff=fock_cutoff)
    return compiler.compile_to_qasm(bb_circuit)


def compute_theoretical_probs(theta: float) -> dict[str, float]:
    return filter_probs(
        {
            "1000": float(0.5 * np.sin(2 * theta) ** 2),
            "0101": float(np.cos(2 * theta) ** 2),
            "0010": float(0.5 * np.sin(2 * theta) ** 2),
        }
    )


def compute_noiseless_probs(open_qasm: str) -> dict[str, float]:

    template = qml.from_qasm(open_qasm, measurements=[])

    dev = qml.device("default.qubit", wires=4, shots=None)

    @qml.qnode(dev)
    def circuit():
        template()
        return qml.probs(wires=[0, 1, 2, 3])

    probs = circuit()

    probs_dict = {format(i, f"0{len(dev.wires)}b"): float(p) for i, p in enumerate(probs)}

    return filter_probs(probs_dict)


def compute_noiseless_probs_from_counts(qasm: str, shots: int = 10_000) -> dict[str, float]:
    template = qml.from_qasm(qasm, measurements=[])

    dev = qml.device("default.qubit", wires=4, shots=shots)

    @qml.qnode(dev)
    def circuit():
        template()
        return qml.counts(wires=range(4))

    counts = circuit()
    total = sum(counts.values())

    return filter_probs({str(state): float(count / total) for state, count in counts.items()})


def filter_probs(probs: dict[str, float]) -> dict[str, float]:

    return {k: v for k, v in probs.items() if v > 1e-20}


def main() -> None:
    # print(create_bb_circuit(np.pi / 4))
    # print(create_openqasm_circuit(create_bb_circuit(np.pi / 4)))
    # print(theoretical_counts(np.pi / 4))

    results: dict[int, dict[str, dict[str, float]]] = {}
    n_samples: int = 10_000

    for n in range(17):
        results[n] = {}
        theta = n * np.pi / 32

        print(f"theta = {n} pi / 32")
        open_qasm_circuit: str = create_openqasm_circuit(create_bb_circuit(theta))
        results[n]["theoretical"] = compute_theoretical_probs(theta)
        results[n]["noiseless"] = compute_noiseless_probs(open_qasm_circuit)
        results[n][f"noiseless-{n_samples}-samples"] = compute_noiseless_probs_from_counts(open_qasm_circuit, n_samples)

        print(f"\t{results[n]['theoretical']}")
        print(f"\t{results[n]['noiseless']}")
        print(f"\t{results[n][f'noiseless-{n_samples}-samples']}")

    output = Path(__file__).with_name("tilted_bs_results.json")

    with open(output, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
