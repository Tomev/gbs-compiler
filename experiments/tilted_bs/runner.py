import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import (
    IQMFakeBackend,
)
from iqm_benchmark.simulator import FakeFromDicts
from mthree import M3Mitigation
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from gbs_compiler import CompilerGBS


def create_noisy_vlq_sim() -> IQMFakeBackend:
    architecture_dict_path = Path(__file__).with_name("vlq_dynamic_architecture.json")
    calibration_dict_path = Path(__file__).with_name("vlq_calibration.json")
    metrics_dict_path = Path(__file__).with_name("vlq_metrics.json")

    architecture_dict: dict[str, Any] = json.loads(architecture_dict_path.read_text())
    calibration_dict: dict[str, Any] = json.loads(calibration_dict_path.read_text())
    metrics_dict: dict[str, Any] = json.loads(metrics_dict_path.read_text())

    return FakeFromDicts(architecture_dict, calibration_dict, metrics_dict, "FakeVLQ")


def _get_qubit_measurement_error_matrices() -> dict[str, NDArray[np.floating]]:
    """TODO(TR): Docstring"""

    quality_dict: dict[str, Any]
    readout_errors: dict[str, dict[str, float]] = defaultdict(lambda: {})
    metrics_dict_path = Path(__file__).with_name("vlq_metrics.json")

    with metrics_dict_path.open() as f:
        quality_dict = json.load(f)

    for observation in quality_dict["observations"]:
        if "error_0_to_1" in observation["dut_field"]:
            component_name = observation["dut_field"].split(".")[-2]
            readout_errors[component_name]["0"] = observation["value"]

        if "error_1_to_0" in observation["dut_field"]:
            component_name = observation["dut_field"].split(".")[-2]
            readout_errors[component_name]["1"] = observation["value"]

    error_matrices: dict[str, NDArray[np.floating]] = {}

    for k, v in readout_errors.items():
        error_matrices[k] = np.array([[1 - v["0"], v["1"]], [v["0"], 1 - v["1"]]])

    return error_matrices


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


def compute_noiseless_probs_from_shots(qasm: str, shots: int = 10_000) -> dict[str, float]:
    template = qml.from_qasm(qasm, measurements=[])

    dev = qml.device("default.qubit", wires=4, shots=shots)

    @qml.qnode(dev)
    def circuit():
        template()
        return qml.counts(wires=range(4))

    counts = circuit()
    total = sum(counts.values())

    return filter_probs({str(state): float(count / total) for state, count in counts.items()})


def compute_noisy_probs_from_shots(open_qasm: str, n_shots: int = 10_000) -> dict[str, float]:
    backend: IQMFakeBackend = create_noisy_vlq_sim()
    circuit: QuantumCircuit = transpile_to_IQM(QuantumCircuit.from_qasm_str(open_qasm), backend)

    counts: dict[str, int] = backend.run(circuit, shots=n_shots).result().get_counts()
    total: int = sum(counts.values())

    # Qiskit bitstrings are little-endian; reverse them to match the q0q1q2q3 keys of the PennyLane-based results.
    return filter_probs({state[::-1]: count / total for state, count in counts.items()})


def apply_noise_mitigation(probs: dict[str, float], n_shots: int = 10_000) -> dict[str, Any]:

    layout: list[int] = [0, 1, 2, 3]  # We don't optimize qubit selection, so layout is [0, 1, 2, 3]
    counts = {k: int(v * n_shots) for k, v in probs.items()}

    error_matrices: dict[str, NDArray[np.floating]] = _get_qubit_measurement_error_matrices()

    # for k, v in error_matrices.items():
    #     print(f"\n\n{k}:\n\n{v}")
    backend = create_noisy_vlq_sim()
    matrices_for_layout: list[NDArray[np.floating]] = []

    for qubit_index in layout:
        qubit_name: str = backend.index_to_qubit_name(qubit_index)
        matrices_for_layout.append(error_matrices[qubit_name])

    m3: M3Mitigation = M3Mitigation()
    m3.cals_from_matrices(matrices_for_layout)

    # counts: list[dict[str, int]] = _to_qiskit_counts(summed_counts.raw_results)

    quasi = m3.apply_correction(counts, range(len(layout)))
    probs = quasi.nearest_probability_distribution()

    return filter_probs({k: float(v) for k, v in probs.items()})


def filter_probs(probs: dict[str, float]) -> dict[str, float]:

    return {k: v for k, v in probs.items() if v > 1e-20}


def counts_to_statevector(counts: dict[str, float]) -> NDArray[np.complexfloating]:
    """
    Creates a statevector with amplitudes ``sqrt(p)`` out of ``counts`` (or probabilities; the values get normalized).

    The keys are expected to be ``q0q1...`` bitstrings, as in the rest of the runner, so the result follows the
    PennyLane ordering (wire 0 is the most significant bit of the index). Counts carry no phase information, hence
    all the amplitudes are real and non-negative.
    """
    if not counts:
        raise ValueError("Cannot create a statevector out of empty counts.")

    n_qubits: int = len(next(iter(counts)))
    state: NDArray[np.complexfloating] = np.zeros(2**n_qubits, dtype=np.complex128)

    for bitstring, count in counts.items():
        if len(bitstring) != n_qubits:
            raise ValueError(f"Inconsistent bitstring lengths: expected {n_qubits}, got '{bitstring}'.")
        if count < 0:
            raise ValueError(f"Negative count for '{bitstring}': {count}.")
        state[int(bitstring, 2)] = np.sqrt(count)

    norm = np.linalg.norm(state)

    if norm == 0:
        raise ValueError("Cannot create a statevector out of all-zero counts.")

    return state / norm


def compute_fidelity_with_theory(probs: dict[str, float], theta: float) -> float:
    """
    Computes the fidelity between the state reconstructed from ``probs`` (e.g. the raw or the noise-mitigated VLQ
    simulation results) and the theoretical state for the tilted beam-splitter with parameter ``theta``.

    Both states are created with :func:`counts_to_statevector`, so they carry no phases and the result is the
    classical fidelity ``(sum_i sqrt(p_i * q_i))^2`` of the two distributions.
    """
    state: NDArray[np.complexfloating] = counts_to_statevector(probs)
    theoretical_state: NDArray[np.complexfloating] = counts_to_statevector(compute_theoretical_probs(theta))

    return float(qml.math.fidelity_statevector(state, theoretical_state))


def compute_fidelity_stderr(fidelity: float, n_samples: int) -> float:
    """
    Computes the standard error ``sqrt(F * (1 - F) / n_samples)`` of the ``fidelity`` estimated from ``n_samples``.

    It's the first-order (delta method) propagation of the multinomial shot noise, assuming that every outcome
    allowed by the theory has been observed. For the noise-mitigated results it's only an approximation.
    """
    # The fidelity can land a rounding error above 1, hence the max.
    return float(np.sqrt(max(fidelity * (1 - fidelity), 0.0) / n_samples))


def add_multinomial_stdevs(step_results: dict[str, dict[str, float]], n_samples: int) -> None:
    """Extends ``step_results`` with a ``<name>-stdev`` entry for every series other than the theoretical one."""
    for name, probs in list(step_results.items()):
        if name != "theoretical":
            # Exact simulations can land a rounding error above 1, hence the max.
            step_results[f"{name}-stdev"] = {
                state: float(np.sqrt(max(p * (1 - p), 0.0) / n_samples)) for state, p in probs.items()
            }


def main() -> None:
    # print(create_bb_circuit(np.pi / 4))
    # print(create_openqasm_circuit(create_bb_circuit(np.pi / 4)))
    # print(theoretical_counts(np.pi / 4))

    results: dict[int, dict[str, dict[str, float]]] = {}
    fidelity: dict[int, dict[str, float]] = {}
    n_samples: int = 10_000

    for n in range(17):
        results[n] = {}
        theta = n * np.pi / 32

        print(f"theta = {n} pi / 32")
        open_qasm_circuit: str = create_openqasm_circuit(create_bb_circuit(theta))
        results[n]["theoretical"] = compute_theoretical_probs(theta)
        results[n]["noiseless"] = compute_noiseless_probs(open_qasm_circuit)
        results[n][f"noiseless-{n_samples}-samples"] = compute_noiseless_probs_from_shots(open_qasm_circuit, n_samples)
        results[n][f"noisy-vlq-{n_samples}-samples"] = compute_noisy_probs_from_shots(open_qasm_circuit, n_samples)
        results[n]["noise_mitigated_noisy_vql"] = apply_noise_mitigation(
            results[n][f"noisy-vlq-{n_samples}-samples"], n_samples
        )
        add_multinomial_stdevs(results[n], n_samples)

        fidelity[n] = {
            "noiseless": compute_fidelity_with_theory(results[n][f"noiseless-{n_samples}-samples"], theta),
            "noisy": compute_fidelity_with_theory(results[n][f"noisy-vlq-{n_samples}-samples"], theta),
            "mitigated": compute_fidelity_with_theory(results[n]["noise_mitigated_noisy_vql"], theta),
        }

        for name in ("noiseless", "noisy", "mitigated"):
            fidelity[n][f"{name}-stderr"] = compute_fidelity_stderr(fidelity[n][name], n_samples)

        print(f"\t{results[n]['theoretical']}")
        print(f"\t{results[n]['noiseless']}")
        print(f"\t{results[n][f'noiseless-{n_samples}-samples']}")
        print(f"\t{results[n][f'noisy-vlq-{n_samples}-samples']}")
        print(f"\t{results[n]['noise_mitigated_noisy_vql']}")

        print(f"\t{fidelity[n]}")

    output = Path(__file__).with_name("tilted_bs_results.json")

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    fidelity_output = Path(__file__).with_name("tilted_bs_fidelity.json")

    with open(fidelity_output, "w") as f:
        json.dump(fidelity, f, indent=4)


if __name__ == "__main__":
    print("Start")
    main()
    print("End")
