from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2

OUTPUT_DIR: Path = Path(__file__).parent / "openqasm_scripts" / "optimized" / "manual"

# Number of the circuits, the same as in ``runner.py``: ``tilted_bs_n_00.qasm`` ... ``tilted_bs_n_16.qasm``.
N_CIRCUITS: int = 17


def generate_circuit(n: int) -> QuantumCircuit:
    """
    Returns the manually optimized tilted BS circuit with the index ``n``, including the measurements, with the same
    register layout as the original files (``qreg q[4]; creg c[4];``).
    """
    theta: float = n * np.pi / 32

    tilted_bs_qc: QuantumCircuit = QuantumCircuit(4, 4)
    tilted_bs_qc = tilted_bs_qc & _state_preparation()
    tilted_bs_qc.append(_M(), [2, 3])
    tilted_bs_qc = tilted_bs_qc & _V_even(theta)
    tilted_bs_qc.append(_M().inverse(), [2, 3])
    tilted_bs_qc = tilted_bs_qc & _final_cxs()
    tilted_bs_qc.measure(range(4), range(4))
    return tilted_bs_qc


def _state_preparation() -> QuantumCircuit:
    """ """
    qc: QuantumCircuit = QuantumCircuit(4, 4)
    qc.x(1)
    qc.x(3)
    qc.cx(3, 1)
    qc.cx(3, 0)
    return qc


def _M() -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(2)
    qc.s(0)
    qc.s(1)
    qc.h(1)
    qc.cx(1, 0)
    return qc


def _V_even(theta: float) -> QuantumCircuit:

    xi_p: float = -theta * (np.sqrt(3) + 1) / 2
    xi_m: float = -theta * (np.sqrt(3) - 1) / 2

    qc: QuantumCircuit = QuantumCircuit(4, 4)
    qc.cy(0, 2)

    qc.rx(-np.pi / 4, 2)
    qc.rx(np.pi / 4, 3)

    qc.ry(2 * xi_p, 2)
    qc.ry(2 * xi_p, 3)

    qc.cx(0, 2)
    qc.cx(0, 3)

    qc.ry(2 * xi_m, 2)
    qc.ry(2 * xi_m, 3)

    qc.cx(0, 2)
    qc.cx(0, 3)

    qc.rx(np.pi / 4, 2)
    qc.rx(-np.pi / 4, 3)

    qc.cy(0, 2)

    return qc


def _final_cxs() -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(4, 4)
    qc.cx(3, 0)
    qc.cx(3, 1)
    qc.cx(2, 0)
    return qc


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in range(N_CIRCUITS):
        target: Path = OUTPUT_DIR / f"tilted_bs_n_{n:02d}.qasm"
        circuit: QuantumCircuit = generate_circuit(n)
        # print(circuit)
        target.write_text(qasm2.dumps(circuit), encoding="utf-8")
        print(f"Saved {target.name}")


if __name__ == "__main__":
    main()
