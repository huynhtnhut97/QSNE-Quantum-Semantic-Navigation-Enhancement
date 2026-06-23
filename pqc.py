"""Six-qubit parameterized quantum circuit (PQC).

Implements the PQC described in Section 2.3 of the paper:
    - 6 qubits.
    - 4 variational layers.
    - Each layer: per-qubit R_X and R_Y rotations, then a CNOT linear chain
      between adjacent qubits.
    - Angle encoding: theta_i = pi * u_tilde_i for the first layer.
    - Measurement: <Z_i> on each qubit, returning a 6-D feature vector.

The PQC is built with PennyLane's torch interface so that gradients with
respect to the rotation angles flow through the same autograd graph as the
LSTM and the policy/value heads. PennyLane uses the parameter-shift rule
to differentiate the quantum kernel.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn

# Paper hyperparameters (Section 2.3 and the consolidated table).
NUM_QUBITS: int = 6
NUM_LAYERS: int = 4


def _build_device(num_qubits: int) -> qml.Device:
    """Construct a noiseless statevector simulator on `num_qubits` qubits."""
    return qml.device("default.qubit", wires=num_qubits)


def _make_circuit(num_qubits: int, num_layers: int):
    """Return a callable PennyLane QNode that evaluates the PQC.

    The QNode signature is `circuit(inputs, weights)` where:
        inputs : tensor of shape (num_qubits,)
            Angle-encoded values theta_i in [0, pi].
        weights : tensor of shape (num_layers, num_qubits, 2)
            Variational R_X and R_Y angles for each layer and qubit.
    """
    dev = _build_device(num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        # Angle encoding on the first layer: R_X by theta_i = pi * u_tilde_i.
        for q in range(num_qubits):
            qml.RX(inputs[q], wires=q)
        # Variational layers.
        for l in range(num_layers):
            for q in range(num_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
            # CNOT linear chain between adjacent qubits.
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        # Pauli-Z measurement on every qubit.
        return [qml.expval(qml.PauliZ(q)) for q in range(num_qubits)]

    return circuit


class PQCFeatureExtractor(nn.Module):
    """Quantum feature extractor wrapped as a PyTorch nn.Module.

    Input  : tensor of shape (batch, 6), entries in [0, 1].
    Output : tensor of shape (batch, 6), the Z-expectation values per qubit.

    The variational parameters are stored as a single nn.Parameter so that
    they are optimized jointly with the rest of the network under the PPO
    clipped objective.
    """

    def __init__(
        self,
        num_qubits: int = NUM_QUBITS,
        num_layers: int = NUM_LAYERS,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Small random initialization keeps the early policy near uniform
        # while still breaking symmetry between qubits.
        self.weights = nn.Parameter(
            init_scale * torch.randn(num_layers, num_qubits, 2)
        )
        self._circuit = _make_circuit(num_qubits, num_layers)

    @staticmethod
    def _angle_encode(u: torch.Tensor) -> torch.Tensor:
        """Map u in [0, 1]^6 to angles theta in [0, pi]^6."""
        return math.pi * u

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate the PQC for a batch of normalized inputs.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, 6)
            Normalized PQC inputs in [0, 1].

        Returns
        -------
        torch.Tensor, shape (batch, 6)
            Quantum feature vector f_t.
        """
        if u.dim() == 1:
            u = u.unsqueeze(0)
        assert u.shape[-1] == self.num_qubits
        theta = self._angle_encode(u)

        outs = []
        for i in range(theta.shape[0]):
            z_list = self._circuit(theta[i], self.weights)
            # PennyLane returns a list of tensors; stack into a row.
            row = torch.stack([z if torch.is_tensor(z) else torch.tensor(z)
                               for z in z_list])
            outs.append(row)
        return torch.stack(outs).to(dtype=u.dtype)
