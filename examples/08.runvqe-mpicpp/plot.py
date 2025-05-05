import json
import sys

import matplotlib.pyplot as plt
import numpy as np

assert len(sys.argv) == 2
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

filename = sys.argv[1]
assert filename.endswith(".ndjson")
label = filename[:-7]

vqe_energy = []
time_data = []
t_sum = 0

with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data["type"] == "info":
            if "mol_name" in data:
                mol_name = data["mol_name"]
            if "hf" in data:
                hf = data["hf"]["e_tot"]
            if "ccsd" in data:
                ccsd = data["ccsd"]["e_tot"]
            if "casci" in data:
                casci = data["casci"]["e_tot"]
            if "ansatz_name" in data:
                ansatz_name = data["ansatz_name"]
            if "n_qubits" in data:
                n_qubits = data["n_qubits"]
            if "mpi_size" in data:
                mpi_size = data["mpi_size"]
        elif data["type"] == "cost":
            cost = data["cost"]
            t = data["time"]["cost"]
            t_sum += t
            vqe_energy.append(cost)
            time_data.append(t_sum)
        elif data["type"] == "grad":
            t = data["time"]["grad"]
            t_sum += t


abs_energy = np.array(vqe_energy) - casci

plt.rcParams["font.size"] = 14
plt.figure()
plt.grid()
plt.ylim(min(1e-4, np.min(abs_energy) * 0.5), max(hf - casci, np.max(abs_energy)) * 2)
plt.yscale("log")
plt.title(f"Ansatz={ansatz_name},#Qubits={n_qubits},#GPUs={mpi_size}")
plt.plot(time_data, abs_energy, ".-", label="VQE")
plt.plot(time_data, [hf - casci] * len(time_data), linestyle="dashed", label="HF")
plt.plot(time_data, [ccsd - casci] * len(time_data), linestyle="dashed", label="CCSD")
plt.plot(time_data, [0.0016] * len(time_data), linestyle="dashed", label="CA")
plt.xlabel("Time [s]")
plt.ylabel("VQE Energy - CASCI")
plt.legend(
    bbox_to_anchor=(1, 1),
    loc="upper right",
)
plt.savefig(
    f"{label}.png",
    bbox_inches="tight",
    pad_inches=0.1,
)
