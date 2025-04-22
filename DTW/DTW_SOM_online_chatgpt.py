import numpy as np
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt

# ====================
# Parameter
# ====================
n_rows, n_cols = 2, 2              # SOM-Gitter (2x2)
n_nodes = n_rows * n_cols
burst_length = 40                 # Zeitpunkte pro Burst
dimensions = 4                    # 4D-Vektoren
learning_rate = 0.1
neighborhood_radius = 1           # Einflussreichweite (in Gitterdistanz)
np.random.seed(42)

# ====================
# Helper: Gridposition & Nachbarschaft
# ====================
def grid_position(index):
    return np.array([index // n_cols, index % n_cols])

def neighborhood_strength(winner_idx, other_idx):
    d = np.linalg.norm(grid_position(winner_idx) - grid_position(other_idx))
    return np.exp(-d**2 / (2 * neighborhood_radius**2))

# ====================
# Initialisierung der Prototypen
# ====================
prototypes = [np.random.randn(burst_length, dimensions) for _ in range(n_nodes)]

# ====================
# Beispielhafte Bursts (Echtzeit-Daten simuliert)
# ====================
def generate_burst():
    base = np.cumsum(np.random.randn(burst_length, dimensions), axis=0)
    return base + np.random.normal(scale=0.5, size=(burst_length, dimensions))

# ====================
# DTW-Update Schritt
# ====================
def update_prototype(prototype, burst, lr):
    path, _ = dtw_path(prototype, burst)
    updated = prototype.copy()
    for (i, j) in path:
        updated[i] += lr * (burst[j] - prototype[i])
    return updated

# ====================
# Trainingsschleife (Online)
# ====================
classified_labels = []  # speichert Zuordnungen von Bursts zu Prototypen

for t in range(100):  # 100 eingehende Datenbursts
    new_burst = generate_burst()

    # 1. Finde besten Prototyp (BMU)
    distances = [dtw_path(p, new_burst)[1] for p in prototypes]
    bmu_idx = np.argmin(distances)
    classified_labels.append(bmu_idx)

    # 2. Aktualisiere alle Prototypen mit Nachbarschaftsgewichtung
    new_prototypes = []
    for i, proto in enumerate(prototypes):
        h = neighborhood_strength(bmu_idx, i)
        updated = update_prototype(proto, new_burst, learning_rate * h)
        new_prototypes.append(updated)
    prototypes = new_prototypes

# ====================
# Visualisierung: Erste Dimension der Prototypen
# ====================
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 6))
for i, proto in enumerate(prototypes):
    row, col = grid_position(i)
    axs[row, col].plot(proto[:, 0])  # nur erste Dimension
    axs[row, col].set_title(f"Knoten {i} (1. Dim)")
plt.tight_layout()
plt.show()

# ====================
# Klassifikation neuer Bursts nach Training
# ====================
def classify_burst(burst):
    distances = [dtw_path(p, burst)[1] for p in prototypes]
    return np.argmin(distances)

# Beispiel
test_burst = generate_burst()
assigned_cluster = classify_burst(test_burst)
print(f"Test-Burst wurde Cluster {assigned_cluster} zugeordnet.")
