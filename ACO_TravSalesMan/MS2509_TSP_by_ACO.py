# Traveling Salesman Problem solved by Ant Colony Optimization (ACO)
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- 1. Read data: each line -> CityName,latitude,longitude ----------
def load_cities(file_path):
    names, coords = [], []
    with open(file_path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                longitude = float(parts[-1])
                latitude = float(parts[-2])
            except ValueError:
                continue
            city_name = " ".join(parts[:-2]).strip()
            if not city_name:
                city_name = f"City_{len(names) + 1}"
            names.append(city_name)
            coords.append([longitude, latitude])

    if not names:
        raise ValueError(f"No valid city data found in {file_path}")

    return names, np.asarray(coords, dtype=float)


file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "India_cities.txt")
names, xy = load_cities(file)
print(f"Loaded {len(names)} cities")
D = np.maximum(np.linalg.norm(xy[:, None] - xy[None], axis=2) * 111.0, 1e-6)  # approx. km
n = len(names)

# ---------- 2. ACO parameters ----------
ants, iters = 30, 100
alpha, beta, rho, Q = 1.0, 5.0, 0.5, 100.0
tau = np.ones((n, n))
eta = 1.0 / D
best_path, best_len, history, snaps = None, np.inf, [], []
show_at = {1, 5, 20, 50, iters}


def length(path):
    return float(sum(D[path[i], path[(i + 1) % n]] for i in range(n)))


# ---------- 3. Main ACO loop ----------
rng = np.random.default_rng(42)
for it in range(1, iters + 1):
    paths = []
    for _ in range(ants):
        path = [int(rng.integers(n))]
        while len(path) < n:
            cur = path[-1]
            prob = (tau[cur] ** alpha) * (eta[cur] ** beta)
            prob[np.array(path, dtype=int)] = 0.0

            if np.all(prob <= 0):
                remaining = [idx for idx in range(n) if idx not in path]
                if not remaining:
                    break
                path.append(int(rng.choice(remaining)))
                continue

            p = prob / prob.sum()
            next_city = int(rng.choice(n, p=p))
            if next_city in path:
                remaining = [idx for idx in range(n) if idx not in path]
                if remaining:
                    next_city = int(rng.choice(remaining))
                else:
                    break
            path.append(next_city)

        if len(path) == n:
            paths.append(path)

    tau *= (1 - rho)
    for path in paths:
        L = length(path)
        for i in range(n):
            a, b = path[i], path[(i + 1) % n]
            tau[a, b] += Q / L
            tau[b, a] += Q / L
        if L < best_len:
            best_len = L
            best_path = path[:]

    history.append(best_len)
    if it in show_at and best_path is not None:
        snaps.append((it, best_path[:], best_len))
    print(f"Iteration {it:3d}  best distance = {best_len:.1f} km")

if best_path is None:
    raise RuntimeError("ACO failed to build a valid tour.")

# ---------- 4. Path tracing graphs with city names at intermediate stages ----------
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
axs = axs.ravel()
for ax, (it, p, L) in zip(axs, snaps):
    q = p + [p[0]]
    ax.plot(xy[q, 0], xy[q, 1], "b-o", lw=1.5)
    ax.plot(*xy[p[0]], "rs", ms=10)
    for c in p:
        ax.annotate(names[c], xy[c], fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"Iteration {it}: distance = {L:.0f} km")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

for ax in axs[len(snaps):]:
    ax.axis("off")

axs[-1].plot(range(1, iters + 1), history, "g-")
axs[-1].set_title("Convergence")
axs[-1].set_xlabel("Iteration")
axs[-1].set_ylabel("Best distance (km)")

plt.tight_layout()
plt.savefig("TSP_ACO_result.png", dpi=120)
plt.close(fig)

# ---------- 5. Final result ----------
route = best_path + [best_path[0]]
print("\nBest route:", " -> ".join(names[c] for c in route))
print(f"Total distance = {best_len:.1f} km")
print("Saved plot to TSP_ACO_result.png")