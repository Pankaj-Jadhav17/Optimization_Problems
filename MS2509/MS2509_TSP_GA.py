"""
=====================================================================
MS2509 - Traveling Salesman Problem using Genetic Algorithm
=====================================================================
Objective : Minimize the total distance traveled visiting all cities
Data File : india_cities_GA.txt  (Indian cities with coordinates)
Outputs   :
    1. Path-tracing graphs at intermediate generations with city names
    2. Average fitness vs. generation plot at the end
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ======================== CONFIGURATION ========================
POPULATION_SIZE   = 200       # Number of chromosomes in the population
NUM_GENERATIONS   = 500       # Total number of generations
ELITE_FRACTION    = 0.10      # Top fraction preserved via elitism
MUTATION_RATE     = 0.02      # Per-gene swap mutation probability
TOURNAMENT_SIZE   = 5         # Tournament selection pool size
DATA_FILE         = "india_cities_GA.txt"
SNAPSHOT_GENS     = [1, 50, 100, 200, 350, 500]  # Generations to plot route
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ======================== DATA LOADING ========================
def load_cities(filepath):
    """
    Read city data from a text file.
    Expected format per line:  CityName  X_coord  Y_coord
    Lines starting with '#' are treated as comments and skipped.
    Returns:
        city_names : list of str
        coords     : numpy array of shape (n, 2)
    """
    city_names = []
    coords = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            city_names.append(parts[0])
            coords.append([float(parts[1]), float(parts[2])])
    return city_names, np.array(coords)


# ======================== FITNESS / DISTANCE ========================
def euclidean_distance_matrix(coords):
    """Compute pairwise Euclidean distance matrix."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i][j] = d
            dist[j][i] = d
    return dist


def route_distance(route, dist_matrix):
    """Total round-trip distance for a given route (returns to start)."""
    total = 0.0
    for i in range(len(route)):
        total += dist_matrix[route[i]][route[(i + 1) % len(route)]]
    return total


def fitness(route, dist_matrix):
    """Fitness = 1 / total distance  (higher is better)."""
    return 1.0 / route_distance(route, dist_matrix)


# ======================== GA OPERATORS ========================
def create_individual(n):
    """Create a random permutation of city indices."""
    ind = list(range(n))
    random.shuffle(ind)
    return ind


def initialize_population(pop_size, n_cities):
    """Generate initial random population."""
    return [create_individual(n_cities) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """Select one individual via tournament selection."""
    indices = random.sample(range(len(population)), k)
    best = max(indices, key=lambda i: fitnesses[i])
    return population[best][:]


def order_crossover(parent1, parent2):
    """
    Order Crossover (OX):
    Select a subsequence from parent1 and fill remaining positions
    from parent2 preserving relative order.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    # Copy segment from parent1
    child[start:end + 1] = parent1[start:end + 1]

    # Fill from parent2
    fill_values = [gene for gene in parent2 if gene not in child[start:end + 1]]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[idx]
            idx += 1
    return child


def mutate(individual, mutation_rate=MUTATION_RATE):
    """Swap mutation: with given probability, swap two random cities."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


# ======================== VISUALIZATION ========================
def plot_route(route, coords, city_names, generation, distance, ax=None, save=False):
    """Plot the TSP route on a 2D map with city names labeled."""
    show_standalone = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Build ordered coordinates (close the loop)
    ordered = [route[i] for i in range(len(route))]
    ordered.append(route[0])
    xs = [coords[c][0] for c in ordered]
    ys = [coords[c][1] for c in ordered]

    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=1.8, markersize=8,
            markerfacecolor="#FF5722", markeredgecolor="white", markeredgewidth=1.2)

    # Annotate city names
    for idx in route:
        ax.annotate(
            city_names[idx],
            (coords[idx][0], coords[idx][1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="bold",
            color="#1a1a2e",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffffcc", ec="#cccccc", lw=0.5),
        )

    ax.set_title(
        f"Generation {generation}  |  Distance = {distance:.2f}",
        fontsize=13, fontweight="bold", color="#1a1a2e",
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#f5f5f5")

    if show_standalone and save:
        plt.tight_layout()
        plt.savefig(f"route_gen_{generation}.png", dpi=150)
        plt.close()


def plot_fitness_history(avg_fitness_list, best_fitness_list):
    """Plot average and best fitness across generations."""
    gens = range(1, len(avg_fitness_list) + 1)

    fig, ax1 = plt.subplots(figsize=(11, 6))

    color_avg = "#2196F3"
    color_best = "#FF5722"

    ax1.plot(gens, avg_fitness_list, color=color_avg, linewidth=1.5,
             label="Average Fitness", alpha=0.85)
    ax1.plot(gens, best_fitness_list, color=color_best, linewidth=1.5,
             label="Best Fitness", alpha=0.85)
    ax1.set_xlabel("Generation", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Fitness  (1 / Distance)", fontsize=12, fontweight="bold")
    ax1.set_title("Fitness Progression over Generations", fontsize=14, fontweight="bold",
                  color="#1a1a2e")
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_facecolor("#f5f5f5")

    # Secondary y-axis: show distance equivalent for best
    ax2 = ax1.twinx()
    best_distances = [1.0 / f for f in best_fitness_list]
    ax2.plot(gens, best_distances, color="#4CAF50", linewidth=1.5,
             linestyle="--", label="Best Distance", alpha=0.7)
    ax2.set_ylabel("Best Distance", fontsize=12, fontweight="bold", color="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("fitness_progression.png", dpi=150)
    plt.show()


# ======================== MAIN GA LOOP ========================
def genetic_algorithm_tsp():
    """Run the Genetic Algorithm for the Traveling Salesman Problem."""

    # --- Load data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, DATA_FILE)
    city_names, coords = load_cities(filepath)
    n_cities = len(city_names)

    print("=" * 65)
    print("   TRAVELING SALESMAN PROBLEM — GENETIC ALGORITHM")
    print("=" * 65)
    print(f"   Data file       : {DATA_FILE}")
    print(f"   Number of cities: {n_cities}")
    print(f"   Population size : {POPULATION_SIZE}")
    print(f"   Generations     : {NUM_GENERATIONS}")
    print(f"   Mutation rate   : {MUTATION_RATE}")
    print(f"   Elite fraction  : {ELITE_FRACTION}")
    print(f"   Tournament size : {TOURNAMENT_SIZE}")
    print("=" * 65)
    print(f"\n   Cities: {', '.join(city_names)}\n")

    # --- Precompute distance matrix ---
    dist_matrix = euclidean_distance_matrix(coords)

    # --- Initialize population ---
    population = initialize_population(POPULATION_SIZE, n_cities)
    n_elite = max(1, int(ELITE_FRACTION * POPULATION_SIZE))

    # --- Tracking ---
    avg_fitness_history = []
    best_fitness_history = []
    snapshots_taken = set()

    # Determine subplot layout for route snapshots
    n_snapshots = len(SNAPSHOT_GENS)
    ncols = 3
    nrows = (n_snapshots + ncols - 1) // ncols
    fig_routes, axes_routes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes_routes = axes_routes.flatten()
    snapshot_idx = 0

    # --- Evolution ---
    for gen in range(1, NUM_GENERATIONS + 1):
        # Evaluate fitness
        fit_values = [fitness(ind, dist_matrix) for ind in population]
        distances = [route_distance(ind, dist_matrix) for ind in population]

        avg_fit = np.mean(fit_values)
        best_idx = np.argmax(fit_values)
        best_fit = fit_values[best_idx]
        best_dist = distances[best_idx]

        avg_fitness_history.append(avg_fit)
        best_fitness_history.append(best_fit)

        # Print progress
        if gen == 1 or gen % 50 == 0 or gen == NUM_GENERATIONS:
            print(f"   Gen {gen:>4d}  |  Best Distance: {best_dist:>10.4f}  |"
                  f"  Avg Fitness: {avg_fit:.6f}  |  Best Fitness: {best_fit:.6f}")

        # Snapshot route at specified generations
        if gen in SNAPSHOT_GENS and gen not in snapshots_taken:
            snapshots_taken.add(gen)
            if snapshot_idx < len(axes_routes):
                plot_route(population[best_idx], coords, city_names,
                           gen, best_dist, ax=axes_routes[snapshot_idx])
                snapshot_idx += 1

        # --- Selection & reproduction ---
        # Sort population by fitness (descending)
        sorted_pairs = sorted(zip(fit_values, population), key=lambda x: x[0], reverse=True)
        sorted_population = [ind for _, ind in sorted_pairs]
        sorted_fitnesses = [f for f, _ in sorted_pairs]

        new_population = []

        # Elitism: carry over top individuals
        for i in range(n_elite):
            new_population.append(sorted_population[i][:])

        # Fill rest via crossover + mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fit_values)
            parent2 = tournament_selection(population, fit_values)
            child = order_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # --- Hide unused subplot axes ---
    for i in range(snapshot_idx, len(axes_routes)):
        axes_routes[i].set_visible(False)

    fig_routes.suptitle(
        "TSP Route Evolution via Genetic Algorithm",
        fontsize=16, fontweight="bold", color="#1a1a2e", y=1.01,
    )
    fig_routes.tight_layout()
    fig_routes.savefig("tsp_route_snapshots.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Final results ---
    final_fitnesses = [fitness(ind, dist_matrix) for ind in population]
    best_idx = np.argmax(final_fitnesses)
    best_route = population[best_idx]
    best_distance = route_distance(best_route, dist_matrix)

    print("\n" + "=" * 65)
    print("   FINAL RESULT")
    print("=" * 65)
    print(f"   Best Distance : {best_distance:.4f}")
    print(f"   Best Route    :")
    route_names = " → ".join(city_names[c] for c in best_route)
    route_names += f" → {city_names[best_route[0]]}"
    print(f"   {route_names}")
    print("=" * 65)

    # --- Plot fitness history ---
    plot_fitness_history(avg_fitness_history, best_fitness_history)

    return best_route, best_distance


# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    best_route, best_distance = genetic_algorithm_tsp()
