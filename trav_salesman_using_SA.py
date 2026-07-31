"""
Traveling salesman problem solved using Simulated Annealing.
"""

# isort: off
import random

from geopy.geocoders import Nominatim
import numpy as np
# isort: on


###------------------------------------------------------------------
###------------------------------------------------------------------
def Distance(P1, P2):
    if P1 == P2:
        return 0.0

    d = np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2)
    return d


###------------------------------------------------------------------
###------------------------------------------------------------------
def TotalDistance(P, seq):
    dist = 0.0
    N = len(seq)
    for i in range(N):
        j = (i + 1) % N
        dist += Distance(P[seq[i]], P[seq[j]])
    return dist


###------------------------------------------------------------------
###------------------------------------------------------------------
def readCities(PNames):
    P = []  # coordinates of cities

    # Get coordinates of cities
    geolocator = Nominatim(user_agent="BWG_App")

    j = 0
    try:
        with open("India_cities.txt") as file:
            for line in file:
                city = line.rstrip("\n")
                if not city:
                    break

                theLocation = city + ", India"
                pt = geolocator.geocode(theLocation, timeout=10)
                if pt is None:
                    print(f"Warning: could not geocode {theLocation}")
                    continue

                y = round(pt.latitude, 2)
                x = round(pt.longitude, 2)
                print(f"City[{j:2d}] = {city} ({x:5.2f}, {y:5.2f})")
                P.append([x, y])
                PNames.append(city)
                j += 1
    except FileNotFoundError:
        print("City file 'India_cities.txt' not found.")

    return P


###------------------------------------------------------------------
###------------------------------------------------------------------
def Plot(seq, P, dist, PNames):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        print(f"Total distance = {dist}")
        return

    # Plot
    Pt = [P[seq[i]] for i in range(len(seq))]
    Pt += [P[seq[0]]]
    Pt = np.array(Pt)
    plt.figure(figsize=(10, 6))
    plt.title("Total distance = " + str(dist))
    plt.plot(Pt[:, 0], Pt[:, 1], "-o")
    plt.gca().set_aspect("equal", adjustable="box")

    for i in range(len(P)):
        plt.annotate(PNames[i], (P[i][0], P[i][1]))

    output_path = "trav_salesman_output.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    try:
        plt.show()
    except (RuntimeError, AttributeError, OSError):
        print("Plot displayed may not be available in this environment.")


###------------------------------------------------------------------
###------------------------------------------------------------------
# Moves N1 (a position in seq) to sit right after N2 (another position in
# seq).  This is the "city transport" move used alongside segment reversal.
# Returns (seq, dist, accepted) where accepted tells the caller whether the
# Metropolis criterion accepted the move (so the outer loop can count it
# towards maxAccepted).
def swap(P, seq, dist, N1, N2, temp, nCity):

    N1L = N1 - 1
    if N1L < 0:
        N1L += nCity
    N1R = N1 + 1
    if N1R >= nCity:
        N1R = 0

    N2L = N2 - 1
    if N2L < 0:
        N2L += nCity
    N2R = N2 + 1
    if N2R >= nCity:
        N2R = 0

    # Nothing to do if the same position, or if they are already adjacent
    # (moving a city next to itself doesn't change the tour)
    if N1 == N2 or N2 == N1R or N1 == N2R:
        return seq, dist, False

    c1 = seq[N1]
    c1L = seq[N1L]
    c1R = seq[N1R]
    c2 = seq[N2]
    c2R = seq[N2R]

    # Cost of removing c1 from between c1L/c1R, and inserting it
    # between c2 and c2R
    oldCost = (
        Distance(P[c1L], P[c1])
        + Distance(P[c1], P[c1R])
        + Distance(P[c2], P[c2R])
    )
    newCost = (
        Distance(P[c1L], P[c1R]) + Distance(P[c2], P[c1]) + Distance(P[c1], P[c2R])
    )

    deltaDist = newCost - oldCost

    accept = False
    if deltaDist < 0.0:
        accept = True
    else:
        if random.random() < np.exp(-deltaDist / temp):
            accept = True

    if accept:
        newSeq = list(seq)
        newSeq.remove(c1)
        pos2 = newSeq.index(c2)
        newSeq.insert(pos2 + 1, c1)
        seq = np.array(newSeq)
        dist += deltaDist
        return seq, dist, True

    return seq, dist, False


###------------------------------------------------------------------
###------------------------------------------------------------------
# Reverses the segment of seq lying between positions N1 and N2 (inclusive).
# Only the 4 edges touching the ends of the segment change, so the distance
# change can be computed in O(1) instead of recomputing the whole tour.
def reverse(P, seq, dist, N1, N2, temp, nCity):

    N1L = N1 - 1
    if N1L < 0:
        N1L += nCity
    N2R = N2 + 1
    if N2R >= nCity:
        N2R = 0

    if N1 == N2:
        return seq, dist, False

    c1L = seq[N1L]
    c1 = seq[N1]
    c2 = seq[N2]
    c2R = seq[N2R]

    oldCost = Distance(P[c1L], P[c1]) + Distance(P[c2], P[c2R])
    newCost = Distance(P[c1L], P[c2]) + Distance(P[c1], P[c2R])
    deltaDist = newCost - oldCost

    accept = False
    if deltaDist < 0.0:
        accept = True
    else:
        if random.random() < np.exp(-deltaDist / temp):
            accept = True

    if accept:
        seq[N1 : N2 + 1] = seq[N1 : N2 + 1][::-1]
        dist += deltaDist
        return seq, dist, True

    return seq, dist, False


###------------------------------------------------------------------
###------------------------------------------------------------------
if __name__ == "__main__":
    PNames = []  # names of cities

    P = readCities(PNames)
    nCity = len(P)  # Number of cities to visit

    if nCity < 2:
        print("Need at least two cities to solve TSP.")
        raise SystemExit(1)

    maxTsteps = 250  # Temperature is lowered maxTsteps times
    fCool = 0.9  # Factor to multiply temperature at each cooling step
    maxSwaps = 2000  # Number of swaps at constant temperature

    maxAccepted = 10 * nCity  # Number of accepted steps at constant temperature

    seq = np.arange(0, nCity, 1)

    dist = TotalDistance(P, seq)

    temp = 10.0 * dist  # Starting temperature - has to be high enough

    print("\n\n")
    print(seq)
    print(f"\n nCity= {nCity:3d}  dist= {dist:f}  temp= {temp:f} \n")
    input("Press Enter to continue...")

    Plot(seq, P, dist, PNames)

    oldDist = 0.0
    convergenceCount = 0

    for t in range(1, maxTsteps + 1):

        if temp < 1.0e-6:
            break

        nAccepted = 0

        for k in range(maxSwaps):

            # Pick two distinct random positions in seq, N1 < N2
            N1 = random.randint(0, nCity - 1)
            N2 = random.randint(0, nCity - 1)
            if N1 == N2:
                continue
            if N1 > N2:
                N1, N2 = N2, N1

            if random.random() < 0.5:
                # Try a segment-reversal move
                seq, dist, accepted = reverse(P, seq, dist, N1, N2, temp, nCity)
            else:
                # Try a city-transport ("swap") move
                seq, dist, accepted = swap(P, seq, dist, N1, N2, temp, nCity)

            if accepted:
                nAccepted += 1

            if nAccepted >= maxAccepted:
                break

        ## dist = TotalDistance(P, seq)

        print(f"Iteration: {t:d}  temp={temp:f} dist={dist:f}")
        print("seq = ")
        np.set_printoptions(precision=3)
        print(seq)
        print("\n\n")

        if abs(dist - oldDist) < 1.0e-4:
            convergenceCount += 1
        else:
            convergenceCount = 0

        if convergenceCount >= 4:
            break

        if (t % 25) == 0:
            Plot(seq, P, dist, PNames)

        temp *= fCool
        oldDist = dist

    Plot(seq, P, dist, PNames)