import csv
import numpy as np

from paths import SEEDS, source_path, ensure_dirs

N_DIM = 10

def source(number_of_dim, distr="dirichlet", rng=None):
    # rng = None -> globala np.random (används av träningen, ingen extra kostnad per anrop).
    # rng = Generator -> seedad dragning, används när batch-filerna skapas.
    draw = rng if rng is not None else np.random
    if distr == "dirichlet":
        alpha_source = np.ones(number_of_dim)
        x0 = draw.dirichlet(alpha_source)
    elif distr == "gaussian":
        x0 = draw.normal(loc=0.0, scale=1.0, size=number_of_dim)
    else:
        raise ValueError(f"Unknown distr: {distr}")
    return x0

def create_csv_source(number_of_sources, filename, distr, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for _ in range(number_of_sources):
            one_list = source(N_DIM, distr, rng)
            writer.writerow(one_list)


def create_all_batches(distr, number_of_sources=10000, seeds=SEEDS):
    # Skapar alla källbatchar för en fördelning. Batch N får seed N, så filerna
    # alltid går att återskapa från koden.
    ensure_dirs()
    for seed in seeds:
        filename = source_path(seed, distr)
        create_csv_source(number_of_sources, filename, distr, seed=seed)
        print(f"wrote {filename}")
