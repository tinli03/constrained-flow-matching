# Single source of truth for experiment filenames.
# Both the producer (generate_samples.py) and the consumer (evaluation.py)
# build every path through here, so the two can never drift apart.

import os

SEEDS = [1, 2, 3, 4, 5]

# method key -> (filename stem, label used in printouts)
METHODS = {
    "unc": ("unconstrained", "unconstrained generated points"),
    "fpr": ("finalprojection", "final projection generated points"),
    "sbs": ("stepbystepprojection", "step by step generated points"),
}

DISTR_TAG = {"dirichlet": "d", "gaussian": "g"}

# Kataloger. Varje fördelning får en egen undermapp, t.ex. batches/gaussian/.
# Skapas automatiskt av ensure_dirs() innan något skrivs.
BATCH_DIR = "batches"        # källpunkterna (input till samplingen)
GENERATED_DIR = "generated"  # genererade punkter (output från samplingen)

CHECKPOINT_DIR = "checkpoints"

TARGET_PATH = "target.csv"


def checkpoint_path(distr="dirichlet"):
    # En modell per källfördelning. Modellen lär sig ett hastighetsfält från EN
    # bestämd startfördelning, så dirichlet- och gaussmodellen får aldrig blandas ihop.
    return os.path.join(CHECKPOINT_DIR, f"model_{distr}.pt")


def loss_curve_path(distr="dirichlet"):
    return f"loss_curve_{distr}.png"


def source_path(seed, distr="dirichlet"):
    name = f"{DISTR_TAG[distr]}_batch{seed}_dat.csv"
    return os.path.join(BATCH_DIR, distr, name)


def generated_path(seed, method, distr="dirichlet"):
    name = f"{DISTR_TAG[distr]}_{seed}_{METHODS[method][0]}_generated.csv"
    return os.path.join(GENERATED_DIR, distr, name)


def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for distr in DISTR_TAG:
        os.makedirs(os.path.join(BATCH_DIR, distr), exist_ok=True)
        os.makedirs(os.path.join(GENERATED_DIR, distr), exist_ok=True)


def has_generated(distr):
    # Finns det en färdig körning för den här fördelningen?
    return all(
        os.path.exists(generated_path(seed, method, distr))
        for seed in SEEDS
        for method in METHODS
    )


def row_index(seed, method):
    # Position of one (seed, method) run inside the method-major all_lists table
    # written by evaluation_list_to_csv(). Keep in sync with that loop order.
    return list(METHODS).index(method) * len(SEEDS) + SEEDS.index(seed)
