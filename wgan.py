import csv
from itertools import product

import torch
from torch import nn


####################################################
#                    CONSTANTS                     #
####################################################

# Device and paths
DEVICE = "cpu"
RESULTS = "eight_gaussian_final_results.csv"
SUMMARY = "eight_gaussian_final_summary.csv"

# Problem
RADIUS = 2.0
DATA_STD = 0.08
LAMBDA_GP = 1e-4
LAMBDA_D = 0.02
LAMBDA_G = 0.02
DIRECTIONS = 64              # Number of unit directions used for estimating SWD
QUALITY_RADIUS = 0.35        # Radius for which we consider a generated point to be close to a center
MINIMUM_MODE_FRACTION = 0.01 # Minimum fraction of generated points close to a center to consider that mode covered

angles = torch.arange(8, dtype=torch.float32, device=DEVICE) * 2 * torch.pi / 8
CENTERS = RADIUS * torch.stack((angles.cos(), angles.sin()), dim=1)

# Optimizer
STEPS = 1024
PREDICTOR_BATCH = 32
CORRECTOR_BATCH = 1024
BETA = 0.9

# Grid search
GAMMAS = (0.0125, 0.025, 0.05, 0.1, 0.2)
TAUS = (0.5, 1.0, 2.0)
DECAYS = (0.2, 0.5, 1.0)
REFINEMENT = (2 ** -0.5, 1.0, 2 ** 0.5)
TOP = 5
COARSE_SEEDS = range(10)
REFINE_SEEDS = range(10, 20)
TUNING_SAMPLES = 2000

# Evaluation
TEST_SAMPLES = 3000
TEST_SEEDS = range(20, 40)

STAGES = {
    "coarse": (COARSE_SEEDS, TUNING_SAMPLES),
    "refinement": (REFINE_SEEDS, TUNING_SAMPLES),
    "test": (TEST_SEEDS, TEST_SAMPLES),
}

METRICS = (
    "swd",
    "modes",
    "near_mode_fraction",
)

####################################################
#                      UTILS                       #
####################################################

# Random generator
def random_generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)

####################################################
#              PROBLEM DEFINITION                  #
####################################################

# Generator and discriminator network
def network(outputs):
    return nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, outputs),
    ).to(DEVICE)

# Sample from the real distribution
def real_data(size, rng):
    indices = torch.randint(8, (size,), generator=rng, device=DEVICE)
    noise = torch.randn(size, 2, generator=rng, device=DEVICE)
    return CENTERS[indices] + DATA_STD * noise

# Generate problem (operator F and evaluation)
def problem(seed):
    torch.manual_seed(4 * seed)
    generator, discriminator = network(2), network(1)

    phi = list(generator.parameters())
    psi = list(discriminator.parameters())
    parameters = phi + psi

    # Operator
    def F(theta, size, rng):
        nn.utils.vector_to_parameters(theta, parameters)

        # Draw and score real and generated data
        real = real_data(size, rng)
        latent = torch.randn(size, 2, generator=rng, device=DEVICE)
        fake = generator(latent)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake)

        # Interpolate with stop-gradient on generated points
        epsilon = torch.rand(size, 1, generator=rng, device=DEVICE)
        interpolated = (epsilon * real + (1 - epsilon) * fake.detach()).requires_grad_(True)
        interpolated_scores = discriminator(interpolated)

        # Input gradients
        input_grad = torch.autograd.grad(interpolated_scores.sum(), interpolated, create_graph=True)[0]
        gradient_penalty = (input_grad.norm(dim=1) - 1).square().mean()

        # Objective
        payoff = real_scores.mean() - fake_scores.mean()
        payoff -= LAMBDA_GP * gradient_penalty
        payoff -= 0.5 * LAMBDA_D * sum(p.square().sum() for p in psi)
        payoff += 0.5 * LAMBDA_G * sum(p.square().sum() for p in phi)

        grad_phi = torch.autograd.grad(payoff, phi, retain_graph=True)
        grad_psi = torch.autograd.grad(payoff, psi)
        return torch.cat([g.flatten() for g in grad_phi] + [-g.flatten() for g in grad_psi])

    # Compute SWD, covered modes, and near-mode fraction
    @torch.no_grad()
    def evaluate(theta, samples):
        nn.utils.vector_to_parameters(theta, parameters)
        rng = random_generator(4 * seed + 3)

        # Draw generated and real samples
        latent = torch.randn(samples, 2, generator=rng, device=DEVICE)
        generated = generator(latent)
        target = real_data(samples, rng)

        # Estimate sliced Wasserstein-1 using random unit directions
        directions = torch.randn(DIRECTIONS, 2, generator=rng, device=DEVICE)
        directions = directions / directions.norm(dim=1, keepdim=True)
        generated_projections = (generated @ directions.T).sort(dim=0).values
        target_projections = (target @ directions.T).sort(dim=0).values
        swd = (generated_projections - target_projections).abs().mean()

        # Fraction of generated points near a mode center
        distances = (generated[:, None] - CENTERS[None]).norm(dim=2)
        nearest_distance, nearest_center = distances.min(dim=1)
        near_mode = nearest_distance < QUALITY_RADIUS
        near_mode_fraction = near_mode.float().mean()

        # Number of covered modes
        counts = torch.bincount(nearest_center[near_mode], minlength=8)
        modes = (counts > MINIMUM_MODE_FRACTION * samples).sum()

        return float(swd), int(modes), float(near_mode_fraction)

    initial = nn.utils.parameters_to_vector(parameters).detach()
    return initial, F, evaluate

####################################################
#                   OPTIMIZER                      #
####################################################

def clip(vector, tau):
    norm = torch.linalg.vector_norm(vector)
    if norm == 0:
        return torch.zeros_like(vector)
    return (tau / norm).clamp(max=1) * vector

def softclip(vector, tau):
    return vector / (1 + torch.linalg.vector_norm(vector) / tau)

def hyper(vector, tau):
    norm = torch.linalg.vector_norm(vector)
    if norm == 0:
        return torch.zeros_like(vector)
    return (torch.asinh(norm) / norm) * vector

METHODS = {
    "SEG": (lambda vector, _: vector, False),
    "Clip-SEG": (clip, True),
    "Softclip-SEG": (softclip, True),
    "Hyper-SEG": (hyper, False),
}

# Run one experiment
def run(method, settings, seed, samples):
    gamma, tau, decay = settings
    prec, _ = METHODS[method]

    theta, F, evaluate = problem(seed)
    pred_rng = random_generator(4 * seed + 1)
    corr_rng = random_generator(4 * seed + 2)

    for iteration in range(STEPS):
        stepsize = gamma * (decay if iteration >= STEPS // 2 else 1.0)

        value = F(theta, PREDICTOR_BATCH, pred_rng)
        predictor = theta - stepsize * prec(value, tau)

        value = F(predictor, CORRECTOR_BATCH, corr_rng)
        theta = theta - BETA * stepsize * prec(value, tau)

    return evaluate(theta, samples)

####################################################
#                    TUNING                        #
####################################################

# Evaluate and rank hyperparameter configurations
def search(method, grid, stage, output):
    seeds, samples = STAGES[stage]
    scores = {}

    for settings in dict.fromkeys(grid):
        results = []

        for seed in seeds:
            metrics = run(method, settings, seed, samples)
            print(stage, method, *settings, seed, *metrics, sep=",", file=output)
            results.append(metrics)

        # Omit configurations containing NaN or Inf
        values = torch.tensor(results, dtype=torch.float64)
        if not torch.isfinite(values).all():
            print(f"{stage}: {method} {settings} | skipped (NaN or Inf)", flush=True)
            continue

        # Rank by mean SWD
        score = float(values[:, 0].mean())
        scores[settings] = score
        print(f"{stage}: {method} {settings} | mean SWD {score:.6f}", flush=True)

    if stage != "test" and not scores:
        raise RuntimeError(f"No valid configuration for {method} during {stage}.")

    return sorted(scores, key=scores.get)

# Coarse search and refinement
def tune(method, output):
    _, uses_tau = METHODS[method]

    # Coarse search
    grid = product(GAMMAS, TAUS if uses_tau else (1.0,), DECAYS)
    best = search(method, grid, "coarse", output)

    # Refine around the best coarse configurations
    grid = [
        (gamma * gamma_factor, tau * tau_factor, decay)
        for gamma, tau, decay in best[:TOP]
        for gamma_factor, tau_factor in product(REFINEMENT, REFINEMENT if uses_tau else (1.0,))
    ]
    best = search(method, grid, "refinement", output)

    print(f"Selected: {method} {best[0]}", flush=True)
    return best[0]

####################################################
#              REPORTING and MAIN                  #
####################################################

# Summarize final test results
def summarize():
    groups = {}

    # Collect test results
    with open(RESULTS, newline="") as source:
        for row in csv.DictReader(source):
            if row["stage"] == "test":
                settings = tuple(row[name] for name in ("method", "gamma", "threshold", "decay"))
                groups.setdefault(settings, []).append([float(row[name]) for name in METRICS])

    print("Final metrics: mean +/- sample std over finite test runs", flush=True)

    with open(SUMMARY, "w", newline="") as output:
        writer = csv.writer(output)
        columns = tuple(f"{name}_{stat}" for name in METRICS for stat in ("mean", "std"))
        writer.writerow(("method", "gamma", "threshold", "decay", "runs", "valid", "failed") + columns)

        # Compute statistics using finite runs only
        for settings, rows in groups.items():
            values = torch.tensor(rows, dtype=torch.float64)
            values = values[torch.isfinite(values).all(dim=1)]

            mean = values.mean(dim=0)
            std = values.std(dim=0, correction=1) if len(values) > 1 else torch.full_like(mean, float("nan"))
            statistics = torch.stack((mean, std), dim=1).flatten().tolist()

            writer.writerow((*settings, len(rows), len(values), len(rows) - len(values), *statistics))
            print(
                f"{settings[0]}: SWD {mean[0]:.4f} +/- {std[0]:.4f} | "
                f"modes {mean[1]:.2f} +/- {std[1]:.2f} / 8 | "
                f"near-mode fraction {mean[2]:.4f} +/- {std[2]:.4f} | "
                f"valid runs {len(values)}/{len(rows)}",
                flush=True,
            )

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    with open(RESULTS, "w", buffering=1) as output:
        header = ("stage", "method", "gamma", "threshold", "decay", "seed") + METRICS
        print(*header, sep=",", file=output)

        # Tune each method
        selected = {method: tune(method, output) for method in METHODS}

        # Evaluate selected settings on separate test seeds
        for method, settings in selected.items():
            search(method, [settings], "test", output)

    summarize()

if __name__ == "__main__":
    main()