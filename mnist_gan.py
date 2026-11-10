import csv
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torchvision.datasets import MNIST
from torchvision.utils import save_image

####################################################
#                    CONSTANTS                     #
####################################################

# Device and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data")
OUTPUT = Path("mnist_gan_final_results")
CLASSIFIER = Path(__file__).with_name("classifier_mnist.pt")

# Architecture
LATENT_DIM = 128
CHANNELS = 64
OUTPUT_REGULARIZATION = 1e-3
WEIGHT_DECAY = 1e-5

# Optimizer
STEPS = 1024
PREDICTOR_BATCH = 32
CORRECTOR_BATCH = 1024
BETA = 0.9

# Grid search
GAMMAS = (0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2)
THRESHOLDS = (0.5, 1.0, 2.0)
DECAYS = (0.2, 0.5, 1.0)
REFINEMENT = (2 ** -0.5, 1.0, 2 ** 0.5)
TOP = 3
COARSE_SEEDS = range(0, 5)
REFINE_SEEDS = range(5, 15)
TUNING_SAMPLES = 2000

# Evaluation
EVAL_BATCH = 256
TEST_SAMPLES = 5000
FIGURE_SCALE = 4
TEST_SEEDS = range(15, 20)

STAGES = {
    "coarse": (COARSE_SEEDS, TUNING_SAMPLES),
    "refinement": (REFINE_SEEDS, TUNING_SAMPLES),
    "test": (TEST_SEEDS, TEST_SAMPLES),
}

METRICS = (
    "mfid",
    "class_coverage",
)

####################################################
#                    CLASSIFIER                    #
####################################################

# Load the MNIST classifier used for evaluation
def load_classifier():
    classifier = nn.ModuleDict({
        "conv": nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        ),
        "feature": nn.Linear(64 * 7 * 7, 128),
        "classifier": nn.Linear(128, 10),
    }).to(DEVICE)

    checkpoint = torch.load(CLASSIFIER, map_location=DEVICE, weights_only=True)
    classifier.load_state_dict(checkpoint["state_dict"])
    return classifier.eval()

# Return output of classifer (10-class probabilities and 128-D hidden representations, the latter is used for calculating mFID)
@torch.no_grad()
def classifier_outputs(classifier, images):
    probabilities, features = [], []

    for batch in images.split(EVAL_BATCH):
        hidden = classifier["conv"](batch.to(DEVICE)).flatten(1)
        hidden = functional.relu(classifier["feature"](hidden))
        probabilities.append(classifier["classifier"](hidden).softmax(dim=1).cpu())
        features.append(hidden.cpu().double())

    return torch.cat(probabilities), torch.cat(features)

####################################################
#                      UTILS                       #
####################################################

# sqrtm for PSD matrix
def square_root(matrix):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T

# Normalize images from [0, 255] to [-1, 1]
def normalize_images(images):
    return images.unsqueeze(1).to(DEVICE, dtype=torch.float32).div(127.5).sub(1)

# Random generator
def random_generator(seed, device=DEVICE):
    return torch.Generator(device=device).manual_seed(seed)

####################################################
#              PROBLEM DEFINITION                  #
####################################################

# Generator and discriminator networks
def networks():
    generator = nn.Sequential(
        nn.Linear(LATENT_DIM, 2 * CHANNELS * 7 * 7),
        nn.Unflatten(1, (2 * CHANNELS, 7, 7)),
        nn.ConvTranspose2d(2 * CHANNELS, CHANNELS, 4, 2, 1, bias=False),
        nn.GroupNorm(min(8, CHANNELS), CHANNELS), nn.ReLU(),
        nn.Conv2d(CHANNELS, CHANNELS, 3, 1, 1, bias=False),
        nn.GroupNorm(min(8, CHANNELS), CHANNELS), nn.ReLU(),
        nn.ConvTranspose2d(CHANNELS, 1, 4, 2, 1), nn.Tanh(),
    ).to(DEVICE)

    discriminator = nn.Sequential(
        nn.Conv2d(1, CHANNELS, 4, 2, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(CHANNELS, 2 * CHANNELS, 4, 2, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(2 * CHANNELS, 2 * CHANNELS, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Flatten(), nn.Linear(2 * CHANNELS * 7 * 7, 1),
    ).to(DEVICE)
    return generator, discriminator

# Generate problem (operator F and evaluation)
def problem(seed, data, classifier):
    torch.manual_seed(6 * seed)
    generator, discriminator = networks()

    phi = list(generator.parameters())
    psi = list(discriminator.parameters())
    parameters = phi + psi

    # Operator
    def F(theta, size, index_rng, latent_rng):
        nn.utils.vector_to_parameters(theta, parameters)

        # Draw and score real and generated images
        indices = torch.randint(len(data), (size,), generator=index_rng)
        real = normalize_images(data[indices])
        latent = torch.randn(size, LATENT_DIM, generator=latent_rng, device=DEVICE)
        fake = generator(latent)

        real_scores = discriminator(real)
        fake_scores = discriminator(fake)

        # Generator and discriminator losses
        generator_loss = functional.softplus(-fake_scores).mean()
        discriminator_loss = functional.softplus(-real_scores).mean() + functional.softplus(fake_scores).mean()
        discriminator_loss += 0.5 * OUTPUT_REGULARIZATION * (real_scores.square().mean() + fake_scores.square().mean())
        discriminator_loss += 0.5 * WEIGHT_DECAY * sum(p.square().sum() for p in psi)

        grad_phi = torch.autograd.grad(generator_loss, phi, retain_graph=True)
        grad_psi = torch.autograd.grad(discriminator_loss, psi)
        return torch.cat([g.flatten() for g in grad_phi + grad_psi])

    # Compute mFID and other metrics
    @torch.no_grad()
    def evaluate(theta, samples, real_stats):
        nn.utils.vector_to_parameters(theta, parameters)
        rng = random_generator(6 * seed + 5)

        # Generate images and classify them
        generated = []
        for start in range(0, samples, EVAL_BATCH):
            size = min(EVAL_BATCH, samples - start)
            latent = torch.randn(size, LATENT_DIM, generator=rng, device=DEVICE)
            generated.append(generator(latent).cpu())
        images = torch.cat(generated)
        probabilities, features = classifier_outputs(classifier, images)

        # Frechet distance
        real_mean, real_cov = real_stats
        fake_mean = features.mean(dim=0)
        fake_cov = torch.cov(features.T)

        difference = real_mean - fake_mean
        root = square_root(real_cov)
        cross = square_root(root @ fake_cov @ root)
        mfid = difference.square().sum() + real_cov.trace() + fake_cov.trace() - 2 * cross.trace()

        # Number of digit classes receiving at least 1% of generated predictions
        counts = torch.bincount(probabilities.argmax(dim=1), minlength=10)
        class_coverage = (counts / samples >= 0.01).sum()

        return (float(mfid), int(class_coverage)), images[:64]

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

# Make an algorithm (to avoid passing data, classifier, and the image stats each time)
def make_runner(data, classifier, real_stats):
    def run(method, settings, seed, samples):
        gamma, tau, decay = settings
        prec, _ = METHODS[method]

        theta, F, evaluate = problem(seed, data, classifier)

        pred_index_rng = random_generator(6 * seed + 1, 'cpu')
        corr_index_rng = random_generator(6 * seed + 2, 'cpu')
        pred_latent_rng = random_generator(6 * seed + 3)
        corr_latent_rng = random_generator(6 * seed + 4)

        for iteration in range(STEPS):
            stepsize = gamma * (decay if iteration >= STEPS // 2 else 1.0)

            value = F(theta, PREDICTOR_BATCH, pred_index_rng, pred_latent_rng)
            predictor = theta - stepsize * prec(value, tau)

            value = F(predictor, CORRECTOR_BATCH, corr_index_rng, corr_latent_rng)
            theta = theta - BETA * stepsize * prec(value, tau)

        return evaluate(theta, samples, real_stats)
    return run

####################################################
#                    TUNING                        #
####################################################

# Evaluate and rank hyperparameter configurations
def search(method, grid, stage, run, output):
    seeds, samples = STAGES[stage]
    scores = {}

    for settings in dict.fromkeys(grid):
        results = []

        for seed in seeds:
            metrics, images = run(method, settings, seed, samples)
            print(stage, method, *settings, seed, *metrics, sep=",", file=output)
            results.append(metrics)

            # Save generated images on the first test seed
            if stage == "test" and seed == TEST_SEEDS[0]:
                digits = functional.interpolate(images, scale_factor=FIGURE_SCALE, mode="nearest")
                save_image(digits, OUTPUT / f"{method}_seed{seed}.png", nrow=8, padding=FIGURE_SCALE, normalize=True, value_range=(-1, 1))

        # Omit configurations containing NaN or Inf
        values = torch.tensor(results, dtype=torch.float64)
        if not torch.isfinite(values).all():
            print(f"{stage}: {method} {settings} | skipped (NaN or Inf)", flush=True)
            continue

        # Rank by mean mFID
        score = float(values[:, 0].mean())
        scores[settings] = score
        print(f"{stage}: {method} {settings} | mean mFID {score:.3f}", flush=True)

    if stage != "test" and not scores:
        raise RuntimeError(f"No valid configuration for {method} during {stage}.")

    return sorted(scores, key=scores.get)

# Coarse search and refinement
def tune(method, run, output):
    _, uses_threshold = METHODS[method]

    # Coarse search
    thresholds = THRESHOLDS if uses_threshold else (1.0,)
    grid = product(GAMMAS, thresholds, DECAYS)
    best = search(method, grid, "coarse", run, output)

    # Refine around the best coarse configurations
    threshold_factors = REFINEMENT if uses_threshold else (1.0,)
    grid = [(gamma * gamma_factor, tau * tau_factor, decay) for gamma, tau, decay in best[:TOP] for gamma_factor, tau_factor in product(REFINEMENT, threshold_factors)]
    best = search(method, grid, "refinement", run, output)
    print(f"Selected: {method} {best[0]}", flush=True)
    return best[0]

####################################################
#              REPORTING and MAIN                  #
####################################################

# Summarize final test results
def summarize():
    groups = {}

    # Collect test results
    with (OUTPUT / "results.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["stage"] == "test":
                settings = tuple(row[name] for name in ("method", "gamma", "threshold", "decay"))
                groups.setdefault(settings, []).append([float(row[name]) for name in METRICS])

    print("Final metrics: mean +/- sample std over finite test runs", flush=True)

    with (OUTPUT / "summary.csv").open("w", newline="") as output:
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
            print(f"{settings[0]}: mFID {mean[0]:.3f} +/- {std[0]:.3f} | valid runs {len(values)}/{len(rows)}", flush=True)

# Run tuning, final testing, and reporting
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load classifier and MNIST data
    classifier = load_classifier()
    train = MNIST(DATA, train=True, download=True)
    test = MNIST(DATA, train=False, download=True)

    # Image statistics for training
    _, features = classifier_outputs(classifier, normalize_images(train.data))
    tuning_stats = features.mean(dim=0), torch.cov(features.T)
    tuning_run = make_runner(train.data, classifier, tuning_stats)

    # Image statistics for test
    probabilities, features = classifier_outputs(classifier, normalize_images(test.data))
    test_stats = features.mean(dim=0), torch.cov(features.T)
    test_run = make_runner(train.data, classifier, test_stats)

    with (OUTPUT / "results.csv").open("w", buffering=1) as output:
        header = ("stage", "method", "gamma", "threshold", "decay", "seed") + METRICS
        print(*header, sep=",", file=output)

        # Tune each method
        selected = {method: tune(method, tuning_run, output) for method in METHODS}

        # Evaluate selected settings on separate test seeds
        for method, settings in selected.items():
            search(method, [settings], "test", test_run, output)
    summarize()

if __name__ == "__main__":
    main()