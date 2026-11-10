import numpy as np

BETA = 0.5
TOL = 1e-6
TRAIN_N, TEST_N = 200, 2000
TRAIN_ITERS, TEST_ITERS = 2500, 5000
GAMMAS = np.arange(0.05, 2.01, 0.05)
TAUS = (1, 2, 5, 10, 20)
METHODS = ("CEG+", "Softclip-CEG+", "Hyper-CEG+")

def polar_game(x):
    r2 = np.sum(x**2, axis=1)
    q = (r2 - 1) * (16 * r2 - 9) / 16
    return np.column_stack((q * x[:, 0] - x[:, 1],
                            q * x[:, 1] + x[:, 0]))

def forsaken(x):
    x1, x2 = x[:, 0], x[:, 1]
    g1 = 0.5 * x1 - 2 * x1**3 + x1**5
    g2 = 0.5 * x2 - 2 * x2**3 + x2**5
    return np.column_stack((x2 - 0.45 + g1, -x1 + g2))

def precondition(y, method, tau=1):
    if method == "CEG+":
        return y

    r = np.linalg.norm(y, axis=1, keepdims=True)
    if method == "Softclip-CEG+":
        return y / (1 + r / tau)

    scale = np.divide(np.arcsinh(r), r, out=np.ones_like(r), where=r > 0)
    return scale * y

def residual(x, F, bound):
    return np.linalg.norm(x - np.clip(x - F(x), -bound, bound), axis=1)

def run(starts, F, bound, method, gamma, tau, max_iters):
    """Return the convergence iteration for each start, or -1 on failure."""
    x = starts.copy()
    hits = np.full(len(x), -1, dtype=int)
    active = np.ones(len(x), dtype=bool)

    for k in range(1, max_iters + 1):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break

        # P-CEG+: H = id - gamma K F, z = P_C(Hx).
        xa = x[ids]
        hx = xa - gamma * precondition(F(xa), method, tau)
        z = np.clip(hx, -bound, bound)
        hz = z - gamma * precondition(F(z), method, tau)
        x[ids] = xa + BETA * (hz - hx)

        # Numerical divergence counts as failure.
        xa = x[ids]
        bad = (~np.isfinite(xa).all(axis=1)) | (np.linalg.norm(xa, axis=1) > 1e6)
        active[ids[bad]] = False

        ids = ids[~bad]
        if len(ids):
            done = residual(x[ids], F, bound) <= TOL
            hits[ids[done]] = k
            active[ids[done]] = False
    return hits

def stats(hits):
    successful = hits >= 0
    calls = 2 * hits[successful]  # two F evaluations per P-CEG+ update
    if len(calls) == 0:
        return 0.0, np.inf, np.inf
    return successful.mean(), np.median(calls), np.quantile(calls, 0.9)

def tune(starts, F, bound, method):
    best = None
    taus = TAUS if method == "Softclip-CEG+" else (1,)

    for tau in taus:
        for gamma in GAMMAS:
            success, median, p90 = stats(
                run(starts, F, bound, method, gamma, tau, TRAIN_ITERS)
            )
            score = (-success, median, p90, gamma, tau)
            if best is None or score < best[0]:
                best = (score, gamma, tau)

    return best[1], best[2]

def sample_starts(n, bound, seed):
    return np.random.default_rng(seed).uniform(-bound, bound, (n, 2))

def experiment(name, F, bound):
    train = sample_starts(TRAIN_N, bound, 12345)
    test = sample_starts(TEST_N, bound, 54321)

    print(f"\n{name}")
    print("method              gamma   tau   success   median calls")

    for method in METHODS:
        gamma, tau = tune(train, F, bound, method)
        success, median, _ = stats(run(test, F, bound, method, gamma, tau, TEST_ITERS))
        tau_text = str(tau) if method == "Softclip-CEG+" else "-"
        print(f"{method:18s} {gamma:5.2f}  {tau_text:>4s}   "
              f"{100 * success:6.2f}%   {median:6.0f}")

if __name__ == "__main__":
    experiment("PolarGame", polar_game, 1.1)
    experiment("Forsaken", forsaken, 1.5)
