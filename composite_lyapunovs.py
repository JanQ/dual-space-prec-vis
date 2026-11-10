import numpy as np
from scipy.optimize import root

J = np.array([[0., -1.], [1., 0.]])
N = 10


def phi_star(y):
    r = np.linalg.norm(y)
    return r * np.arcsinh(r) - np.sqrt(1 + r**2) + 1


def K(y):
    r = np.linalg.norm(y)
    return np.zeros_like(y) if r == 0 else np.arcsinh(r) * y / r


def K_inv(y):
    r = np.linalg.norm(y)
    return np.zeros_like(y) if r == 0 else np.sinh(r) * y / r


def D(a, b):
    return phi_star(a) - phi_star(b) - K(b) @ (a - b)


def sample_problem(rng):
    a = rng.uniform(0.001, 0.1)
    s = rng.uniform(0.1, 1)

    M = a * np.eye(2) - s * J
    gamma = rng.uniform(0.05, 0.95) * a / (a**2 + s**2)

    u = rng.uniform(-10, 10, 2)
    Q = rng.uniform(1, 10) * np.outer(u, u)
    Q += rng.uniform(1e-5, 1e-2) * np.eye(2)

    c = rng.uniform(-5, 5, 2)
    q = rng.uniform(-5, 5, 2)
    x0 = rng.uniform(-5, 5, 2)

    return M, c, Q, q, gamma, x0


def run(problem):
    M, c, Q, q, gamma, x0 = problem

    F = lambda x: M @ x + c
    B = lambda x: Q @ x + q

    xstar = -np.linalg.solve(M + Q, c + q)
    fstar = F(xstar)
    vstar = B(xstar)

    xs = [x0]
    vs = []

    for _ in range(N):
        x = xs[-1]
        f = F(x)

        fun = lambda z: z - x + gamma * (K(f) + K(B(z)))
        sol = root(fun, x, tol=1e-12)

        if not sol.success:
            return None

        xs.append(sol.x)
        vs.append(B(sol.x))

    values = {
        "D(v,-F)": [],
        "D(-F,v)": [],
        "distance": [],
        "paired Bregman": [],
        "residual": [],
    }

    for x, v in zip(xs, vs):
        f = F(x)
        r = K_inv(K(f) + K(v))

        values["D(v,-F)"].append(D(v, -f))
        values["D(-F,v)"].append(D(-f, v))
        values["distance"].append(0.5 * np.linalg.norm(x - xstar)**2)
        values["paired Bregman"].append(D(f, fstar) + D(v, vstar))
        values["residual"].append(phi_star(r))

    return values


def is_counterexample(values):
    if values is None:
        return False

    inc = [np.diff(v) > 1e-9 for v in values.values()]

    return all(np.any(d) for d in inc) and np.any(inc[0] & inc[1])


rng = np.random.default_rng(0)

for attempt in range(1, 100000):
    problem = sample_problem(rng)
    values = run(problem)

    if is_counterexample(values):
        break
else:
    raise RuntimeError("No counterexample found.")

M, c, Q, q, gamma, x0 = problem

print("found after", attempt, "draws")
print("M =\n", M)
print("c =", c)
print("Q =\n", Q)
print("q =", q)
print("x0 =", x0)
print("gamma =", gamma)

for name, vals in values.items():
    k = np.flatnonzero(np.diff(vals) > 1e-9)[0]
    print(name, ":", k, "->", k + 1)