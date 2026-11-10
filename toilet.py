import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 24,
})

J = np.array([[0, -1], [1, 0]])

def F(x):
    return np.exp(np.linalg.norm(x)**2 / 2) * J @ x

def K(y):
    return y / (1 + np.linalg.norm(y))

def run(step, n_iter=100):
    x = np.array([1.0, 1.0])
    xs = [x.copy()]

    for _ in range(n_iter):
        x = step(x)
        xs.append(x.copy())

        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 10:
            break

    return np.array(xs)

gamma = omega = .95

def fixed_step(x):
    z = x - gamma * F(x)
    return x - omega * F(z)

def variable_step(x):
    z = x - gamma * K(F(x))
    return x - omega * K(F(z))

colors = [(0.889, 0.436, 0.278), (0.000, 0.606, 0.979)]

fig, ax = plt.subplots(figsize=(6, 5))

for label, step, color in zip(
    ["Fixed", "Variable"],
    [fixed_step, variable_step],
    colors,
):
    xs = run(step)
    ax.plot(xs[:, 0], xs[:, 1], "-o", lw=4, ms=6,
            color=color, label=label)

ax.plot(1, 1, "k*", ms=14)
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.legend()
ax.grid(alpha=.3)

fig.subplots_adjust(left=.18, right=.97, top=.97, bottom=.18)
fig.savefig("toilet.pdf")
plt.show()