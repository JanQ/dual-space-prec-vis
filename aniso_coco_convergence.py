import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 24,
})

def F(x):
    x1, x2 = x
    return np.array([x1 + x1**3, x1 + x2])

def run(K, n_iter=20):
    x = np.array([2.0, 0.0])
    errors = [np.linalg.norm(x)]

    for _ in range(n_iter):
        x -= 0.5 * K(F(x))
        errors.append(np.linalg.norm(x))

    return errors

methods = {
    "Euclidean": lambda y: y,
    "Anisotropic": lambda y: y / (1 + np.abs(y)),
}

colors = [(0.889, 0.436, 0.278), (0.000, 0.606, 0.979)]
fig, ax = plt.subplots(figsize=(8, 4))

for (label, K), color in zip(methods.items(), colors):
    ax.plot(run(K), "-o", color=color, ms=6, lw=4, label=label)

ax.set(xlabel=r"Iteration $k$", ylabel=r"$\|x^k-x^\star\|$", ylim=(-0.5, 4))
ax.legend()
ax.grid(alpha=0.3)

fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.2)
fig.savefig("aniso_coco_convergence.pdf")
plt.show()