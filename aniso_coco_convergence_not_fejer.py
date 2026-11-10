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
    return np.array([x1 - 2*x2, 2*x1 + x2])

x = np.array([2.0, 8.0])
errors = [np.linalg.norm(x)]

for _ in range(100):
    y = F(x)
    x -= 0.2 * y / (1 + np.abs(y))
    errors.append(np.linalg.norm(x))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(errors, "-o", color=(0.000, 0.606, 0.979), ms=6, lw=4, label="Anisotropic")

ax.set(xlabel=r"Iteration $k$", ylabel=r"$\|x^k-x^\star\|$")
ax.legend()
ax.grid(alpha=0.3)

fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.2)
fig.savefig("aniso_coco_convergence_not_fejer.pdf")
plt.show()