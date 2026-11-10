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
    return np.linalg.norm(x)**2 * np.array([x1 - x2, x1 + x2])

x = np.array([5.0, 7.0])
gamma = 1. / 6
errors = [np.linalg.norm(x)]

for _ in range(125):
    y = F(x)
    x -= gamma * y / (1 + np.linalg.norm(y))
    errors.append(np.linalg.norm(x))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(errors, "-o", color=(0.242, 0.643, 0.304), ms=6, lw=4, label="Isotropic")

ax.set(xlabel=r"Iteration $k$", ylabel=r"$\|x^k-x^\star\|$")
ax.legend()
ax.grid(alpha=0.3)

fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.2)
fig.savefig("iso_coco_convergence.pdf")
plt.show()