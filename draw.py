#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import argv
from tqdm import tqdm


def latexfy_2x2_matrix(mat, label="A"):
    code = rf"${label}=\begin{{pmatrix}}"
    for row in mat:
        code += "&".join([f"{x:0.2f}" for x in row])
        code += r"\\"
    code += r"\end{pmatrix}$"
    return code


# Parameters
filename = argv[1]
num_frames = int(argv[2])+1
frames = np.arange(0, num_frames)
num_ticks = int(argv[3])+1
num_vecs = num_ticks**2

# Use LaTeX for labels
plt.rcParams["text.usetex"] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

# Grid
X = np.linspace(-10, 10, num_ticks)
Y = np.linspace(-10, 10, num_ticks)
V = np.array([[x, y] for x in X for y in Y])

# Evolving matrices
A = np.zeros((num_frames, 2, 2))
F = np.zeros((num_frames, num_vecs, 2))
C = np.zeros((num_frames, num_vecs))
A0 = np.array([[1, 0], [0, 1]])
An = np.array([[1, 0], [0, -1]])
# A0 = np.random.uniform(-1.0, 1.0, size=(2, 2))
# An = np.random.uniform(-1.0, 1.0, size=(2, 2))
dA = (An-A0)/(num_frames-1)
for i in frames:
    A[i] = A0+(dA*i)
    F[i] = np.dot(V, A[i])
    C[i] = np.sqrt(F[i, :, 0]*F[i, :, 0] + F[i, :, 1]*F[i, :, 1])

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.xlabel(r"$x$", fontsize=30)
    plt.ylabel(r"$y$", fontsize=30)
    for frame in tqdm(frames):
        ax.cla()
        q = ax.quiver(
            V[:, 0], V[:, 1], F[frame, :, 0], F[frame, :, 1], C[frame],
            angles="xy", scale_units="xy", scale=1.0, cmap=plt.cm.jet
        )
        matrix_label = latexfy_2x2_matrix(A[frame], label="M")
        ax.text(
            X[0], Y[-1]+9, matrix_label, color="black", backgroundcolor="white",
            fontsize=30, horizontalalignment="left", verticalalignment="top"
        )
        fig.savefig(
            f"frames/{filename}_{frame:03d}.png",
            dpi=fig.dpi*2.5, bbox_inches="tight"
        )
    # plt.show()
