import math
import os
import tempfile
from itertools import product

import mpmath as mp
import numpy as np

from common import BUILD, write_table


PQ = (0.75, 1.0, 1.25)
C = (0.8, 1.0, 1.2)
DEGREES = (8, 16, 32, 64, 128)
M_VALUES = (20, 32, 64, 128)


def first_indices():
    counts = {}
    for n in range(1, 129):
        known = sum(k * counts[k] for k in counts if n % k == 0)
        counts[n], rest = divmod(2**n - known, n)
        assert rest == 0
    return {n: 1 + sum(counts[k] for k in range(1, n)) for n in counts}


def factors(indices):
    data = {}
    for p, q, c, n in product(PQ, PQ, C, DEGREES):
        j = indices[n]
        a = mp.mpf(str(c)) * mp.mpf(j) ** (-p) * mp.log(j + 2, 2) ** (-q) * 2**n
        data[p, q, c, n] = a / (1 - mp.mpf(2) ** (-n))
    return data


def phase_table(phase):
    cases = [
        (1.25, 0.75, 1),
        (1, 1.25, 1),
        (1, 1, 0.8),
        (1, 1, 1),
        (1, 1, 1.2),
        (1, 0.75, 1),
        (0.75, 1.25, 1),
    ]
    rows = []
    for p, q, c in cases:
        a8 = float(phase[p, q, c, 8])
        a128 = float(phase[p, q, c, 128])
        rows.append([f"({p:g},{q:g},{c:g})", f"{a8:.4g}", f"{a128:.4g}"])
    write_table(1, ["(p,q,c)", "a_j8", "a_j128"], rows)


def witnesses(indices):
    data = {}
    for c in C:
        W = mp.mpf(1) / 4
        for n in range(20, max(M_VALUES) + 1):
            j = indices[n]
            den = mp.mpf(j) * mp.log(j + 2, 2) * (1 - mp.mpf(2) ** (-n))
            W *= mp.mpf(str(c)) * 2**n / den
            if n in M_VALUES:
                data[n, c] = W
    return data


def witness_table(witness):
    rows = [[M, *(f"{float(witness[M, c]):.4g}" for c in C)] for M in M_VALUES]
    write_table(2, ["M", "c_0.8", "c_1", "c_1.2"], rows)


def draw_figure(phase, witness):
    cache = BUILD / "cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    tempfile.tempdir = str(cache)

    import matplotlib as mpl

    mpl.use("pgf")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

    STYLES = {
        0.8: ("#176b91", "-", "o"),
        1.0: ("#30343b", "--", "s"),
        1.2: ("#ce6a22", "-.", "^"),
    }

    WIDTH = 390 * 0.48 * 72 / 72.27
    SIDE = 138.5
    HEIGHT = 164.5
    RECT = (24 / WIDTH, 24 / HEIGHT, SIDE / WIDTH, SIDE / HEIGHT)

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath,bm}",
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.55,
            "lines.linewidth": 0.80,
            "lines.markersize": 3.4,
            "lines.markeredgewidth": 0.75,
            "lines.markerfacecolor": "white",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )

    def panel():
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
        ax = fig.add_axes(RECT)
        ax.set_box_aspect(1)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            length=2.5,
            width=0.55,
            pad=1.5,
        )
        ax.xaxis.labelpad = 1.5
        ax.yaxis.labelpad = -2
        return fig, ax

    def save(fig, ax, name):
        box = ax.get_position()
        assert box.width * box.height >= 0.62, name
        assert abs(box.width * WIDTH / (box.height * HEIGHT) - 1) < 1.0e-10, name
        fig.savefig(BUILD / name)
        plt.close(fig)

    values = np.array(
        [[math.log10(float(phase[p, q, 1.0, 128])) for p in PQ] for q in PQ]
    )
    fig, ax = panel()
    limit = math.ceil(abs(values).max())
    mesh = ax.pcolormesh(
        np.arange(4) - 0.5,
        np.arange(4) - 0.5,
        values,
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        edgecolors="white",
        linewidth=0.8,
    )
    for row in range(3):
        for col in range(3):
            ax.text(
                col,
                row,
                "$" + format(values[row, col], "+.2f") + "$",
                ha="center",
                va="center",
                color="white" if abs(values[row, col]) > 6 else "#30343b",
                fontsize=7.2,
            )
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec="#30343b", lw=1.1))
    ax.set(
        xlim=(-0.5, 2.5),
        ylim=(-0.5, 2.5),
        xlabel="$p$",
        ylabel="$q$",
        xticks=range(3),
        yticks=range(3),
        xticklabels=["$0.75$", "$1$", "$1.25$"],
        yticklabels=["$0.75$", "$1$", "$1.25$"],
    )
    ax.tick_params(length=0)
    cax = fig.add_axes(
        ((24 + SIDE + 1) / WIDTH, 24 / HEIGHT, 4.5 / WIDTH, SIDE / HEIGHT)
    )
    cb = fig.colorbar(mesh, cax=cax, ticks=[-10, 0, 10])
    cb.ax.tick_params(labelsize=7.2, length=1.5, pad=1)
    cb.ax.get_yticklabels()[-1].set_verticalalignment("top")

    save(fig, ax, "fig1a.pdf")

    fig, ax = panel()
    x = np.array(M_VALUES)
    for c, (color, line, marker) in STYLES.items():
        y = np.array([float(witness[M, c]) for M in M_VALUES])
        ax.plot(x, y, color=color, ls=line, marker=marker, label=rf"$c={c:g}$")
        assert ((x >= 17) & (x <= 132) & (y >= 1e-9) & (y <= 1e13)).all()
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set(
        xlim=(17, 132),
        ylim=(1e-9, 1e13),
        xlabel="$M$",
        ylabel="$W_M(c)$",
        xticks=[20, 64, 128],
    )
    ax.yaxis.set_label_coords((8.5 - 24) / SIDE, 0.5)
    ax.yaxis.set_major_locator(FixedLocator([1e-8, 1e-4, 1, 1e4, 1e8, 1e12]))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(1, color="#a5a9ae", ls=":", lw=0.6, zorder=0)
    legend = ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="#9b9b9b",
        handlelength=1.45,
        labelspacing=0.22,
        borderpad=0.28,
        handletextpad=0.45,
    )
    legend.get_frame().set_linewidth(0.42)
    save(fig, ax, "fig1b.pdf")


def main():
    BUILD.mkdir(exist_ok=True)

    with mp.workdps(100):
        indices = first_indices()
        phase = factors(indices)
        witness = witnesses(indices)

    phase_table(phase)
    witness_table(witness)
    draw_figure(phase, witness)


if __name__ == "__main__":
    main()
