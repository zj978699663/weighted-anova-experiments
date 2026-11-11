from fractions import Fraction
import numpy as np

from common import (
    BUILD,
    L_DIGITS,
    digital_words,
    directions,
    generators,
    write_table,
)

def prefix_variances(gen, degrees):
    values = {}
    for j in (3, 4):
        e = degrees[j - 1]
        for m in range(e, 11):
            a = directions(gen[j - 1], m + 1)
            for off in (-1, 0, 1):
                n = 2**m + off
                words = digital_words(a, np.uint64(0), n)
                _, occ = np.unique(words >> np.uint64(L_DIGITS - e), return_counts=True)
                gain = Fraction(
                    2**e * sum(int(k) ** 2 for k in occ) - n * n, n * (2**e - 1)
                )
                assert n * gain == (0 if off == 0 else 1)
                values[j, m, off] = gain / n
    return values


def main():
    BUILD.mkdir(exist_ok=True)
    values = prefix_variances(*generators(4))
    rows = []
    for m in (3, 5, 7, 9):
        row = [m, 2**m - 1, 2**m + 1]
        for off in (-1, 1):
            a = values[3, m, off]
            assert a == values[4, m, off]
            row.append(f"{float(a):.4g}")
        rows.append(row)
    write_table(
        3,
        ["m", "n_minus", "n_plus", "variance_minus", "variance_plus"],
        rows,
    )


if __name__ == "__main__":
    main()
