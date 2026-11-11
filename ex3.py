import math

import mpmath as mp
import numpy as np

from common import (
    BUILD,
    L_DIGITS,
    digital_words,
    directions,
    generators,
    write_table,
)


ALPHA = 0.5
C_WEIGHT = 0.25
DIMS = (8, 16, 32)
SIZES = (32, 128, 512, 2048)


def direct_diagonal_envelope(degrees: list[int], d: int) -> float:
    factors = [
        j**-4 * 2.0 ** degrees[j - 1] / (1.0 - 2.0 ** (-degrees[j - 1]))
        for j in range(1, d + 1)
    ]
    candidates = []
    for e_min in sorted(set(degrees[:d])):
        eligible = [
            factor
            for factor, degree in zip(factors, degrees[:d])
            if degree >= e_min
        ]
        mandatory = [
            factor
            for factor, degree in zip(factors, degrees[:d])
            if degree == e_min
        ]
        product = math.prod(factor for factor in eligible if factor > 1.0)
        if not any(factor > 1.0 for factor in mandatory):
            product *= max(mandatory)
        candidates.append((1.0 - 2.0 ** (-e_min)) * product)
    return 2.0 ** (-2.0 * ALPHA) * max(candidates)


def product_envelopes(
    degrees: list[int], d: int
) -> tuple[float, float, float, float, float]:
    kh = 18.0 * math.e**2
    bd = math.prod(1.0 + kh * C_WEIGHT**2 * j**-2 for j in range(1, d + 1))
    cd_diag = direct_diagonal_envelope(degrees, d)
    pd_add_envelope = 2.0 ** (-2.0 * ALPHA) * (
        math.prod(
            1.0 + j**-4 * 2.0 ** degrees[j - 1] / (1.0 - 2.0 ** (-degrees[j - 1]))
            for j in range(1, d + 1)
        )
        - 1.0
    )
    return bd, cd_diag, pd_add_envelope, bd * cd_diag, bd * pd_add_envelope


def projected_energy(r):
    M = mp.mpf(2) ** r
    if r == 0:
        return mp.mpf(0)
    A = mp.expm1(1 / M)
    C = mp.exp(1 / M) - M * A

    def geometric(t):
        return mp.expm1(M * t) / mp.expm1(t)

    t = 2 / M
    s0 = geometric(t)
    s1 = mp.diff(geometric, t, 1)
    s2 = mp.diff(geometric, t, 2)
    return (A * A * s2 + 2 * A * C * s1 + C * C * s0) / M - 1


def variance(words, degrees, energies, d, N):
    nu = (mp.exp(2) - 5) / 4
    total = mp.mpf(0)
    for delta in range(N):
        factor = mp.mpf(1)
        for j in range(d):
            h = nu
            if delta:
                leading = L_DIGITS - int(words[j][delta]).bit_length()
                e = degrees[j]
                k, B = leading // e, 2**e
                h = (B * energies[e * k] - energies[e * (k + 1)]) / (B - 1)
            factor *= 1 + h / (16 * (j + 1) ** 6)
        total += factor - 1
    return total / N


def solve(words, degrees, precision):
    with mp.workdps(precision):
        energies = {k: projected_energy(k) for k in range(20)}
        result = {}
        for d in DIMS:
            _, cd, _, U_d, _ = product_envelopes(degrees, d)
            assert cd == 1
            prev = None
            for N in SIZES:
                v = variance(words, degrees, energies, d, N)
                rmse = mp.sqrt(v)
                normalized = N * N * v / mp.mpf(str(U_d))
                assert 0 < normalized < 1
                rate = None if prev is None else mp.log(prev / rmse) / mp.log(4)
                result[d, N] = v, normalized, rmse, rate
                prev = rmse
    return result


def check_precision(low, high):
    with mp.workdps(120):
        for key, expected in high.items():
            for a, b in zip(low[key][:3], expected[:3]):
                assert abs(a / b - 1) < mp.mpf("1e-38"), key
            if expected[3] is not None:
                assert abs(low[key][3] - expected[3]) < mp.mpf("1e-28"), key


def main():
    BUILD.mkdir(exist_ok=True)
    gen, degrees = generators(max(DIMS))
    words = [digital_words(directions(a, 11), np.uint64(0), 2048) for a in gen]
    low = solve(words, degrees, 80)
    values = solve(words, degrees, 120)
    check_precision(low, values)
    print("12 product cases agree at 80 and 120 digits")

    rows = [[d, *(f"{float(values[d, N][1]):.4g}" for N in SIZES)] for d in DIMS]
    write_table(4, ["d", *(f"N_{N}" for N in SIZES)], rows)

    rows = []
    for N in SIZES:
        _, _, rmse, rate = values[32, N]
        rows.append(
            [N, f"{float(rmse):.4g}", "--" if rate is None else f"{float(rate):.4g}"]
        )
    write_table(5, ["N", "RMSE", "rho"], rows)


if __name__ == "__main__":
    main()
