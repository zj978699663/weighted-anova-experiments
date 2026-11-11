import hashlib

import numpy as np

from common import BUILD, L_DIGITS, digital_words, generators, write_table


BASE_SEED = 20260816
BOOTSTRAP_COUNT = 2000
REPS = 1024
LEVELS = (4, 6, 8, 10)


def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy()
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def random_invertible(size: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        a = rng.integers(0, 2, size=(size, size), dtype=np.uint8)
        if gf2_rank(a) == size:
            return a


def seed_from_tuple(method: int, test: int, d: int, m: int, replicate: int) -> int:
    payload = f"{BASE_SEED}|{method}|{test}|{d}|{m}|{replicate}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def scramble_directions(
    base_rows: np.ndarray,
    e: int,
    m: int,
    rng: np.random.Generator,
    coarse: bool,
) -> tuple[np.ndarray, np.uint64]:
    block = e if coarse else 1
    needed = ((L_DIGITS + block - 1) // block) * block
    rows = base_rows[:needed]
    out_rows = np.zeros(L_DIGITS, dtype=np.uint64)
    nblocks = needed // block
    for br in range(nblocks):
        row_start = br * block
        diagonal = random_invertible(block, rng)
        previous = rng.integers(0, 2, size=(block, row_start), dtype=np.uint8)
        for rr in range(block):
            out_r = row_start + rr
            if out_r >= L_DIGITS:
                continue
            idx_prev = np.flatnonzero(previous[rr])
            idx_diag = np.flatnonzero(diagonal[rr]) + row_start
            selected = np.concatenate((idx_prev, idx_diag))
            out_rows[out_r] = np.bitwise_xor.reduce(
                rows[selected], initial=np.uint64(0)
            )
    cols = np.arange(m, dtype=np.uint64)
    bits = ((out_rows[:, None] >> cols[None, :]) & np.uint64(1)).astype(bool)
    weights = np.array(
        [np.uint64(1 << (L_DIGITS - 1 - row)) for row in range(L_DIGITS)]
    )[:, None]
    directions = np.bitwise_or.reduce(np.where(bits, weights, np.uint64(0)), axis=0)
    shift = rng.integers(0, 1 << L_DIGITS, dtype=np.uint64)
    return directions, shift


def haar_values(words: np.ndarray, level: int) -> np.ndarray:
    if level == 0:
        prefix_zero = np.ones(words.size, dtype=bool)
    else:
        prefix_zero = (words >> np.uint64(L_DIGITS - level)) == 0
    signbit = (words >> np.uint64(L_DIGITS - level - 1)) & np.uint64(1)
    sign = 1.0 - 2.0 * signbit.astype(np.float64)
    return np.where(prefix_zero, (2.0 ** (level / 2.0)) * sign, 0.0)


def block_estimate(
    generator_rows: list[np.ndarray],
    degrees: list[int],
    d: int,
    m: int,
    method: int,
    test: int,
    replicate: int,
    j_star: int,
    e_star: int,
) -> float:
    n = 1 << m
    level = m - e_star if test == 2 else m
    rng = np.random.default_rng(seed_from_tuple(method, test, d, m, replicate))
    if method == 3:
        x = rng.random(n)
        left = 2.0 ** (-level - 1)
        right = 2.0 ** (-level)
        vals = np.where(
            x < left,
            2.0 ** (level / 2.0),
            np.where(x < right, -(2.0 ** (level / 2.0)), 0.0),
        )
        return float(np.mean(vals))
    j = j_star - 1
    directions, shift = scramble_directions(
        generator_rows[j], degrees[j], m, rng, coarse=(method == 1)
    )
    words = digital_words(directions, shift, n)
    return float(np.mean(haar_values(words, level)))


def bootstrap_variance_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    variance = float(np.var(values, ddof=1))
    if variance == 0.0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_COUNT, dtype=np.float64)
    batch = 100
    for start in range(0, BOOTSTRAP_COUNT, batch):
        stop = min(start + batch, BOOTSTRAP_COUNT)
        idx = rng.integers(0, values.size, size=(stop - start, values.size))
        samples[start:stop] = np.var(values[idx], axis=1, ddof=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def haar_samples(gen, degrees, m, tid):
    return np.array(
        [
            block_estimate(gen, degrees, 8, m, 1, tid, rep, 3, 2)
            for rep in range(1, REPS + 1)
        ]
    )


def main():
    BUILD.mkdir(exist_ok=True)
    gen, degrees = generators(8)
    rows = []
    for m in LEVELS:
        N = 2**m
        Q0 = haar_samples(gen, degrees, m, 2)
        assert np.all(Q0 == 0)

        Q1 = haar_samples(gen, degrees, m, 3)
        Vhat = float(np.var(Q1, ddof=1))
        lo, hi = bootstrap_variance_ci(Q1, seed_from_tuple(1, 3, 8, m, 99991))
        assert lo <= 1 / N <= hi
        rows.append([N, f"{N * Vhat:.3f}", f"{N * lo:.3f}", f"{N * hi:.3f}"])

    write_table(6, ["N", "N_Vhat", "CI_low", "CI_high"], rows)
    print(f"Generated {2 * len(LEVELS) * REPS:,} Haar estimates")


if __name__ == "__main__":
    main()
