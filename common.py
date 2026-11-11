import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
REFERENCE = ROOT / "reference"
L_DIGITS = 52


def value(cell):
    try:
        return Decimal(cell)
    except InvalidOperation:
        return cell


def write_table(n, header, rows):
    name = f"table{n}.csv"
    actual = [[str(cell) for cell in row] for row in [header, *rows]]
    with (REFERENCE / name).open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    if len(actual) != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected) - 1} rows, got {len(actual) - 1}"
        )
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        if list(map(value, a)) != list(map(value, b)):
            raise ValueError(f"{name}, line {i}: {a} differs from {b}")
    with (BUILD / name).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(actual)
    print(f"{name}: {len(actual) - 1} rows checked")


def gf2_degree(p: int) -> int:
    return p.bit_length() - 1


def gf2_mod(a: int, b: int) -> int:
    db = gf2_degree(b)
    while a and gf2_degree(a) >= db:
        a ^= b << (gf2_degree(a) - db)
    return a


def is_irreducible(p: int) -> bool:
    n = gf2_degree(p)
    if n <= 0:
        return False
    for qdeg in range(1, n // 2 + 1):
        for q in range(1 << qdeg, 1 << (qdeg + 1)):
            if gf2_mod(p, q) == 0:
                return False
    return True


def first_irreducibles(count: int) -> list[int]:
    out: list[int] = []
    degree = 1
    while len(out) < count:
        for p in range(1 << degree, 1 << (degree + 1)):
            if is_irreducible(p):
                out.append(p)
                if len(out) == count:
                    break
        degree += 1
    return out


def series_mul(a: list[int], b: list[int], order: int) -> list[int]:
    c = [0] * (order + 1)
    for i, ai in enumerate(a):
        if not ai:
            continue
        for j, bj in enumerate(b[: order + 1 - i]):
            if bj:
                c[i + j] ^= 1
    return c


def series_inverse(a: list[int], order: int) -> list[int]:
    if not a or a[0] != 1:
        raise ValueError("series must have constant coefficient one")
    inv = [0] * (order + 1)
    inv[0] = 1
    for n in range(1, order + 1):
        bit = 0
        for k in range(1, min(n, len(a) - 1) + 1):
            bit ^= a[k] & inv[n - k]
        inv[n] = bit
    return inv


def niederreiter_rows(poly: int, m_max: int, rows: int) -> np.ndarray:
    e = gf2_degree(poly)
    p_rev = [((poly >> (e - i)) & 1) for i in range(e + 1)]
    result = np.zeros(rows, dtype=np.uint64)

    for row1 in range(1, rows + 1):
        quotient, deg_num = divmod(row1 - 1, e)
        denom = [1] + [0] * rows
        for _ in range(quotient + 1):
            denom = series_mul(denom, p_rev, rows)
        inv = series_inverse(denom, rows)
        shift = e * (quotient + 1) - deg_num
        for col1 in range(1, m_max + 1):
            if col1 >= shift and inv[col1 - shift]:
                result[row1 - 1] |= np.uint64(1 << (col1 - 1))
    return result


def digital_words(directions: np.ndarray, shift: np.uint64, n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint64)
    words = np.full(n, shift, dtype=np.uint64)
    for col, direction in enumerate(directions):
        take = ((idx >> np.uint64(col)) & np.uint64(1)).astype(bool)
        words[take] ^= direction
    return words


def generators(d):
    polys = first_irreducibles(d)
    gen = [niederreiter_rows(p, 11, 56) for p in polys]
    return gen, [gf2_degree(p) for p in polys]


def directions(a, m):
    return np.array(
        [
            sum(((int(a[k]) >> col) & 1) << (L_DIGITS - 1 - k) for k in range(L_DIGITS))
            for col in range(m)
        ],
        dtype=np.uint64,
    )
