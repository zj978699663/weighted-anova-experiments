# Weighted ANOVA experiments

## Scope

This package reproduces Tables 1–6 and Figure 1 of *Sharp Dimension-Independent Variance Bounds for Coarsely Scrambled Niederreiter Sequences in Weighted Multiresolution ANOVA Spaces*.

## Run

Use Python 3.13.9. Figure 1 also needs pdfLaTeX with Computer Modern, `amsmath` and `bm`. Run from this directory.

```text
python -m pip install numpy==2.3.5 mpmath==1.3.0 matplotlib==3.10.6
python ex1.py
python ex2.py
python ex3.py
python ex4.py
```

## Tables, figures and checks

Each command can run on its own. The scripts generate the data and write the tables and figures to `build/`.

| Paper | Command | Calculation and checks | Output |
|---|---|---|---|
| §7.1 Witnesses for critical weights, Tables 1–2 and Figure 1 | `python ex1.py` | Compute the factors and witness products at 100 digits. Use the same values for the tables and figure. | `table1.csv`, `table2.csv`, `fig1a.pdf`, `fig1b.pdf` |
| §7.2 Prefix saturation, Table 3 | `python ex2.py` | Build the Niederreiter matrices and check the gains by exact occupancy counts. | `table3.csv` |
| §7.3 Product integrand, Tables 4–5 | `python ex3.py` | Compute the 12 product variances from pair covariances at 80 and 120 digits and compare the results. | `table4.csv`, `table5.csv` |
| §7.4 Haar block gains, Table 6 | `python ex4.py` | Generate all 8,192 estimates and compute the variances and bootstrap intervals. | `table6.csv` |

The Haar intervals use 2,000 bootstrap draws. `ex4.py` derives the sample and bootstrap seeds from base seed `20260816`. The six CSV files in `reference/` hold the values printed in the paper and are read for the table checks. The two PDFs show the reference figure panels.
