# AAA approximation with algebraic normalization for ZS scattering

This standalone package contains the paper's core weighted independent AAA
fitting and algebraic normalization. The portable example computes scattering
samples with SciPy's DOP853 integrator. **The paper's reported experiments use
ES8; this example does not reproduce their timings, sample counts, or tables.**

## Run

Use Python 3.10 or newer and install NumPy and SciPy:

```sh
python -m pip install -r requirements.txt
python demo.py --sigma 1
python demo.py --sigma=-1
python check.py
```

The example solves the real-axis scattering problem for
`q(t) = 1.2 * exp(-(1 + 1j) * log(cosh(t)))` on `[-24, 24]`. The sign is
`sigma=1` for the compact case and `sigma=-1` for the noncompact case. DOP853
uses `rtol=2e-11`, `atol=2e-13`, and batches of at most 64 spectral points.
The spectral interval is `[-6, 6]`, with 513 output points. The full-line
analytic solution is used only for the final error comparison; the DOP853
samples solve the truncated problem. This comparison includes both propagation
and truncation errors and is not a rigorous error certificate.

## Files and interface

- `zs_aaa.py`: numerical and analytic scattering routines, shared sample cache,
  independent weighted scalar AAA fits, adaptive sampling, and stable normalization.
- `demo.py`: complete sampling, fitting, output evaluation, and normalization example.
- `check.py`: zero-potential/cache/budget checks, rational fitting and wide Loewner
  matrix checks, normalization at large amplitudes and excluded inputs, and both
  example signs.
- `requirements.txt`: the two runtime dependencies.

The main calls are:

```python
import numpy as np
from zs_aaa import CachedOracle, adaptive_aaa, normalize

oracle = CachedOracle(1, mode='direct', A=1.2, C=1., L=24.)
model, info = adaptive_aaa(oracle, tol=1e-6/3, halfwidth=6.)
if info['validated']:
    x = np.linspace(-6., 6., 513)
    values = normalize(model(x), 1)
```

To supply another analytically evaluable potential, pass `potential=q` to
`CachedOracle`, where `q(t)` returns its complex value at scalar `t`. The AAA
routine can also use another cached scattering callable with attributes
`cache`, `seconds`, and `nfev`. Each call must return an `(N, 2)` complex array
containing `a` and `b`. `mode='exact'` selects the chirped-secant analytic
benchmark instead of numerical propagation.

The adaptive defaults retain 65 initial equispaced points, at most 90 supports
per coefficient, six fitting rounds, 64 new uniform random points plus all
training-grid midpoints for the first check, and 128 new uniform random points
for the second check. The seed is `20261027` and can be supplied to
`adaptive_aaa`. The internal training threshold is `tol/20`; both check
thresholds are `tol/2`. Failed checks enter the training set, and the fits are
rebuilt. The cache counts all distinct training and check samples, including
successful final checks. The default sample budget is 8192; exceeding it raises
an exception before the batch is evaluated.

`info['validated']` means that the finite residual checks against numerical
scattering data passed. It does not establish the final coefficient error or
a bound over the whole spectral interval. Check this flag before evaluating
the returned model. Nonfinite output or an excluded normalization input is a
failure. The extracted AAA implementation has no explicit pole filtering,
near-zero-denominator threshold, or Froissart cleanup. Scaling in `normalize`
avoids unnecessary overflow in the norm calculation; it does not certify
coefficient accuracy. The example reports the scaled quadratic-invariant
defect separately from coefficient error.

## Source and verification

The scattering routines and AAA calculations were extracted from
`experiments/zs_aaa_study.py` of the manuscript sources. The method dispatcher
was restricted to the `weighted_independent` branch and its seed was exposed.
The cache gained a sample budget and defaults to numerical propagation.
`normalize` was copied unchanged from `experiments/algebraic_normalization.py`.
The AAA support selection, weighted Loewner SVD (including wide matrices),
barycentric evaluation, and acceptance calculations are retained.

`python check.py` passed with Python 3.12.10, NumPy 2.4.6, and SciPy 1.18.0.
For the supplied example, both signs passed the residual checks and the
`1e-6` target on the stated output grid. Numerical results can vary with
library versions and platforms. This package contains no compiled libraries,
external data, or complete reproduction scripts for the paper's experiments.
