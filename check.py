"""Small functional checks; run with python check.py."""
import json
import platform
import numpy as np
import scipy
from demo import run
from zs_aaa import CachedOracle, WeightedVectorAAA, normalize


def main():
    zero = CachedOracle(1, A=0, L=1, rtol=1e-11, atol=1e-13)
    x = np.linspace(-2, 2, 5)
    np.testing.assert_array_equal(zero(x), np.column_stack((np.ones(5), np.zeros(5))))
    count = len(zero.cache)
    zero(x[::-1])
    assert len(zero.cache) == count
    capped = CachedOracle(1, A=0, L=1, sample_budget=2)
    try:
        capped(x)
    except RuntimeError:
        assert not capped.cache
    else:
        raise AssertionError('The sample budget was not enforced')

    train = np.linspace(-1, 1, 33)
    f = (1 + 1j*train) / (2 + train)
    model = WeightedVectorAAA(train, f[:, None], tol=1e-12)
    test = np.linspace(-1, 1, 57)
    rational_error = float(np.max(np.abs(model(test)[:, 0] - (1+1j*test)/(2+test))))
    assert rational_error < 1e-11

    # Four supports and one remaining data row exercise the full right nullspace.
    wide_x = np.linspace(-1, 1, 5)
    wide = WeightedVectorAAA(wide_x, np.exp((1+1j)*wide_x)[:, None],
                             tol=0, max_terms=4)
    assert wide.terms == 4 and np.all(np.isfinite(wide(wide_x)))

    large = np.array([[1e300+1e300j, 1e300-1e300j]])
    compact = normalize(large, 1)
    assert np.all(np.isfinite(compact))
    np.testing.assert_allclose(np.sum(np.abs(compact)**2, axis=1), 1, atol=5e-16)
    noncompact = normalize(large, -1)
    assert np.all(np.isfinite(noncompact))
    np.testing.assert_array_equal(noncompact[:, 1], large[:, 1])
    for sigma, pair in [(1, [[0, 0]]), (-1, [[0, 1]])]:
        try:
            normalize(pair, sigma)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('The normalization domain was not checked')

    examples = [run(1), run(-1)]
    assert all(r['residual_checks_passed'] and r['output_grid_target_met'] for r in examples)
    print(json.dumps(dict(status='passed', python=platform.python_version(),
                          numpy=np.__version__, scipy=scipy.__version__,
                          rational_test_error=rational_error,
                          wide_matrix_supports=wide.terms,
                          zero_potential_cache_and_budget_checks='passed',
                          large_value_normalization_and_domain_checks='passed',
                          examples=examples), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
