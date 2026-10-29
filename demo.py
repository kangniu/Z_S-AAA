"""Run AAA + normalization with a portable DOP853 scattering backend."""
import argparse
import json
import numpy as np
from zs_aaa import CachedOracle, adaptive_aaa, exact_scattering, metrics, normalize


def run(sigma=1, target=1e-6, points=513):
    # The analytic full-line solution is used only for the final comparison.
    amplitude, chirp, half_length, half_width = 1.2, 1.0, 24.0, 6.0
    oracle = CachedOracle(sigma, mode='direct', sample_budget=8192,
                          A=amplitude, C=chirp, L=half_length,
                          rtol=2e-11, atol=2e-13, batch_size=64)
    model, info = adaptive_aaa(oracle, tol=target/3, halfwidth=half_width)
    result = dict(sigma=sigma, target=target, theta=target/3,
                  backend='DOP853', half_length=half_length,
                  spectral_halfwidth=half_width, output_points=points,
                  amplitude=amplitude, chirp=chirp,
                  samples=info['samples'], support_points=model.terms,
                  residual_checks_passed=bool(info['validated']),
                  rounds=info['history'])
    if not info['validated']:
        result['termination'] = 'maximum fitting rounds reached'
        return result
    x = np.linspace(-half_width, half_width, points)
    raw = model(x)
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError('Nonfinite rational output')
    normalized = normalize(raw, sigma)
    reference = exact_scattering(x, sigma, A=amplitude, C=chirp)
    raw_metrics = metrics(raw, reference, sigma)
    final_metrics = metrics(normalized, reference, sigma)
    raw_error = max(raw_metrics['max_a'], raw_metrics['max_b'])
    final_error = max(final_metrics['max_a'], final_metrics['max_b'])
    result.update(termination='residual checks passed',
                  raw_error_against_full_line=raw_error,
                  normalized_error_against_full_line=final_error,
                  scaled_invariant_defect=final_metrics['invariant_scaled'],
                  output_grid_target_met=bool(final_error <= target))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigma', type=int, choices=(1, -1), default=1)
    parser.add_argument('--target', type=float, default=1e-6)
    parser.add_argument('--points', type=int, default=513)
    args = parser.parse_args()
    if not np.isfinite(args.target) or args.target <= 0 or args.points < 2:
        parser.error('The target must be positive and finite; points must be at least two.')
    print(json.dumps(run(args.sigma, args.target, args.points), indent=2,
                     allow_nan=False))
