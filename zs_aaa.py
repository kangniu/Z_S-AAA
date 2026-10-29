"""Weighted independent AAA and algebraic normalization for real-axis ZS scattering.

The AAA and normalization calculations are extracted from the manuscript sources.
The self-contained numerical scattering example uses SciPy DOP853, not ES8.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import loggamma

def exact_scattering(x, sigma=1, A=5.2, C=4.0):
    """Medvedev et al., Opt. Express 28 (2020), Eq. (60).

    Full-line chirped-sech potential; phase convention matches the IVP below.
    This benchmark routine is not the proposed numerical scattering solver.
    """
    x = np.asarray(x, dtype=float)
    if A == 0:
        return np.column_stack((np.ones_like(x, complex), np.zeros_like(x, complex)))
    D = np.sqrt(complex(sigma*A*A-C*C/4))
    g = loggamma
    a = np.exp(g(.5-1j*(x+C/2))+g(.5-1j*(x-C/2))
               -g(.5-1j*x-D)-g(.5-1j*x+D))
    if C == 0 and sigma == 1 and A == int(A):
        b = np.zeros_like(x, complex)
    else:
        b = np.exp(-1j*C*np.log(2)-np.log(A)
                   +g(.5-1j*(x+C/2))+g(.5+1j*(x-C/2))
                   -g(-1j*C/2-D)-g(-1j*C/2+D))
    return np.column_stack((a, b))


def direct_scattering(x, sigma=1, A=5.2, C=4., L=24.,
                      rtol=2e-11, atol=2e-13, batch_size=64, potential=None):
    """DOP853 applied to interaction-picture ZS equations.

    u=exp(i*x*t)*psi1, v=exp(-i*x*t)*psi2, (u,v)(-L)=(1,0).
    Then (a,b)=(u,v)(L). q is evaluated analytically, not reconstructed
    from equispaced samples. This is an explicit limitation of this pilot.
    """
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty((len(x), 2), complex)
    nfev = 0
    for start in range(0, len(x), batch_size):
        xx = x[start:start+batch_size]
        n = len(xx)
        def rhs(t, y):
            q = (potential(t) if potential is not None else
                 A*np.exp(-(1+1j*C)*np.log(np.cosh(t))))
            z = q*np.exp(2j*xx*t)
            return np.concatenate((z*y[n:], -sigma*z.conj()*y[:n]))
        y0 = np.concatenate((np.ones(n, complex), np.zeros(n, complex)))
        sol = solve_ivp(rhs, (-L, L), y0, method='DOP853',
                        rtol=rtol, atol=atol, t_eval=[L])
        if not sol.success:
            raise RuntimeError(sol.message)
        out[start:start+n] = sol.y[:, -1].reshape(2, n).T
        nfev += sol.nfev
    return out, nfev


def point_errors(y, ref):
    return np.abs(y-ref)/np.maximum(1., np.abs(ref))


def metrics(y, ref, sigma):
    e = point_errors(y, ref)
    h = np.abs(y[:, 0])**2+sigma*np.abs(y[:, 1])**2-1
    # In the indefinite case raw invariant subtraction is ill-conditioned.
    scale = np.maximum(1., np.sum(np.abs(y)**2, axis=1))
    return dict(max_a=float(e[:, 0].max()), max_b=float(e[:, 1].max()),
                nmse_a=float(np.mean(e[:, 0]**2)),
                nmse_b=float(np.mean(e[:, 1]**2)),
                invariant_absolute=float(np.max(np.abs(h))),
                invariant_scaled=float(np.max(np.abs(h)/scale)))


class CachedOracle:
    def __init__(self, sigma, mode='direct', sample_budget=8192, **kwargs):
        if mode not in ('direct', 'exact') or sigma not in (1, -1):
            raise ValueError('Use sigma=1/-1 and mode=direct/exact')
        self.sigma, self.mode, self.kwargs = sigma, mode, kwargs
        self.sample_budget = sample_budget
        self.cache = {}
        self.seconds = 0.
        self.nfev = 0
    def __call__(self, x):
        x = np.atleast_1d(x)
        missing = np.array(sorted(set(float(t) for t in x)-self.cache.keys()))
        if self.sample_budget is not None and len(self.cache) + len(missing) > self.sample_budget:
            raise RuntimeError('Scattering sample budget exceeded')
        if len(missing):
            t0 = time.perf_counter()
            if self.mode == 'exact':
                f = exact_scattering(missing, self.sigma, **self.kwargs)
            else:
                f, nf = direct_scattering(missing, self.sigma, **self.kwargs)
                self.nfev += nf
            self.seconds += time.perf_counter()-t0
            self.cache.update(zip(missing, f))
        return np.array([self.cache[float(t)] for t in x])


class WeightedVectorAAA:
    """Shared-denominator AAA with componentwise mixed-error weighting.

    A proposed variant, not a claim about standard AAA. Fixed row weights
    1/max(1,abs(f_l(x))) enter the stacked Loewner least-squares problem.
    No exact invariant enforcement and no Froissart cleanup are included.
    """
    def __init__(self, x, f, tol=1e-10, max_terms=90):
        self.x, self.f = np.asarray(x), np.asarray(f)
        self.scale = np.maximum(1., np.abs(f))
        residual = f-np.mean(f, axis=0)
        selected = []
        self.history = []
        for _ in range(min(max_terms, len(x)-1)):
            score = np.max(np.abs(residual)/self.scale, axis=1)
            score[selected] = -1
            selected.append(int(np.argmax(score)))
            j = np.array(selected)
            rest = np.ones(len(x), bool); rest[j] = False
            z, fj = x[j], f[j]
            cauchy = 1/(x[rest, None]-z[None, :])
            blocks = [(f[rest, l, None]-fj[None, :, l])*cauchy
                      /self.scale[rest, l, None] for l in range(f.shape[1])]
            loewner = np.vstack(blocks)
            if loewner.shape[0] < loewner.shape[1]:
                # The full right factor contains the nullspace of a wide matrix.
                _, _, vh = np.linalg.svd(loewner, full_matrices=True)
            else:
                _, _, vh = np.linalg.svd(loewner, full_matrices=False)
            self.z, self.fj, self.w = z, fj, vh[-1].conj()
            r = self(x)
            residual = f-r
            err = float(np.max(np.abs(residual)/self.scale))
            self.history.append(err)
            if err <= tol:
                break
        self.sample_error = self.history[-1]
    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        r = np.empty((len(x), self.fj.shape[1]), complex)
        for start in range(0, len(x), 2048):
            xx = x[start:start+2048]
            diff = xx[:, None]-self.z
            matches = diff == 0
            hit = np.any(matches, axis=1)
            rr = np.empty((len(xx), self.fj.shape[1]), complex)
            rr[hit] = self.fj[np.argmax(matches[hit], axis=1)]
            if np.any(~hit):
                cw = self.w[None, :]/diff[~hit]
                rr[~hit] = (cw@self.fj)/np.sum(cw, axis=1)[:, None]
            r[start:start+len(xx)] = rr
        return r
    @property
    def terms(self):
        return len(self.z)


class WeightedIndependentAAA:
    def __init__(self, x, f, tol=1e-10):
        self.models = [WeightedVectorAAA(x, f[:, j:j+1], tol=tol)
                       for j in range(f.shape[1])]
    def __call__(self, x):
        return np.column_stack([r(x)[:, 0] for r in self.models])
    @property
    def terms(self):
        return [r.terms for r in self.models]


def adaptive_aaa(oracle, tol=1e-8, halfwidth=20., max_rounds=6, seed=20261027):
    """Empirical adaptive mesh validation, NOT certified continuum AAA.

    All oracle calls (including validation and successful stopping checks)
    count toward sampling cost. Failed validation nodes join the fit set.
    Two consecutive successful validation rounds are required.
    """
    x = np.linspace(-halfwidth, halfwidth, 65)
    f = oracle(x)
    rng = np.random.default_rng(seed)
    log = []; good = 0
    start = time.perf_counter()
    for level in range(max_rounds):
        model = WeightedIndependentAAA(x, f, tol=tol*.05)
        mid = (x[:-1]+x[1:])/2
        probe = np.unique(np.r_[mid, rng.uniform(-halfwidth, halfwidth, 64)])
        fp = oracle(probe)
        err = float(np.max(point_errors(model(probe), fp)))
        log.append(dict(round=level, fit_nodes=len(x), validation_nodes=len(probe),
                        unique_samples=len(oracle.cache), terms=model.terms,
                        validation_error=err))
        if err <= tol*.5:
            good += 1
            if good == 2:
                break
            # Keep the fit fixed and test another independently shifted mesh.
            probe2 = rng.uniform(-halfwidth, halfwidth, 128)
            e2 = float(np.max(point_errors(model(probe2), oracle(probe2))))
            log[-1]['second_validation_error'] = e2
            if e2 <= tol*.5:
                good = 2
                break
            good = 0  # A refitted model must pass both validation batches.
            probe = np.r_[probe, probe2]
        else:
            good = 0
        x = np.unique(np.r_[x, probe]); f = oracle(x)
    return model, dict(history=log, samples=len(oracle.cache),
                       oracle_seconds=oracle.seconds,
                       construction_seconds=time.perf_counter()-start,
                       validated=good == 2, nfev=oracle.nfev)


def normalize(f,sigma):
    f=np.asarray(f,dtype=complex)
    if sigma not in (1,-1) or f.ndim!=2 or f.shape[1]!=2:
        raise ValueError('Invalid sign or coefficient shape')
    if not np.all(np.isfinite(f)):raise FloatingPointError('Nonfinite input')
    if sigma==1:
        scale=np.max(np.maximum(abs(f.real),abs(f.imag)),axis=1)
        if np.any(scale==0):raise FloatingPointError('Zero compact pair')
        z=f/scale[:,None]
        norm=np.sqrt(np.sum(z.real*z.real+z.imag*z.imag,axis=1))
        out=z/norm[:,None]
    else:
        sa=np.maximum(abs(f[:,0].real),abs(f[:,0].imag))
        if np.any(sa==0):raise FloatingPointError('Zero first coefficient')
        az=f[:,0]/sa;phase=az/np.hypot(az.real,az.imag)
        sb=np.maximum(1.,np.maximum(abs(f[:,1].real),abs(f[:,1].imag)))
        bz=f[:,1]/sb
        amplitude=np.hypot(1./sb,np.hypot(bz.real,bz.imag))
        out=f.copy()
        with np.errstate(over='ignore',invalid='ignore'):
            out[:,0]=(phase*amplitude)*sb
    if not np.all(np.isfinite(out)):raise FloatingPointError('Nonfinite normalized output')
    return out
