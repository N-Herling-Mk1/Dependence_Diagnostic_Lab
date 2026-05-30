"""
core_metrics.py -- distribution-free dependence metrics.

Everything here works on vectors (n, d>=1), so the same function serves
scalar score-vs-score and whole-feature-set-vs-feature-set questions.

Implemented:
  distance_correlation        biased dCor in [0,1]; 0 iff independent (population)
  dcor_permutation_test       p-value of dCor against the independence null
  hsic / hsic_permutation     kernel (RBF) independence, normalized to [0,1]
  mutual_information          I(x;y) via KSG (sklearn) with binned fallback
  partial_distance_corr       Szekely-Rizzo partial dCor (conditional dependence)
  canonical_correlations      CCA spectrum (linear set-to-set baseline)
"""
import numpy as np

try:
    from sklearn.feature_selection import mutual_info_regression
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# ----------------------------------------------------------------------
def _as2d(x):
    x = np.asarray(x, float)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def _pdist(X):
    """Euclidean pairwise distance matrix for rows of X (n, d)."""
    G = X @ X.T
    sq = np.diag(G).copy()
    d2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)

def subsample(*arrays, n=2000, seed=0):
    """Jointly subsample rows (dCor/HSIC are O(n^2) memory)."""
    m = len(arrays[0])
    if m <= n:
        return arrays if len(arrays) > 1 else arrays[0]
    idx = np.random.default_rng(seed).choice(m, size=n, replace=False)
    out = tuple(np.asarray(a)[idx] for a in arrays)
    return out if len(out) > 1 else out[0]


# ----------------------------------------------------------------------
# Distance correlation (biased; the standard estimator)
# ----------------------------------------------------------------------
def _dcenter(D):
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()

def distance_correlation(x, y):
    """dCor in [0,1]. 0 <=> independence at the population level."""
    A = _dcenter(_pdist(_as2d(x)))
    B = _dcenter(_pdist(_as2d(y)))
    dcov2 = (A * B).mean()
    vx, vy = (A * A).mean(), (B * B).mean()
    denom = np.sqrt(vx * vy)
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 0 else 0.0

def dcor_permutation_test(x, y, n_perm=500, seed=0, progress=None):
    # precompute centered distance matrices once; permutation only reindexes
    A = _dcenter(_pdist(_as2d(x)))
    B = _dcenter(_pdist(_as2d(y)))
    vx, vy = (A * A).mean(), (B * B).mean()
    denom = np.sqrt(vx * vy)
    obs = float(np.sqrt(max((A * B).mean(), 0.0) / denom)) if denom > 0 else 0.0
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]            # = double-center of permuted y
        null[i] = (np.sqrt(max((A * Bp).mean(), 0.0) / denom)
                   if denom > 0 else 0.0)
        if progress:
            progress(i, n_perm, "permute")
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(p)


# ----------------------------------------------------------------------
# HSIC (RBF kernel), normalized to a [0,1] correlation-like quantity
# ----------------------------------------------------------------------
def _rbf(X, sigma=None):
    D = _pdist(_as2d(X))
    if sigma is None:
        med = np.median(D[D > 0])
        sigma = med if med > 0 else 1.0
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2))

def _kcenter(K):
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()

def normalized_hsic(x, y):
    Kc = _kcenter(_rbf(x))
    Lc = _kcenter(_rbf(y))
    num = np.sum(Kc * Lc)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(num / den) if den > 0 else 0.0

def hsic_permutation_test(x, y, n_perm=500, seed=0, progress=None):
    Kc = _kcenter(_rbf(x))
    Lc = _kcenter(_rbf(y))
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    obs = float(np.sum(Kc * Lc) / den) if den > 0 else 0.0
    rng = np.random.default_rng(seed)
    n = Kc.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        Lp = Lc[np.ix_(perm, perm)]
        null[i] = float(np.sum(Kc * Lp) / den) if den > 0 else 0.0
        if progress:
            progress(i, n_perm, "hsic-perm")
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(p)


# ----------------------------------------------------------------------
# Mutual information  (KSG via sklearn; binned fallback)
# ----------------------------------------------------------------------
def mutual_information(x, y, seed=0):
    x = np.ravel(np.asarray(x, float))
    y = np.ravel(np.asarray(y, float))
    if _HAVE_SK:
        a = mutual_info_regression(x.reshape(-1, 1), y, random_state=seed)[0]
        b = mutual_info_regression(y.reshape(-1, 1), x, random_state=seed)[0]
        return float(max(0.0, 0.5 * (a + b)))  # nats
    # histogram fallback
    bins = max(8, int(np.sqrt(len(x) / 5)))
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(1, keepdims=True)
    p_y = p_xy.sum(0, keepdims=True)
    nz = p_xy > 0
    return float(np.sum(p_xy[nz] * np.log(p_xy[nz] / (p_x @ p_y)[nz])))


# ----------------------------------------------------------------------
# Partial distance correlation (Szekely & Rizzo 2014)
#   pdCor(X, Y ; Z): dependence between X and Y with Z projected out.
#   Uses U-centered distance matrices (unbiased inner product).
# ----------------------------------------------------------------------
def _u_center(D):
    n = D.shape[0]
    rs = D.sum(1)
    total = D.sum()
    U = (D
         - rs[:, None] / (n - 2)
         - rs[None, :] / (n - 2)
         + total / ((n - 1) * (n - 2)))
    np.fill_diagonal(U, 0.0)
    return U

def _u_dot(A, B):
    n = A.shape[0]
    return float(np.sum(A * B) / (n * (n - 3)))

def partial_distance_corr(x, y, z):
    """pdCor(x, y ; z) in roughly [-1,1]; ~0 => no dependence beyond z."""
    Ax = _u_center(_pdist(_as2d(x)))
    Ay = _u_center(_pdist(_as2d(y)))
    Az = _u_center(_pdist(_as2d(z)))
    czz = _u_dot(Az, Az)
    if czz <= 0:
        return distance_correlation(x, y)  # z carries no info -> plain dCor
    Px = Ax - (_u_dot(Ax, Az) / czz) * Az
    Py = Ay - (_u_dot(Ay, Az) / czz) * Az
    nx, ny = np.sqrt(max(_u_dot(Px, Px), 0)), np.sqrt(max(_u_dot(Py, Py), 0))
    if nx <= 0 or ny <= 0:
        return 0.0
    return float(_u_dot(Px, Py) / (nx * ny))


# ----------------------------------------------------------------------
# Canonical correlations (linear set-to-set; CCA spectrum)
# ----------------------------------------------------------------------
def canonical_correlations(X, Y, ridge=1e-6):
    X = _as2d(X) - _as2d(X).mean(0)
    Y = _as2d(Y) - _as2d(Y).mean(0)
    n = len(X)
    Sxx = X.T @ X / (n - 1) + ridge * np.eye(X.shape[1])
    Syy = Y.T @ Y / (n - 1) + ridge * np.eye(Y.shape[1])
    Sxy = X.T @ Y / (n - 1)
    M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
    eig = np.linalg.eigvals(M).real
    eig = np.clip(eig, 0.0, 1.0)
    return np.sqrt(np.sort(eig)[::-1])


# ----------------------------------------------------------------------
# Redundancy summaries
# ----------------------------------------------------------------------
def participation_ratio(corr):
    """Effective dimensionality from a correlation/covariance matrix."""
    lam = np.linalg.eigvalsh(corr)
    lam = np.clip(lam.real, 0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0, np.sort(lam)[::-1]

def gaussian_total_correlation(X):
    """TC = -0.5 log det(R): multivariate redundancy under a Gaussian model."""
    R = np.corrcoef(_as2d(X), rowvar=False)
    R = np.atleast_2d(R)
    sign, logdet = np.linalg.slogdet(R + 1e-9 * np.eye(R.shape[0]))
    return float(-0.5 * logdet) if sign > 0 else float("nan")

def dcor_matrix(X, cols, sub=1500, seed=0, progress=None):
    """Pairwise dCor matrix over the columns of X."""
    X = _as2d(X)
    p = X.shape[1]
    M = np.eye(p)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    for k, (i, j) in enumerate(pairs):
        xi, xj = subsample(X[:, i], X[:, j], n=sub, seed=seed)
        M[i, j] = M[j, i] = distance_correlation(xi, xj)
        if progress:
            progress(k, len(pairs), "dcor-mat")
    return M
