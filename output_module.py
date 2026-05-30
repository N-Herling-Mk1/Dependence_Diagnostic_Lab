"""
output_module.py -- OBJECT B: NN1 vs NN2 output-score correlation.

Pearson (the floor) + Spearman, dCor + permutation, mutual information,
sliced dCor (locality), conditional mean AND variance profiles (the
heteroscedastic channel), and the bootstrapped ABCD closure ratio (arbiter).
"""
import numpy as np
from scipy.stats import spearmanr
import core_metrics as cm
import helix_ui as ui


def _abcd(s1, s2, c1, c2):
    hi1, hi2 = s1 > c1, s2 > c2
    A = int(np.sum(hi1 & hi2)); B = int(np.sum(hi1 & ~hi2))
    C = int(np.sum(~hi1 & hi2)); D = int(np.sum(~hi1 & ~hi2))
    return A, B, C, D

def _ratio(s1, s2, c1, c2):
    A, B, C, D = _abcd(s1, s2, c1, c2)
    return np.nan if (D == 0 or A == 0) else A / (B * C / D)


def analyze(s1, s2, cfg):
    ui.section("B  output correlation : NN1 score vs NN2 score")
    pear = float(np.corrcoef(s1, s2)[0, 1])
    spear = float(spearmanr(s1, s2).correlation)

    xa, ya = cm.subsample(s1, s2, n=cfg.dcor_subsample, seed=cfg.seed)
    ui.status("dCor + permutation test")
    dcor, p = cm.dcor_permutation_test(xa, ya, n_perm=cfg.n_perm,
                                       seed=cfg.seed, progress=ui.bar)
    mi = cm.mutual_information(s1, s2, seed=cfg.seed)

    ui.warn(f"Pearson rho = {pear:+.3f}  (linear floor only)")
    ui.result("Spearman rho", f"{spear:+.3f}")
    ui.result("dCor", f"{dcor:.3f}  (p_indep={p:.3f})")
    ui.result("mutual information", f"{mi:.3f} nats")

    # ---- sliced dCor + conditional mean / variance profiles ----
    ui.status("sliced dCor + conditional profiles across NN1")
    edges = np.quantile(s1, np.linspace(0, 1, cfg.n_slices + 1))
    slices = []
    for k in range(cfg.n_slices):
        m = (s1 >= edges[k]) & (s1 <= edges[k + 1])
        if m.sum() < 50:
            slices.append((edges[k], edges[k + 1], np.nan, np.nan,
                           np.nan, int(m.sum()))); continue
        xs, ys = cm.subsample(s1[m], s2[m], n=cfg.matrix_subsample, seed=cfg.seed)
        slices.append((edges[k], edges[k + 1],
                       cm.distance_correlation(xs, ys),
                       float(s2[m].mean()), float(s2[m].var()), int(m.sum())))
    for lo, hi, d, mu, var, n in slices:
        ui.result(f"NN1 [{lo:.2f},{hi:.2f}]",
                  f"dCor={d:.3f}  E[s2]={mu:.3f}  Var[s2]={var:.4f}  n={n}")

    # ---- bootstrapped ABCD closure ----
    ui.status("bootstrapped ABCD closure")
    A, B, C, D = _abcd(s1, s2, cfg.cut_nn1, cfg.cut_nn2)
    obs = _ratio(s1, s2, cfg.cut_nn1, cfg.cut_nn2)
    rng = np.random.default_rng(cfg.seed); n = len(s1)
    boot = np.empty(cfg.n_boot)
    for i in range(cfg.n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _ratio(s1[idx], s2[idx], cfg.cut_nn1, cfg.cut_nn2)
        ui.bar(i, cfg.n_boot, "bootstrap")
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [16, 84])
    ui.result("ABCD regions", f"A={A} B={B} C={C} D={D}")
    ui.result("closure ratio", f"{obs:.3f}  [{lo:.3f},{hi:.3f}]  (1=closes)")

    return dict(pearson=pear, spearman=spear, dcor=dcor, dcor_p=p, mi=mi,
                slices=slices, closure=obs, closure_ci=(float(lo), float(hi)),
                regions=(A, B, C, D))
