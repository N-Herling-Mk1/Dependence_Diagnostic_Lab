"""
feature_module.py -- OBJECT A: input feature correlation.

A1 within-set redundancy : Pearson + dCor matrices, PCA spectrum +
                           participation ratio, Gaussian total correlation.
A2 cross-set overlap     : Jaccard of feature lists, CCA spectrum,
                           vector dCor and normalized HSIC between the two
                           input sets (this sets the DPI ceiling for outputs).
"""
import numpy as np
import core_metrics as cm
import helix_ui as ui


def within_set(name, X, cols, cfg):
    ui.section(f"A1 within-set redundancy : {name}")
    R = np.atleast_2d(np.corrcoef(X, rowvar=False))
    pr, spectrum = cm.participation_ratio(R)
    tc = cm.gaussian_total_correlation(X)
    ui.status(f"computing dCor matrix over {len(cols)} features")
    D = cm.dcor_matrix(X, cols, sub=cfg.matrix_subsample,
                       seed=cfg.seed, progress=ui.bar)

    # off-diagonal summaries
    iu = np.triu_indices(len(cols), 1)
    ui.result("nominal features", len(cols))
    ui.result("participation ratio (eff. dim)", f"{pr:.2f}")
    ui.result("max |Pearson| off-diag", f"{np.abs(R[iu]).max():.3f}")
    ui.result("max dCor off-diag", f"{D[iu].max():.3f}")
    ui.result("Gaussian total correlation", f"{tc:.3f} nats")
    return dict(name=name, pearson=R, dcor=D, pr=pr, spectrum=spectrum,
                total_corr=tc, cols=cols)


def cross_set(X1, c1, X2, c2, cfg):
    ui.section("A2 cross-set overlap : NN1 inputs vs NN2 inputs  (the ceiling)")
    inter = set(c1) & set(c2)
    union = set(c1) | set(c2)
    jacc = len(inter) / len(union) if union else 0.0

    Xa, Xb = cm.subsample(X1, X2, n=cfg.dcor_subsample, seed=cfg.seed)
    ui.status("vector dCor between input sets")
    vdcor, vp = cm.dcor_permutation_test(Xa, Xb, n_perm=cfg.n_perm,
                                         seed=cfg.seed, progress=ui.bar)
    ui.status("normalized HSIC between input sets")
    nhsic, hp = cm.hsic_permutation_test(Xa, Xb, n_perm=cfg.n_perm,
                                         seed=cfg.seed, progress=ui.bar)
    ccs = cm.canonical_correlations(X1, X2)

    ui.result("shared raw features", f"{len(inter)} ({sorted(inter)})")
    ui.result("Jaccard(feature lists)", f"{jacc:.3f}")
    ui.result("top canonical correlation", f"{ccs[0]:.3f}")
    ui.result("vector dCor (sets)", f"{vdcor:.3f}  (p={vp:.3f})")
    ui.result("normalized HSIC (sets)", f"{nhsic:.3f}  (p={hp:.3f})")
    return dict(jaccard=jacc, shared=sorted(inter), cca=ccs,
                vector_dcor=vdcor, vector_dcor_p=vp,
                hsic=nhsic, hsic_p=hp)
