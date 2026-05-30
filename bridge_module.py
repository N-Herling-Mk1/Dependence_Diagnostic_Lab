"""
bridge_module.py -- connects Object A and Object B.

Question: is the output dependence INHERITED (rode in through shared input
physics) or MANUFACTURED (the architectures create it)?

partial_dcor(s1, s2 ; shared)  projects the shared features out of the
score-score dependence:
  * collapses toward 0  -> inherited; feature-level fixes (more orthogonal
                            inputs) will help.
  * survives            -> manufactured; go to output-level decorrelation
                            (DisCo) or model it (MaxEnt tilt).

event_mixing_dcor is the rigorous complement (needs the trained models): it
rebuilds NN1/NN2 inputs from DIFFERENT events to destroy the shared latent,
then re-measures output dCor. The drop = dependence due to the shared event.
"""
import numpy as np
import core_metrics as cm
import helix_ui as ui


def analyze(s1, s2, shared_Z, cfg):
    ui.section("BRIDGE  inherited vs manufactured dependence")
    xa, ya, za = cm.subsample(s1, s2, shared_Z,
                              n=cfg.dcor_subsample, seed=cfg.seed)
    plain = cm.distance_correlation(xa, ya)
    pdc = cm.partial_distance_corr(xa, ya, za)

    ui.result("dCor(s1, s2)", f"{plain:.3f}")
    ui.result("partial dCor(s1, s2 ; shared)", f"{pdc:.3f}")
    drop = plain - max(pdc, 0.0)
    frac = drop / plain if plain > 0 else 0.0
    ui.result("dependence removed by conditioning", f"{frac*100:.0f}%")
    verdict = ("INHERITED (feature-level fix viable)" if pdc < 0.1
               else "MANUFACTURED (needs output-level fix)")
    ui.result("verdict", verdict)
    return dict(dcor=plain, partial_dcor=pdc, explained_fraction=float(frac),
                verdict=verdict)


def event_mixing_dcor(predict_nn1, predict_nn2, X1, X2, sub=2000, seed=0):
    """
    Rigorous shared-latent test (requires trained models).
      predict_nn1, predict_nn2 : callables features -> score
      X1, X2 : aligned per-event input feature arrays for the two nets
    Returns (dcor_aligned, dcor_mixed). A large drop => dependence was driven
    by the shared event; a small drop => architecture-level coupling.
    """
    rng = np.random.default_rng(seed)
    X1, X2 = cm.subsample(X1, X2, n=sub, seed=seed)
    s1 = np.ravel(predict_nn1(X1)); s2 = np.ravel(predict_nn2(X2))
    d_aligned = cm.distance_correlation(s1, s2)
    perm = rng.permutation(len(X2))            # NN2 reads a different event
    s2_mixed = np.ravel(predict_nn2(X2[perm]))
    d_mixed = cm.distance_correlation(s1, s2_mixed)
    return float(d_aligned), float(d_mixed)
