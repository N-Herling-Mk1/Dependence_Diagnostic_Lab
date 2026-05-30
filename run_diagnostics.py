#!/usr/bin/env python3
"""
run_diagnostics.py -- orchestrates the full NN dependence suite.

    python run_diagnostics.py --demo
    python run_diagnostics.py --features features.csv --scores scores.csv

Reads ROOT->CSV exports (edit config.py for column names), runs:
  A  input feature correlation (within-set + cross-set ceiling)
  B  output score correlation (dCor / MI / sliced / closure)
  bridge  partial dCor -> inherited vs manufactured
Writes a JSON summary + TRON plots to cfg.outdir.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

import config as _cfg
import helix_ui as ui
import core_metrics as cm
import feature_module as fm
import output_module as om
import bridge_module as bm


def synth_demo(n=6000, seed=42):
    """Shared-latent generator: two nets reading the same physics through
    different features -> high input cross-dCor, high (nonlinear) output dCor,
    dependence that should collapse under conditioning on the shared latent."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                       # shared event latent
    w = rng.normal(size=n)                        # NN1-only nuisance
    v = rng.normal(size=n)                        # NN2-only nuisance
    f = pd.DataFrame({
        "vtx_mass":   z + 0.3*rng.normal(size=n),
        "vtx_ntrk":   0.8*z + w + 0.2*rng.normal(size=n),
        "vtx_DR":     w + 0.2*rng.normal(size=n),
        "mu_pt":      0.7*z + 0.4*rng.normal(size=n),
        "ms_seg":     z**2 + 0.5*rng.normal(size=n),     # nonlinear in z
        "iso":        v + 0.3*rng.normal(size=n),
        "eta":        0.6*z + v + 0.2*rng.normal(size=n),
        "phi_spread": v + 0.3*rng.normal(size=n),
    })
    sig = lambda a: 1/(1+np.exp(-a))
    s1 = sig(2.0*np.tanh(0.9*z + 0.5*w) + 0.3*rng.normal(size=n))
    s2 = sig(2.0*np.tanh(0.8*z + 0.6*v) + 0.3*rng.normal(size=n))
    sc = pd.DataFrame({"nn1_score": s1, "nn2_score": s2})
    cfg = _cfg.Config(
        nn1_features=["vtx_mass", "vtx_ntrk", "vtx_DR", "mu_pt"],
        nn2_features=["ms_seg", "iso", "eta", "phi_spread"],
        shared_features=["vtx_mass", "mu_pt"],   # proxies for the latent z
        n_perm=300, n_boot=1500, outdir="helix_demo_report")
    return f, sc, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--features"); ap.add_argument("--scores")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    ui.banner("HELIX  ::  NN DEPENDENCE DIAGNOSTIC SUITE")

    if args.demo:
        feats, scores, cfg = synth_demo()
        ui.status("DEMO mode -- synthetic shared-latent data")
    else:
        cfg = _cfg.Config()
        if args.features: cfg.features_csv = args.features
        if args.scores:   cfg.scores_csv = args.scores
        if not cfg.nn1_features or not cfg.nn2_features:
            ui.warn("set nn1_features / nn2_features in config.py first"); sys.exit(1)
        ui.status(f"loading {cfg.features_csv} / {cfg.scores_csv}")
        feats = pd.read_csv(cfg.features_csv)
        scores = pd.read_csv(cfg.scores_csv)
    if args.outdir: cfg.outdir = args.outdir
    os.makedirs(cfg.outdir, exist_ok=True)

    X1 = feats[cfg.nn1_features].to_numpy(float)
    X2 = feats[cfg.nn2_features].to_numpy(float)
    s1 = scores[cfg.nn1_score].to_numpy(float)
    s2 = scores[cfg.nn2_score].to_numpy(float)
    Z  = feats[cfg.shared_features].to_numpy(float) if cfg.shared_features else None
    ui.status(f"{len(s1):,} events | NN1 {len(cfg.nn1_features)} feats | "
              f"NN2 {len(cfg.nn2_features)} feats")

    R = {}
    R["A1_nn1"] = fm.within_set("NN1", X1, cfg.nn1_features, cfg)
    R["A1_nn2"] = fm.within_set("NN2", X2, cfg.nn2_features, cfg)
    R["A2_cross"] = fm.cross_set(X1, cfg.nn1_features, X2, cfg.nn2_features, cfg)
    R["B_output"] = om.analyze(s1, s2, cfg)
    if Z is not None:
        R["bridge"] = bm.analyze(s1, s2, Z, cfg)

    # ---- DPI sanity line: outputs cannot exceed input ceiling (in MI spirit) ----
    ui.section("THEORY CHECK  ::  data-processing ceiling")
    ui.result("input-set vector dCor (ceiling-ish)", f"{R['A2_cross']['vector_dcor']:.3f}")
    ui.result("output dCor (realized)", f"{R['B_output']['dcor']:.3f}")

    # ---- plots ----
    if cfg.make_plots:
        try:
            import plots
            plots.heatmap(R["A1_nn1"]["dcor"], cfg.nn1_features,
                          "NN1 within-set dCor", f"{cfg.outdir}/nn1_dcor.png", cfg)
            plots.heatmap(R["A1_nn2"]["dcor"], cfg.nn2_features,
                          "NN2 within-set dCor", f"{cfg.outdir}/nn2_dcor.png", cfg)
            plots.spectrum(R["A1_nn1"]["spectrum"], "NN1 PCA spectrum",
                           f"{cfg.outdir}/nn1_spectrum.png", cfg)
            plots.slice_profile(R["B_output"]["slices"],
                                f"{cfg.outdir}/output_profiles.png", cfg)
            ui.status(f"plots written to {cfg.outdir}/")
        except Exception as e:
            ui.warn(f"plotting skipped: {e}")

    # ---- JSON summary (numpy-safe) ----
    def clean(o):
        if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    with open(f"{cfg.outdir}/summary.json", "w") as fh:
        json.dump(clean({k: {kk: vv for kk, vv in v.items()
                             if kk not in ("pearson", "dcor", "spectrum", "cols")}
                         for k, v in R.items()}), fh, indent=2)
    ui.banner("COMPLETE")
    ui.status(f"summary.json + plots in {cfg.outdir}/")


if __name__ == "__main__":
    main()
