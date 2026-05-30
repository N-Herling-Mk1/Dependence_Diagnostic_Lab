# HELIX :: NN Dependence Diagnostic Suite

Distribution-free dependence diagnostics for the Run 3 LLP MSVtx ABCD analysis.
Measures **two distinct objects** and connects them:

- **Object A — input feature correlation** (within-set redundancy + cross-set overlap)
- **Object B — output score correlation** (NN1 vs NN2 discriminants)
- **Bridge** — partial distance correlation: is output dependence *inherited*
  from shared inputs, or *manufactured* by the architectures?

Pearson is reported only as a linear floor / non-Gaussianity cross-check. The
independence decision rests on dCor (zero **iff** independent, for any
distribution), with closure as the final physics arbiter. See
`docs/` for the rationale and the full pipeline writeup.

## Quick start

```bash
pip install -r requirements.txt
python run_diagnostics.py --demo            # synthetic shared-latent example
```

For real data, edit `config.py` (feature/score columns, ABCD cuts), then:

```bash
python run_diagnostics.py --features features.csv --scores scores.csv
```

Outputs land in `helix_dep_report/`: `summary.json` + TRON-styled plots
(within-set dCor heatmaps, PCA spectrum, sliced profiles).

## Files

| file | role |
|------|------|
| `core_metrics.py`   | dCor (vector), permutation test, HSIC, MI, partial dCor, CCA, PR, total correlation |
| `feature_module.py` | Object A: within-set matrices + cross-set ceiling |
| `output_module.py`  | Object B: dCor / MI / sliced / conditional profiles / bootstrapped closure |
| `bridge_module.py`  | partial dCor (inherited vs manufactured) + event-mixing hook |
| `run_diagnostics.py`| orchestrator + `--demo` |
| `helix_ui.py`       | TRON Ares terminal status / progress |
| `plots.py`          | dark-theme report plots |
| `config.py`         | column names, cuts, sampling, palette |
| `docs/`             | `pearson_vs_dcor` (metric rationale) + `nn_dependence_pipeline` (.tex/.pdf) |

## Notes

- dCor / HSIC are O(n^2) in memory — the suite subsamples (see `config.py`).
- Run on **background-only**, **continuous (pre-quantization)** scores: the
  barrel NN-output quantization biases dCor/MI estimation and manufactures
  spurious structure at the cut lattice.
- `mutual_information` uses the KSG estimator via scikit-learn when available,
  with a histogram fallback.
- The event-mixing diagnostic in `bridge_module.py` is a hook: it needs the
  trained `predict` callables to be rigorous (re-inference on mixed features).
