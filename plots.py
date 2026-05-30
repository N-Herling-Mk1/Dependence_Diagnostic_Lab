"""
plots.py -- TRON Ares dark plots for the report (Agg backend, saves PNGs).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style(cfg):
    plt.rcParams.update({
        "figure.facecolor": cfg.bg, "axes.facecolor": cfg.bg,
        "savefig.facecolor": cfg.bg, "text.color": cfg.fg,
        "axes.labelcolor": cfg.fg, "xtick.color": cfg.fg,
        "ytick.color": cfg.fg, "axes.edgecolor": cfg.accent,
        "font.size": 9,
    })

def heatmap(M, labels, title, path, cfg):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(max(4, .5*len(labels)), max(3.5, .5*len(labels))))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, color=cfg.accent)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def spectrum(spec, title, path, cfg):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, len(spec)+1), spec, color=cfg.fg, edgecolor=cfg.accent)
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue")
    ax.set_title(title, color=cfg.accent)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def slice_profile(slices, path, cfg):
    _style(cfg)
    mids = [0.5*(lo+hi) for lo, hi, *_ in slices]
    dc = [s[2] for s in slices]; mu = [s[3] for s in slices]; var = [s[4] for s in slices]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for a, y, t, col in zip(ax, [dc, mu, var],
                            ["sliced dCor", "E[s2 | s1]", "Var[s2 | s1]"],
                            [cfg.hot, cfg.fg, cfg.accent]):
        a.plot(mids, y, "o-", color=col)
        a.set_xlabel("NN1 score bin"); a.set_title(t, color=cfg.accent)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def closure_hist(boot, obs, ci, path, cfg):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(boot, bins=40, color=cfg.fg, alpha=.7)
    ax.axvline(1.0, color=cfg.hot, lw=2, label="closure=1")
    ax.axvline(obs, color=cfg.accent, lw=2, ls="--", label=f"obs={obs:.2f}")
    ax.set_xlabel("ABCD closure ratio"); ax.legend()
    ax.set_title("bootstrapped closure", color=cfg.accent)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
