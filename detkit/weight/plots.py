"""Report-quality plots for the weight features (matplotlib is imported lazily; ``pyplot()`` raises ImportError
when it is missing). Used by ``detkit weight features``.

Photos of one pile (``pile_id``) share a colour and a marker everywhere. They also share one measured weight, so in a
weight-against-feature plot they sit in one horizontal row; the diamond is the mean of the pile. A photo without a
``pile_id`` is a pile of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MARKERS = "osv^PX<>*hp"                    # no "D": the diamond is the pile mean
RC = {"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "axes.spines.top": False,
      "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
      "legend.fontsize": 8, "savefig.dpi": 200}
FIT_COLOR = "#d62728"


def pyplot(show: bool = False):
    """``matplotlib.pyplot`` (headless backend unless ``show``)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def unit_of(feat: str) -> str:
    if feat.startswith("count"):
        return "kernels"
    if feat.endswith("_mm2"):
        return "mm²"
    if feat.endswith("area_px"):
        return "px²"
    if feat.endswith("_px"):
        return "px"
    return "–" if feat.endswith("ratio") else ""


def label_of(feat: str) -> str:
    u = unit_of(feat)
    return f"{feat} ({u})" if u else feat


def pile_of(s) -> str:
    """Name of the pile a photo belongs to."""
    pid = s.meta.get("pile_id")
    return f"pile {pid}" if pid not in (None, "") else f"photo {Path(s.file).stem}"


def _natural(label: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", label)]


def pile_styles(samples, plt) -> dict:
    """``{pile name: (colour, marker)}``; every pile gets its own colour / marker pair."""
    keys = sorted({pile_of(s) for s in samples}, key=_natural)
    cmap = plt.get_cmap("tab10" if len(keys) <= 10 else "tab20")
    return {k: (cmap(i % cmap.N), MARKERS[i % len(MARKERS)]) for i, k in enumerate(keys)}


def _groups(rows) -> dict:
    g: dict = {}
    for s in rows:
        g.setdefault(pile_of(s), []).append(s)
    return dict(sorted(g.items(), key=lambda kv: _natural(kv[0])))


def pile_legend(fig, styles: dict, counts: dict | None = None, max_piles: int = 16) -> None:
    """One legend of the piles at the bottom of a figure (skipped when there are too many)."""
    from matplotlib.lines import Line2D
    if not styles or len(styles) > max_piles:
        return
    h = [Line2D([0], [0], marker=mk, ls="", mfc=col, mec="white", ms=7,
                label=k + (f" ({counts[k]})" if counts and k in counts else ""))
         for k, (col, mk) in styles.items()]
    h.append(Line2D([0], [0], marker="D", ls="", mfc="none", mec="k", ms=8, label="mean of the pile"))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), fontsize=8, frameon=False)


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def spread_text(groups: dict, feat: str) -> str:
    """Within-pile spread (photos of one pile: photographing / detector noise) against between-pile spread."""
    vals = {k: np.array([s.get(feat) for s in g], dtype=float) for k, g in groups.items()}
    allv = np.concatenate(list(vals.values()))
    multi = [v for v in vals.values() if len(v) > 1]
    lines = []
    if multi:
        w = float(np.sqrt(np.mean([v.var(ddof=1) for v in multi])))
        pct = f" = {100 * w / abs(allv.mean()):.1f} % of the mean" if allv.mean() else ""
        lines.append(f"within a pile: sd {_fmt(w)}{pct}")
    else:
        lines.append("one photo per pile: no within-pile spread")
    if len(vals) > 1:
        b = float(np.std([v.mean() for v in vals.values()], ddof=1))
        lines.append(f"between piles: sd {_fmt(b)} (of the pile means)")
    return "   |   ".join(lines)


def draw_feature(ax, feat: str, allp, usable, e: dict | None, styles: dict, big: bool = True) -> None:
    """One feature. With a correlation ``e`` (a row of ``feature_correlations``): measured weight against the
    feature, with the free least-squares line (red) and the line through the origin (dashed) that
    ``weight = feature x constant`` uses (its slope is the constant). Without one (no weights, or a feature that
    does not vary): the value of every photo, grouped by pile, with the pile means and the spread."""
    sz = 52 if big else 26
    if e:
        groups = _groups(usable)
        for k, g in groups.items():
            col, mk = styles[k]
            x = np.array([s.get(feat) for s in g], dtype=float)
            y = np.array([s.weight_g for s in g], dtype=float)
            ax.scatter(x, y, color=col, marker=mk, s=sz * (1.9 if mk == "*" else 1.0), edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
            if len(g) > 1:
                ax.scatter([x.mean()], [y.mean()], marker="D", s=sz * 2.8, facecolor="none", edgecolor="k",
                           linewidth=1.3, zorder=4)
        x = np.array([s.get(feat) for s in usable], dtype=float)
        y = np.array([s.weight_g for s in usable], dtype=float)
        xs = np.array([min(0.0, x.min()), x.max()])
        ax.plot(xs, e["slope"] * xs + e["intercept"], "-", color=FIT_COLOR, lw=1.5,
                label=f"least-squares fit: slope {e['slope']:.3g}, intercept {e['intercept']:.3g} g")
        c0 = float((x * y).sum() / (x * x).sum()) if (x * x).sum() else 0.0
        ax.plot(xs, c0 * xs, "k--", lw=1.1, label=f"through the origin: {c0:.3g} g per {unit_of(feat) or 'unit'}")
        ci = f"  (95% CI {e['r_lo']:+.2f} .. {e['r_hi']:+.2f})" if e.get("r_lo") is not None else ""
        ax.set_title(f"{feat}\nr = {e['r']:+.3f}{ci}, R² = {e['r2']:.3f}; {len(x)} photos, {len(groups)} piles",
                     loc="left", fontsize=9 if not big else 10.5)
        ax.set_xlabel(label_of(feat))
        ax.set_ylabel("measured weight (g)")
        ax.set_xlim(left=min(0.0, x.min()))
        ax.set_ylim(bottom=min(0.0, y.min()))
        if big:
            ax.legend(loc="upper left", fontsize=8)
        return
    groups = _groups(allp)
    pos, ticks, names, means = 0, [], [], []
    for k, g in groups.items():
        col, mk = styles[k]
        v = np.array([s.get(feat) for s in g], dtype=float)
        xs = np.arange(pos, pos + len(g))
        ax.scatter(xs, v, color=col, marker=mk, s=sz * (1.9 if mk == "*" else 1.0), edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        ax.hlines(v.mean(), xs[0] - 0.4, xs[-1] + 0.4, colors="k", lw=1.8, zorder=4)
        ticks.append(xs.mean())
        names.append(k.split(" ", 1)[1])
        means.append(v.mean())
        pos += len(g) + 1
    ax.set_title(f"{feat}\nno weights: every photo grouped by pile (black bar = mean of the pile)\n"
                 f"{spread_text(groups, feat)}", loc="left", fontsize=8.5 if not big else 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels(names if len(names) <= 25 else [""] * len(names), fontsize=8)
    ax.set_xlabel("pile (each dot is one photo)")
    ax.set_ylabel(label_of(feat))


def feature_figure(plt, feat, allp, usable, e, styles):
    """One feature, large; with weights the piles are listed in a legend at the right."""
    from matplotlib.lines import Line2D
    side = bool(e) and 1 < len(styles) <= 16
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10 if side else 8.2, 5.4))
        draw_feature(ax, feat, allp, usable, e, styles, big=True)
        if side:
            counts = {k: len(g) for k, g in _groups(usable).items()}
            h = [Line2D([0], [0], marker=styles[k][1], ls="", mfc=styles[k][0], mec="white", ms=8,
                        label=f"{k} ({n} photo{'s' * (n > 1)})") for k, n in counts.items()]
            h.append(Line2D([0], [0], marker="D", ls="", mfc="none", mec="k", ms=8, label="mean of the pile"))
            fig.legend(handles=h, loc="center right", fontsize=8, frameon=False, title="photos of one pile\nshare colour, marker\nand weight", title_fontsize=8)
            fig.tight_layout(rect=(0, 0, 0.80, 1))
        else:
            fig.tight_layout()
    return fig


def overview_figure(plt, feats, allp, usable, by_feat, styles, cols: int = 3):
    """All features in one grid, one shared pile legend."""
    rows_n = max(1, -(-len(feats) // cols))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 4.3 * rows_n + 0.6), squeeze=False)
        for ax, k in zip(axes.ravel(), feats):
            draw_feature(ax, k, allp, usable, by_feat.get(k), styles, big=False)
        for ax in axes.ravel()[len(feats):]:
            ax.axis("off")
        if by_feat:
            pile_legend(fig, {k: styles[k] for k in _groups(usable)},
                        {k: len(g) for k, g in _groups(usable).items()})
        fig.tight_layout(rect=(0, 0.06 if by_feat else 0, 1, 1))
    return fig


def learning_curve_figure(plt, out: list[dict], feat: str, total: int, repeats: int):
    """Learning curve of ``learning_curve``: r, slope and the width of the 95 % range of r for every number of
    photos. The sizes are spaced evenly (the labels give the photos and the piles they come from). The full-size
    entry has no spread by construction (every repeat uses all the data), so it is drawn as the dashed reference
    line / hollow marker, not as part of the band."""
    ok = [e for e in out if e["r_mean"] is not None]
    at = {e["n"]: i for i, e in enumerate(ok)}
    part = [e for e in ok if e["n"] < total]
    full = [e for e in ok if e["n"] >= total]
    pos = [at[e["n"]] for e in part]

    def tick(e):
        k = e.get("n_piles")
        return f"{e['n']}\n({k:.0f} pile{'s' if round(k) != 1 else ''})" if k else str(e["n"])

    unit = unit_of(feat)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, key, label in ((axes[0], "r", "correlation r"), (axes[1], "slope", f"slope (g per {unit or 'unit'})")):
            if part:
                ax.fill_between(pos, [e[f"{key}_lo"] for e in part], [e[f"{key}_hi"] for e in part], alpha=0.25,
                                color="C0", label=f"95 % of the {repeats} random subsets")
                ax.plot(pos, [e[f"{key}_mean"] for e in part], "o-", color="C0", label="mean of the subsets")
            if full:
                v = full[0][f"{key}_mean"]
                ax.axhline(v, color="k", ls="--", lw=1, label=f"all {total} photos: {v:+.3g}")
                ax.plot([at[full[0]["n"]]], [v], "o", mfc="white", mec="k", ms=8, zorder=5)
            ax.set_ylabel(label)
            ax.set_title(f"{label} of {feat}" if key == "r" else label, fontsize=10)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_ylim(-1.05, 1.05)
        axes[2].plot(pos, [e["r_hi"] - e["r_lo"] for e in part], "o-", color="C3")
        axes[2].set_ylabel("width of the 95 % range of r")
        axes[2].set_title("how uncertain is r? (smaller is better)", fontsize=10)
        axes[2].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xticks(range(len(ok)))
            ax.set_xticklabels([tick(e) for e in ok], fontsize=8)
            ax.set_xlabel("photos in the subset, whole piles (sizes spaced evenly)")
        fig.suptitle(f"Learning curve of {feat}: the estimate has settled when the band stops getting narrower",
                     fontsize=11)
        fig.tight_layout()
    return fig


def curves_overview_figure(plt, curves: dict, total: int, repeats: int, cols: int = 3):
    """The learning curves of several features side by side (r with its 95 % range), same scale, to compare them."""
    names = list(curves)
    rows_n = max(1, -(-len(names) // cols))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 3.9 * rows_n + 0.4), squeeze=False, sharey=True)
        for ax, k in zip(axes.ravel(), names):
            ok = [e for e in curves[k] if e["r_mean"] is not None]
            at = {e["n"]: i for i, e in enumerate(ok)}
            part = [e for e in ok if e["n"] < total]
            full = [e for e in ok if e["n"] >= total]
            pos = [at[e["n"]] for e in part]
            if part:
                ax.fill_between(pos, [e["r_lo"] for e in part], [e["r_hi"] for e in part], alpha=0.25, color="C0")
                ax.plot(pos, [e["r_mean"] for e in part], "o-", color="C0", ms=4)
            if full:
                ax.axhline(full[0]["r_mean"], color="k", ls="--", lw=1)
                ax.plot([at[full[0]["n"]]], [full[0]["r_mean"]], "o", mfc="white", mec="k", ms=7, zorder=5)
            ax.set_title(k, fontsize=10, loc="left")
            ax.set_xticks(range(len(ok)))
            ax.set_xticklabels([f"{e['n']}\n{e['n_piles']:.0f}p" if e.get("n_piles") else str(e["n"]) for e in ok],
                               fontsize=7)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("correlation r")
            ax.set_xlabel("photos / piles (p)")
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.suptitle(f"Learning curves of the features: mean r and the 95 % range of {repeats} random subsets of whole "
                     f"piles (dashed / hollow = all {total} photos)", fontsize=10)
        fig.tight_layout()
    return fig


def kernel_examples_figure(plt, bgr, bboxes, title: str = "", n_zoom: int = 8, pad: int = 6):
    """One photo with its boxes and the kernel pixels found inside them, plus close-ups of single kernels.

    Left: the whole photo, boxes in blue, kernel pixels tinted green. Right: ``n_zoom`` kernels cut out of the photo
    (the ones whose box is least filled first, i.e. the diagonal ones, then the most filled and the median one),
    each with the box (blue) and the outline of the kernel pixels (green) and the numbers in the title.
    """
    import cv2
    import numpy as np

    from .features import kernel_mask

    mask, areas = kernel_mask(bgr, bboxes)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    box_area = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in bboxes]
    fill = [a / ba if ba > 0 else np.nan for a, ba in zip(areas, box_area)]
    order = [i for i in np.argsort(fill) if not np.isnan(fill[i])]
    k = min(n_zoom, len(order))
    low = order[:max(0, k - 3)]
    pick = list(low) + ([order[-1]] if k > len(low) else []) + ([order[len(order) // 2]] if k > len(low) + 1 else [])
    pick = list(dict.fromkeys(pick))[:k]
    with plt.rc_context(RC):
        cols = 4
        rows = max(1, -(-len(pick) // cols))
        fig = plt.figure(figsize=(15, 5.2 + 1.0 * max(0, rows - 1)))
        gs = fig.add_gridspec(rows, cols + 3, wspace=0.25, hspace=0.45)
        ax = fig.add_subplot(gs[:, :3])
        shown = rgb.copy()
        shown[mask] = (0.45 * shown[mask] + 0.55 * np.array([0, 200, 60])).astype(np.uint8)
        ax.imshow(shown)
        for x1, y1, x2, y2 in bboxes:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1f77b4", lw=0.8))
        tot_box, tot_k = sum(box_area), int(sum(areas))
        ax.set_title(f"{title}  -  {len(bboxes)} kernels: box area {tot_box:.0f} px, kernel pixels {tot_k} px "
                     f"({100 * tot_k / tot_box:.0f} %)" if tot_box else title, fontsize=10, loc="left")
        ax.axis("off")
        ax.grid(False)
        for j, i in enumerate(pick):
            a2 = fig.add_subplot(gs[j // cols, 3 + j % cols])
            x1, y1, x2, y2 = (int(round(v)) for v in bboxes[i])
            cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
            a2.imshow(rgb[cy1:cy2, cx1:cx2], extent=(cx1, cx2, cy2, cy1))
            m = mask[cy1:cy2, cx1:cx2].astype(float)
            if m.any():
                a2.contour(np.arange(cx1, cx2) + 0.5, np.arange(cy1, cy2) + 0.5, m, levels=[0.5],
                           colors="#00c83c", linewidths=1.4)
            a2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#1f77b4", lw=1.4))
            a2.set_title(f"box {box_area[i]:.0f} px, kernel {areas[i]} px\n{100 * fill[i]:.0f} % of the box",
                         fontsize=8)
            a2.axis("off")
            a2.grid(False)
        fig.suptitle("Blue: detection box (axis aligned). Green: kernel pixels found inside it "
                     "(paper background removed).", fontsize=10, y=0.99)
    return fig


def save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
