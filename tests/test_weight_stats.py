"""feature_correlations, bootstrap_ci, learning_curve (task 2, detkit/weight/stats.py)."""

import numpy as np
import pytest

from detkit.cli import main
from detkit.weight import SampleTable, feature_correlations, bootstrap_ci, learning_curve


def table(n_piles=20, per_pile=2, noise=0.25, seed=1, with_pile_id=True, share_noise=False):
    """Piles of ``per_pile`` photos. Weight ~ box area, so count and area follow it, the mean box area does not.
    ``share_noise``: the photos of one pile also share their error, so they are not independent."""
    rng = np.random.default_rng(seed)
    t = SampleTable()
    for p in range(n_piles):
        k = int(rng.integers(5, 40))
        pile_err = rng.normal(0, noise)
        for j in range(per_pile):
            side = float(rng.uniform(18, 26))
            boxes = [{"bbox": [i * 30, 0, i * 30 + side, side], "label": "wheat", "score": 0.9} for i in range(k)]
            err = pile_err if share_noise else rng.normal(0, noise)
            meta = {"crop": "wheat", "total_weight_g": 0.002 * k * side * side * (1 + err)}
            if with_pile_id:
                meta["pile_id"] = f"p{p}"
            t.add(f"w{p}_{j}.jpg", boxes, meta)
    return t


def test_feature_correlations_rank_and_columns():
    res = feature_correlations(table(noise=0.05), n_boot=100)
    names = [e["feature"] for e in res]
    assert names[0] in ("total_area_px", "count") and abs(res[0]["r"]) > 0.95
    assert names.index("total_area_px") < names.index("mean_area_px")      # mean box size says little on its own
    assert all(abs(res[i]["r"]) >= abs(res[i + 1]["r"]) for i in range(len(res) - 1))
    e = res[0]
    assert e["n"] == 40 and e["r2"] == pytest.approx(e["r"] ** 2) and e["r_lo"] <= e["r"] <= e["r_hi"]
    assert all("r_lo" not in x for x in feature_correlations(table()))
    assert feature_correlations(table(), crop="sunflower") == []
    assert [e["feature"] for e in feature_correlations(table(), features=["count"])] == ["count"]


def test_constant_feature_is_left_out():
    t = SampleTable()
    for i in range(5):
        t.add(f"a{i}.jpg", [{"bbox": [0, 0, 10, 10], "label": "wheat", "score": 1}] * 3,
              {"crop": "wheat", "total_weight_g": 1.0 + i})
    names = [e["feature"] for e in feature_correlations(t)]
    assert "count" not in names and "total_area_px" not in names          # same in every photo: r undefined


def test_bootstrap_ci_resamples_piles_not_photos():
    # photos of one pile share their error: the 'same' pile twice must not look like twice the evidence
    grouped = table(share_noise=True)
    ungrouped = table(share_noise=True, with_pile_id=False)               # same data, every photo its own pile
    lo_g, hi_g = bootstrap_ci(grouped, "total_area_px", n_boot=600)
    lo_u, hi_u = bootstrap_ci(ungrouped, "total_area_px", n_boot=600)
    assert (hi_g - lo_g) > 1.15 * (hi_u - lo_u)
    lo, hi = bootstrap_ci(grouped, "total_area_px", stat="slope", n_boot=300)
    assert lo < 0.002 < hi * 1.5 and lo > 0                                # slope near the true 0.002 g/px^2


def test_bootstrap_ci_reproducible_and_edge_cases():
    t = table()
    assert bootstrap_ci(t, "count", n_boot=100) == bootstrap_ci(t, "count", n_boot=100)
    assert bootstrap_ci(t, "count", n_boot=100, seed=1) != bootstrap_ci(t, "count", n_boot=100, seed=2)
    with pytest.raises(ValueError):
        bootstrap_ci(t, "count", stat="median")
    assert bootstrap_ci(table(n_piles=1, per_pile=5), "count") is None   # one pile: nothing to resample
    assert bootstrap_ci(t, "count", crop="sunflower") is None
    assert bootstrap_ci(t, "count", n_boot=100, alpha=0.5)[0] > bootstrap_ci(t, "count", n_boot=100, alpha=0.05)[0]


def test_learning_curve_settles_with_more_samples():
    out = learning_curve(table(n_piles=30), "total_area_px", sizes=[6, 12, 30, 60], repeats=150)
    assert [e["n"] for e in out] == [6, 12, 30, 60]
    width = [e["r_hi"] - e["r_lo"] for e in out]
    assert width[0] > width[1] > width[2] > width[3]
    assert width[3] == pytest.approx(0, abs=1e-9)                          # full size: all the data, no spread
    assert out[0]["n_piles"] == pytest.approx(3, abs=0.6) and out[3]["n_piles"] == 30
    assert out[-1]["slope_mean"] == pytest.approx(0.002, rel=0.2)


def test_learning_curve_edge_cases():
    t = table(n_piles=4)
    out = learning_curve(t, "count", sizes=[4, 100], repeats=10)
    assert out[0]["r_mean"] is not None and out[1] == {**out[1], "r_mean": None, "slope_mean": None, "n": 100}
    assert learning_curve(table(n_piles=1, per_pile=2), "count", sizes=[2], repeats=5) is None


def test_features_cli(tmp_path, capsys):
    from detkit.project import Project
    p = Project(workdir=tmp_path, classes=["wheat"])
    p.save()
    st = p.store()
    rng = np.random.default_rng(0)
    for i in range(8):
        k = int(rng.integers(3, 20))
        st.upsert_image(f"w{i}.jpg", 1000, 1000, [{"bbox": [j * 20, 0, j * 20 + 10, 20], "class": "wheat"}
                                                  for j in range(k)],
                        {"crop": "wheat", "total_weight_g": 0.05 * k, "pile_id": str(i // 2)}, reviewed=True)
    assert main(["weight", "features", "--workdir", str(tmp_path), "--boot", "50",
                 "--plot", str(tmp_path / "all.png"), "--plot-dir", str(tmp_path / "per")]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "all.png").stat().st_size > 1000
    assert (tmp_path / "per" / "total_area_px.png").exists() and (tmp_path / "per" / "count.png").exists()
    assert "8 photos, 4 piles, 8 with a weight > 0" in out
    assert "p95" in out and "median" in out                              # the value table is always there and "total_area_px" in out and "95% CI of r" in out
    assert main(["weight", "features", "--workdir", str(tmp_path), "--crop", "sunflower"]) == 1
    assert main(["weight", "features", "--workdir", str(tmp_path), "--format", "pdf", "--curve", "count",
                 "--plot-dir", str(tmp_path / "pdf")]) == 0
    assert (tmp_path / "pdf" / "count.pdf").stat().st_size > 1000
    assert (tmp_path / "pdf" / "learning_curve_count.pdf").stat().st_size > 1000


def test_features_cli_plots_without_any_weight(tmp_path, capsys):
    """No (usable) weights: no correlation, but every feature is still plotted per photo."""
    from detkit.project import Project
    p = Project(workdir=tmp_path, classes=["wheat"])
    p.save()
    st = p.store()
    for i in range(5):
        st.upsert_image(f"w{i}.jpg", 1000, 1000, [{"bbox": [j * 20, 0, j * 20 + 10, 20 + i], "class": "wheat"}
                                                  for j in range(3 + i)],
                        {"crop": "wheat", "total_weight_g": -15.0 if i == 0 else None}, reviewed=True)
    assert main(["weight", "features", "--workdir", str(tmp_path), "--plot", str(tmp_path / "all.png"),
                 "--plot-dir", str(tmp_path / "per")]) == 0
    out = capsys.readouterr().out
    assert "5 photos" in out and "0 with a weight > 0 (1 with weight <= 0 left out)" in out
    assert "no correlation" in out and "total_area_px" in out
    assert "p95" in out and "median" in out                              # the value table is there without weights too
    assert (tmp_path / "all.png").stat().st_size > 1000
    assert (tmp_path / "per" / "total_area_px.png").exists() and (tmp_path / "per" / "mean_diag_px.png").exists()


def test_features_cli_learning_curve(tmp_path, capsys):
    from detkit.project import Project
    p = Project(workdir=tmp_path, classes=["wheat"])
    p.save()
    st = p.store()
    rng = np.random.default_rng(0)
    for i in range(12):
        k = int(rng.integers(5, 40))
        st.upsert_image(f"w{i}.jpg", 1000, 1000, [{"bbox": [j * 20, 0, j * 20 + 10, 20], "class": "wheat"}
                                                  for j in range(k)],
                        {"crop": "wheat", "total_weight_g": 0.05 * k * (1 + rng.normal(0, 0.1)),
                         "pile_id": str(i // 2)}, reviewed=True)
    base = ["weight", "features", "--workdir", str(tmp_path)]
    assert main(base + ["--curve", "count", "--sizes", "4,8,12", "--repeats", "40",
                        "--plot-dir", str(tmp_path / "per")]) == 0
    out = capsys.readouterr().out
    assert "learning curve of count" in out and "r 95% range" in out
    rows = [ln for ln in out.splitlines() if ln.split() and ln.split()[0] in ("4", "8", "12") and ".." in ln]
    assert len(rows) == 3
    assert (tmp_path / "per" / "learning_curve_count.png").stat().st_size > 1000
    # without any plot option the curves are still drawn (default folder), several features at once
    assert main(base + ["--curve", "count,total_area_px", "--sizes", "4,8,12", "--repeats", "20"]) == 0
    capsys.readouterr()
    rep = tmp_path / "reports" / "features"
    for f in ("learning_curve_count.png", "learning_curve_total_area_px.png", "learning_curve_all.png"):
        assert (rep / f).stat().st_size > 1000
    assert main(base + ["--curve", "nope"]) == 0
    assert "not a numeric feature" in capsys.readouterr().out


def test_plots_piles_get_their_own_colour_and_marker():
    from detkit.weight import plots as WP
    plt = WP.pyplot()
    t = table(n_piles=25)
    rows = list(t)
    st = WP.pile_styles(rows, plt)
    assert len(st) == 25 and len({(c, m) for c, m in st.values()}) == 25
    assert list(st)[:3] == ["pile p0", "pile p1", "pile p2"] and list(st)[10] == "pile p10"   # natural order
    assert WP.pile_of(rows[0]) == WP.pile_of(rows[1]) != WP.pile_of(rows[2])                  # 2 photos per pile
    txt = WP.spread_text(WP._groups(rows), "count")
    assert "within a pile" in txt and "between piles" in txt
    assert "no within-pile" in WP.spread_text(WP._groups(rows[::2]), "count")                 # one photo per pile
    assert WP.unit_of("kernel_area_px") == "px²" and WP.unit_of("count") == "kernels" and WP.unit_of("mean_diag_px") == "px"
