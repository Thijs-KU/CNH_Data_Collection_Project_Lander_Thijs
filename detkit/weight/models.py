"""Weight models: features -> Estimate. weight = feature x constant per crop, for kernel count
(``CountTimesConstant``, the baseline) or box area (``AreaTimesConstant``). Bounds (student task 3): ``fit`` stores
the per-pile spread of the ratios, ``predict`` turns it into a 95 % prediction interval ``low_g`` / ``high_g``."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG
from .features import SampleTable


COVERAGE = 0.95
# Two-sided 95 % Student-t factor by degrees of freedom (numpy only; no scipy in the core dependencies).
# Between table rows above 30 the next lower df is used, i.e. the slightly wider (conservative) factor.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_factor(df: int) -> float:
    """Student-t factor for a two-sided ``COVERAGE`` (95 %) interval with ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError("need at least 1 degree of freedom")
    return _T95[max(k for k in _T95 if k <= df)]          # above 120: 1.980 (true limit 1.960, so conservative)


@dataclass
class Estimate:
    weight_g: float
    low_g: float | None = None       # lower/upper bound of an interval, if the model can give one
    high_g: float | None = None
    note: str = ""


class WeightModel:
    """Interface. Subclass, set ``name``/``description``, implement ``fit`` and ``predict``.

    ``params()``/``set_params()`` hold everything needed to save and restore a fitted model as JSON.
    """
    name = "base"
    description = ""
    implemented = False

    def fit(self, samples: SampleTable) -> "WeightModel":
        raise NotImplementedError

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params()}

    @staticmethod
    def from_dict(d: dict) -> "WeightModel":
        m = get_model(d["name"])
        m.set_params(d.get("params", {}))
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "WeightModel":
        return WeightModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FeatureTimesConstant(WeightModel):
    """weight = feature x constant, one constant per crop (``feature`` is a key of ``image_features``).

    ``constants`` = {"default": c, "<crop>": c}. Subclasses pick the feature; ``config_key`` is where the
    constants live in weight_config.json.

    ``spreads`` = {"default": {"sd": s, "n": n, "photos": m}, "<crop>": {...}}, filled by ``fit``. Photos of one
    pile (same ``pile_id``) share one measured weight and are not independent, so the unit of the spread is the
    pile: ``sd`` is the sample standard deviation of the per-pile mean ratios weight / feature (same unit as the
    constant), ``n`` the number of piles (the sample size to use for degrees of freedom) and ``photos`` the number of
    photos behind them. A photo without ``pile_id`` is a pile of its own. A crop is absent when fewer than 2 piles
    were weighed (no spread can be estimated); ``predict`` has to cope with that.
    """
    name = "feature_x_constant"
    feature = "count"
    unit = "g per unit"
    config_key = "constants"
    implemented = True

    def __init__(self, constants: dict[str, float] | None = None, spreads: dict[str, dict] | None = None):
        self.constants = dict(DEFAULT_CONFIG[self.config_key] if constants is None else constants)
        self.spreads: dict[str, dict] = {k: dict(v) for k, v in (spreads or {}).items()}

    def fit(self, samples: SampleTable) -> "FeatureTimesConstant":
        """Calibrate: per crop, constant = mean(weight / feature) over weighed photos.
        Crops without data keep their current constant. ``default`` = pooled mean over all crops.

        Task 3: also store the spread in ``self.spreads`` (per crop, and pooled for ``default``), so ``predict`` can
        give ``low_g`` / ``high_g``. Photos of one pile share one weight, so they are averaged per pile first and
        the standard deviation (ddof=1) is taken over the piles; ``n`` counts piles, not photos. Crops without data
        keep their current spread; a crop with a single pile has no spread (entry removed)."""
        ratios: dict[str, list[float]] = {}
        piles: dict[tuple[str, str], list[float]] = {}          # (crop, pile) -> ratios of the photos of that pile
        for s in samples.with_weight():
            if s.features.get(self.feature):
                r = s.weight_g / s.features[self.feature]
                crop = s.crop or "default"
                ratios.setdefault(crop, []).append(r)
                piles.setdefault((crop, str(s.meta.get("pile_id") or s.file)), []).append(r)
        for crop, r in ratios.items():
            self.constants[crop] = statistics.fmean(r)
            self._set_spread(crop, [v for (c, _), v in piles.items() if c == crop])
        if ratios:
            self.constants["default"] = statistics.fmean([x for r in ratios.values() for x in r])
            self._set_spread("default", list(piles.values()))
        return self

    def _set_spread(self, crop: str, piles: list[list[float]]) -> None:
        """``piles`` = the per-photo ratios of each pile of this crop."""
        means = [statistics.fmean(v) for v in piles]
        if len(means) >= 2:
            self.spreads[crop] = {"sd": statistics.stdev(means), "n": len(means), "photos": sum(map(len, piles))}
        else:
            self.spreads.pop(crop, None)

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        known = crop in self.constants
        if not known and "default" not in self.constants:
            raise ValueError(f"no {self.unit} constant for crop {crop!r}: "
                             f"run `detkit weight calibrate --model {self.name}`")
        c = self.constants[crop] if known else self.constants["default"]
        note = "" if known or crop is None else f"no constant for crop {crop!r}, used default"
        x = features.get(self.feature, 0)
        weight = x * c
        # Task 3: 95 % prediction interval for the weight of a NEW pile, from the spread stored by ``fit``:
        #   weight +- t(n-1) * sqrt(1 + 1/n) * sd * feature      (sd of the per-pile ratios, n = number of piles)
        # sqrt(1 + 1/n) = the scatter of one new pile plus the uncertainty of the constant itself. The same crop
        # (or ``default`` when the crop has no constant) supplies constant and spread, so they always match.
        spread = self.spreads.get(crop if known else "default")
        if not spread:
            return Estimate(weight, note="; ".join(filter(None, [note, "no spread (fewer than 2 piles): no bounds"])))
        n = spread["n"]
        u = t_factor(n - 1) * math.sqrt(1 + 1 / n) * spread["sd"] * x
        return Estimate(weight, max(0.0, weight - u), weight + u, note=note)

    def params(self) -> dict:
        return {"constants": self.constants, "spreads": self.spreads}

    def set_params(self, params: dict) -> None:
        self.constants = dict(params["constants"])
        self.spreads = {k: dict(v) for k, v in params.get("spreads", {}).items()}   # files from before task 3: none


class CountTimesConstant(FeatureTimesConstant):
    """Baseline: weight = number of detected kernels x grams per kernel of the crop."""
    name = "count_x_constant"
    description = "weight = kernel count x g/kernel per crop (baseline)"
    feature = "count"
    unit = "g/kernel"
    config_key = "constants"


class AreaTimesConstant(FeatureTimesConstant):
    """weight = total box area (px^2) x grams per px^2 of the crop. The constant depends on the camera
    distance (pixels per mm), so it is only valid for photos taken the same way as the calibration photos."""
    name = "area_x_constant"
    description = "weight = total box area (px^2) x g/px^2 per crop"
    feature = "total_area_px"
    unit = "g/px^2"
    config_key = "area_constants"


class KernelAreaTimesConstant(FeatureTimesConstant):
    """weight = kernel pixels (px^2) x grams per px^2 of the crop. Like ``area_x_constant`` but only the pixels
    that really are kernel count (paper inside the boxes is left out), so tilted kernels and loose boxes no longer
    inflate the area. Needs the photo when the features are made (``kernel_area_px``); same distance caveat."""
    name = "kernel_area_x_constant"
    description = "weight = kernel pixels (px^2, box background removed) x g/px^2 per crop"
    feature = "kernel_area_px"
    unit = "g/px^2 (kernel)"
    config_key = "kernel_area_constants"


MODELS: dict[str, type[WeightModel]] = {
    m.name: m for m in (CountTimesConstant, AreaTimesConstant, KernelAreaTimesConstant)
}


def get_model(name: str, **kw) -> WeightModel:
    if name not in MODELS:
        raise KeyError(f"unknown weight model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
