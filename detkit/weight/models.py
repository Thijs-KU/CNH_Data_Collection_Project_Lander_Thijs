"""Weight models: features -> Estimate. weight = feature x constant per crop, for kernel count
(``CountTimesConstant``, the baseline) or box area (``AreaTimesConstant``). Bounds: student task 3."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG
from .features import SampleTable


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
    """
    name = "feature_x_constant"
    feature = "count"
    unit = "g per unit"
    config_key = "constants"
    implemented = True

    def __init__(self, constants: dict[str, float] | None = None):
        self.constants = dict(DEFAULT_CONFIG[self.config_key] if constants is None else constants)

    def fit(self, samples: SampleTable) -> "FeatureTimesConstant":
        """Calibrate: per crop, constant = mean(weight / feature) over weighed photos.
        Crops without data keep their current constant. ``default`` = pooled mean over all crops."""
        ratios: dict[str, list[float]] = {}
        for s in samples.with_weight():
            if s.features.get(self.feature):
                ratios.setdefault(s.crop or "default", []).append(s.weight_g / s.features[self.feature])
        for crop, r in ratios.items():
            self.constants[crop] = statistics.fmean(r)
        allr = [x for r in ratios.values() for x in r]
        if allr:
            self.constants["default"] = statistics.fmean(allr)
        # TODO(student) task 3: also keep what predict() needs for low_g / high_g
        return self

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        known = crop in self.constants
        if not known and "default" not in self.constants:
            raise ValueError(f"no {self.unit} constant for crop {crop!r}: "
                             f"run `detkit weight calibrate --model {self.name}`")
        c = self.constants[crop] if known else self.constants["default"]
        note = "" if known or crop is None else f"no constant for crop {crop!r}, used default"
        # TODO(student) task 3: low_g / high_g (the bounds of this prediction)
        return Estimate(features.get(self.feature, 0) * c, note=note)

    def params(self) -> dict:
        return {"constants": self.constants}

    def set_params(self, params: dict) -> None:
        self.constants = dict(params["constants"])


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


MODELS: dict[str, type[WeightModel]] = {
    m.name: m for m in (CountTimesConstant, AreaTimesConstant)
}


def get_model(name: str, **kw) -> WeightModel:
    if name not in MODELS:
        raise KeyError(f"unknown weight model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
