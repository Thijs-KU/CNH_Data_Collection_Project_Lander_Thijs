"""Weight estimation: detections -> features -> weight (g).

The `TODO(student)` markers show what to implement. Tasks are listed in docs/STUDENT_TASKS.md.

    features.py  image_features, Sample, SampleTable   (working)
    models.py    WeightModel, CountTimesConstant, AreaTimesConstant (working; bounds TODO)
    stats.py     evaluate_model, feature_correlations, learning_curve, bootstrap_ci (working); propagate (TODO)
    config.py    load_config / save_config (weight_config.json)
"""

from .config import load_config, save_config
from .features import Sample, SampleTable, image_features, kernel_mask
from .models import (MODELS, AreaTimesConstant, CountTimesConstant, Estimate, FeatureTimesConstant,
                     KernelAreaTimesConstant, WeightModel, get_model)
from .stats import bootstrap_ci, evaluate_model, feature_correlations, learning_curve, numeric_features, propagate

__all__ = ["image_features", "kernel_mask", "Sample", "SampleTable", "Estimate", "WeightModel", "FeatureTimesConstant",
           "CountTimesConstant", "AreaTimesConstant", "KernelAreaTimesConstant", "MODELS", "get_model",
           "evaluate_model", "feature_correlations", "numeric_features", "learning_curve", "bootstrap_ci", "propagate", "load_config", "save_config"]
