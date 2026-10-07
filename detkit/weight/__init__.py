"""Weight estimation: detections -> features -> weight (g).

The `TODO(student)` markers show what to implement. Tasks are listed in docs/STUDENT_TASKS.md.

    features.py  image_features, Sample, SampleTable   (working)
    models.py    WeightModel, CountTimesConstant, AreaTimesConstant (working; bounds TODO)
    stats.py     evaluate_model (working); learning_curve, bootstrap_ci, propagate (TODO)
    config.py    load_config / save_config (weight_config.json)
"""

from .config import load_config, save_config
from .features import Sample, SampleTable, image_features
from .models import (MODELS, AreaTimesConstant, CountTimesConstant, Estimate, FeatureTimesConstant,
                     WeightModel, get_model)
from .stats import bootstrap_ci, evaluate_model, learning_curve, propagate

__all__ = ["image_features", "Sample", "SampleTable", "Estimate", "WeightModel", "FeatureTimesConstant",
           "CountTimesConstant", "AreaTimesConstant", "MODELS", "get_model",
           "evaluate_model", "learning_curve", "bootstrap_ci", "propagate", "load_config", "save_config"]
