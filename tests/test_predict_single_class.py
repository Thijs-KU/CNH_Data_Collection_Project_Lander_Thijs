"""predict_image keeps one class per photo (torch and the model are stubbed)."""

import sys
import types

from PIL import Image


class _Arr(list):
    def tolist(self):
        return list(self)


class _Batch(dict):
    def to(self, device):
        return self


def _stubs(monkeypatch):
    torch = types.ModuleType("torch")

    class no_grad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Proc:
        def __call__(self, images, return_tensors):
            return _Batch(n=len(images))

        def post_process_object_detection(self, out, threshold, target_sizes):
            # one kernel with two labels, a second kernel with the main label, a third seen only as "pumpkin"
            boxes = [[10, 10, 60, 60], [12, 12, 62, 62], [100, 10, 150, 60], [200, 10, 250, 60]]
            labels, scores = [1, 2, 1, 2], [0.9, 0.6, 0.8, 0.5]
            return [{"boxes": _Arr(map(_Arr, boxes)), "scores": _Arr(scores), "labels": _Arr(labels)}]

    class Model:
        config = types.SimpleNamespace(id2label={1: "popcorn_corn", 2: "pumpkin"})

        def __call__(self, **kw):
            return None

    return Model(), Proc()


def test_predict_image_single_class(monkeypatch):
    model, proc = _stubs(monkeypatch)
    from detkit.predict import predict_image
    img = Image.new("RGB", (300, 100))
    kw = dict(tile=300, overlap=0.0, score=0.05)
    out = predict_image(img, model, proc, "cpu", **kw)
    assert [d["label"] for d in out] == ["popcorn_corn", "popcorn_corn"]
    allc = predict_image(img, model, proc, "cpu", single_class=False, **kw)
    assert sorted(d["label"] for d in allc) == ["popcorn_corn", "popcorn_corn", "pumpkin", "pumpkin"]


def _patched_model(monkeypatch):
    model, proc = _stubs(monkeypatch)
    import detkit.predict as pr
    monkeypatch.setattr(pr, "load_model", lambda ckpt, device=None: (model, proc, {}, "cpu"))


def test_predict_paths_normalizes_raw_photos(monkeypatch, tmp_path):
    import json

    from detkit.preprocess import brightness
    from detkit.predict import predict_paths
    _patched_model(monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (300, 100), (120, 120, 120)).save(raw / "a.png")
    out = tmp_path / "out"
    res = predict_paths("ck", [raw], out, tile=300, overlap=0.0, score=0.05, normalize=(220, 1.0))
    assert res["a.png"]["normalized"]["target"] == 220 and abs(res["a.png"]["normalized"]["after"] - 220) <= 1
    assert (out / "a_pre.jpg").exists() and (out / "a_pred.jpg").exists()               # before / after + overlay
    assert json.loads((out / "predictions.json").read_text())["a.png"]["normalized"]["before"] < 150
    res = predict_paths("ck", [raw], tmp_path / "o2", tile=300, overlap=0.0, score=0.05)
    assert res["a.png"]["normalized"] is None


def test_predict_paths_leaves_project_photos(monkeypatch, tmp_path):
    from detkit.predict import predict_paths
    _patched_model(monkeypatch)
    ph = tmp_path / "w" / "photos"
    ph.mkdir(parents=True)
    (tmp_path / "w" / "project.json").write_text("{}")
    Image.new("RGB", (300, 100), (120, 120, 120)).save(ph / "a.png")
    res = predict_paths("ck", [ph], tmp_path / "o", tile=300, overlap=0.0, score=0.05, normalize=(220, 1.0))
    assert res["a.png"]["normalized"] is None and not (tmp_path / "o" / "a_pre.jpg").exists()


def test_eval_run_model_normalizes_raw_images(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from detkit.cmd_eval import _run_model
    _patched_model(monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (300, 100), (120, 120, 120)).save(raw / "a.png")
    gt = {"images": [{"file_name": "a.png", "width": 300, "height": 100}]}
    a = SimpleNamespace(model="ck", overlap=0.0, crop=None, tile=300, score=0.05, keep_all_classes=False,
                        canonicalize=False, normalize=True, target_brightness=220, contrast=1.0)
    res, _ = _run_model(a, gt, raw, "source", None, tmp_path / "out")
    assert abs(res["a.png"]["normalized"]["after"] - 220) <= 1 and (tmp_path / "out" / "normalized" / "a_pre.jpg").exists()
    a.normalize = False
    assert _run_model(a, gt, raw, "source", None, tmp_path / "o2")[0]["a.png"]["normalized"] is None
