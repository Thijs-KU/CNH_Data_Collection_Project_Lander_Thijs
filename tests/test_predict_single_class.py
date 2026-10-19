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
