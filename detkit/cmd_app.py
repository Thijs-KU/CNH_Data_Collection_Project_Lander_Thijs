"""``detkit app``: launch the annotation / review app on a workdir."""

from __future__ import annotations


def cmd_app(a) -> int:
    try:
        from .app.ui import HAVE_EDITOR, launch
    except ImportError as e:
        print(f"The app needs Gradio: uv sync --extra app  ({e})")
        return 1
    if not HAVE_EDITOR:
        print("note: gradio_image_annotation is not installed -> boxes are edited in a table only "
              "(uv sync --extra app for drawing on the image).")
    launch(a.workdir, host=a.host, port=a.port, ckpt=a.ckpt, device=a.device)
    return 0


def register(sub) -> None:
    sp = sub.add_parser("app", help="launch the annotation/review web app")
    sp.add_argument("--workdir", default="runs/default")
    sp.add_argument("--port", type=int, default=7860)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--ckpt", help="RF-DETR checkpoint dir to preselect for box proposals")
    sp.add_argument("--device", choices=["auto", "cpu", "cuda"], help="override the project device")
    sp.set_defaults(fn=cmd_app)
