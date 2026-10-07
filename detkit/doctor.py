"""Environment check: torch/CUDA actually works on this GPU, token present/valid."""

from __future__ import annotations

import os
import sys
import time


def check_gpu() -> dict:
    import torch

    info: dict = {"python": sys.version.split()[0], "torch": torch.__version__,
                  "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not info["cuda_available"]:
        info["hint"] = (("CPU-only torch build: fine for the cpu flow." if "+cpu" in torch.__version__ else
                         "CUDA unavailable. A +cu130 torch needs a newer driver: `uv sync --extra gpu` "
                         "pins the cu126 build; without a GPU use `--extra cpu`."))
        return info
    cap = torch.cuda.get_device_capability(0)
    info.update(device=torch.cuda.get_device_name(0), capability=f"sm_{cap[0]}{cap[1]}",
                arch_list=torch.cuda.get_arch_list())
    free, total = torch.cuda.mem_get_info()
    info["vram_free_total_gb"] = [round(free / 2**30, 2), round(total / 2**30, 2)]
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=dt)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(10):
                a @ a
            torch.cuda.synchronize()
            info[f"matmul_{str(dt)[6:]}_tflops"] = round(2 * 2048**3 * 10 / (time.time() - t) / 1e12, 2)
        except Exception as e:
            info[f"matmul_{str(dt)[6:]}"] = f"FAIL {type(e).__name__}"
    try:
        conv = torch.nn.Conv2d(3, 8, 3).cuda()
        x = torch.randn(2, 3, 64, 64, device="cuda", requires_grad=True)
        conv(x).sum().backward()
        info["conv_backward"] = "ok"
    except Exception as e:
        info["conv_backward"] = f"FAIL {type(e).__name__}: {e}"
    return info


def check_token(online: bool = False) -> dict:
    tok = os.environ.get("HF_TOKEN")
    info: dict = {"HF_TOKEN_set": bool(tok)}
    if tok and online:
        import requests

        r = requests.get("https://huggingface.co/api/whoami-v2",
                         headers={"Authorization": f"Bearer {tok}"}, timeout=20)
        info["whoami_status"] = r.status_code
    return info
