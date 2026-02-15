from __future__ import annotations


def resolve_torch_device(*, torch: object, device_str: str):
    """Resolve a requested device string into a torch.device.

    Supported:
      - "auto": cuda if available else cpu
      - "cuda": cuda (errors if unavailable)
      - "cpu": cpu
      - any torch-compatible string (e.g. cuda:0)
    """

    device_str = str(device_str).strip().lower()

    if device_str == "auto":
        if bool(torch.cuda.is_available()):
            return torch.device("cuda")
        return torch.device("cpu")

    if device_str == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("Requested device='cuda' but CUDA is not available. Use --device cpu or --device auto.")

    return torch.device(device_str)
