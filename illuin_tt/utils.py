import torch


def get_device(pref: str | torch.device | None = "auto") -> torch.device:
    if pref is None or pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(pref)
