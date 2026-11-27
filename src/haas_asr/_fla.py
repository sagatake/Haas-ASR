"""Choose the Kimi Delta Attention implementation: fla's Triton kernels or a PyTorch port.

Two backends build the same layer, with the same parameters and cache:

- ``"fla"``: flash-linear-attention's ``KimiDeltaAttention``, with a strict version
  check. It needs Triton and a CUDA GPU.
- ``"torch"``: the pure-PyTorch port in ``haas_asr._kda_torch``. It runs anywhere,
  including the CPU and macOS.

``"auto"`` picks fla when a CUDA GPU and Triton are available, and the port otherwise.
The backend is chosen when the decoder is built, so set it before loading a model:
pass ``kda_backend`` to ``haas_asr.load_model``, set the ``HAAS_KDA_BACKEND``
environment variable, or wrap ``ASRModel.restore_from`` in ``use_kda_backend()``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import metadata
from typing import Literal

import torch

FLA_DISTRIBUTION = "flash-linear-attention"
FLA_REQUIRED_VERSION = "0.5.2"

KDABackend = Literal["auto", "fla", "torch"]
KDA_BACKENDS: tuple[str, ...] = ("auto", "fla", "torch")
KDA_BACKEND_ENV = "HAAS_KDA_BACKEND"

_backend: ContextVar[str | None] = ContextVar("haas_kda_backend", default=None)


def check_fla_version() -> None:
    """Raise unless the tested flash-linear-attention release is installed."""
    try:
        installed = metadata.version(FLA_DISTRIBUTION)
    except metadata.PackageNotFoundError as e:
        raise ImportError(
            f"{FLA_DISTRIBUTION}=={FLA_REQUIRED_VERSION} is required but not installed."
        ) from e
    if installed != FLA_REQUIRED_VERSION:
        raise ImportError(
            f"{FLA_DISTRIBUTION}=={FLA_REQUIRED_VERSION} is required, found {installed}. "
            "Its layer APIs change between releases, so other versions are refused."
        )


def fla_available() -> bool:
    """Whether fla's kernels can run here: a CUDA GPU and an importable Triton."""
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def _validate(name: str) -> str:
    if name not in KDA_BACKENDS:
        raise ValueError(f"unknown KDA backend {name!r}; expected one of {KDA_BACKENDS}")
    return name


def resolve_kda_backend(
    backend: KDABackend | None = None, device: str | torch.device | None = None
) -> Literal["fla", "torch"]:
    """Turn a requested backend into ``"fla"`` or ``"torch"``.

    The request is, in order: `backend`, the one set by ``use_kda_backend()``, the
    ``HAAS_KDA_BACKEND`` environment variable, and ``"auto"``. With ``"auto"``, a
    `device` that is not a CUDA device selects the port, since fla's kernels only run
    on CUDA tensors; without a `device`, fla is used whenever it is available.
    """
    name = _validate(backend or _backend.get() or os.environ.get(KDA_BACKEND_ENV) or "auto")
    if name != "auto":
        return name  # type: ignore[return-value]
    if device is not None and torch.device(device).type != "cuda":
        return "torch"
    return "fla" if fla_available() else "torch"


@contextmanager
def use_kda_backend(backend: KDABackend) -> Iterator[None]:
    """Build KDA layers with `backend` inside this block (e.g. around ``restore_from``)."""
    token = _backend.set(_validate(backend))
    try:
        yield
    finally:
        _backend.reset(token)


def load_kda_class() -> type:
    """Return the KDA layer class of the current backend (see ``resolve_kda_backend``).

    fla is imported on first use rather than at package import, because it needs
    Triton, which is unavailable without a CUDA build (e.g. on macOS).
    """
    if resolve_kda_backend() == "torch":
        from haas_asr._kda_torch import TorchKimiDeltaAttention

        return TorchKimiDeltaAttention
    check_fla_version()
    from fla.layers.kda import KimiDeltaAttention

    return KimiDeltaAttention
