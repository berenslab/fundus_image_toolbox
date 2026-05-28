"""FIT wrapper for vessel segmentation checkpoint resolution."""

from pathlib import Path
from typing import Optional, Union

from fundus_image_toolbox.utils.model_cache import component_cache_dir
from segmentation_quality_control.utils import (
    download_weights as segqc_download_weights,
    ensure_models_dir as segqc_ensure_models_dir,
)

from .default import VESSEL_SEGMENTATION_COMPONENT_NAME


def _fit_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    if models_dir is not None:
        return Path(models_dir).expanduser().resolve()
    return component_cache_dir(VESSEL_SEGMENTATION_COMPONENT_NAME, cache_dir=cache_dir)


def ensure_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return FR-UNet ensemble weights under FIT cache or an explicit directory."""
    target = _fit_models_dir(cache_dir=cache_dir, models_dir=models_dir)
    return segqc_ensure_models_dir(models_dir=target)


def download_weights(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Download vessel segmentation weights into FIT cache or an explicit directory."""
    target = _fit_models_dir(cache_dir=cache_dir, models_dir=models_dir)
    return segqc_download_weights(models_dir=target)
