"""Weight resolution and git-based download for vessel segmentation."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from fundus_image_toolbox.utils.model_cache import (
    FIT_CACHE_ENV_VAR,
    component_cache_dir,
    has_all_paths,
    _remove_file_if_exists,
)

from .default import (
    ENSEMBLE_WEIGHT_FILES,
    VESSEL_SEGMENTATION_COMPONENT_NAME,
    VESSEL_WEIGHTS_GIT_REF,
    VESSEL_WEIGHTS_GIT_SUBDIR,
    VESSEL_WEIGHTS_GIT_URL,
)


def _has_ensemble_weights(models_dir: Path) -> bool:
    return has_all_paths(models_dir, ENSEMBLE_WEIGHT_FILES)


def resolve_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the directory containing FR-UNet ensemble checkpoints."""
    if models_dir is not None:
        resolved = Path(models_dir).expanduser().resolve()
        if not _has_ensemble_weights(resolved):
            missing = [
                name
                for name in ENSEMBLE_WEIGHT_FILES
                if not (resolved / name).exists()
            ]
            raise FileNotFoundError(
                f"[FIT:vessel_segmentation] Expected weights missing in {resolved}: "
                f"{', '.join(missing)}"
            )
        return resolved

    cache_models_dir = component_cache_dir(
        VESSEL_SEGMENTATION_COMPONENT_NAME, cache_dir=cache_dir
    )
    cache_models_dir.mkdir(parents=True, exist_ok=True)
    if _has_ensemble_weights(cache_models_dir):
        return cache_models_dir

    return cache_models_dir


def _manual_git_instructions(models_dir: Path) -> str:
    return (
        f"Manual workaround to get the weights for {VESSEL_SEGMENTATION_COMPONENT_NAME}:\n"
        f"1) git clone --depth 1 --filter=blob:none --sparse "
        f"{VESSEL_WEIGHTS_GIT_URL} ~/tmp/segmentation_weights\n"
        f"2) cd ~/tmp/segmentation_weights\n"
        f"3) git fetch --depth 1 origin {VESSEL_WEIGHTS_GIT_REF}\n"
        f"4) git checkout {VESSEL_WEIGHTS_GIT_REF}\n"
        f"5) git sparse-checkout set {VESSEL_WEIGHTS_GIT_SUBDIR}\n"
        f"6) Copy {', '.join(ENSEMBLE_WEIGHT_FILES)} into {models_dir}\n"
        f"7) Re-run your command.\n"
        f"You can change the cache root with the environment variable {FIT_CACHE_ENV_VAR}."
    )


def _run_git(args: list[str], *, cwd: Optional[Path] = None) -> None:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")


def _fetch_weights_from_git(
    destination_dir: Path,
    git_url: str = VESSEL_WEIGHTS_GIT_URL,
    git_ref: str = VESSEL_WEIGHTS_GIT_REF,
    subdir: str = VESSEL_WEIGHTS_GIT_SUBDIR,
) -> None:
    """Fetch ensemble checkpoints from a pinned git commit into destination_dir."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    try:
        with tempfile.TemporaryDirectory(prefix="fit-vessel-weights-") as tmp:
            repo_dir = Path(tmp) / "repo"
            _run_git(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    git_url,
                    str(repo_dir),
                ]
            )
            _run_git(["git", "sparse-checkout", "set", subdir], cwd=repo_dir)
            _run_git(
                ["git", "fetch", "--depth", "1", "origin", git_ref],
                cwd=repo_dir,
            )
            _run_git(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

            source_dir = (repo_dir / subdir).resolve()
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    f"[FIT:vessel_segmentation] Missing '{subdir}/' directory at git ref "
                    f"{git_ref} in {git_url}"
                )

            for name in ENSEMBLE_WEIGHT_FILES:
                source = (source_dir / name).resolve()
                if not source.is_file():
                    raise FileNotFoundError(
                        f"[FIT:vessel_segmentation] Missing checkpoint {name} at git ref "
                        f"{git_ref} in {git_url}"
                    )
                if not str(source).startswith(str(source_dir)):
                    raise RuntimeError(
                        f"[FIT:vessel_segmentation] Unsafe checkpoint path for {name}"
                    )

                target = (destination_dir / name).resolve()
                if not str(target).startswith(str(destination_dir.resolve())):
                    raise RuntimeError(
                        f"[FIT:vessel_segmentation] Unsafe destination path for {name}"
                    )

                shutil.copy2(source, target)
                copied.append(target)
    except Exception:
        for path in copied:
            _remove_file_if_exists(path)
        raise


def download_weights(
    git_url: str = VESSEL_WEIGHTS_GIT_URL,
    git_ref: str = VESSEL_WEIGHTS_GIT_REF,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Download vessel segmentation ensemble weights from git into the FIT cache."""
    models_dir = resolve_models_dir(cache_dir=cache_dir)
    if _has_ensemble_weights(models_dir):
        return models_dir

    print(
        f"[FIT:vessel_segmentation] Downloading weights from git "
        f"({git_url} @ {git_ref})..."
    )
    try:
        _fetch_weights_from_git(
            destination_dir=models_dir,
            git_url=git_url,
            git_ref=git_ref,
        )
        print("[FIT:vessel_segmentation] Done.")
    except Exception as exc:
        raise RuntimeError(
            f"[FIT:vessel_segmentation] Failed to download weights from git.\n"
            f"Last error: {exc}\n\n{_manual_git_instructions(models_dir)}"
        ) from exc

    if not _has_ensemble_weights(models_dir):
        raise FileNotFoundError(
            f"[FIT:vessel_segmentation] Expected checkpoints were not found after git "
            f"download in {models_dir}: {', '.join(ENSEMBLE_WEIGHT_FILES)}"
        )
    return models_dir


def ensure_models_dir(
    cache_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return a models directory, downloading weights into cache if needed."""
    if models_dir is not None:
        return resolve_models_dir(models_dir=models_dir)

    resolved = resolve_models_dir(cache_dir=cache_dir)
    if _has_ensemble_weights(resolved):
        return resolved

    return download_weights(cache_dir=cache_dir)
