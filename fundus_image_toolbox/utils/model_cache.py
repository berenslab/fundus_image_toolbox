"""Utilities for cache-based model artifact handling."""

import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import requests
from tqdm import tqdm


FIT_CACHE_ENV_VAR = "FIT_CACHE_DIR"
APP_CACHE_DIRNAME = "fundus_image_toolbox"
N_RETRY_DL = 2


def resolve_fit_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the cache root directory.

    Priority:
    1) explicit cache_dir argument,
    2) FIT_CACHE_DIR environment variable,
    3) platform-specific cache default.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()

    env_cache_dir = os.environ.get(FIT_CACHE_ENV_VAR)
    if env_cache_dir:
        return Path(env_cache_dir).expanduser().resolve()

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root).resolve() / APP_CACHE_DIRNAME
        return (Path.home() / "AppData" / "Local" / APP_CACHE_DIRNAME).resolve()

    if os.name == "posix" and Path.home().exists():
        if os.uname().sysname == "Darwin":
            return (Path.home() / "Library" / "Caches" / APP_CACHE_DIRNAME).resolve()
        return (Path.home() / ".cache" / APP_CACHE_DIRNAME).resolve()

    return (Path.cwd() / ".cache" / APP_CACHE_DIRNAME).resolve()


def component_cache_dir(
    component_name: str, cache_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Return component-specific cache directory."""
    return resolve_fit_cache_dir(cache_dir=cache_dir) / component_name


def download(
    url: str,
    target_path: Union[str, Path],
    component_name: str,
    manual_file_name: str,
    manual_target_dir: Union[str, Path],
) -> Path:
    """Download a file with progress and actionable errors.

    Raises RuntimeError with clear instructions when download fails.
    """
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    manual_dir = Path(manual_target_dir).expanduser().resolve()
    manual_hint = _manual_download_instructions(
        url=url,
        file_name=manual_file_name,
        destination=manual_dir,
        component_name=component_name,
    )

    total_attempts = 1 + N_RETRY_DL
    last_error: Optional[Exception] = None

    for attempt in range(1, total_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                if not response.ok:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} while downloading from {url}"
                    )

                total_size = int(response.headers.get("content-length", 0))
                with target.open("wb") as out_file, tqdm(
                    desc=f"Downloading {target.name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 8):
                        if not chunk:
                            continue
                        out_file.write(chunk)
                        progress.update(len(chunk))
            return target
        except (requests.RequestException, requests.HTTPError) as exc:
            last_error = exc
            _remove_file_if_exists(target)
            if attempt < total_attempts:
                continue

    raise RuntimeError(
        f"Failed to download weights from {url} after {total_attempts} attempts.\n"
        f"Last error: {last_error}\n"
        "This is unexpected. If reproducible, please open an issue.\n\n"
        f"{manual_hint}"
    ) from last_error


def extract_tar_safely(
    archive_path: Union[str, Path],
    destination_dir: Union[str, Path],
    replace_colon_with: str = "_",
) -> None:
    """Extract tar archive safely with path sanitization."""
    extract_tar_safely_with_manifest(
        archive_path=archive_path,
        destination_dir=destination_dir,
        replace_colon_with=replace_colon_with,
    )


def extract_tar_safely_with_manifest(
    archive_path: Union[str, Path],
    destination_dir: Union[str, Path],
    replace_colon_with: str = "_",
) -> Dict[str, List[Path]]:
    """Extract tar archive and return created files/dirs for rollback."""
    archive = Path(archive_path).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Path]] = {"files": [], "dirs": []}

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            safe_name = _safe_archive_member_name(
                member.name, replace_colon_with=replace_colon_with
            )
            if safe_name is None:
                continue

            target_path = (destination / safe_name).resolve()
            if not _is_relative_to(target_path, destination):
                continue

            member.name = safe_name
            if member.isdir():
                if not target_path.exists():
                    manifest["dirs"].append(target_path)
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            parents_to_create: List[Path] = []
            parent = target_path.parent
            while parent != destination and _is_relative_to(parent, destination):
                if not parent.exists():
                    parents_to_create.append(parent)
                parent = parent.parent
            for p in reversed(parents_to_create):
                p.mkdir(parents=False, exist_ok=True)
                manifest["dirs"].append(p)
            if not target_path.parent.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            try:
                if not target_path.exists():
                    manifest["files"].append(target_path)
                with target_path.open("wb") as out:
                    shutil.copyfileobj(source, out)
            finally:
                source.close()

    return manifest


def cleanup_extraction_artifacts(manifest: Dict[str, List[Path]]) -> None:
    """Best-effort cleanup for files and empty directories from one extraction."""
    for path in manifest.get("files", []):
        _remove_file_if_exists(path)

    dirs = sorted(manifest.get("dirs", []), key=lambda p: len(p.parts), reverse=True)
    for directory in dirs:
        try:
            if directory.exists() and directory.is_dir():
                directory.rmdir()
        except OSError:
            # Directory can be non-empty when shared content already exists.
            continue


def _remove_file_if_exists(path: Union[str, Path]) -> None:
    p = Path(path)
    try:
        if p.exists() and p.is_file():
            os.remove(str(p))
    except OSError:
        pass


def has_all_paths(
    base_dir: Union[str, Path], relative_paths: Iterable[Union[str, Path]]
) -> bool:
    """Check if all expected paths exist under base_dir."""
    base = Path(base_dir).expanduser().resolve()
    return all((base / Path(p)).exists() for p in relative_paths)


def _manual_download_instructions(
    url: str, file_name: str, destination: Path, component_name: str
) -> str:
    return (
        f"Manual workaround for {component_name}:\n"
        f"1) Download this file in your browser: {url}\n"
        f"2) Create directory if needed: {destination}\n"
        f"3) Place the file at: {destination / file_name}\n"
        "4) Re-run your command. The toolbox will use the local files.\n"
        f"You can change this destination by exporting the environment variable {FIT_CACHE_ENV_VAR}."
    )


def _safe_archive_member_name(
    member_name: str, replace_colon_with: str = "_"
) -> Optional[str]:
    name = member_name.lstrip("/\\")
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]

    safe_parts: list[str] = []
    for part in parts:
        if part == "..":
            continue
        clean = part.replace(":", replace_colon_with).rstrip(" .")
        if not clean:
            continue
        safe_parts.append(clean)

    if not safe_parts:
        return None
    return "/".join(safe_parts)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
