"""Utilities for exporting and loading a unified MEG data bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_scene_view_from_filename(path: Path) -> tuple[int, int] | None:
    """Parse ``(SYNSscene, SYNSView)`` from names like ``S07_Im11.jpg``."""

    stem = path.stem
    if not stem.startswith("S") or "_Im" not in stem:
        return None
    left, right = stem.split("_Im", 1)
    try:
        return int(left[1:]), int(right)
    except ValueError:
        return None


def _category_from_relative_path(relative_path: Path) -> tuple[int, str]:
    """Extract ``(category_id, category_name)`` from folder like ``2_Nature``."""

    parent = relative_path.parent.name
    if "_" in parent:
        prefix, name = parent.split("_", 1)
        if prefix.isdigit():
            return int(prefix), name
    return -1, parent


def _load_meg_subject_tensor(meg_rdm_path: Path) -> np.ndarray:
    """Load MEG RDM tensor as ``[N, N, T, S]`` from MATLAB file."""

    mat = loadmat(meg_rdm_path, squeeze_me=True)
    if "MEGRDMs_2D" not in mat:
        keys = [key for key in mat if not key.startswith("__")]
        raise SystemExit(f"MEG RDM variable 'MEGRDMs_2D' not found in {meg_rdm_path}. Found keys: {keys}")
    meg = np.asarray(mat["MEGRDMs_2D"], dtype=np.float64)
    if meg.ndim == 3:
        if meg.shape[0] != meg.shape[1]:
            raise SystemExit(f"MEG RDM must be square on first two dimensions, got {meg.shape}")
        return meg[:, :, :, np.newaxis]
    if meg.ndim == 4:
        if meg.shape[0] != meg.shape[1]:
            raise SystemExit(f"MEG RDM must be square on first two dimensions, got {meg.shape}")
        return meg
    raise SystemExit(f"Expected MEG RDM shape [N, N, T] or [N, N, T, S], got {meg.shape}")


def _load_time_vector(time_mat_path: Path, n_time: int) -> np.ndarray:
    """Load time vector from MATLAB file, or fallback to ``-200..`` ms."""

    if time_mat_path.exists():
        mat = loadmat(time_mat_path, squeeze_me=True)
        if "time" in mat:
            time = np.asarray(mat["time"], dtype=np.float64).reshape(-1)
            if time.size == n_time:
                return time
    return np.arange(n_time, dtype=np.float64) - 200.0


def _build_pair_lookup(images_dir: Path) -> dict[tuple[int, int], Path]:
    """Build mapping from ``(scene, view)`` pair to image file path."""

    lookup: dict[tuple[int, int], Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        parsed = _parse_scene_view_from_filename(path)
        if parsed is None:
            continue
        if parsed in lookup:
            raise SystemExit(f"Duplicate SYNS scene/view pair found for {parsed}: {lookup[parsed]} and {path}")
        lookup[parsed] = path
    return lookup


def build_meg_bundle(
    meg_rdm_path: Path,
    time_mat_path: Path,
    image_struct_path: Path,
    images_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export MEG signals + metadata into one compressed ``.npz`` file."""

    meg_subjects = _load_meg_subject_tensor(meg_rdm_path)
    n_images, _, n_time, n_subjects = meg_subjects.shape
    time_ms = _load_time_vector(time_mat_path, n_time=n_time)

    mat = loadmat(image_struct_path, squeeze_me=True, struct_as_record=False)
    if "imF" not in mat or "imI" not in mat:
        keys = [key for key in mat if not key.startswith("__")]
        raise SystemExit(f"Expected keys 'imF' and 'imI' in {image_struct_path}. Found: {keys}")
    imf = np.asarray(mat["imF"], dtype=np.float64).reshape(-1)
    imi = np.asarray(mat["imI"], dtype=np.float64).reshape(-1)
    finite = np.isfinite(imf) & np.isfinite(imi)
    finite_indices = np.where(finite)[0]
    if finite_indices.size != n_images:
        raise SystemExit(
            f"Finite imF/imI entries ({finite_indices.size}) do not match MEG image size ({n_images})."
        )

    lookup = _build_pair_lookup(images_dir)
    ordered_abs: list[Path] = []
    ordered_scene: list[int] = []
    ordered_view: list[int] = []
    for idx in finite_indices:
        scene = int(imf[idx])
        view = int(imi[idx])
        path = lookup.get((scene, view))
        if path is None:
            raise SystemExit(
                f"Could not map imF/imI pair ({scene}, {view}) from {image_struct_path} to {images_dir}."
            )
        ordered_abs.append(path.resolve())
        ordered_scene.append(scene)
        ordered_view.append(view)

    ordered_rel = [str(path.relative_to(images_dir.resolve())) for path in ordered_abs]
    if len(set(ordered_rel)) != len(ordered_rel):
        raise SystemExit("Duplicate image entries detected while building MEG bundle.")

    category_ids: list[int] = []
    category_names: list[str] = []
    for rel in ordered_rel:
        cat_id, cat_name = _category_from_relative_path(Path(rel))
        category_ids.append(cat_id)
        category_names.append(cat_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bundle_version=np.array(1, dtype=np.int32),
        created_utc=np.array(datetime.now(timezone.utc).isoformat()),
        source_meg_rdm_path=np.array(str(meg_rdm_path.resolve())),
        source_time_mat_path=np.array(str(time_mat_path.resolve())),
        source_image_struct_path=np.array(str(image_struct_path.resolve())),
        source_images_dir=np.array(str(images_dir.resolve())),
        meg_rdm_subjects=meg_subjects.astype(np.float64),
        meg_rdm_avg=meg_subjects.mean(axis=3).astype(np.float64),
        time_ms=time_ms.astype(np.float64),
        stimulus_file_name=np.array(ordered_rel, dtype=np.str_),
        category_id=np.array(category_ids, dtype=np.int32),
        category_name=np.array(category_names, dtype=np.str_),
        syns_scene=np.array(ordered_scene, dtype=np.int32),
        syns_view=np.array(ordered_view, dtype=np.int32),
        imF_raw=imf.astype(np.float64),
        imI_raw=imi.astype(np.float64),
        finite_im_indices=finite_indices.astype(np.int32),
    )

    return {
        "output_path": str(output_path.resolve()),
        "n_images": int(n_images),
        "n_time": int(n_time),
        "n_subjects": int(n_subjects),
        "category_blocks": category_names,
    }


def load_meg_bundle(bundle_path: Path, images_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """Load MEG tensors, time vector, and stimulus order from a bundle file."""

    if not bundle_path.exists():
        raise SystemExit(f"MEG bundle file not found: {bundle_path}")

    with np.load(bundle_path, allow_pickle=False) as bundle:
        for key in ("meg_rdm_subjects", "time_ms", "stimulus_file_name"):
            if key not in bundle:
                raise SystemExit(f"Missing key '{key}' in MEG bundle: {bundle_path}")

        meg_subjects = np.asarray(bundle["meg_rdm_subjects"], dtype=np.float64)
        if meg_subjects.ndim == 3:
            meg_subjects = meg_subjects[:, :, :, np.newaxis]
        if meg_subjects.ndim != 4 or meg_subjects.shape[0] != meg_subjects.shape[1]:
            raise SystemExit(
                f"Bundle key 'meg_rdm_subjects' must have shape [N,N,T,S], got {meg_subjects.shape}"
            )

        n_images = meg_subjects.shape[0]
        n_time = meg_subjects.shape[2]

        time_ms = np.asarray(bundle["time_ms"], dtype=np.float64).reshape(-1)
        if time_ms.size != n_time:
            raise SystemExit(
                f"Bundle time vector length ({time_ms.size}) does not match MEG time dimension ({n_time})."
            )

        tokens = [str(x) for x in np.asarray(bundle["stimulus_file_name"]).reshape(-1)]
        if len(tokens) != n_images:
            raise SystemExit(
                f"Bundle stimulus list length ({len(tokens)}) does not match MEG image dimension ({n_images})."
            )

        images_root = images_dir.resolve()
        stimulus_order: list[Path] = []
        for token in tokens:
            path = Path(token)
            if not path.is_absolute():
                path = (images_root / path).resolve()
            else:
                path = path.resolve()
            stimulus_order.append(path)

        if len(set(map(str, stimulus_order))) != len(stimulus_order):
            raise SystemExit(f"MEG bundle contains duplicate stimulus entries: {bundle_path}")

        if "meg_rdm_avg" in bundle:
            meg_avg = np.asarray(bundle["meg_rdm_avg"], dtype=np.float64)
        else:
            meg_avg = meg_subjects.mean(axis=3)
        if meg_avg.shape != (n_images, n_images, n_time):
            raise SystemExit(f"Bundle key 'meg_rdm_avg' has unexpected shape: {meg_avg.shape}")

    return meg_avg, meg_subjects, time_ms, stimulus_order
