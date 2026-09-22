"""
CivitAI model downloads + local model deletion.

Streams model files from CivitAI into the right ComfyUI model folder, tracks
progress in memory (polled via the API — no WebSocket), and hands finished
downloads to `hashing.py` so they land in `/imagelab/hashes` immediately.

The destination folder is normally derived from CivitAI's own model type, so
callers only need a version id. A folder can still be forced when CivitAI's
type doesn't map cleanly.

Server-side only — a browser can't write multi-GB files to the ComfyUI host.
The CivitAI API token is read from the `CIVITAI_API_KEY` env var; without it,
download still works for public models but may hit auth/rate limits.

Startup auto-download: set `IMAGELAB_AUTO_DOWNLOAD` to a comma/newline list of
version ids, CivitAI URLs, or `model:<id>` / `<folder>::<entry>` forms — see
`parse_auto_download`. Queued from `__init__.py` once the server loop is up.

Download/cancel/delete only, wired into the hash index in `hashing.py`.
"""
import os
import re
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import folder_paths

from . import hashing

# Configurable for proxying / mirrors.
CIVITAI_API_BASE = os.environ.get("CIVITAI_API_BASE", "https://civitai.com/api/v1")
CIVITAI_DOWNLOAD_BASE = os.environ.get("CIVITAI_DOWNLOAD_BASE", "https://civitai.com/api/download/models")

# CivitAI model type → ComfyUI model folder.
MODEL_TYPE_TO_FOLDER = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "TextualInversion": "embeddings",
    "Hypernetwork": "hypernetworks",
    "Controlnet": "controlnet",
    "VAE": "vae",
    "Upscaler": "upscale_models",
    "MotionModule": "animatediff_motion_lora",
}

_MODEL_EXTS = (".safetensors", ".pt", ".ckpt", ".bin", ".pth")
# Record progress at most every this many bytes.
_PROGRESS_STEP = 512 * 1024


class CivitaiError(Exception):
    """A CivitAI/download failure carrying an HTTP status for the API layer."""

    def __init__(self, message: str, code: str = "CIVITAI_ERROR", status: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass
class DownloadProgress:
    version_id: int
    model_id: int
    folder: str  # ComfyUI folder, e.g. "checkpoints" — "?" until resolved
    filename: str
    downloaded_bytes: int
    total_bytes: int
    status: str  # 'downloading' | 'completed' | 'failed' | 'cancelled'
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        percent = round(self.downloaded_bytes / self.total_bytes * 100, 1) if self.total_bytes else 0.0
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "folder": self.folder,
            "filename": self.filename,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": percent,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


# In-memory download state. Keyed by CivitAI version_id. Entries persist after
# completion so the downloads panel can show finished/failed rows until the
# client dismisses them (DELETE /imagelab/downloads/<id>).
_progress: Dict[int, DownloadProgress] = {}
_tasks: Dict[int, asyncio.Task] = {}
_cancelled: set = set()


def _token() -> Optional[str]:
    return os.environ.get("CIVITAI_API_KEY")


def _api_headers() -> Dict[str, str]:
    token = _token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _target_dir(folder: str) -> str:
    """Resolve a ComfyUI folder name to its on-disk path."""
    paths = folder_paths.folder_names_and_paths.get(folder)
    if not paths or not paths[0]:
        raise CivitaiError(f"No folder configured for '{folder}'", "CONFIG_ERROR", 500)
    first = paths[0][0] if isinstance(paths[0], list) else paths[0]
    return first


def _folder_for_type(civitai_type: str) -> str:
    """Map a CivitAI model type ('Checkpoint', 'LORA', …) to a ComfyUI folder."""
    folder = MODEL_TYPE_TO_FOLDER.get(civitai_type)
    if not folder:
        raise CivitaiError(
            f"CivitAI type '{civitai_type}' has no folder mapping — "
            f"prefix the entry with '<folder>::' to choose one",
            "UNMAPPED_TYPE",
            422,
        )
    return folder


async def _fetch_version(version_id: int) -> Dict:
    """Fetch a model version's metadata (file list, download URL, model type)."""
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_api_headers(), timeout=aiohttp.ClientTimeout(total=30)) as res:
            if res.status in (401, 403):
                raise CivitaiError("CivitAI rejected the API token", "AUTH_ERROR", 401)
            if res.status == 404:
                raise CivitaiError(f"Version {version_id} not found", "NOT_FOUND", 404)
            if res.status == 429:
                raise CivitaiError("CivitAI rate limit hit", "RATE_LIMITED", 429)
            if res.status != 200:
                raise CivitaiError(f"CivitAI API error {res.status}", "CIVITAI_ERROR", 502)
            return await res.json()


async def _fetch_model(model_id: int) -> Dict:
    """Fetch a model's metadata — used to resolve a model id to its latest version."""
    url = f"{CIVITAI_API_BASE}/models/{model_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_api_headers(), timeout=aiohttp.ClientTimeout(total=30)) as res:
            if res.status in (401, 403):
                raise CivitaiError("CivitAI rejected the API token", "AUTH_ERROR", 401)
            if res.status == 404:
                raise CivitaiError(f"Model {model_id} not found", "NOT_FOUND", 404)
            if res.status != 200:
                raise CivitaiError(f"CivitAI API error {res.status}", "CIVITAI_ERROR", 502)
            return await res.json()


def _pick_file(version: Dict) -> Dict:
    """Choose the file to download — the primary, else the first safetensors."""
    files = version.get("files") or []
    if not files:
        raise CivitaiError("This version has no downloadable files", "NO_FILES", 422)
    primary = next((f for f in files if f.get("primary")), None)
    safet = next((f for f in files if str(f.get("name", "")).endswith(".safetensors")), None)
    return primary or safet or files[0]


async def _run_download(version_id: int, folder: Optional[str], filename: Optional[str]) -> None:
    """The actual streamed download. Runs as a background task; updates `_progress`."""
    progress = _progress[version_id]
    target_dir: Optional[str] = None

    try:
        version = await _fetch_version(version_id)
        file_info = _pick_file(version)

        # Derive the destination folder from CivitAI's model type unless forced.
        if folder is None:
            folder = _folder_for_type((version.get("model") or {}).get("type", ""))
        target_dir = _target_dir(folder)

        name = filename or file_info["name"]
        if not name.endswith(_MODEL_EXTS):
            name += ".safetensors"

        progress.model_id = version.get("modelId", 0)
        progress.folder = folder
        progress.filename = name
        progress.total_bytes = int(file_info.get("sizeKB", 0) * 1024)

        target_path = os.path.join(target_dir, name)

        # Already on disk (persistent volume / re-run) — just make sure it's indexed.
        if os.path.exists(target_path):
            size = os.path.getsize(target_path)
            progress.downloaded_bytes = progress.total_bytes = size
            progress.status = "completed"
            progress.updated_at = time.time()
            await asyncio.to_thread(hashing.hash_model, target_path)
            print(f"[ImageLab] {name} already present — skipped")
            return

        temp_path = target_path + ".downloading"
        download_url = file_info.get("downloadUrl") or f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
        token = _token()
        if token:
            download_url += ("&" if "?" in download_url else "?") + f"token={token}"

        os.makedirs(target_dir, exist_ok=True)
        last_mark = 0

        async with aiohttp.ClientSession() as session:
            async with session.get(
                download_url,
                timeout=aiohttp.ClientTimeout(total=None, connect=30),
                allow_redirects=True,
            ) as res:
                if res.status in (401, 403):
                    raise CivitaiError("Download requires a valid CIVITAI_API_KEY", "AUTH_ERROR", 401)
                if res.status != 200:
                    raise CivitaiError(f"Download failed: HTTP {res.status}", "DOWNLOAD_FAILED", 502)

                content_length = res.headers.get("Content-Length")
                if content_length:
                    progress.total_bytes = int(content_length)

                with open(temp_path, "wb") as fh:
                    async for chunk in res.content.iter_chunked(65536):
                        if version_id in _cancelled:
                            progress.status = "cancelled"
                            progress.updated_at = time.time()
                            fh.close()
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return
                        fh.write(chunk)
                        progress.downloaded_bytes += len(chunk)
                        if progress.downloaded_bytes - last_mark >= _PROGRESS_STEP:
                            last_mark = progress.downloaded_bytes
                            progress.updated_at = time.time()

        # Atomically swap the completed file into place.
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

        # Hash it into the index so it shows up in /imagelab/hashes right away.
        # hash_model is blocking (reads the whole file) — keep it off the loop.
        await asyncio.to_thread(hashing.hash_model, target_path)

        progress.status = "completed"
        progress.updated_at = time.time()
        print(f"[ImageLab] Downloaded {name}")

    except asyncio.CancelledError:
        progress.status = "cancelled"
        progress.updated_at = time.time()
        _cleanup_temp(target_dir, progress.filename)
        raise
    except CivitaiError as e:
        progress.status = "failed"
        progress.error = e.message
        progress.updated_at = time.time()
        _cleanup_temp(target_dir, progress.filename)
        print(f"[ImageLab] Download failed ({version_id}): {e.message}")
    except Exception as e:  # noqa: BLE001 — surface any failure to the panel
        progress.status = "failed"
        progress.error = str(e)
        progress.updated_at = time.time()
        _cleanup_temp(target_dir, progress.filename)
        print(f"[ImageLab] Download error ({version_id}): {e}")
    finally:
        _cancelled.discard(version_id)
        _tasks.pop(version_id, None)


def _cleanup_temp(target_dir: Optional[str], filename: str) -> None:
    if not target_dir:
        return
    temp_path = os.path.join(target_dir, filename + ".downloading")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


def start_download(version_id: int, folder: Optional[str] = None, filename: Optional[str] = None) -> None:
    """
    Begin downloading a CivitAI model version in the background.

    `folder` is a ComfyUI folder name (e.g. "checkpoints"); when omitted it's
    derived from CivitAI's model type. Raises CivitaiError synchronously for
    bad input or an already-running download; anything that fails mid-download
    is reported via `list_downloads()`.
    """
    if folder is not None:
        if folder not in folder_paths.folder_names_and_paths:
            raise CivitaiError(f"Unknown folder: {folder}", "VALIDATION_ERROR", 400)
        _target_dir(folder)  # fail fast if the folder isn't configured

    existing = _tasks.get(version_id)
    if existing and not existing.done():
        raise CivitaiError(f"Version {version_id} is already downloading", "ALREADY_RUNNING", 409)

    _cancelled.discard(version_id)
    _progress[version_id] = DownloadProgress(
        version_id=version_id,
        model_id=0,
        folder=folder or "?",
        filename=filename or str(version_id),
        downloaded_bytes=0,
        total_bytes=0,
        status="downloading",
    )
    _tasks[version_id] = asyncio.create_task(_run_download(version_id, folder, filename))


def cancel_download(version_id: int) -> Dict[str, bool]:
    """
    Cancel an in-flight download, or dismiss a finished/failed row. Returns
    `{cancelled, cleared}` — `cancelled` if a running task was stopped,
    `cleared` if a completed entry was removed from the list.
    """
    task = _tasks.get(version_id)
    if task and not task.done():
        _cancelled.add(version_id)
        task.cancel()
        return {"cancelled": True, "cleared": False}
    if version_id in _progress:
        del _progress[version_id]
        return {"cancelled": False, "cleared": True}
    return {"cancelled": False, "cleared": False}


def list_downloads() -> List[Dict]:
    """All tracked downloads — active, completed, failed, cancelled."""
    return [p.to_dict() for p in _progress.values()]


def delete_model(model_type: str, filename: str) -> None:
    """Delete a local model file and drop it from the hash index."""
    if model_type not in folder_paths.folder_names_and_paths:
        raise CivitaiError(f"Invalid model type: {model_type}", "VALIDATION_ERROR", 400)
    filepath = folder_paths.get_full_path(model_type, filename)
    if not filepath or not os.path.exists(filepath):
        raise CivitaiError(f"Model not found: {filename}", "NOT_FOUND", 404)
    os.remove(filepath)
    hashing.forget(filepath)
    print(f"[ImageLab] Deleted {model_type}/{filename}")


# ── startup auto-download ──────────────────────────────────────────────────

@dataclass
class _AutoEntry:
    """One parsed `IMAGELAB_AUTO_DOWNLOAD` entry — exactly one id is set."""
    version_id: Optional[int]
    model_id: Optional[int]
    folder: Optional[str]  # forced folder via `<folder>::` prefix, else None


def parse_auto_download(raw: str) -> List[_AutoEntry]:
    """
    Parse the `IMAGELAB_AUTO_DOWNLOAD` value into entries. Accepts a comma- or
    newline-separated list; each entry is one of:

        128713                                    a CivitAI version id
        https://civitai.com/models/4384?modelVersionId=128713   a version URL
        https://civitai.com/models/4384           a model URL → latest version
        model:4384                                a model id  → latest version
        upscale_models::128713                    any of the above, forced into
                                                  the named ComfyUI folder

    Unrecognized entries are skipped with a warning rather than aborting.
    """
    entries: List[_AutoEntry] = []
    for chunk in re.split(r"[,\n]", raw):
        chunk = chunk.strip()
        if not chunk:
            continue

        folder: Optional[str] = None
        if "::" in chunk:
            folder, chunk = (part.strip() for part in chunk.split("::", 1))

        url = re.search(r"civitai\.com/models/(\d+)", chunk)
        if url:
            ver = re.search(r"modelVersionId=(\d+)", chunk)
            if ver:
                entries.append(_AutoEntry(int(ver.group(1)), None, folder))
            else:
                entries.append(_AutoEntry(None, int(url.group(1)), folder))
        elif chunk.lower().startswith("model:") and chunk[6:].strip().isdigit():
            entries.append(_AutoEntry(None, int(chunk[6:].strip()), folder))
        elif chunk.isdigit():
            entries.append(_AutoEntry(int(chunk), None, folder))
        else:
            print(f"[ImageLab] auto-download: skipping unrecognized entry {chunk!r}")

    return entries


async def queue_startup_downloads() -> None:
    """
    Read `IMAGELAB_AUTO_DOWNLOAD` and queue every entry. Runs once, from the
    server's startup hook (so the event loop exists). Errors are logged per
    entry — one bad id never blocks the rest.
    """
    raw = os.environ.get("IMAGELAB_AUTO_DOWNLOAD", "").strip()
    if not raw:
        return

    entries = parse_auto_download(raw)
    if not entries:
        return

    print(f"[ImageLab] auto-download: queueing {len(entries)} model(s)")
    for entry in entries:
        try:
            version_id = entry.version_id
            if version_id is None:
                model = await _fetch_model(entry.model_id)
                versions = model.get("modelVersions") or []
                if not versions:
                    print(f"[ImageLab] auto-download: model {entry.model_id} has no versions")
                    continue
                version_id = versions[0]["id"]  # newest first
            start_download(version_id, folder=entry.folder)
        except CivitaiError as e:
            print(f"[ImageLab] auto-download failed: {e.message}")
        except Exception as e:  # noqa: BLE001
            print(f"[ImageLab] auto-download error: {e}")
