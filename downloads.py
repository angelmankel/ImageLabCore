"""
CivitAI model downloads + local model deletion.

Streams model files from CivitAI into the right ComfyUI model folder, tracks
progress in memory (polled via the API — no WebSocket), and hands finished
downloads to `hashing.py` so they land in `/imagelab/api/hashes` immediately.

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

Files come down over up to `_PARTS` connections at once when the server takes Range requests
(CivitAI's CDN and Hugging Face both do): one connection from CivitAI runs at about 1 MB/s, which
made a 16 GB video model a four-hour wait. Each part retries on its own.

Besides CivitAI versions, any https URL can be fetched into a named folder (`start_url_download`)
— text encoders and VAEs live on Hugging Face, not CivitAI. `HF_TOKEN`, when set, is sent to
huggingface.co for gated repos.

A "Checkpoint" from CivitAI that turns out to be a diffusion model only is moved to
`diffusion_models/` once it lands (`model_sort`).
"""
import os
import re
import time
import zlib
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import folder_paths

from . import hashing
from . import model_sort

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

_MODEL_EXTS = (".safetensors", ".pt", ".ckpt", ".bin", ".pth", ".sft", ".gguf")
# Record progress at most every this many bytes.
_PROGRESS_STEP = 512 * 1024
# Parallel connections for one file, and the smallest file worth splitting.
_PARTS = 8
_SPLIT_MIN = 64 * 1024 * 1024
_PART_RETRIES = 4


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
# client dismisses them (DELETE /imagelab/api/downloads/<id>).
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


def _invalidate_folder_cache(folder: Optional[str]) -> None:
    """Drop ComfyUI's cached filename list for a folder so a just-added file shows in /object_info.

    ComfyUI caches each folder's listing. Some builds only re-scan when a directory's mtime looks
    changed, so a freshly downloaded LoRA (or any model) could stay missing from the model pickers
    until a restart. Popping the cache entry forces the next `get_filename_list` to re-scan. Written
    to tolerate every folder_paths version: a missing attribute or a changed shape is a no-op.
    """
    if not folder:
        return
    for attr in ("filename_list_cache", "cache"):
        cache = getattr(folder_paths, attr, None)
        if isinstance(cache, dict):
            cache.pop(folder, None)
    helper = getattr(folder_paths, "cache_helper", None)
    try:
        if helper is not None and hasattr(helper, "pop"):
            helper.pop(folder, None)
    except Exception:  # noqa: BLE001 — a cache clear must never break a finished download.
        pass


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


def _refusal(status: int, body: object) -> "CivitaiError":
    """Turn a 401/403 from the download URL into CivitAI's actual reason.

    A 403 is not always the token: Early Access models answer 403 with
    {"error": "Early Access", "deadline": ...} to a perfectly valid key, and the old blanket
    "needs a valid CIVITAI_API_KEY" sent people chasing a key that was fine.
    """
    info = body if isinstance(body, dict) else {}
    reason = str(info.get("error") or "")
    if reason.lower() == "early access":
        deadline = str(info.get("deadline") or "")[:10]
        when = f" It becomes free on {deadline}." if deadline else ""
        return CivitaiError(
            f"Early Access on CivitAI: the creator charges Buzz to download it now.{when}",
            "EARLY_ACCESS", 403)
    if status == 401 or not os.environ.get("CIVITAI_API_KEY"):
        return CivitaiError("CivitAI rejected the API key (CIVITAI_API_KEY)", "AUTH_ERROR", 401)
    detail = str(info.get("message") or reason or "").strip()
    return CivitaiError(f"CivitAI refused the download{': ' + detail if detail else ' (HTTP 403)'}",
                        "FORBIDDEN", 403)


def _pick_file(version: Dict) -> Dict:
    """Choose the file to download — the primary, else the first safetensors."""
    files = version.get("files") or []
    if not files:
        raise CivitaiError("This version has no downloadable files", "NO_FILES", 422)
    primary = next((f for f in files if f.get("primary")), None)
    safet = next((f for f in files if str(f.get("name", "")).endswith(".safetensors")), None)
    return primary or safet or files[0]


def _host_headers(url: str) -> Dict[str, str]:
    """Auth for the first request only: Hugging Face gated repos want HF_TOKEN; CivitAI takes
    its token in the URL. Redirected CDN URLs are signed and get no header."""
    hf = os.environ.get("HF_TOKEN")
    if hf and re.match(r"https://([\w.-]+\.)?huggingface\.co/", url):
        return {"Authorization": f"Bearer {hf}"}
    return {}


async def _stream(url: str, headers: Dict[str, str], target_path: str, progress: DownloadProgress,
                  civitai: bool = False) -> bool:
    """
    Download `url` to `target_path` (via `<target>.downloading`, renamed into place at the end).
    Splits into `_PARTS` Range requests when the server allows it, else one stream. Returns False
    if the download was cancelled (the partial file is removed), True when the file is in place.
    """
    key = progress.version_id
    temp_path = target_path + ".downloading"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Probe with a one-byte range: follows redirects to the real (signed) URL and says whether
        # the server takes ranges and how big the file is.
        async with session.get(url, headers={**headers, "Range": "bytes=0-0"}, allow_redirects=True) as res:
            if res.status in (401, 403):
                try:
                    body = await res.json(content_type=None)
                except Exception:
                    body = None
                if civitai:
                    raise _refusal(res.status, body)
                raise CivitaiError(f"Refused (HTTP {res.status}) — gated? Set HF_TOKEN for Hugging Face.", "FORBIDDEN", res.status)
            if res.status not in (200, 206):
                raise CivitaiError(f"Download failed: HTTP {res.status}", "DOWNLOAD_FAILED", 502)
            final_url = str(res.url)
            total = 0
            ranged = res.status == 206
            if ranged:
                m = re.search(r"/(\d+)$", res.headers.get("Content-Range", ""))
                total = int(m.group(1)) if m else 0
            else:
                total = int(res.headers.get("Content-Length") or 0)
            if res.status == 206:
                await res.read()
        if total:
            progress.total_bytes = total

        cancelled = lambda: key in _cancelled  # noqa: E731
        progress.downloaded_bytes = 0

        if not (ranged and total >= _SPLIT_MIN):
            # One stream from the start. The probe's redirect target is reused when there was one.
            async with session.get(final_url if ranged else url, headers={} if ranged else headers, allow_redirects=True) as res:
                if res.status != 200:
                    raise CivitaiError(f"Download failed: HTTP {res.status}", "DOWNLOAD_FAILED", 502)
                last_mark = 0
                with open(temp_path, "wb") as fh:
                    async for chunk in res.content.iter_chunked(1 << 20):
                        if cancelled():
                            break
                        fh.write(chunk)
                        progress.downloaded_bytes += len(chunk)
                        if progress.downloaded_bytes - last_mark >= _PROGRESS_STEP:
                            last_mark = progress.downloaded_bytes
                            progress.updated_at = time.time()
        else:
            with open(temp_path, "wb") as fh:
                fh.truncate(total)
            step = -(-total // _PARTS)

            async def part(start: int, end: int) -> None:
                pos = start
                for attempt in range(_PART_RETRIES):
                    try:
                        async with session.get(final_url, headers={"Range": f"bytes={pos}-{end}"}) as res:
                            if res.status != 206:
                                raise CivitaiError(f"Part {start}: HTTP {res.status}", "DOWNLOAD_FAILED", 502)
                            with open(temp_path, "r+b") as fh:
                                fh.seek(pos)
                                async for chunk in res.content.iter_chunked(1 << 20):
                                    if cancelled():
                                        return
                                    fh.write(chunk)
                                    pos += len(chunk)
                                    progress.downloaded_bytes += len(chunk)
                                    progress.updated_at = time.time()
                        if pos > end:
                            return
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt == _PART_RETRIES - 1:
                            raise CivitaiError(f"Download failed after {_PART_RETRIES} tries: {e}", "DOWNLOAD_FAILED", 502)
                        await asyncio.sleep(2 * (attempt + 1))
                if pos <= end and not cancelled():
                    raise CivitaiError("Download ended early", "DOWNLOAD_FAILED", 502)

            parts = [asyncio.create_task(part(i, min(total, i + step) - 1)) for i in range(0, total, step)]
            try:
                await asyncio.gather(*parts)
            except BaseException:
                # One part failed (or the whole download was cancelled): stop the others before
                # the caller removes the file they are writing to.
                for t in parts:
                    t.cancel()
                await asyncio.gather(*parts, return_exceptions=True)
                raise

    if key in _cancelled:
        progress.status = "cancelled"
        progress.updated_at = time.time()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    if total and os.path.getsize(temp_path) != total:
        raise CivitaiError("Downloaded size does not match", "DOWNLOAD_FAILED", 502)
    # Atomically swap the completed file into place.
    os.replace(temp_path, target_path)
    return True


async def _run_download(version_id: int, folder: Optional[str], filename: Optional[str]) -> None:
    """The actual streamed download. Runs as a background task; updates `_progress`."""
    progress = _progress[version_id]
    target_dir: Optional[str] = None
    forced_folder = folder is not None

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

        # A diffusion model downloaded before was moved out of checkpoints/: that counts too.
        if folder == "checkpoints" and not forced_folder and not os.path.exists(target_path):
            sorted_dir = model_sort.diffusion_models_dir()
            if sorted_dir and os.path.exists(os.path.join(sorted_dir, name)):
                target_path = os.path.join(sorted_dir, name)
                progress.folder = "diffusion_models"

        # Already on disk (persistent volume / re-run) — just make sure it's indexed.
        if os.path.exists(target_path):
            size = os.path.getsize(target_path)
            progress.downloaded_bytes = progress.total_bytes = size
            progress.status = "completed"
            progress.updated_at = time.time()
            await asyncio.to_thread(hashing.hash_model, target_path)
            _invalidate_folder_cache(progress.folder)
            print(f"[ImageLab] {name} already present — skipped")
            return

        download_url = file_info.get("downloadUrl") or f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
        token = _token()
        if token:
            download_url += ("&" if "?" in download_url else "?") + f"token={token}"

        if not await _stream(download_url, {}, target_path, progress, civitai=True):
            return  # cancelled

        # CivitAI calls every base model a Checkpoint; a diffusion model alone belongs elsewhere.
        if folder == "checkpoints" and not forced_folder:
            moved = await asyncio.to_thread(model_sort.sort_file, target_path)
            if moved:
                target_path = moved
                progress.folder = "diffusion_models"

        # Hash it into the index so it shows up in /imagelab/api/hashes right away.
        # hash_model is blocking (reads the whole file) — keep it off the loop.
        await asyncio.to_thread(hashing.hash_model, target_path)

        # Clear ComfyUI's cached folder listing so the new file appears in /object_info (and so the
        # app's model/LoRA pickers) without waiting for a restart. Clear both the folder it landed
        # in and `progress.folder`: model_sort may have moved a checkpoint to diffusion_models/, so
        # the source list is stale too.
        _invalidate_folder_cache(folder)
        _invalidate_folder_cache(progress.folder)

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


def url_key(url: str) -> int:
    """Download rows are keyed by CivitAI version id; a URL gets a stable negative id instead."""
    return -(zlib.crc32(url.encode()) & 0x7FFFFFFF) - 1


async def _run_url_download(key: int, url: str, folder: str, name: str) -> None:
    progress = _progress[key]
    target_dir = _target_dir(folder)
    try:
        target_path = os.path.join(target_dir, name)
        if os.path.exists(target_path):
            progress.downloaded_bytes = progress.total_bytes = os.path.getsize(target_path)
        elif not await _stream(url, _host_headers(url), target_path, progress):
            return  # cancelled
        await asyncio.to_thread(hashing.hash_model, target_path)
        _invalidate_folder_cache(folder)
        progress.status = "completed"
        progress.updated_at = time.time()
        print(f"[ImageLab] Downloaded {folder}/{name}")
    except asyncio.CancelledError:
        progress.status = "cancelled"
        progress.updated_at = time.time()
        _cleanup_temp(target_dir, name)
        raise
    except Exception as e:  # noqa: BLE001 — surface any failure to the panel
        progress.status = "failed"
        progress.error = e.message if isinstance(e, CivitaiError) else str(e)
        progress.updated_at = time.time()
        _cleanup_temp(target_dir, name)
        print(f"[ImageLab] Download failed ({url}): {progress.error}")
    finally:
        _cancelled.discard(key)
        _tasks.pop(key, None)


def start_url_download(url: str, folder: str, filename: Optional[str] = None) -> int:
    """
    Download any https URL (Hugging Face, mostly) into a ComfyUI folder. Returns the row's key.
    The name defaults to the URL's last path segment.
    """
    if not re.match(r"https?://", url or ""):
        raise CivitaiError("url must be http(s)", "VALIDATION_ERROR", 400)
    if folder not in folder_paths.folder_names_and_paths:
        raise CivitaiError(f"Unknown folder: {folder}", "VALIDATION_ERROR", 400)
    _target_dir(folder)
    name = os.path.basename(filename or url.split("?", 1)[0].rstrip("/"))
    if not name or not name.endswith(_MODEL_EXTS):
        raise CivitaiError("Give a filename ending in a model extension", "VALIDATION_ERROR", 400)
    key = url_key(url)
    existing = _tasks.get(key)
    if existing and not existing.done():
        raise CivitaiError(f"{name} is already downloading", "ALREADY_RUNNING", 409)
    _cancelled.discard(key)
    _progress[key] = DownloadProgress(version_id=key, model_id=0, folder=folder, filename=name,
                                      downloaded_bytes=0, total_bytes=0, status="downloading")
    _tasks[key] = asyncio.create_task(_run_url_download(key, url, folder, name))
    return key


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
