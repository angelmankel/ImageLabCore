"""
ImageLabCore — ComfyUI custom node entrypoint.

It:
  - hashes all model files on startup (background) into an index
  - exposes that index + CivitAI download/delete over a small HTTP API
  - optionally auto-downloads models on startup (IMAGELAB_AUTO_DOWNLOAD)

Other modules in this directory (`routes.py`, `workflows.py`, `imagelab_nodes.py`,
`resources.py`, ...) are reference only and not loaded.
"""
import os
import threading


def _load_env_file() -> None:
    """Populate os.environ from the extension's `.env` file.

    ComfyUI is launched directly via `python main.py`, so nothing injects
    these vars. Without this, CIVITAI_API_KEY / IMAGELAB_AUTO_DOWNLOAD sit
    unread in .env and downloads fail with a 401.

    Existing env vars win — the shell/pod template can still override.
    Must run before importing modules that read env at import time.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"[ImageLab] Failed to read .env: {e}")


_load_env_file()

from server import PromptServer

from . import hashing
from . import api
from . import downloads

# No custom nodes yet — ComfyUI still expects these symbols to exist.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register the HTTP API (synchronous — routes must exist before the server
# finishes starting; this is just dict insertion, it does not block).
api.setup_routes()


def _hash_on_startup():
    print("[ImageLab] Building model-hash index in the background ...")
    try:
        counts = hashing.build_index()
        if counts:
            print(f"[ImageLab] Hashed: {counts}")
        else:
            print("[ImageLab] No new models to hash")
        version, entries = hashing.get_snapshot()
        print(f"[ImageLab] Index ready: {len(entries)} models (version {version})")
    except Exception as e:
        print(f"[ImageLab] Error during startup hashing: {e}")


threading.Thread(target=_hash_on_startup, daemon=True).start()


async def _auto_download_on_startup(_app):
    """aiohttp on_startup hook — runs inside the server loop, so downloads
    (which create asyncio tasks) can be queued safely."""
    try:
        await downloads.queue_startup_downloads()
    except Exception as e:
        print(f"[ImageLab] Error queueing startup downloads: {e}")


# Defer auto-downloads until the server's event loop is running.
PromptServer.instance.app.on_startup.append(_auto_download_on_startup)

print("[ImageLabCore] loaded")
