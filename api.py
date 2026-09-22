"""
HTTP API surface.

Registered on ComfyUI's PromptServer (aiohttp). Kept deliberately minimal.
`routes.py` / `civitai_routes.py` are reference only, not loaded.
"""
from aiohttp import web

from server import PromptServer

from . import hashing
from . import downloads
from . import favorites


def setup_routes() -> None:
    """Register API routes with the running ComfyUI server."""
    routes = PromptServer.instance.routes

    # -- model hash index --------------------------------------------------

    @routes.get("/imagelab/api/hashes")
    async def get_hashes(request: web.Request) -> web.Response:
        """
        Return the full model-hash cache in one shot:

            { "version": "<etag>", "models": [ {key, filename, hash, hashed_at}, ... ] }

        `version` is a content token over every (key, hash) pair. Clients
        should send it back via `If-None-Match`; when nothing has changed the
        endpoint replies `304 Not Modified` with no body, so repeat loads are
        essentially free. The payload itself is served straight from memory --
        no per-request disk I/O.
        """
        version, entries = hashing.get_snapshot()
        etag = f'"{version}"'

        if request.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers={"ETag": etag})

        return web.json_response(
            {"version": version, "models": entries},
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    # -- CivitAI downloads -------------------------------------------------

    @routes.post("/imagelab/api/downloads")
    async def start_download(request: web.Request) -> web.Response:
        """
        Start a CivitAI download.

            body: { version_id: int, folder?: str, filename?: str }

        `folder` is a ComfyUI folder name (e.g. "checkpoints"); omit it to let
        the destination be derived from CivitAI's model type. Returns
        immediately -- the download runs in the background; poll
        `GET /imagelab/api/downloads` for progress.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body", "code": "VALIDATION_ERROR"}, status=400)

        version_id = body.get("version_id")
        if version_id is None:
            return web.json_response(
                {"error": "version_id is required", "code": "VALIDATION_ERROR"},
                status=400,
            )

        try:
            downloads.start_download(int(version_id), body.get("folder"), body.get("filename"))
        except downloads.CivitaiError as e:
            return web.json_response({"error": e.message, "code": e.code}, status=e.status)
        except (TypeError, ValueError):
            return web.json_response({"error": "version_id must be an integer", "code": "VALIDATION_ERROR"}, status=400)

        return web.json_response({"version_id": int(version_id), "status": "started"})

    @routes.get("/imagelab/api/downloads")
    async def list_downloads(request: web.Request) -> web.Response:
        """List every tracked download -- active, completed, failed, cancelled."""
        return web.json_response({"downloads": downloads.list_downloads()})

    @routes.delete("/imagelab/api/downloads/{version_id}")
    async def cancel_download(request: web.Request) -> web.Response:
        """Cancel an in-flight download, or dismiss a finished/failed row."""
        try:
            version_id = int(request.match_info["version_id"])
        except ValueError:
            return web.json_response({"error": "version_id must be an integer", "code": "VALIDATION_ERROR"}, status=400)
        return web.json_response(downloads.cancel_download(version_id))

    # -- local model deletion ---------------------------------------------

    @routes.delete("/imagelab/api/models/{model_type}/{filename:.*}")
    async def delete_model(request: web.Request) -> web.Response:
        """Delete a local model file and drop it from the hash index."""
        try:
            downloads.delete_model(request.match_info["model_type"], request.match_info["filename"])
        except downloads.CivitaiError as e:
            return web.json_response({"error": e.message, "code": e.code}, status=e.status)
        return web.json_response({"deleted": True})

    # -- favorites ---------------------------------------------------------
    #
    # Persist explicitly starred images so they survive ComfyUI restarts.
    # See `favorites.py` for the layout and storage rationale.

    @routes.post("/imagelab/api/favorites")
    async def create_favorite(request: web.Request) -> web.Response:
        """
        Save a comfy-managed image (typically a PreviewImage output in `temp/`)
        into the favorites tree under today's MM-DD-YYYY folder.

            body: { filename, subfolder?, type? }   # type defaults to "temp"
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body", "code": "VALIDATION_ERROR"}, status=400)

        filename = body.get("filename")
        if not filename or not isinstance(filename, str):
            return web.json_response(
                {"error": "filename is required", "code": "VALIDATION_ERROR"},
                status=400,
            )
        subfolder = body.get("subfolder") or ""
        image_type = body.get("type") or "temp"
        if image_type not in ("output", "temp", "input"):
            return web.json_response(
                {"error": "type must be one of output/temp/input", "code": "VALIDATION_ERROR"},
                status=400,
            )

        try:
            fav = favorites.save_favorite(filename, subfolder, image_type)
        except FileNotFoundError as e:
            return web.json_response({"error": str(e), "code": "NOT_FOUND"}, status=404)
        except Exception as e:  # noqa: BLE001 -- surface any unexpected I/O failure
            return web.json_response({"error": str(e), "code": "SAVE_FAILED"}, status=500)

        return web.json_response({"favorite": fav})

    @routes.get("/imagelab/api/favorites")
    async def list_favorites(request: web.Request) -> web.Response:
        """
        List every favorite. ETag'd so polling clients don't repeatedly
        re-download a list that hasn't changed.
        """
        version, entries = favorites.list_favorites()
        etag = f'"{version}"'
        if request.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers={"ETag": etag})
        return web.json_response(
            {"version": version, "favorites": entries},
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    @routes.delete("/imagelab/api/favorites/{date}/{filename:.*}")
    async def delete_favorite(request: web.Request) -> web.Response:
        """Remove a favorite from disk."""
        date = request.match_info["date"]
        filename = request.match_info["filename"]
        ok = favorites.delete_favorite(date, filename)
        if not ok:
            return web.json_response({"error": "Favorite not found", "code": "NOT_FOUND"}, status=404)
        return web.json_response({"deleted": True})

    @routes.get("/imagelab/api/favorites/view")
    async def view_favorite(request: web.Request) -> web.Response:
        """Serve the image bytes for a single favorite."""
        date = request.query.get("date", "")
        filename = request.query.get("filename", "")
        resolved = favorites.open_favorite(date, filename)
        if not resolved:
            return web.json_response({"error": "Favorite not found", "code": "NOT_FOUND"}, status=404)
        path, content_type = resolved
        # FileResponse handles Range, Content-Length, ETag from mtime.
        return web.FileResponse(path, headers={"Content-Type": content_type, "Cache-Control": "no-cache"})

    # -- what is running -------------------------------------------------------------
    #
    # A node that can be changed on a live pod needs a way to say which version answered. Without
    # it "did the push land?" is guesswork, and the honest test of a hot-push loop is a route that
    # did not exist before the push.

    @routes.get("/imagelab/api/version")
    async def get_version(request: web.Request) -> web.Response:
        """Identify the running node: its name, and the mtime of this file."""
        import os
        import time

        return web.json_response({
            "name": "ImageLabCore",
            "source": os.path.dirname(os.path.abspath(__file__)),
            "api_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(__file__))),
        })

    print("[ImageLab] API routes registered")
