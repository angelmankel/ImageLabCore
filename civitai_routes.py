"""
CivitAI API routes for model management.
"""
from aiohttp import web

from server import PromptServer

from . import civitai


def SetupCivitAIRoutes(server: PromptServer):
    """Register CivitAI API routes with the ComfyUI server."""

    # =========== SEARCH & DISCOVERY ===========

    @server.instance.routes.get("/imagelab/civitai/models")
    async def search_models(request):
        """
        Search CivitAI models.

        Query params:
            - query: Search term
            - types: Comma-separated model types (Checkpoint,LORA,etc)
            - sort: Highest Rated, Most Downloaded, Newest
            - period: AllTime, Year, Month, Week, Day
            - nsfw: true/false
            - limit: 1-100 (default 20)
            - page: Page number
            - baseModels: Comma-separated base models (SD 1.5, SDXL, etc)
            - tag: Filter by tag
        """
        try:
            client = civitai.get_client()

            # Parse query params
            query = request.query.get("query")
            types = request.query.get("types")
            sort = request.query.get("sort", "Most Downloaded")
            period = request.query.get("period", "AllTime")
            nsfw_str = request.query.get("nsfw")
            limit = int(request.query.get("limit", 20))
            page = int(request.query.get("page", 1))
            base_models = request.query.get("baseModels")
            tag = request.query.get("tag")

            # Parse comma-separated values
            types_list = types.split(",") if types else None
            base_models_list = base_models.split(",") if base_models else None
            nsfw = nsfw_str.lower() == "true" if nsfw_str else None

            result = await client.search_models(
                query=query,
                types=types_list,
                sort=sort,
                period=period,
                nsfw=nsfw,
                limit=limit,
                page=page,
                base_models=base_models_list,
                tag=tag,
            )

            return web.json_response({
                "success": True,
                "items": result.get("items", []),
                "metadata": result.get("metadata", {}),
            })

        except civitai.CivitAIAuthError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=401
            )
        except civitai.CivitAIRateLimitError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code, "retry_after": e.retry_after},
                status=429
            )
        except civitai.CivitAIError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=500
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.get("/imagelab/civitai/models/{model_id}")
    async def get_model_details(request):
        """Get detailed info for a specific model."""
        try:
            model_id = int(request.match_info["model_id"])
            client = civitai.get_client()

            result = await client.get_model(model_id)

            return web.json_response({
                "success": True,
                "model": result,
            })

        except civitai.CivitAINotFoundError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=404
            )
        except civitai.CivitAIAuthError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=401
            )
        except civitai.CivitAIError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=500
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.get("/imagelab/civitai/versions/{version_id}")
    async def get_version_details(request):
        """Get detailed info for a specific model version."""
        try:
            version_id = int(request.match_info["version_id"])
            client = civitai.get_client()

            result = await client.get_model_version(version_id)

            return web.json_response({
                "success": True,
                "version": result,
            })

        except civitai.CivitAINotFoundError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=404
            )
        except civitai.CivitAIAuthError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=401
            )
        except civitai.CivitAIError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=500
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    # =========== DOWNLOADS ===========

    @server.instance.routes.post("/imagelab/civitai/download")
    async def download_model(request):
        """
        Start a model download.

        Body:
            - version_id: CivitAI version ID (required)
            - model_type: Target type - Checkpoint, LORA, etc. (required)
            - filename: Optional custom filename
        """
        try:
            body = await request.json()

            version_id = body.get("version_id")
            model_type = body.get("model_type")
            filename = body.get("filename")

            if not version_id:
                return web.json_response(
                    {"success": False, "error": "version_id is required", "code": "VALIDATION_ERROR"},
                    status=400
                )

            if not model_type:
                return web.json_response(
                    {"success": False, "error": "model_type is required", "code": "VALIDATION_ERROR"},
                    status=400
                )

            if model_type not in civitai.MODEL_TYPE_TO_FOLDER:
                return web.json_response(
                    {"success": False, "error": f"Invalid model_type: {model_type}", "code": "VALIDATION_ERROR"},
                    status=400
                )

            client = civitai.get_client()

            # Start download as background task
            task = client.start_download_task(
                version_id=int(version_id),
                model_type=model_type,
                filename=filename,
            )

            return web.json_response({
                "success": True,
                "download_id": str(version_id),
                "status": "started",
                "message": f"Download started for version {version_id}",
            })

        except civitai.CivitAIAuthError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=401
            )
        except civitai.CivitAIError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=500
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.get("/imagelab/civitai/downloads")
    async def list_downloads(request):
        """List all active and recent downloads with progress."""
        try:
            client = civitai.get_client()
            downloads = client.get_all_downloads()

            items = []
            for progress in downloads:
                percent = 0
                if progress.total_bytes > 0:
                    percent = round(progress.downloaded_bytes / progress.total_bytes * 100, 1)

                items.append({
                    "version_id": progress.version_id,
                    "model_id": progress.model_id,
                    "filename": progress.filename,
                    "downloaded_bytes": progress.downloaded_bytes,
                    "total_bytes": progress.total_bytes,
                    "percent": percent,
                    "status": progress.status,
                    "error": progress.error,
                })

            return web.json_response({
                "success": True,
                "downloads": items,
            })

        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.delete("/imagelab/civitai/downloads/{version_id}")
    async def cancel_download(request):
        """Cancel an in-progress download."""
        try:
            version_id = int(request.match_info["version_id"])
            client = civitai.get_client()

            cancelled = await client.cancel_download(version_id)

            return web.json_response({
                "success": True,
                "cancelled": cancelled,
                "version_id": version_id,
            })

        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    # =========== LOCAL MODEL MANAGEMENT ===========

    @server.instance.routes.get("/imagelab/civitai/local")
    async def list_local_models(request):
        """
        List local models with CivitAI metadata.

        Query params:
            - type: Filter by model folder type (checkpoints, loras, etc.)
            - indexed_only: If 'true', only return models with CivitAI metadata
        """
        try:
            model_type = request.query.get("type")
            indexed_only = request.query.get("indexed_only", "false").lower() == "true"

            result = civitai.list_local_models_with_metadata(model_type)

            # Filter indexed only if requested
            if indexed_only:
                for folder_name, models in result.items():
                    result[folder_name] = [m for m in models if m.get("indexed")]

            return web.json_response({
                "success": True,
                "models": result,
            })

        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.delete("/imagelab/civitai/local/{model_type}/{filename:.*}")
    async def delete_model(request):
        """Delete a local model file."""
        try:
            model_type = request.match_info["model_type"]
            filename = request.match_info["filename"]

            civitai.delete_local_model(model_type, filename)

            return web.json_response({
                "success": True,
                "deleted": filename,
                "type": model_type,
            })

        except civitai.CivitAINotFoundError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=404
            )
        except civitai.CivitAIError as e:
            return web.json_response(
                {"success": False, "error": e.message, "code": e.code},
                status=500
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    # =========== STATS & CONFIG ===========

    @server.instance.routes.get("/imagelab/civitai/stats")
    async def get_stats(request):
        """
        Get CivitAI statistics.

        Returns:
            - local_model_counts: Count by type
            - indexed_count: Models with CivitAI metadata
            - total_size_bytes: Total disk usage
            - active_downloads: Current download count
        """
        try:
            stats = civitai.get_stats()

            return web.json_response({
                "success": True,
                "stats": stats,
            })

        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )

    @server.instance.routes.get("/imagelab/civitai/config")
    async def get_config(request):
        """
        Get CivitAI configuration status.

        Returns:
            - token_configured: Whether CIVITAI_API_KEY is set
            - model_directories: Available model directories
            - supported_types: Supported model types
        """
        try:
            config = civitai.get_config()

            return web.json_response({
                "success": True,
                "config": config,
            })

        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e), "code": "INTERNAL_ERROR"},
                status=500
            )
