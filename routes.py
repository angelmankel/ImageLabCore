"""
API routes for external workflow execution.
"""

from aiohttp import web

from server import PromptServer

from . import resources
from .civitai_routes import SetupCivitAIRoutes
from .workflows import (
    get_workflow,
    get_workflow_metadata,
    get_workflow_parameters,
    invalidate_workflow_cache,
    list_workflows,
)


def get_comfy_script_client_id():
    """
    Get the comfy_script client_id lazily.
    This must be called after comfy_script's load() has been invoked,
    which happens when any workflow is first imported/executed.
    """
    try:
        import comfy_script.runtime as runtime

        return getattr(runtime, "_client_id", None)
    except ImportError:
        return None


def SetupRoutes(server: PromptServer):
    """Register API routes with the ComfyUI server."""

    @server.instance.routes.get("/imagelab/workflows")
    async def list_available_workflows(request):
        """
        List all available workflows with metadata.
        Query params:
            - refresh: If 'true', force refresh the workflow cache
        """
        refresh = request.query.get("refresh", "false").lower() == "true"
        return web.json_response(
            {"success": True, "workflows": list_workflows(refresh=refresh)}
        )

    @server.instance.routes.post("/imagelab/workflows/refresh")
    async def refresh_workflows(request):
        """Force refresh workflow cache and return updated list."""
        invalidate_workflow_cache()
        workflows = list_workflows(refresh=True)
        return web.json_response(
            {
                "success": True,
                "message": f"Refreshed {len(workflows)} workflows",
                "workflows": workflows,
            }
        )

    @server.instance.routes.get("/imagelab/workflow/{workflow_name}/metadata")
    async def get_workflow_info(request):
        """Get metadata for a specific workflow."""
        workflow_name = request.match_info["workflow_name"]
        try:
            metadata = get_workflow_metadata(workflow_name)
            return web.json_response(
                {"success": True, "workflow": workflow_name, "metadata": metadata}
            )
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=404)

    @server.instance.routes.get("/imagelab/workflow/{workflow_name}/parameters")
    async def get_workflow_params(request):
        """Get parameter schema for a specific workflow."""
        workflow_name = request.match_info["workflow_name"]
        try:
            parameters = get_workflow_parameters(workflow_name)

            if parameters is None:
                return web.json_response(
                    {
                        "success": True,
                        "workflow": workflow_name,
                        "parameters": [],
                        "message": "Workflow does not define parameter schema",
                    }
                )

            return web.json_response(
                {"success": True, "workflow": workflow_name, "parameters": parameters}
            )
        except ValueError as e:
            return web.json_response({"success": False, "error": str(e)}, status=404)

    @server.instance.routes.post("/imagelab/workflow/{workflow_name}")
    async def execute_workflow(request):
        """
        Execute a specific workflow by name.

        Query params:
            - client_id: Optional client ID for WebSocket connection.
                         Pass this to receive binary preview frames on your WebSocket.
        """
        workflow_name = request.match_info["workflow_name"]

        # Extract client_id from query params (used for WebSocket preview routing)
        client_id = request.query.get("client_id")

        try:
            # Get the workflow function (passing client_id to configure comfy_script)
            workflow_func = get_workflow(workflow_name, client_id=client_id)

            # Parse request body for parameters
            try:
                params = await request.json()
            except:
                params = {}

            # Execute workflow
            result = workflow_func(**params)

            return web.json_response(
                {"success": True, "workflow": workflow_name, "result": result}
            )

        except ValueError as e:
            return web.json_response(
                {"success": False, "type": "ValueError", "error": str(e)}, status=404
            )

        except Exception as e:
            return web.json_response(
                {
                    "success": False,
                    "type": "Exception",
                    "error": f"Workflow execution failed: {str(e)}",
                },
                status=500,
            )

    @server.instance.routes.get("/imagelab/resources")
    async def get_all_resources(request):
        """Get all available resources (models, samplers, etc.) - fast, no indexing"""
        try:
            models = resources.get_all_models()

            return web.json_response(
                {
                    "success": True,
                    "resources": {
                        **models,
                        "samplers": resources.get_samplers(),
                        "schedulers": resources.get_schedulers(),
                        "upscale_methods": resources.get_upscale_methods(),
                    },
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @server.instance.routes.get("/imagelab/resources/{resource_type}")
    async def get_resource_by_type(request):
        """Get resources of a specific type - fast, no indexing"""
        resource_type = request.match_info["resource_type"]

        # Map API keys to ComfyUI folder names
        folder_map = {
            "vaes": "vae",  # API uses plural, folder is singular
        }
        folder_name = folder_map.get(resource_type, resource_type)

        try:
            # Handle special resource types
            if resource_type == "samplers":
                result = resources.get_samplers()
            elif resource_type == "schedulers":
                result = resources.get_schedulers()
            elif resource_type == "upscale_methods":
                result = resources.get_upscale_methods()
            else:
                # Try to get as a model type
                result = resources.get_models_by_type(folder_name)

            return web.json_response(
                {"success": True, "resource_type": resource_type, "items": result}
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @server.instance.routes.post("/imagelab/resources/{resource_type}/index/{filename}")
    async def index_model_file(request):
        """Index a specific model file - calculate hash and fetch metadata"""
        resource_type = request.match_info["resource_type"]
        filename = request.match_info["filename"]

        try:
            metadata = resources.index_model_by_name(resource_type, filename)

            return web.json_response(
                {
                    "success": True,
                    "resource_type": resource_type,
                    "filename": filename,
                    "metadata": metadata,
                }
            )
        except FileNotFoundError as e:
            return web.json_response({"success": False, "error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @server.instance.routes.post("/imagelab/resources/index-all")
    async def index_all_models(request):
        """Index all models that don't have index files yet"""
        try:
            indexed_counts = resources.index_all_new_models()

            return web.json_response(
                {
                    "success": True,
                    "message": "Indexing complete",
                    "indexed": indexed_counts,
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @server.instance.routes.get("/imagelab/hello")
    async def get_hello(request):
        """Test endpoint."""
        return web.json_response({"message": "Hello from external API!"})

    @server.instance.routes.get("/imagelab/client-id")
    async def get_client_id(request):
        """
        Get the comfy_script client ID for WebSocket connections.
        This ID is used by ComfyUI to route execution events.
        Connect to /ws?clientId={client_id} to receive workflow events.

        Note: This endpoint requires that at least one workflow has been
        loaded (which initializes comfy_script's connection to ComfyUI).
        """
        client_id = get_comfy_script_client_id()
        if client_id is None:
            return web.json_response(
                {
                    "success": False,
                    "error": "comfy_script not initialized - run a workflow first",
                },
                status=503,
            )
        return web.json_response({"success": True, "client_id": client_id})

    @server.instance.routes.get("/imagelab/input-images")
    async def get_input_images(request):
        """List all images in the ComfyUI input folder."""
        import os

        import folder_paths

        try:
            input_dir = folder_paths.get_input_directory()
            images = []

            # Supported image extensions
            image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

            for filename in os.listdir(input_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    filepath = os.path.join(input_dir, filename)
                    stat = os.stat(filepath)
                    images.append(
                        {
                            "filename": filename,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )

            # Sort by modified time (newest first)
            images.sort(key=lambda x: x["modified"], reverse=True)

            return web.json_response({"success": True, "images": images})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # Register CivitAI routes
    SetupCivitAIRoutes(server)
