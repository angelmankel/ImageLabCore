"""
CivitAI model management module.
Provides search, download, and management of CivitAI models.
"""
import os
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Literal, Callable
from dataclasses import dataclass, field
from enum import Enum

import folder_paths

from . import resources
from .websocket import broadcast_progress

# Constants - configurable via environment variables for proxying
CIVITAI_API_BASE = os.environ.get('CIVITAI_API_BASE', "https://civitai.com/api/v1")
CIVITAI_DOWNLOAD_BASE = os.environ.get('CIVITAI_DOWNLOAD_BASE', "https://civitai.com/api/download/models")

# Model type to ComfyUI folder mapping
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
    "Poses": "poses",
    "AestheticGradient": "embeddings",
    "MotionModule": "animatediff_motion_lora",
}

# Reverse mapping for API
FOLDER_TO_MODEL_TYPE = {v: k for k, v in MODEL_TYPE_TO_FOLDER.items()}


class CivitAIError(Exception):
    """Base exception for CivitAI operations."""
    def __init__(self, message: str, code: str = "CIVITAI_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class CivitAIAuthError(CivitAIError):
    """Authentication error."""
    def __init__(self, message: str = "API token required or invalid"):
        super().__init__(message, "AUTH_ERROR")


class CivitAINotFoundError(CivitAIError):
    """Resource not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, "NOT_FOUND")


class CivitAIRateLimitError(CivitAIError):
    """Rate limited."""
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limited. Retry after {retry_after}s", "RATE_LIMITED")
        self.retry_after = retry_after


class CivitAIDownloadError(CivitAIError):
    """Download failed."""
    def __init__(self, message: str, version_id: int):
        super().__init__(message, "DOWNLOAD_FAILED")
        self.version_id = version_id


@dataclass
class DownloadProgress:
    """Download progress information."""
    model_id: int
    version_id: int
    filename: str
    downloaded_bytes: int
    total_bytes: int
    status: Literal["downloading", "completed", "failed", "cancelled"]
    error: Optional[str] = None
    speed_bps: float = 0.0


class CivitAIClient:
    """Async client for CivitAI API operations."""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("CIVITAI_API_KEY")
        self._active_downloads: Dict[int, asyncio.Task] = {}
        self._download_progress: Dict[int, DownloadProgress] = {}
        self._cancelled_downloads: set = set()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional auth."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        if response.status == 200:
            return await response.json()
        elif response.status == 401:
            raise CivitAIAuthError("Invalid or missing API token")
        elif response.status == 403:
            raise CivitAIAuthError("Access forbidden - check API token permissions")
        elif response.status == 404:
            raise CivitAINotFoundError()
        elif response.status == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise CivitAIRateLimitError(retry_after)
        else:
            text = await response.text()
            raise CivitAIError(f"API error {response.status}: {text}")

    async def search_models(
        self,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        nsfw: Optional[bool] = None,
        limit: int = 20,
        page: int = 1,
        base_models: Optional[List[str]] = None,
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search CivitAI models.

        Args:
            query: Search term
            types: Model types (Checkpoint, LORA, etc.)
            sort: Sort order (Highest Rated, Most Downloaded, Newest)
            period: Time period (AllTime, Year, Month, Week, Day)
            nsfw: Filter NSFW content (None = user preference)
            limit: Results per page (max 100)
            page: Page number
            base_models: Base model filters (SD 1.5, SDXL 1.0, etc.)
            tag: Filter by tag

        Returns:
            Dict with items and metadata
        """
        params = {
            "limit": min(limit, 100),
            "page": page,
            "sort": sort,
            "period": period,
        }

        if query:
            params["query"] = query
        if types:
            params["types"] = ",".join(types)
        if nsfw is not None:
            params["nsfw"] = str(nsfw).lower()
        if base_models:
            params["baseModels"] = ",".join(base_models)
        if tag:
            params["tag"] = tag

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/models",
                params=params,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return await self._handle_response(response)

    async def get_model(self, model_id: int) -> Dict[str, Any]:
        """Get model details by ID."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/models/{model_id}",
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return await self._handle_response(response)

    async def get_model_version(self, version_id: int) -> Dict[str, Any]:
        """Get model version details."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/model-versions/{version_id}",
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                return await self._handle_response(response)

    async def download_model(
        self,
        version_id: int,
        model_type: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> str:
        """
        Download a model version to the appropriate folder.

        Args:
            version_id: CivitAI version ID
            model_type: Target model type (Checkpoint, LORA, etc.)
            filename: Optional custom filename
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded file
        """
        # Get target directory
        folder_name = MODEL_TYPE_TO_FOLDER.get(model_type)
        if not folder_name:
            raise CivitAIError(f"Unknown model type: {model_type}")

        if folder_name not in folder_paths.folder_names_and_paths:
            raise CivitAIError(f"Folder not configured: {folder_name}")

        paths = folder_paths.folder_names_and_paths[folder_name]
        if not paths or not paths[0]:
            raise CivitAIError(f"No path configured for: {folder_name}")

        target_dir = paths[0][0] if isinstance(paths[0], list) else paths[0]

        # Get version details for filename
        version = await self.get_model_version(version_id)

        if not version.get("files"):
            raise CivitAIError("No files available for this version")

        # Find primary file (prefer safetensors)
        primary_file = None
        for f in version["files"]:
            if f.get("primary"):
                primary_file = f
                break
            if f.get("name", "").endswith(".safetensors") and primary_file is None:
                primary_file = f

        if not primary_file:
            primary_file = version["files"][0]

        if not filename:
            filename = primary_file["name"]

        # Ensure .safetensors extension
        if not filename.endswith((".safetensors", ".pt", ".ckpt", ".bin")):
            filename += ".safetensors"

        target_path = os.path.join(target_dir, filename)
        temp_path = target_path + ".downloading"

        # Build download URL
        download_url = primary_file.get("downloadUrl") or f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
        if self.api_token:
            separator = "&" if "?" in download_url else "?"
            download_url += f"{separator}token={self.api_token}"

        # Initialize progress
        total_bytes = primary_file.get("sizeKB", 0) * 1024
        progress = DownloadProgress(
            model_id=version.get("modelId", 0),
            version_id=version_id,
            filename=filename,
            downloaded_bytes=0,
            total_bytes=total_bytes,
            status="downloading"
        )
        self._download_progress[version_id] = progress
        self._cancelled_downloads.discard(version_id)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    download_url,
                    timeout=aiohttp.ClientTimeout(total=None, connect=30),
                    allow_redirects=True
                ) as response:
                    if response.status == 401:
                        raise CivitAIAuthError("Download requires authentication")
                    if response.status != 200:
                        raise CivitAIDownloadError(
                            f"Download failed: HTTP {response.status}",
                            version_id
                        )

                    # Update total from Content-Length header
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        progress.total_bytes = int(content_length)

                    # Create temp file
                    os.makedirs(target_dir, exist_ok=True)

                    last_broadcast = 0
                    with open(temp_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(65536):
                            # Check for cancellation
                            if version_id in self._cancelled_downloads:
                                progress.status = "cancelled"
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                await self._broadcast_download_event(progress)
                                return None

                            f.write(chunk)
                            progress.downloaded_bytes += len(chunk)

                            # Broadcast progress every 500KB
                            if progress.downloaded_bytes - last_broadcast >= 524288:
                                last_broadcast = progress.downloaded_bytes
                                await self._broadcast_download_event(progress)

                                if progress_callback:
                                    progress_callback(progress)

            # Move completed download
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)
            progress.status = "completed"

            # Auto-index the downloaded model with CivitAI metadata
            print(f"[CivitAI] Indexing downloaded model: {filename}")
            index_data = resources.index_model(target_path, filename)

            # Store the version metadata directly if we have it
            if not index_data.get("civitai") and version:
                index_data["civitai"] = version
                resources.save_index(filename, index_data)

            # Broadcast completion
            await self._broadcast_download_event(progress, index_data)

            return target_path

        except asyncio.CancelledError:
            progress.status = "cancelled"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await self._broadcast_download_event(progress)
            raise
        except Exception as e:
            progress.status = "failed"
            progress.error = str(e)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await self._broadcast_download_event(progress)
            raise
        finally:
            # Clean up
            self._download_progress.pop(version_id, None)
            self._active_downloads.pop(version_id, None)

    async def _broadcast_download_event(
        self,
        progress: DownloadProgress,
        index_data: Optional[Dict] = None
    ):
        """Broadcast download progress/completion to WebSocket clients."""
        percent = 0
        if progress.total_bytes > 0:
            percent = round(progress.downloaded_bytes / progress.total_bytes * 100, 1)

        event_type = f"civitai:download:{progress.status}"
        if progress.status == "downloading":
            event_type = "civitai:download:progress"

        data = {
            "type": event_type,
            "data": {
                "model_id": progress.model_id,
                "version_id": progress.version_id,
                "filename": progress.filename,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "percent": percent,
                "status": progress.status,
                "error": progress.error
            }
        }

        # Include metadata on completion
        if progress.status == "completed" and index_data:
            data["data"]["civitai"] = index_data.get("civitai")
            data["data"]["hash"] = index_data.get("hash")

        await broadcast_progress(data)

    async def cancel_download(self, version_id: int) -> bool:
        """Cancel an in-progress download."""
        self._cancelled_downloads.add(version_id)

        task = self._active_downloads.get(version_id)
        if task and not task.done():
            task.cancel()
            return True

        return version_id in self._download_progress

    def get_download_progress(self, version_id: int) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self._download_progress.get(version_id)

    def get_all_downloads(self) -> List[DownloadProgress]:
        """Get all active downloads."""
        return list(self._download_progress.values())

    def start_download_task(
        self,
        version_id: int,
        model_type: str,
        filename: Optional[str] = None,
    ) -> asyncio.Task:
        """Start a download as a background task."""
        task = asyncio.create_task(
            self.download_model(version_id, model_type, filename)
        )
        self._active_downloads[version_id] = task
        return task


# Singleton instance
_client: Optional[CivitAIClient] = None


def get_client() -> CivitAIClient:
    """Get or create CivitAI client instance."""
    global _client
    if _client is None:
        _client = CivitAIClient()
    return _client


def get_model_directory(model_type: str) -> Optional[str]:
    """Get the ComfyUI folder path for a model type."""
    folder_name = MODEL_TYPE_TO_FOLDER.get(model_type)
    if folder_name and folder_name in folder_paths.folder_names_and_paths:
        paths = folder_paths.folder_names_and_paths[folder_name]
        if paths and paths[0]:
            return paths[0][0] if isinstance(paths[0], list) else paths[0]
    return None


def list_local_models_with_metadata(model_type: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    List local models with their CivitAI metadata from index files.

    Args:
        model_type: Optional filter by model type

    Returns:
        Dict of model type -> list of models with metadata
    """
    result = {}

    types_to_check = [model_type] if model_type else list(MODEL_TYPE_TO_FOLDER.values())

    for folder_name in types_to_check:
        if folder_name not in folder_paths.folder_names_and_paths:
            continue

        models = []
        files = folder_paths.get_filename_list(folder_name)

        for filename in files:
            filepath = folder_paths.get_full_path(folder_name, filename)
            if not filepath:
                continue

            # Get file info
            try:
                stat = os.stat(filepath)
                size_bytes = stat.st_size
            except:
                size_bytes = 0

            # Get index data if available
            index = resources.get_index(filename)

            models.append({
                "filename": filename,
                "type": folder_name,
                "size_bytes": size_bytes,
                "indexed": index.get("hash") is not None,
                "hash": index.get("hash"),
                "civitai": index.get("civitai"),
            })

        if models:
            result[folder_name] = models

    return result


def delete_local_model(model_type: str, filename: str) -> bool:
    """
    Delete a local model file and its index.

    Args:
        model_type: The model type/folder name
        filename: The filename to delete

    Returns:
        True if deleted successfully
    """
    if model_type not in folder_paths.folder_names_and_paths:
        raise CivitAIError(f"Invalid model type: {model_type}")

    filepath = folder_paths.get_full_path(model_type, filename)
    if not filepath or not os.path.exists(filepath):
        raise CivitAINotFoundError(f"Model not found: {filename}")

    # Delete the model file
    os.remove(filepath)

    # Delete the index file if it exists
    index_path = os.path.join(resources.RESOURCES_DIR, f"{filename}.index")
    if os.path.exists(index_path):
        os.remove(index_path)

    return True


def get_stats() -> Dict[str, Any]:
    """Get CivitAI statistics."""
    client = get_client()

    stats = {
        "local_model_counts": {},
        "indexed_count": 0,
        "total_count": 0,
        "total_size_bytes": 0,
        "active_downloads": len(client.get_all_downloads()),
    }

    for folder_name in MODEL_TYPE_TO_FOLDER.values():
        if folder_name not in folder_paths.folder_names_and_paths:
            continue

        files = folder_paths.get_filename_list(folder_name)
        count = len(files)

        if count > 0:
            stats["local_model_counts"][folder_name] = count
            stats["total_count"] += count

            for filename in files:
                filepath = folder_paths.get_full_path(folder_name, filename)
                if filepath:
                    try:
                        stats["total_size_bytes"] += os.path.getsize(filepath)
                    except:
                        pass

                index_path = os.path.join(resources.RESOURCES_DIR, f"{filename}.index")
                if os.path.exists(index_path):
                    stats["indexed_count"] += 1

    return stats


def get_config() -> Dict[str, Any]:
    """Get CivitAI configuration status."""
    token = os.environ.get("CIVITAI_API_KEY")

    directories = {}
    for civitai_type, folder_name in MODEL_TYPE_TO_FOLDER.items():
        if folder_name in folder_paths.folder_names_and_paths:
            paths = folder_paths.folder_names_and_paths[folder_name]
            if paths and paths[0]:
                path = paths[0][0] if isinstance(paths[0], list) else paths[0]
                directories[civitai_type] = path

    return {
        "token_configured": bool(token),
        "token_preview": f"{token[:8]}..." if token and len(token) > 8 else None,
        "model_directories": directories,
        "supported_types": list(MODEL_TYPE_TO_FOLDER.keys()),
    }
