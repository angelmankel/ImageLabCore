"""
ImageLab Progress Broadcasting Module

This module patches ComfyUI's WebSocket send_sync method to broadcast
progress updates, workflow events, and preview images to ALL connected
WebSocket clients, not just the client that submitted the prompt.

Events broadcasted:
- IMAGELAB_WORKFLOW_EVENT (102): Workflow lifecycle events (executing, executed, etc.)
- IMAGELAB_PROGRESS_EVENT (100): Sampler step progress updates
- IMAGELAB_PREVIEW_EVENT (101): Preview images with metadata

Binary format for IMAGELAB_PREVIEW_EVENT:
    [4 bytes: metadata length][JSON metadata][image bytes]
"""

import os
import sys
import json
import struct
from io import BytesIO
from typing import Any, Optional

# Ensure ComfyUI root is in path for imports
_comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _comfyui_root not in sys.path:
    sys.path.insert(0, _comfyui_root)

from PIL import Image, ImageOps
from server import PromptServer
from protocol import BinaryEventTypes


# =============================================================================
# Constants
# =============================================================================

# Custom binary event types for ImageLab broadcasts
IMAGELAB_PROGRESS_EVENT = 100
IMAGELAB_PREVIEW_EVENT = 101
IMAGELAB_WORKFLOW_EVENT = 102

# Events that trigger workflow broadcasts
WORKFLOW_EVENTS = frozenset({
    'executing',
    'execution_start', 
    'executed',
    'execution_error',
    'execution_cached'
})


# =============================================================================
# Internal State
# =============================================================================

_original_send_sync = PromptServer.instance.send_sync
_is_broadcasting = False  # Prevents infinite recursion during broadcasts


# =============================================================================
# Helper Functions
# =============================================================================

def _get_resampling_filter():
    """Get the appropriate PIL resampling filter for the installed version."""
    if hasattr(Image, 'Resampling'):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def _resize_image_if_needed(pil_image: Image.Image, max_size: Optional[int]) -> Image.Image:
    """Resize image to fit within max_size while maintaining aspect ratio."""
    if max_size is None:
        return pil_image
    resampling = _get_resampling_filter()
    return ImageOps.contain(pil_image, (max_size, max_size), resampling)


def _encode_image(pil_image: Image.Image, img_format: str) -> bytes:
    """Encode a PIL image to bytes in the specified format."""
    buffer = BytesIO()
    save_kwargs = {'format': img_format}
    
    if img_format == 'JPEG':
        save_kwargs['quality'] = 80
    elif img_format == 'PNG':
        save_kwargs['compress_level'] = 1
        
    pil_image.save(buffer, **save_kwargs)
    return buffer.getvalue()


def _pack_image_with_metadata(image_bytes: bytes, metadata: dict) -> bytes:
    """Pack image bytes with JSON metadata for transmission.
    
    Format: [4 bytes: metadata length (big-endian)][JSON metadata][image bytes]
    """
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    return struct.pack(">I", len(metadata_bytes)) + metadata_bytes + image_bytes


def _broadcast(event_type: int, data: bytes) -> None:
    """Broadcast binary data to all connected WebSocket clients."""
    _original_send_sync(event_type, data, sid=None)


def _broadcast_json(event_type: int, data: dict) -> None:
    """Broadcast JSON data to all connected WebSocket clients."""
    _broadcast(event_type, json.dumps(data).encode('utf-8'))


# =============================================================================
# Event Handlers
# =============================================================================

def _handle_workflow_event(event: str, data: Any) -> None:
    """Broadcast workflow lifecycle events to all clients."""
    _broadcast_json(IMAGELAB_WORKFLOW_EVENT, {
        'event': event,
        'data': data
    })


def _handle_progress_event(data: dict) -> None:
    """Broadcast sampler progress updates to all clients."""
    value = data.get('value', 0)
    max_val = data.get('max', 1)
    
    progress_data = {
        'event': 'sampler_progress',
        'node_id': data.get('node'),
        'prompt_id': data.get('prompt_id'),
        'value': value,
        'max': max_val,
        'progress_percent': round((value / max_val * 100), 2) if max_val > 0 else 0
    }
    _broadcast_json(IMAGELAB_PROGRESS_EVENT, progress_data)


def _handle_unencoded_preview(data: tuple) -> None:
    """Broadcast unencoded preview images to all clients.
    
    Args:
        data: Tuple of (format, PIL.Image, max_size)
    """
    img_format = data[0]
    pil_image = data[1]
    max_size = data[2] if len(data) > 2 else None
    
    # Resize and encode image
    pil_image = _resize_image_if_needed(pil_image, max_size)
    image_bytes = _encode_image(pil_image, img_format)
    
    # Get context from server instance
    metadata = {
        'node_id': getattr(PromptServer.instance, 'last_node_id', None),
        'prompt_id': getattr(PromptServer.instance, 'last_prompt_id', None)
    }
    
    combined = _pack_image_with_metadata(image_bytes, metadata)
    _broadcast(IMAGELAB_PREVIEW_EVENT, combined)


def _handle_preview_with_metadata(data: tuple) -> None:
    """Broadcast preview images with metadata to all clients.
    
    Args:
        data: Tuple of ((format, PIL.Image, max_size), metadata_dict)
    """
    image_data, metadata = data
    img_format = image_data[0]
    pil_image = image_data[1]
    max_size = image_data[2] if len(image_data) > 2 else None
    
    # Resize and encode image
    pil_image = _resize_image_if_needed(pil_image, max_size)
    image_bytes = _encode_image(pil_image, img_format)
    
    combined = _pack_image_with_metadata(image_bytes, metadata)
    _broadcast(IMAGELAB_PREVIEW_EVENT, combined)


# =============================================================================
# Main Patch Function
# =============================================================================

def patched_send_sync(event: Any, data: Any, sid: Optional[str] = None) -> None:
    """Patched send_sync that broadcasts events to all connected clients.
    
    This function wraps the original PromptServer.send_sync method to intercept
    and broadcast progress updates, workflow events, and preview images to ALL
    connected WebSocket clients, regardless of which client submitted the prompt.
    
    Args:
        event: The event type (string for JSON events, int for binary events)
        data: The event data
        sid: Optional session ID to send to specific client (passed to original)
    """
    global _is_broadcasting
    
    # Always forward to original handler first
    _original_send_sync(event, data, sid)
    
    # Prevent infinite recursion during our broadcasts
    if _is_broadcasting:
        return
    
    try:
        _is_broadcasting = True
        
        # Route to appropriate handler based on event type
        if event in WORKFLOW_EVENTS:
            _handle_workflow_event(event, data)
            
        elif event == 'progress' and isinstance(data, dict):
            _handle_progress_event(data)
            
        elif event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            _handle_unencoded_preview(data)
            
        elif event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            _handle_preview_with_metadata(data)
            
    except Exception as e:
        print(f"[ImageLab] Broadcast error: {e}")
        
    finally:
        _is_broadcasting = False


# =============================================================================
# Public API
# =============================================================================

def add_progress_hook() -> None:
    """Apply the send_sync patch to broadcast progress to all clients.
    
    This should be called once during module initialization.
    """
    PromptServer.instance.send_sync = patched_send_sync
