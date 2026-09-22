from __future__ import annotations
import os
import sys
import json
import random
import struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import numpy as np
import torch
from io import BytesIO
from server import PromptServer, BinaryEventTypes


class ImageLabSaveImage:
    """
    Unified image output node supporting three save modes:
    - temporary: Save to temp folder (deleted on server shutdown)
    - disk: Save to output folder (permanent)
    - stream: Send via websocket (no server storage, client stores)
    
    All modes support embedding metadata (parametersRaw) in PNG.
    """
    
    SAVE_MODES = ["disk", "temporary", "stream"]
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "save_mode": (s.SAVE_MODES, {"default": "disk", "tooltip": "Where to save: disk (permanent), temporary (cleared on restart), stream (websocket only)"}),
                "filename_prefix": ("STRING", {"default": "ImageLab", "tooltip": "The prefix for the file to save."})
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "imagelab"
    DESCRIPTION = "Saves images with embedded metadata. Supports disk, temporary, and websocket streaming modes."

    def _create_metadata(self, prompt=None, extra_pnginfo=None):
        """Create PngInfo metadata object."""
        metadata = PngInfo()
        
        # Add ComfyUI workflow metadata
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        
        # Add ImageLab parameters and any other extra metadata
        if extra_pnginfo is not None:
            for key in extra_pnginfo:
                metadata.add_text(key, json.dumps(extra_pnginfo[key]))
        
        return metadata

    def _save_to_folder(self, images, output_dir, filename_prefix, file_type, metadata):
        """Save images to a folder (disk or temp)."""
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        results = []
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file), 
                pnginfo=metadata, 
                compress_level=self.compress_level
            )
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": file_type
            })
            counter += 1
        
        return results

    def _stream_to_websocket(self, images, metadata):
        """
        Stream images via websocket with embedded PNG metadata.
        
        Sends PNG bytes with metadata embedded via PIL's pnginfo.
        Uses PREVIEW_IMAGE event type but with pre-encoded PNG bytes.
        
        Note: The PNG bytes contain the full metadata, so the client can:
        1. Display the image immediately
        2. Save the PNG with all metadata preserved
        3. Extract imagelab_parameters for recall functionality
        """
        results = []
        server = PromptServer.instance
        
        for image in images:
            # Convert tensor to PIL Image
            array = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            # Encode PNG with embedded metadata
            buffer = BytesIO()
            # Add type header (2 = PNG) that ComfyUI expects
            buffer.write(struct.pack(">I", 2))  # PNG type
            img.save(buffer, format="PNG", pnginfo=metadata, compress_level=1)
            png_bytes = buffer.getvalue()
            
            # Send as PREVIEW_IMAGE (already encoded) using raw bytes
            # When send_sync receives bytes, it calls send_bytes directly
            server.send_sync(
                BinaryEventTypes.PREVIEW_IMAGE,
                png_bytes,
                server.client_id,
            )
            
            results.append({
                "source": "websocket",
                "content-type": "image/png",
                "type": "output",
                "has_metadata": True,
            })
        
        return results

    def save_images(self, images, save_mode="disk", filename_prefix="ImageLab", prompt=None, extra_pnginfo=None):
        """Main entry point - routes to appropriate save method."""
        metadata = self._create_metadata(prompt, extra_pnginfo)
        
        if save_mode == "disk":
            results = self._save_to_folder(
                images, self.output_dir, filename_prefix, "output", metadata
            )
        elif save_mode == "temporary":
            prefix = filename_prefix + "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
            results = self._save_to_folder(
                images, self.temp_dir, prefix, "temp", metadata
            )
        elif save_mode == "stream":
            results = self._stream_to_websocket(images, metadata)
        else:
            raise ValueError(f"Unknown save_mode: {save_mode}")
        
        return {"ui": {"images": results}}

class ColorFillImage:
    """
    A simple node that creates a solid color image of specified size.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "color_hex": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_image"
    CATEGORY = "imagelab"
    DESCRIPTION = "Creates a solid color image of specified size from a hex color."

    def create_image(self, width, height, color_hex):
        # Parse hex color (handle with or without #)
        hex_color = color_hex.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Create solid color image
        img = Image.new("RGB", (width, height), (r, g, b))

        # Convert to tensor format ComfyUI expects: [batch, height, width, channels], float32, 0-1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        return (img_tensor,)

import hashlib


def is_changed_file(filepath):
    """Check if file has changed using MD5 hash."""
    try:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if not hasattr(is_changed_file, "file_hashes"):
            is_changed_file.file_hashes = {}
        if filepath in is_changed_file.file_hashes:
            if is_changed_file.file_hashes[filepath] == file_hash:
                return False
        is_changed_file.file_hashes[filepath] = file_hash
        return float("NaN")  # Signals ComfyUI that the node needs re-execution
    except Exception as e:
        print(f"Error checking file {filepath}: {e}")
        return False

class WatchImageFile:
    """Watches a specific image file for changes and loads it when changed."""
    
    _watched_path = None  # Class variable to store path for IS_CHANGED
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "C:/path/to/your/image.png",
                    "multiline": False
                }),
                "wait_for_changes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "load_image"
    CATEGORY = "imagelab"

    def load_image(self, file_path, wait_for_changes):
        # Store path for IS_CHANGED to access
        WatchImageFile._watched_path = file_path
        
        if not os.path.exists(file_path):
            # Return a blank image if file doesn't exist
            print(f"File not found: {file_path}")
            blank = Image.new("RGB", (512, 512), (0, 0, 0))
            img_array = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(img_array)[None,], 512, 512)
        
        try:
            with open(file_path, "rb") as f:
                img_data = f.read()
            
            img = Image.open(BytesIO(img_data))
            img = img.convert("RGB")
            width, height = img.size
            
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            
            return (img_tensor, width, height)
        except Exception as e:
            print(f"Error loading image: {e}")
            blank = Image.new("RGB", (512, 512), (0, 0, 0))
            img_array = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(img_array)[None,], 512, 512)

    @classmethod
    def IS_CHANGED(cls, file_path, wait_for_changes):
        if not wait_for_changes:
            return float("NaN")  # Always re-execute if not waiting
        return is_changed_file(file_path)


NODE_CLASS_MAPPINGS = {
    "WatchImageFile": WatchImageFile,
    "ImageLabSaveImage": ImageLabSaveImage,
    "ColorFillImage": ColorFillImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatchImageFile": "Watch Image File",
    "ImageLabSaveImage": "ImageLab Save Image",
    "ColorFillImage": "Color Fill Image",
}