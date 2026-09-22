"""
Base workflow class with common functionality.
All workflows should inherit from this class.
"""
from comfy_script.runtime import *
from comfy_script.runtime.nodes import *
from typing import Literal

# Type alias for save modes
SaveMode = Literal["disk", "temporary", "stream"]


class BaseWorkflow:
    """Base class for ComfyScript workflows with common utilities."""
    
    def __init__(self, params, params_raw: dict | None = None, save_mode: SaveMode = "disk"):
        self.params = params
        self.params_raw = params_raw  # Original parameters from frontend (for metadata)
        self.save_mode = save_mode
    
    def load_checkpoints(self):
        """
        Load and optionally merge checkpoints.
        Automatically merges if multiple checkpoints are provided.
        
        Returns:
            tuple: (model, clip, vae)
        """
        checkpoints = self.params.checkpoints
        
        if not checkpoints:
            raise ValueError("At least one checkpoint is required")
        
        if len(checkpoints) == 1:
            # Single checkpoint - load directly
            self.model, self.clip, self.vae = CheckpointLoaderSimple(checkpoints[0])
            # self.clip = CLIPSetLastLayer(self.clip, -2)
        else:
            # Multiple checkpoints - merge them
            self.model, self.clip, self.vae = self._merge_checkpoints(checkpoints)
        
        return self.model, self.clip, self.vae
    
    def _merge_checkpoints(self, checkpoints):
        # Load first checkpoint
        model, clip, vae = CheckpointLoaderSimple(checkpoints[0])
        
        # Merge with each additional checkpoint
        merge_ratio = 1.0 / len(checkpoints)
        
        for i in range(1, len(checkpoints)):
            model_next, clip_next, _ = CheckpointLoaderSimple(checkpoints[i])
            model = ModelMergeSimple(model, model_next, merge_ratio)
            clip = CLIPMergeSimple(clip, clip_next, merge_ratio)
        
        return model, clip, vae
    
    def apply_loras(self, model=None, clip=None):      
        for lora in self.params.loras:
            if not lora.disabled:
                model, clip = LoraLoader(
                    model, clip, lora.apiName,
                    lora.parameters.strength_model,
                    lora.parameters.strength_clip
                )
        
        return model, clip
    
    def run(self):
        """
        Main execution method. Override this in subclasses.
        
        Returns:
            dict: Result with prompt_id, status, and output data
        """
        raise NotImplementedError("Subclasses must implement run() method")
    
    def save_outputs(self, image, filename_prefix: str = "ImageLab"):
        """
        Save output images using the configured save mode.
        
        Uses ImageLabSaveImage node which supports:
        - disk: Permanent storage in output folder
        - temporary: Temp folder (cleared on server restart)  
        - stream: WebSocket streaming to client (no server storage)
        
        Automatically embeds params_raw as PNG metadata for parameter recall.
        
        Args:
            image: The image tensor to save
            filename_prefix: Prefix for saved files (only used for disk/temporary modes)
        """
        # Build extra_pnginfo with imagelab parameters for recall
        extra_pnginfo = {}
        if self.params_raw is not None:
            extra_pnginfo["imagelab_parameters"] = self.params_raw
        
        # Use our custom ImageLabSaveImage node
        ImageLabSaveImage(
            image,
            save_mode=self.save_mode,
            filename_prefix=filename_prefix,
            extra_pnginfo=extra_pnginfo if extra_pnginfo else None
        )
