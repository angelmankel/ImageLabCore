"""
Basic SDXL Text to Image workflow.
Schema is defined in basic_sdxl.yaml (auto-generated from WorkflowParams).
"""
import os
from comfy_script.runtime import *
from comfy_script.runtime import _client_id

# ComfyUI server URL - configurable via environment variable
COMFYUI_SERVER_URL = os.environ.get('COMFYUI_SERVER_URL', 'http://127.0.0.1:8188/')
load(COMFYUI_SERVER_URL)

from comfy_script.runtime.nodes import *

from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base import BaseWorkflow
from parameters import (
    Prompt, NegativePrompt, Checkpoint, Seed, Sampler, Scheduler,
    Loras, Dimensions, Loopback, InputImage, MaskImage
)

# ============================================
# Workflow Metadata
# ============================================

WORKFLOW_METADATA = {
    "name": "Basic SDXL Text to Image",
    "description": "Workflow with SDXL presets for quick iteration.",
    "model_types": ["SDXL"],
    "category": "Image Generation",
}

# ============================================
# Workflow Parameters Model
# ============================================

class WorkflowParams(BaseModel):
    """
    Parameters for SDXL text-to-image generation.
    Schema is auto-generated to basic_sdxl.yaml from these type annotations.
    """
    # Use annotated types for automatic UI schema
    prompt: Prompt
    negativePrompt: NegativePrompt = "text, watermark"
    vae: str = Field(default="", json_schema_extra={"category": "models", "label": "VAE", "options_source": "vaes"})
    checkpoints: Checkpoint
    loras: Loras  # Uses shared type with category: models
    sampler: Sampler = "euler"
    scheduler: Scheduler = "normal"
    seed: Seed  # Required, no default
    
    # Nested objects using shared types
    dimensions: Dimensions
    loopback: Loopback
    
    # Inline Field() for one-off constraints
    steps: int = Field(default=20, ge=1, le=150)
    cfg: float = Field(default=8.0, ge=1.0, le=20.0, json_schema_extra={"step": 0.5})
    inputImage: InputImage = ""  # Optional input image for img2img
    mask: MaskImage = ""  # Optional inpainting mask (white = regenerate, black = preserve)
    inpaintMode: str = Field(default="noise_mask", json_schema_extra={
        "category": "composition",
        "label": "Inpaint Mode",
        "depends_on": {"mask": True},
        "options": [
            {"label": "Noise Mask", "apiName": "noise_mask", "description": "Preserves image data under the mask, blending new content with existing pixels"},
            {"label": "VAE Inpaint", "apiName": "vae_inpaint", "description": "Uses empty latent under the mask, generating completely new content in masked areas"}
        ]
    })
    inpaintFullRes: bool = Field(default=False, json_schema_extra={
        "category": "composition",
        "label": "Inpaint at Full Resolution",
        "depends_on": {"mask": True},
        "description": "Crops and upscales the masked area for higher detail inpainting, then composites back"
    })
    inpaintFullResSize: int = Field(default=0, ge=0, le=2048, json_schema_extra={
        "category": "composition",
        "label": "Full Res Size",
        "depends_on": {"inpaintFullRes": True},
        "description": "Target size for cropped region (0 = use input image dimensions)"
    })
    inpaintPadding: float = Field(default=1.5, ge=1.0, le=5.0, json_schema_extra={
        "step": 0.1,
        "category": "composition",
        "label": "Inpaint Padding",
        "depends_on": {"inpaintFullRes": True},
        "description": "Padding multiplier around the mask for context (higher = more context)"
    })
    inpaintFeather: int = Field(default=20, ge=0, le=100, json_schema_extra={
        "category": "composition",
        "label": "Inpaint Feather",
        "depends_on": {"inpaintFullRes": True},
        "description": "Feathering radius for blending the inpainted region back"
    })
    denoise: float = Field(default=1.0, ge=0.0, le=1.0, json_schema_extra={"step": 0.01, "category": "composition", "depends_on": {"inputImage": True}})
    upscale: float = Field(default=1.0, ge=-2.0, le=3.0, json_schema_extra={"step": 0.05, "category": "composition", "depends_on": {"inputImage": True}})

    # Output processing parameters
    upscaleModel: str = Field(default="", json_schema_extra={
        "category": "output",
        "label": "Upscale Model",
        "options_source": "upscale_models",
        "clearable": True,
        "description": "AI upscaler model for enhancing output resolution"
    })
    maxOutputSize: str = Field(default="0", json_schema_extra={
        "category": "output",
        "label": "Max Output Size",
        "description": "Maximum dimension for final output. Downscales if larger.",
        "options": [
            {"label": "No Limit", "apiName": "0"},
            {"label": "1024 px", "apiName": "1024"},
            {"label": "1536 px", "apiName": "1536"},
            {"label": "2048 px", "apiName": "2048"},
            {"label": "3072 px", "apiName": "3072"},
            {"label": "4096 px", "apiName": "4096"},
            {"label": "6144 px", "apiName": "6144"},
            {"label": "8192 px", "apiName": "8192"}
        ]
    })
    removeBackground: bool = Field(default=False, json_schema_extra={
        "category": "output",
        "label": "Remove Background",
        "description": "Remove background using BRIA RMBG AI model"
    })

# Workflow definition
class SDXLWorkflow(BaseWorkflow):
    def run(self):
        params: WorkflowParams = self.params

        with Workflow(wait=False) as wf:
            model, clip, vae = self.load_checkpoints()
            model, clip = self.apply_loras(model, clip)
            clip = CLIPSetLastLayer(clip, -2)
            # Use custom VAE if specified, otherwise use checkpoint's VAE
            if params.vae and params.vae != "":
                vae = VAELoader(params.vae)

            # Build conditioning early (needed for all paths)
            conditioning = CLIPTextEncode(params.prompt, clip)
            conditioning2 = CLIPTextEncode(params.negativePrompt, clip)

            # Track final output image
            output_image = None

            if params.inputImage != "" and params.inputImage is not None:
                input_image, _ = LoadImage(params.inputImage)
                input_image = ResizeImagesByLongerEdge(input_image, max(params.dimensions.width, params.dimensions.height))

                # Check for inpainting mask
                if params.mask and params.mask != "":
                    mask_image, _ = LoadImage(params.mask)
                    inpaint_mask = ImageToMask(mask_image, 'red')

                    # Full Resolution Inpainting using Impact Pack's MaskDetailer
                    if params.inpaintFullRes:
                        # Determine guide_size: 0 means use input image dimensions
                        guide_size = params.inpaintFullResSize
                        if guide_size == 0:
                            guide_size = max(params.dimensions.width, params.dimensions.height)

                        # Create BASIC_PIPE for MaskDetailer
                        basic_pipe = ToBasicPipe(model, clip, vae, conditioning, conditioning2)

                        # MaskDetailerPipe: crops masked region, upscales, inpaints, composites back
                        # Returns the composited image as first output
                        detailed_image, _, _, _, _ = MaskDetailerPipe(
                            image=input_image,
                            mask=inpaint_mask,
                            basic_pipe=basic_pipe,
                            guide_size=float(guide_size),
                            guide_size_for=True,  # Use mask bbox for sizing
                            max_size=float(guide_size * 1.5),  # Allow some headroom
                            mask_mode=True,  # Masked only (not whole image)
                            seed=params.seed,
                            steps=params.steps,
                            cfg=params.cfg,
                            sampler_name=params.sampler,
                            scheduler=params.scheduler,
                            denoise=params.denoise,
                            feather=params.inpaintFeather,
                            crop_factor=params.inpaintPadding,
                            drop_size=10,  # Min segment size
                            refiner_ratio=0.0,  # No refiner
                            batch_size=1,
                            cycle=1,
                            inpaint_model=(params.inpaintMode == "vae_inpaint"),
                            noise_mask_feather=params.inpaintFeather
                        )

                        # Apply upscale if specified
                        if params.upscale != 1.0:
                            detailed_image = ImageScaleBy(detailed_image, 'lanczos', params.upscale)

                        output_image = detailed_image

                    else:
                        # Standard inpainting path (non-full-res)
                        if params.inpaintMode == "vae_inpaint":
                            latent = VAEEncodeForInpaint(input_image, vae, inpaint_mask)
                        else:
                            latent = VAEEncode(input_image, vae)
                            latent = SetLatentNoiseMask(latent, inpaint_mask)

                        # Apply upscale if specified
                        if params.upscale != 1.0:
                            latent = LatentUpscaleBy(latent, 'nearest-exact', params.upscale)

                        # Sample and decode
                        latent = KSampler(model, params.seed, params.steps, params.cfg, params.sampler, params.scheduler, conditioning, conditioning2, latent, params.denoise)
                        output_image = VAEDecode(latent, vae)

                else:
                    # Img2img without mask
                    latent = VAEEncode(input_image, vae)

                    if params.upscale != 1.0:
                        latent = LatentUpscaleBy(latent, 'nearest-exact', params.upscale)

                    latent = KSampler(model, params.seed, params.steps, params.cfg, params.sampler, params.scheduler, conditioning, conditioning2, latent, params.denoise)

                    if params.loopback.enabled:
                        for _ in range(params.loopback.iterations):
                            latent = LatentUpscaleBy(latent, 'nearest-exact', params.loopback.upscale)
                            latent = KSampler(model, params.seed + 1, params.loopback.steps, params.loopback.cfg, params.sampler, params.scheduler, conditioning, conditioning2, latent, params.loopback.denoise)

                    output_image = VAEDecode(latent, vae)

            else:
                # Text-to-image (no input image)
                latent = EmptyLatentImage(params.dimensions.width, params.dimensions.height, 1)
                latent = KSampler(model, params.seed, params.steps, params.cfg, params.sampler, params.scheduler, conditioning, conditioning2, latent, params.denoise)

                if params.loopback.enabled:
                    for _ in range(params.loopback.iterations):
                        latent = LatentUpscaleBy(latent, 'nearest-exact', params.loopback.upscale)
                        latent = KSampler(model, params.seed + 1, params.loopback.steps, params.loopback.cfg, params.sampler, params.scheduler, conditioning, conditioning2, latent, params.loopback.denoise)

                output_image = VAEDecode(latent, vae)

            # =============================================
            # Output Processing Pipeline
            # =============================================

            # 1. Upscale with AI model if specified
            if params.upscaleModel and params.upscaleModel != "":
                upscale_model = UpscaleModelLoader(params.upscaleModel)
                output_image = ImageUpscaleWithModel(upscale_model, output_image)

            # 2. Cap output size if maxOutputSize is set (scales down if larger)
            max_size = int(params.maxOutputSize) if params.maxOutputSize else 0
            if max_size > 0:
                output_image = ResizeImagesByLongerEdge(output_image, max_size)

            # 3. Remove background if enabled
            if params.removeBackground:
                rmbg_model = BRIARMBGModelLoaderZho()
                output_image, _ = BRIARMBGZho(rmbg_model, output_image)

            # Save outputs using configured save mode (disk/temporary/stream)
            self.save_outputs(output_image)

        return {
            'prompt_id': wf.task.prompt_id,
            'client_id': _client_id,
            'status': 'queued',
            'parameters': self.params.model_dump(mode='json')
        }


# Module-level entry point
def run_workflow(**kwargs):
    """
    Entry point for the workflow.

    Special kwargs (not passed to WorkflowParams):
        save_mode: "disk" | "temporary" | "stream" - how to save output images
        params_raw: dict - original parameters from frontend (for PNG metadata/recall)
    """
    # Extract special parameters before creating WorkflowParams
    save_mode = kwargs.pop('save_mode', 'disk')
    params_raw = kwargs.pop('params_raw', None)

    required = ['prompt', 'checkpoints', 'seed', 'steps', 'negativePrompt', 'dimensions', 'cfg']
    missing = [f for f in required if f not in kwargs]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    params = WorkflowParams(**kwargs)

    # Create and run workflow with save settings
    workflow = SDXLWorkflow(params, params_raw=params_raw, save_mode=save_mode)
    return workflow.run()
