"""
Shared type definitions for workflows.
Keep all reusable parameter types here.
"""
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional

# =============================================================================
# Annotated Types for UI Schema Generation
# =============================================================================
# 
# Each type specifies:
#   - component: The frontend UI component to render (optional, for custom UX)
#   - category: Which tab/section to display the field in
#   - options_source: For selections, where to get the options from
#
# If no component is specified, frontend uses generic components based on type.

# Text fields
Prompt = Annotated[str, Field(
    json_schema_extra={"component": "prompt", "category": "prompts", "multiline": True, "rows": 4}
)]
NegativePrompt = Annotated[str, Field(
    json_schema_extra={"component": "negative_prompt", "category": "prompts", "multiline": True, "rows": 2}
)]

# Special types
Seed = Annotated[int, Field(
    json_schema_extra={"component": "seed", "category": "parameters"}
)]

# Input image (for img2img, controlnet, etc.)
# Value is the filename string (e.g., "uploaded_image.png")
InputImage = Annotated[Optional[str], Field(
    default=None,
    json_schema_extra={"component": "input_image", "category": "composition"}
)]

# Inpainting mask (white = regenerate, black = preserve)
# Value is the filename string (e.g., "mask.png")
MaskImage = Annotated[Optional[str], Field(
    default=None,
    json_schema_extra={"component": "mask", "category": "composition", "depends_on": {"inputImage": True}}
)]

# Enum types (selections from options_source)
Sampler = Annotated[str, Field(
    json_schema_extra={"component": "sampler", "options_source": "samplers", "category": "parameters"}
)]
Scheduler = Annotated[str, Field(
    json_schema_extra={"component": "scheduler", "options_source": "schedulers", "category": "parameters"}
)]

# Model selectors
Checkpoint = Annotated[List[str], Field(
    json_schema_extra={"component": "checkpoints", "options_source": "checkpoints", "category": "models"}
)]

# Loras - uses Lora class defined below
# The actual type is defined after Lora class, see Loras below

# =============================================================================
# Nested Parameter Models
# =============================================================================

class DimensionsParams(BaseModel):
    """Image dimensions parameters."""
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=8)

class LoopbackParams(BaseModel):
    """Loopback/Hires Fix parameters."""
    enabled: bool = False
    iterations: int = Field(default=1, ge=1, le=5)
    upscale: float = Field(default=1.25, ge=1.0, le=2.0)
    denoise: float = Field(default=0.5, ge=0.0, le=1.0)
    steps: int = Field(default=10, ge=1, le=50)
    cfg: float = Field(default=7.0, ge=1.0, le=20.0)

# class InputImageParams(BaseModel):
#     """Input Image parameters."""
#     use_image_size: bool = False
#     width: int = Field(default=1024, ge=64, le=2048, multiple_of=8)
#     height: int = Field(default=1024, ge=64, le=2048, multiple_of=8)

# Annotated types for nested objects
Dimensions = Annotated[DimensionsParams, Field(
    default_factory=DimensionsParams,
    json_schema_extra={"component": "dimensions", "category": "composition"}
)]

Loopback = Annotated[LoopbackParams, Field(
    default_factory=LoopbackParams,
    json_schema_extra={"component": "loopback", "category": "advanced"}
)]

# InputImage = Annotated[InputImageParams, Field(
#     default_factory=InputImageParams,
#     json_schema_extra={"component": "input_image", "category": "composition"}
# )]

# =============================================================================
# Models with Parameters (for model_with_params type)
# =============================================================================

# Legacy - kept for reference, use LoopbackParams instead
# class LoopbackParameters(BaseModel):
#     iterations: int = 1
#     upscale: float = 1.0
#     denoise: float = 0.5
#     steps: int = 10
#     cfg: float = 7.0

class LoraParameters(BaseModel):
    strength_model: float = 1.0
    strength_clip: float = 1.0

class Lora(BaseModel):
    apiName: str
    disabled: bool = False
    parameters: LoraParameters = LoraParameters()

class ControlnetParameters(BaseModel):
    strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0

class Controlnet(BaseModel):
    apiName: str
    disabled: bool = False
    parameters: ControlnetParameters = ControlnetParameters()

# =============================================================================
# Annotated List Types (for model_with_params with category)
# =============================================================================

# Loras - List of Lora models with parameters
Loras = Annotated[List[Lora], Field(
    default=[],
    json_schema_extra={"component": "loras", "options_source": "loras", "category": "models"}
)]

# Controlnets - List of Controlnet models with parameters
Controlnets = Annotated[List[Controlnet], Field(
    default=[],
    json_schema_extra={"component": "controlnets", "options_source": "controlnets", "category": "models"}
)]
