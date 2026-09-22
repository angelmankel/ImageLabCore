"""
Parameter Schema for ImageLab Workflows
========================================

This module defines the parameter schema system for workflows.
Each parameter is a dictionary that describes:
- What data it holds (type, name)
- How it should be rendered (component)
- Where it belongs (category)
- Validation rules (min, max, required, etc.)

Usage in workflow files:
------------------------
from ..parameter_schema import param, ParameterType, ParameterCategory, ComponentType

WORKFLOW_PARAMETERS = [
    param.text_area(
        name="prompt",
        label="Prompt",
        category=ParameterCategory.PROMPTS,
        required=True,
        placeholder="Enter your prompt...",
    ),
    param.slider(
        name="steps",
        label="Steps",
        category=ParameterCategory.PARAMETERS,
        min_val=1,
        max_val=150,
        default=20,
    ),
    param.model_select(
        name="checkpoint",
        label="Checkpoint",
        category=ParameterCategory.MODELS,
        model_type="checkpoints",
        required=True,
    ),
]
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum


# ============================================
# Enums for Type Safety
# ============================================

class ParameterType(str, Enum):
    """Data types for parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"  # For lists of items (e.g., multiple loras)


class ComponentType(str, Enum):
    """
    UI component types for rendering parameters.
    Maps to frontend React components.
    """
    # Text inputs
    TEXT_INPUT = "text_input"          # Single line text
    TEXT_AREA = "text_area"            # Multi-line text (prompts)
    
    # Numeric inputs
    SLIDER = "slider"                  # Range slider with number
    NUMBER_INPUT = "number_input"      # Direct number input
    SEED_INPUT = "seed_input"          # Seed with randomize button
    
    # Selection inputs
    SELECT = "select"                  # Dropdown with fixed options
    MODEL_SELECT = "model_select"      # Model/resource selector
    MULTI_SELECT = "multi_select"      # Multiple selection dropdown
    
    # Toggle input (use style="switch" or style="checkbox")
    TOGGLE = "toggle"                  # Boolean toggle


class ParameterCategory(str, Enum):
    """
    Categories for grouping parameters in the UI.
    Maps to frontend tabs/sections.
    """
    PROMPTS = "prompts"           # Prompt, negative prompt, helpers
    MODELS = "models"             # Checkpoints, LoRAs, VAE
    PARAMETERS = "parameters"     # Steps, CFG, sampler, scheduler, seed
    COMPOSITION = "composition"   # Dimensions, input images
    ADVANCED = "advanced"         # Advanced/experimental options
    OUTPUT = "output"             # Post processing, filtering, etc.


# ============================================
# Parameter Schema Definition
# ============================================

class ParameterSchema:
    """
    Base schema for all parameters.
    Contains common fields and validation.
    """
    
    @staticmethod
    def _base(
        name: str,
        param_type: ParameterType,
        component: ComponentType,
        category: ParameterCategory,
        label: str,
        description: str = "",
        required: bool = False,
        default: Any = None,
        hidden: bool = False,
        disabled: bool = False,
        depends_on: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create base parameter structure.
        
        Args:
            name: Unique identifier (used as API field name)
            param_type: Data type of the parameter
            component: UI component to render
            category: Which tab/section this belongs to
            label: Display label in UI
            description: Help text / tooltip
            required: Whether the field is required
            default: Default value
            hidden: Hide from UI (for internal params)
            disabled: Show but disable interaction
            depends_on: Conditional visibility based on other params
                        e.g., {"loopback": True} shows only when loopback is enabled
        """
        schema = {
            "name": name,
            "type": param_type.value,
            "component": component.value,
            "category": category.value,
            "label": label,
            "required": required,
        }
        
        if description:
            schema["description"] = description
        if default is not None:
            schema["default"] = default
        if hidden:
            schema["hidden"] = hidden
        if disabled:
            schema["disabled"] = disabled
        if depends_on:
            schema["depends_on"] = depends_on
            
        return schema

    # ============================================
    # Text Components
    # ============================================
    
    @staticmethod
    def text_input(
        name: str,
        label: str,
        category: ParameterCategory,
        description: str = "",
        required: bool = False,
        default: str = "",
        placeholder: str = "",
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Single line text input."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.STRING,
            component=ComponentType.TEXT_INPUT,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default,
            **kwargs
        )
        if placeholder:
            schema["placeholder"] = placeholder
        if max_length:
            schema["max_length"] = max_length
        return schema
    
    @staticmethod
    def text_area(
        name: str,
        label: str,
        category: ParameterCategory,
        description: str = "",
        required: bool = False,
        default: str = "",
        placeholder: str = "",
        rows: int = 4,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Multi-line text area (for prompts)."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.STRING,
            component=ComponentType.TEXT_AREA,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default,
            **kwargs
        )
        if placeholder:
            schema["placeholder"] = placeholder
        schema["rows"] = rows
        if max_length:
            schema["max_length"] = max_length
        return schema

    # ============================================
    # Numeric Components
    # ============================================
    
    @staticmethod
    def slider(
        name: str,
        label: str,
        category: ParameterCategory,
        min_val: Union[int, float],
        max_val: Union[int, float],
        default: Union[int, float],
        step: Union[int, float] = 1,
        description: str = "",
        required: bool = False,
        is_float: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Slider with numeric input."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.FLOAT if is_float else ParameterType.INTEGER,
            component=ComponentType.SLIDER,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default,
            **kwargs
        )
        schema["min"] = min_val
        schema["max"] = max_val
        schema["step"] = step
        return schema
    
    @staticmethod
    def number_input(
        name: str,
        label: str,
        category: ParameterCategory,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        default: Union[int, float] = 0,
        step: Union[int, float] = 1,
        description: str = "",
        required: bool = False,
        is_float: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Direct number input field."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.FLOAT if is_float else ParameterType.INTEGER,
            component=ComponentType.NUMBER_INPUT,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default,
            **kwargs
        )
        if min_val is not None:
            schema["min"] = min_val
        if max_val is not None:
            schema["max"] = max_val
        schema["step"] = step
        return schema
    
    @staticmethod
    def seed_input(
        name: str = "seed",
        label: str = "Seed",
        category: ParameterCategory = ParameterCategory.PARAMETERS,
        default: int = -1,
        description: str = "Random seed (-1 for random)",
        **kwargs
    ) -> Dict[str, Any]:
        """Seed input with randomize button."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.INTEGER,
            component=ComponentType.SEED_INPUT,
            category=category,
            label=label,
            description=description,
            default=default,
            **kwargs
        )
        schema["min"] = -1
        schema["max"] = 2147483647  # Max int32
        return schema

    # ============================================
    # Selection Components
    # ============================================
    
    @staticmethod
    def select(
        name: str,
        label: str,
        category: ParameterCategory,
        options: List[Dict[str, str]],  # [{"value": "x", "label": "X"}, ...]
        default: str = "",
        description: str = "",
        required: bool = False,
        searchable: bool = False,
        clearable: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Dropdown select with fixed options."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.STRING,
            component=ComponentType.SELECT,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default,
            **kwargs
        )
        schema["options"] = options
        schema["searchable"] = searchable
        schema["clearable"] = clearable
        return schema
    
    @staticmethod
    def multi_select(
        name: str,
        label: str,
        category: ParameterCategory,
        options: List[Dict[str, str]],
        default: List[str] = None,
        description: str = "",
        required: bool = False,
        searchable: bool = True,
        max_selections: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Multi-select dropdown."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.ARRAY,
            component=ComponentType.MULTI_SELECT,
            category=category,
            label=label,
            description=description,
            required=required,
            default=default or [],
            **kwargs
        )
        schema["options"] = options
        schema["searchable"] = searchable
        if max_selections:
            schema["max_selections"] = max_selections
        return schema
    
    @staticmethod
    def model_select(
        name: str,
        label: str,
        category: ParameterCategory,
        model_type: str,  # "checkpoints", "loras", "vae", "controlnet", etc.
        default: str = "",
        description: str = "",
        required: bool = False,
        multiple: bool = False,
        max_selections: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Model/resource selector that loads options from API."""
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.ARRAY if multiple else ParameterType.STRING,
            component=ComponentType.MODEL_SELECT,
            category=category,
            label=label,
            description=description,
            required=required,
            default=[] if multiple else default,
            **kwargs
        )
        schema["model_type"] = model_type
        schema["multiple"] = multiple
        if max_selections and multiple:
            schema["max_selections"] = max_selections
        return schema

    # ============================================
    # Toggle Component
    # ============================================
    
    @staticmethod
    def toggle(
        name: str,
        label: str,
        category: ParameterCategory,
        default: bool = False,
        description: str = "",
        style: str = "switch",  # "switch" or "checkbox"
        **kwargs
    ) -> Dict[str, Any]:
        """
        Boolean toggle component.
        
        Args:
            style: Visual style - "switch" for toggle switch, "checkbox" for checkbox
        """
        schema = ParameterSchema._base(
            name=name,
            param_type=ParameterType.BOOLEAN,
            component=ComponentType.TOGGLE,
            category=category,
            label=label,
            description=description,
            default=default,
            **kwargs
        )
        schema["style"] = style
        return schema


# ============================================
# Convenience Alias
# ============================================

# Use `param.text_area(...)` instead of `ParameterSchema.text_area(...)`
param = ParameterSchema


# ============================================
# Validation Helpers
# ============================================

def validate_parameter_schema(schema: Dict[str, Any]) -> List[str]:
    """
    Validate a parameter schema and return list of errors.
    Returns empty list if valid.
    """
    errors = []
    
    required_fields = ["name", "type", "component", "category", "label"]
    for field in required_fields:
        if field not in schema:
            errors.append(f"Missing required field: {field}")
    
    if "type" in schema:
        valid_types = [t.value for t in ParameterType]
        if schema["type"] not in valid_types:
            errors.append(f"Invalid type '{schema['type']}'. Must be one of: {valid_types}")
    
    if "component" in schema:
        valid_components = [c.value for c in ComponentType]
        if schema["component"] not in valid_components:
            errors.append(f"Invalid component '{schema['component']}'. Must be one of: {valid_components}")
    
    if "category" in schema:
        valid_categories = [c.value for c in ParameterCategory]
        if schema["category"] not in valid_categories:
            errors.append(f"Invalid category '{schema['category']}'. Must be one of: {valid_categories}")
    
    return errors


def validate_workflow_parameters(parameters: List[Dict[str, Any]]) -> List[str]:
    """Validate a list of parameter schemas."""
    errors = []
    names_seen = set()
    
    for i, param in enumerate(parameters):
        param_errors = validate_parameter_schema(param)
        for error in param_errors:
            errors.append(f"Parameter {i}: {error}")
        
        if "name" in param:
            if param["name"] in names_seen:
                errors.append(f"Parameter {i}: Duplicate name '{param['name']}'")
            names_seen.add(param["name"])
    
    return errors
