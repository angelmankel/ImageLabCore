# YAML Workflow Schema Implementation Plan

## Overview

Move workflow parameter schemas from Python code to YAML files for better maintainability and potential client-side editing. Auto-generate YAML from `WorkflowParams` Pydantic models with smart merging to preserve customizations.

## File Structure

```
external-api/
  __main__.py              # CLI entry point
  cli.py                   # CLI commands
  schema_generator.py      # Inference + YAML generation
  workflows.py             # Load YAML, validate, execute
  workflows/
    basic_sdxl.py          # WorkflowParams + execution logic
    basic_sdxl.yaml        # Generated/customized schema
```

## YAML Format (Flat)

```yaml
name: Basic SDXL Text to Image
description: Workflow with SDXL presets for quick iteration.
category: Image Generation
model_types:
  - SDXL

parameters:
  - name: prompt
    label: Prompt
    type: text
    category: prompts
    required: true
    multiline: true
    rows: 4
    placeholder: Enter your prompt...

  - name: negativePrompt
    label: Negative Prompt
    type: text
    category: prompts
    multiline: true
    rows: 2
    default: "text, watermark"

  - name: checkpoints
    label: Checkpoint
    type: model
    category: models
    options_source: checkpoints
    required: true
    multiple: true

  - name: loras
    label: LoRAs
    type: model_with_params
    category: models
    options_source: loras
    default: []
    multiple: true
    params:
      - name: strength_model
        type: float
        default: 1.0
      - name: strength_clip
        type: float
        default: 1.0

  - name: steps
    label: Steps
    type: int
    category: parameters
    min: 1
    max: 150
    default: 20

  - name: cfg
    label: CFG Scale
    type: float
    category: parameters
    min: 1.0
    max: 20.0
    step: 0.5
    default: 8.0

  - name: sampler
    label: Sampler
    type: enum
    category: parameters
    options_source: samplers
    default: euler
    searchable: true

  - name: scheduler
    label: Scheduler
    type: enum
    category: parameters
    options_source: schedulers
    default: normal

  - name: seed
    label: Seed
    type: seed
    category: parameters
    required: true

  - name: width
    label: Width
    type: int
    category: composition
    min: 64
    max: 2048
    step: 8
    default: 1024

  - name: height
    label: Height
    type: int
    category: composition
    min: 64
    max: 2048
    step: 8
    default: 1024

  - name: loopback
    label: Enable Loopback
    type: bool
    category: advanced
    default: false
    description: Enable iterative upscaling and refinement

  - name: loopbackIterations
    label: Loopback Iterations
    type: int
    category: advanced
    min: 1
    max: 5
    default: 1
    depends_on:
      loopback: true
```

## CLI Commands

```bash
# Generate YAML for a workflow
python -m external-api generate basic_sdxl

# Generate for all workflows
python -m external-api generate --all

# Preview without writing
python -m external-api generate basic_sdxl --dry-run

# Validate schema matches code
python -m external-api validate basic_sdxl

# List all workflows
python -m external-api list

# Watch mode (requires: pip install watchdog)
python -m external-api watch
```

## Inference Rules

### Annotation-Based Inference

UI schema metadata is defined using `Annotated` types with Pydantic's `Field()` and `json_schema_extra`. This provides explicit, type-safe field definitions.

**Reusable types in `parameters.py`:**

```python
from typing import Annotated, List
from pydantic import BaseModel, Field

# Text fields
Prompt = Annotated[str, Field(
    json_schema_extra={"ui_type": "text", "category": "prompts", "multiline": True, "rows": 4}
)]
NegativePrompt = Annotated[str, Field(
    json_schema_extra={"ui_type": "text", "category": "prompts", "multiline": True, "rows": 2}
)]

# Special types
Seed = Annotated[int, Field(
    json_schema_extra={"ui_type": "seed", "category": "parameters"}
)]

# Enum types
Sampler = Annotated[str, Field(
    json_schema_extra={"ui_type": "enum", "options_source": "samplers", "category": "parameters"}
)]
Scheduler = Annotated[str, Field(
    json_schema_extra={"ui_type": "enum", "options_source": "schedulers", "category": "parameters"}
)]

# Model selectors
Checkpoint = Annotated[List[str], Field(
    json_schema_extra={"ui_type": "model", "options_source": "checkpoints", "category": "models"}
)]

# Dimension fields (uses Pydantic's built-in constraints)
Width = Annotated[int, Field(
    ge=64, le=2048, multiple_of=8,
    json_schema_extra={"category": "composition"}
)]
Height = Annotated[int, Field(
    ge=64, le=2048, multiple_of=8,
    json_schema_extra={"category": "composition"}
)]
```

**Pydantic constraints map to YAML:**

| Pydantic Field | YAML Property    |
| -------------- | ---------------- |
| `ge` (>=)      | `min`            |
| `le` (<=)      | `max`            |
| `multiple_of`  | `step`           |
| `default`      | `default`        |
| No default     | `required: true` |

**Fallback for plain types:**

| Python Type | Inferred YAML Type |
| ----------- | ------------------ |
| `bool`      | `bool`             |
| `int`       | `int`              |
| `float`     | `float`            |
| `str`       | `text`             |
| `List[*]`   | `multiple: true`   |

Default category: `parameters`

### Models with Parameters

For types like `Lora`, `Controlnet`, `IPAdapter` that have embedded parameters, we infer the structure directly from the Pydantic class in `parameters.py`:

```python
# parameters.py
class LoraParameters(BaseModel):
    strength_model: float = 1.0
    strength_clip: float = 1.0

class Lora(BaseModel):
    apiName: str
    disabled: bool = False
    parameters: LoraParameters = LoraParameters()
```

When the generator sees `loras: List[Lora]` in WorkflowParams, it:

1. Detects `Lora` is a Pydantic model from `parameters.py`
2. Inspects `Lora.parameters` to find `LoraParameters`
3. Extracts parameter fields and their defaults
4. Generates YAML with `params` embedded:

```yaml
- name: loras
  label: LoRAs
  type: model_with_params
  category: models
  options_source: loras
  multiple: true
  params:
    - name: strength_model
      type: float
      default: 1.0
    - name: strength_clip
      type: float
      default: 1.0
```

| Pydantic Type     | Inferred `options_source` | Params Source          |
| ----------------- | ------------------------- | ---------------------- |
| `List[Lora]`      | `loras`                   | `LoraParameters`       |
| `Controlnet`      | `controlnets`             | `ControlnetParameters` |
| `List[IPAdapter]` | `ipadapters`              | `IPAdapterParameters`  |

This keeps the source of truth in Python (`parameters.py`) while generating user-friendly YAML schemas.

## Merge Strategy

When merging generated schema with existing YAML:

**Preserve** (user customizations):

- `label`
- `description`
- `placeholder`
- `rows`
- `searchable`
- `options`
- `depends_on`
- `max_selections`

**Update** (from Pydantic model):

- `type`
- `default`
- `required`
- `min`, `max`, `step`
- `category`

**Add**: New fields from WorkflowParams

**Remove**: Fields no longer in WorkflowParams

## Validation

Errors on:

- **Workflow load**: If YAML has fields not in `WorkflowParams`
- **generate-schema**: If validation fails
- **API request**: If parameters don't match schema

## Auto-generation

- Manual CLI command: `python -m external-api generate <workflow_id>`
- Auto-generate on workflow load if YAML missing
- Watch mode available with `watchdog` library

## Dependencies

- `pyyaml` (required)
- `watchdog` (optional, for watch mode)

## YAML Property Reference

Properties are sourced from three places:

- **auto (code)** - Derived from Python code structure (field name, type, default value)
- **auto (annotation)** - From `json_schema_extra` in `Annotated` types or `Field()`
- **manual** - Not auto-generated; user adds to YAML after generation (preserved on regenerate)

### All Fields

| Property      | Source            | Description                                                     |
| ------------- | ----------------- | --------------------------------------------------------------- |
| `name`        | auto (code)       | Field name from WorkflowParams                                  |
| `label`       | auto (code)       | Generated from field name, customizable                         |
| `type`        | auto (code)       | **Data type:** int, float, str, bool, object                    |
| `component`   | auto (annotation) | **UI component:** seed, prompt, model, select, etc. (optional)  |
| `category`    | auto (annotation) | UI grouping: prompts, models, parameters, composition, advanced |
| `required`    | auto (code)       | True if field has no default value                              |
| `default`     | auto (code)       | Default value from Python                                       |
| `multiple`    | auto (code)       | True if Python type is `List[*]`                                |
| `description` | manual            | Help text shown in UI                                           |
| `depends_on`  | manual            | Conditional visibility based on other fields                    |

### type: str (text fields)

| Property      | Source            | Description                        |
| ------------- | ----------------- | ---------------------------------- |
| `multiline`   | auto (annotation) | True for textarea, false for input |
| `rows`        | auto (annotation) | Number of rows for textarea        |
| `placeholder` | manual            | Placeholder text                   |

### type: int / float

| Property | Source            | Description                         |
| -------- | ----------------- | ----------------------------------- |
| `min`    | auto (annotation) | Minimum value (from `ge`)           |
| `max`    | auto (annotation) | Maximum value (from `le`)           |
| `step`   | auto (annotation) | Step increment (from `multiple_of`) |

### component: select / model / model_with_params

| Property         | Source            | Description                                                  |
| ---------------- | ----------------- | ------------------------------------------------------------ |
| `options_source` | auto (annotation) | Resource key: samplers, schedulers, checkpoints, loras, etc. |
| `searchable`     | manual            | Show search filter in dropdown                               |
| `max_selections` | manual            | Limit selections (omit for unlimited)                        |

### component: model_with_params

| Property | Source      | Description                             |
| -------- | ----------- | --------------------------------------- |
| `params` | auto (code) | Inferred from Pydantic parameters class |

## Future Parameter Types

Types to be added as needed:

| Type    | Use Case                                     |
| ------- | -------------------------------------------- |
| `image` | Input images for img2img, inpainting, etc.   |
| `mask`  | Mask images for inpainting, region selection |
| `color` | Color pickers for tinting, background, etc.  |

## Simplified Workflow File

```python
"""
Basic SDXL Text to Image workflow.
Schema is defined in basic_sdxl.yaml (auto-generated from WorkflowParams).
"""
from pydantic import BaseModel, Field
from typing import List
from parameters import (
    Prompt, NegativePrompt, Checkpoint, Seed, Sampler, Scheduler,
    Width, Height, Lora
)

WORKFLOW_METADATA = {
    "name": "Basic SDXL Text to Image",
    "description": "Workflow with SDXL presets for quick iteration.",
    "model_types": ["SDXL"],
    "category": "Image Generation",
}

class WorkflowParams(BaseModel):
    # Use annotated types for automatic UI schema
    prompt: Prompt
    negativePrompt: NegativePrompt = "text, watermark"
    checkpoints: Checkpoint
    loras: List[Lora] = []  # Generator infers params from Lora class
    sampler: Sampler = "euler"
    scheduler: Scheduler = "normal"
    seed: Seed  # Required, no default
    width: Width = 1024
    height: Height = 1024

    # Inline Field() for one-off constraints
    steps: int = Field(default=20, ge=1, le=150)
    cfg: float = Field(default=8.0, ge=1.0, le=20.0, json_schema_extra={"step": 0.5})

    # Plain types fall back to basic inference
    loopback: bool = False
    loopbackIterations: int = Field(default=1, ge=1, le=5, json_schema_extra={"category": "advanced"})
    loopbackUpscale: float = 1.25
    loopbackDenoise: float = 0.5
    loopbackSteps: int = 10
    loopbackCfg: float = 7.0

def run_workflow(**kwargs):
    params = WorkflowParams(**kwargs)
    # ... execution logic
```

## Module Responsibilities

| Module                | Responsibilities                                                               |
| --------------------- | ------------------------------------------------------------------------------ |
| `schema_generator.py` | Load WorkflowParams, infer schema, merge with existing YAML, write YAML        |
| `cli.py`              | CLI commands: generate, validate, list, watch                                  |
| `__main__.py`         | Entry point for `python -m external-api`                                       |
| `workflows.py`        | Load YAML, resolve enums from resources.py, execute workflows, backward compat |

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Update `parameters.py`**
   - Add `Annotated` type definitions (Prompt, NegativePrompt, Seed, Sampler, Scheduler, Checkpoint, Width, Height)
   - Ensure Lora, Controlnet classes have proper `parameters` field with defaults

2. **Create `schema_generator.py`**
   - Function to extract metadata from `Annotated` types and `Field()`
   - Function to infer `options_source` and `params` from model types (Lora, Controlnet, etc.)
   - Function to generate label from field name (camelCase → Title Case)
   - Function to merge generated schema with existing YAML (preserve/update logic)
   - Function to write YAML file

3. **Create `cli.py`**
   - `generate` command (single workflow or `--all`)
   - `--dry-run` flag for preview
   - `validate` command
   - `list` command

4. **Create `__main__.py`**
   - Entry point for `python -m external-api`

### Phase 2: Integration

5. **Update `workflows.py`**
   - Load schema from YAML instead of `WORKFLOW_PARAMETERS`
   - Auto-generate YAML if missing on workflow load
   - Resolve `options_source` references from resources
   - Backward compatibility: support old format during transition

6. **Update existing workflows**
   - Convert `WorkflowParams` to use `Annotated` types
   - Remove `WORKFLOW_PARAMETERS` dict
   - Generate YAML files for each workflow

### Phase 3: Testing & Polish

7. **Validation**
   - Test generate → regenerate preserves customizations
   - Test all field types render correctly in frontend
   - Test `depends_on` conditional visibility
   - Test `options_source` resolution

8. **Documentation**
   - Update workflow creation guide
   - Document available `Annotated` types
   - Document YAML customization options

### Phase 4: Optional Enhancements

9. **Watch mode** (optional)
   - Install `watchdog` dependency
   - Auto-regenerate YAML when `.py` files change

10. **Future types**
    - Add `image`, `mask`, `color` types as needed

### Phase 5: Type/Component Separation

Separate data type (`type`) from UI component (`component`) for cleaner architecture.

**Problem:** Currently `type` conflates data structure with UI rendering:

- `type: seed` - Is it an int? How should it render?
- `type: model` - Is it a string? A list?

**Solution:** Split into two properties:

- `type` - The semantic data type (int, float, str, bool, enum, object)
- `component` - The frontend component to render (seed, model, prompt, select, etc.)

**Key Insight:** If a field has `options_source`, it's a selection from a list, not free text:

- `options_source` present → `type: enum` (even if Python type is `str`)
- This tells frontend to render a select/dropdown, not a text input

11. **Update `parameters.py` annotations**
    - Change `ui_type` to `component` in `json_schema_extra`
    - Keep `options_source` for resource-based selections

    ```python
    # Seed - int with special UI
    Seed = Annotated[int, Field(
        json_schema_extra={"component": "seed", "category": "parameters"}
    )]
    # Generated: type: int, component: seed

    # Sampler - selection from list (NOT free text)
    Sampler = Annotated[str, Field(
        json_schema_extra={"component": "sampler", "options_source": "samplers", "category": "parameters"}
    )]
    # Generated: type: enum, component: sampler, options_source: samplers

    # Checkpoints - multiple selection from list
    Checkpoint = Annotated[List[str], Field(
        json_schema_extra={"component": "checkpoints", "options_source": "checkpoints", "category": "models"}
    )]
    # Generated: type: enum, multiple: true, component: checkpoints, options_source: checkpoints
    ```

12. **Update `schema_generator.py` type inference**

    ```python
    def infer_type(python_type, field_info, json_extra) -> str:
        """Infer YAML type from Python type and metadata."""

        # If has options_source, it's a selection (enum), not free text
        if json_extra.get("options_source"):
            return "enum"

        # If has params, it's a complex object
        if json_extra.get("params") or has_parameters_class(python_type):
            return "object"

        # Otherwise, infer from Python type
        origin = get_origin(python_type)
        if origin is list:
            inner = get_args(python_type)[0]
            return infer_type(inner, field_info, json_extra)

        type_map = {
            int: "int",
            float: "float",
            str: "str",
            bool: "bool",
        }
        return type_map.get(python_type, "str")
    ```

13. **Update YAML generation**
    - `type` = semantic data type (enum for selections, int/float/str/bool for primitives, object for complex)
    - `component` = explicit UI component (optional)
    - `multiple` = true if Python type is `List[*]`

14. **Regenerate all YAML files**
    - Run `python -m external-api generate --all`

**Updated YAML Format:**

```yaml
parameters:
  # Seed - int with custom component
  - name: seed
    type: int # Primitive data type
    component: seed # Custom UI component
    category: parameters
    required: true
    label: Seed

  # Sampler - enum (selection from options_source)
  - name: sampler
    type: enum # Selection type (has options_source)
    component: sampler # Custom UI component (optional)
    options_source: samplers # Where to get options
    category: parameters
    default: euler
    searchable: true
    label: Sampler

  # Scheduler - enum (selection from options_source)
  - name: scheduler
    type: enum # Selection type
    component: scheduler # Custom UI component (optional)
    options_source: schedulers
    category: parameters
    default: normal
    label: Scheduler

  # Checkpoints - enum array (multiple selection)
  - name: checkpoints
    type: enum # Selection type
    multiple: true # List[str] → multiple selections
    component: checkpoints # Custom UI component
    options_source: checkpoints
    category: models
    required: true
    label: Checkpoints

  # LoRAs - object array (complex with params)
  - name: loras
    type: object # Complex type (has params)
    multiple: true
    component: loras # Custom UI component
    options_source: loras
    category: models
    label: LoRAs
    params:
      - name: strength_model
        type: float
        default: 1.0
      - name: strength_clip
        type: float
        default: 1.0

  # Prompt - str with custom component
  - name: prompt
    type: str # Free text (no options_source)
    component: prompt # Custom UI component
    category: prompts
    required: true
    multiline: true
    rows: 4
    label: Prompt

  # Steps - int with constraints (NO custom component)
  - name: steps
    type: int
    category: parameters
    min: 1
    max: 150
    default: 20
    label: Steps
    # No component = frontend uses generic SliderField (int with min/max)

  # Loopback - bool (NO custom component)
  - name: loopback
    type: bool
    category: advanced
    default: false
    label: Enable Loopback
    # No component = frontend uses generic ToggleField
```

**Component Resolution (Frontend):**

| Priority | Check                     | Example                                     |
| -------- | ------------------------- | ------------------------------------------- |
| 1        | `component` property      | `component: "seed"` → SeedField             |
| 2        | `options_source` property | `options_source: "samplers"` → SamplerField |
| 3        | `type` + constraints      | `type: "int"` + min/max → SliderField       |
| 4        | `type` only               | `type: "bool"` → ToggleField                |

**Type Inference Rules (Backend):**

| Condition                        | Inferred `type`              |
| -------------------------------- | ---------------------------- |
| Has `options_source`             | `enum`                       |
| Has `params` (model_with_params) | `object`                     |
| Python `int`                     | `int`                        |
| Python `float`                   | `float`                      |
| Python `str` (no options_source) | `str`                        |
| Python `bool`                    | `bool`                       |
| Python `List[T]`                 | type of T + `multiple: true` |

**Files to modify:**

- `parameters.py` - Change `ui_type` → `component`
- `schema_generator.py` - Update type inference (options_source → enum)
- All workflow YAML files - Regenerate
