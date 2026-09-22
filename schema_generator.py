"""
Schema Generator for Workflow YAML files.

Generates YAML schemas from WorkflowParams Pydantic models with smart merging
to preserve user customizations.
"""
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, get_args, get_origin
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Properties to preserve during merge (user customizations)
PRESERVE_PROPERTIES = {
    "label",
    "description", 
    "placeholder",
    "rows",
    "searchable",
    "options",
    "depends_on",
    "max_selections",
}

# Properties to update from Pydantic model
UPDATE_PROPERTIES = {
    "type",
    "component",  # NEW: explicit UI component
    "default",
    "required",
    "min",
    "max", 
    "step",
    "category",
    "options_source",
    "multiple",
    "params",
    "multiline",
}


def serialize_default(value: Any) -> Any:
    """Convert a default value to a JSON/YAML-serializable format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [serialize_default(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_default(v) for k, v in value.items()}
    if isinstance(value, BaseModel):
        return value.model_dump()
    # Fallback: convert to string
    return str(value)


def camel_to_title(name: str) -> str:
    """Convert camelCase to Title Case.
    
    Examples:
        negativePrompt -> Negative Prompt
        loopbackIterations -> Loopback Iterations
        cfg -> Cfg
    """
    # Insert space before uppercase letters
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Capitalize first letter of each word
    return result.title()


def get_python_type_name(python_type: type) -> str:
    """Get the basic type name from a Python type."""
    origin = get_origin(python_type)
    
    if origin is list or origin is List:
        return "list"
    
    if python_type is bool:
        return "bool"
    elif python_type is int:
        return "int"
    elif python_type is float:
        return "float"
    elif python_type is str:
        return "str"
    
    return str(python_type)


def is_model_with_params(field_type: type) -> bool:
    """Check if a type is a Pydantic model with a 'parameters' field."""
    # Unwrap List[X] to get X
    origin = get_origin(field_type)
    if origin is list or origin is List:
        args = get_args(field_type)
        if args:
            field_type = args[0]
    
    # Check if it's a Pydantic model with 'parameters' field
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        if "parameters" in field_type.model_fields:
            return True
    
    return False


def get_model_params_schema(field_type: type) -> List[Dict[str, Any]]:
    """Extract parameter schema from a model's parameters class."""
    # Unwrap List[X] to get X
    origin = get_origin(field_type)
    if origin is list or origin is List:
        args = get_args(field_type)
        if args:
            field_type = args[0]
    
    if not isinstance(field_type, type) or not issubclass(field_type, BaseModel):
        return []
    
    params_field = field_type.model_fields.get("parameters")
    if not params_field:
        return []
    
    # Get the parameters class
    params_type = params_field.annotation
    if not isinstance(params_type, type) or not issubclass(params_type, BaseModel):
        return []
    
    # Extract fields from parameters class
    params = []
    for name, field_info in params_type.model_fields.items():
        param = {"name": name}
        
        # Get type
        field_annotation = field_info.annotation
        if field_annotation is float:
            param["type"] = "float"
        elif field_annotation is int:
            param["type"] = "int"
        elif field_annotation is bool:
            param["type"] = "bool"
        elif field_annotation is str:
            param["type"] = "str"
        
        # Get default
        if field_info.default is not None:
            param["default"] = field_info.default
        
        params.append(param)
    
    return params


def infer_options_source(field_name: str, field_type: type) -> Optional[str]:
    """Infer options_source from model type name."""
    # Unwrap List[X] to get X
    origin = get_origin(field_type)
    if origin is list or origin is List:
        args = get_args(field_type)
        if args:
            field_type = args[0]
    
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Convert class name to plural lowercase
        # Lora -> loras, Controlnet -> controlnets
        type_name = field_type.__name__.lower()
        if not type_name.endswith('s'):
            type_name += 's'
        return type_name
    
    return None


def extract_field_schema(
    field_name: str,
    field_info: FieldInfo,
    field_type: type
) -> Dict[str, Any]:
    """Extract schema from a Pydantic field.
    
    Type inference rules:
    1. If has options_source -> type: enum (selection from list)
    2. If has params (model_with_params) -> type: object
    3. Otherwise infer from Python type (int, float, str, bool)
    
    Component is optional - frontend falls back to generic rendering based on type.
    """
    schema: Dict[str, Any] = {
        "name": field_name,
        "label": camel_to_title(field_name),
    }
    
    # Check for json_schema_extra first (from Annotated types or Field())
    extra = {}
    if field_info.json_schema_extra:
        if callable(field_info.json_schema_extra):
            # Handle callable json_schema_extra
            extra = {}
        else:
            extra = dict(field_info.json_schema_extra)
    
    # Extract component (NEW: explicit UI component key)
    component = extra.pop("component", None)
    
    # Extract options_source and inline options early - needed for type inference
    options_source = extra.get("options_source")
    inline_options = extra.get("options")  # inline options array

    # Handle List types
    origin = get_origin(field_type)
    inner_type = field_type
    if origin is list or origin is List:
        schema["multiple"] = True
        args = get_args(field_type)
        if args:
            inner_type = args[0]
    
    # Infer type based on priority:
    # 1. model_with_params (has params)
    # 2. enum (has options_source)
    # 3. primitive type (int, float, str, bool)
    
    if is_model_with_params(field_type):
        # Complex type with embedded parameters
        schema["type"] = "object"
        inferred_source = infer_options_source(field_name, field_type)
        if not options_source:
            schema["options_source"] = inferred_source
        # For model_with_params, use options_source as component if not specified
        if not component:
            component = inferred_source or field_name
        schema["params"] = get_model_params_schema(field_type)
    elif options_source or inline_options:
        # Selection from a list of options (either dynamic from options_source or inline)
        schema["type"] = "enum"
    else:
        # Primitive type - infer from Python type
        if inner_type is bool:
            schema["type"] = "bool"
        elif inner_type is int:
            schema["type"] = "int"
        elif inner_type is float:
            schema["type"] = "float"
        elif inner_type is str:
            schema["type"] = "str"
        else:
            schema["type"] = "str"  # Default fallback
    
    # Add component if specified (optional UI hint)
    if component:
        schema["component"] = component
    
    # Copy remaining extra properties (category, multiline, rows, options_source, etc.)
    for key, value in extra.items():
        schema[key] = value
    
    # Extract Pydantic constraints
    metadata = field_info.metadata if hasattr(field_info, 'metadata') else []
    for meta in metadata:
        if hasattr(meta, 'ge') and meta.ge is not None:
            schema["min"] = meta.ge
        if hasattr(meta, 'le') and meta.le is not None:
            schema["max"] = meta.le
        if hasattr(meta, 'multiple_of') and meta.multiple_of is not None:
            schema["step"] = meta.multiple_of
    
    # Handle default value
    # Check for PydanticUndefined and Ellipsis as markers for required fields
    from pydantic_core import PydanticUndefined
    default_val = None
    has_default = False
    
    if field_info.default is not None and field_info.default is not ... and field_info.default is not PydanticUndefined:
        default_val = field_info.default
        has_default = True
    elif field_info.default_factory is not None:
        # Call default_factory to get the default value
        try:
            default_val = field_info.default_factory()
            has_default = True
        except:
            pass
    
    if has_default and default_val is not None:
        # Convert to JSON-serializable format
        schema["default"] = serialize_default(default_val)
    elif has_default and default_val is None:
        # Explicit None default
        schema["default"] = None
    else:
        # No default means required
        schema["required"] = True
    
    # Set default category if not specified
    if "category" not in schema:
        schema["category"] = "parameters"
    
    return schema


def generate_schema(
    workflow_params: Type[BaseModel],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a complete YAML schema from WorkflowParams."""
    schema = {
        "name": metadata.get("name", "Unnamed Workflow"),
        "description": metadata.get("description", ""),
        "category": metadata.get("category", "Uncategorized"),
        "model_types": metadata.get("model_types", []),
        "parameters": [],
    }
    
    for field_name, field_info in workflow_params.model_fields.items():
        field_type = field_info.annotation
        field_schema = extract_field_schema(field_name, field_info, field_type)
        schema["parameters"].append(field_schema)
    
    return schema


def merge_schemas(
    generated: Dict[str, Any],
    existing: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge generated schema with existing YAML, preserving customizations and ordering."""
    merged = {
        "name": generated["name"],
        "description": existing.get("description", generated.get("description", "")),
        "category": generated["category"],
        "model_types": generated["model_types"],
        "parameters": [],
    }
    
    # Create lookups for both
    existing_params = {p["name"]: p for p in existing.get("parameters", [])}
    generated_params = {p["name"]: p for p in generated.get("parameters", [])}
    
    # Track which generated params have been processed
    processed = set()
    
    # First pass: iterate through EXISTING order to preserve user ordering
    for existing_param in existing.get("parameters", []):
        param_name = existing_param["name"]
        
        if param_name in generated_params:
            # Merge: preserve user customizations, update from generated
            gen_param = generated_params[param_name]
            merged_param = {}
            
            # Always use generated values for UPDATE_PROPERTIES
            for key, value in gen_param.items():
                if key in UPDATE_PROPERTIES or key == "name":
                    merged_param[key] = value
            
            # Preserve user customizations from PRESERVE_PROPERTIES
            for key in PRESERVE_PROPERTIES:
                if key in existing_param:
                    merged_param[key] = existing_param[key]
            
            # Keep label from existing if customized, otherwise use generated
            if "label" in existing_param:
                merged_param["label"] = existing_param["label"]
            else:
                merged_param["label"] = gen_param.get("label", camel_to_title(param_name))
            
            merged["parameters"].append(merged_param)
            processed.add(param_name)
    
    # Second pass: add any NEW fields from generated (not in existing YAML)
    for gen_param in generated["parameters"]:
        param_name = gen_param["name"]
        if param_name not in processed:
            merged["parameters"].append(gen_param)
    
    return merged


def load_existing_yaml(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing YAML file if it exists."""
    if not yaml_path.exists():
        return None
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(schema: Dict[str, Any], yaml_path: Path) -> None:
    """Write schema to YAML file with nice formatting."""
    # Custom representer to handle empty lists nicely
    def represent_list(dumper, data):
        if len(data) == 0:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    yaml.add_representer(list, represent_list)
    
    # Build YAML in parts for better formatting
    lines = []
    
    # Metadata section
    lines.append(f"name: {schema['name']}")
    lines.append(f"description: {schema.get('description', '')}")
    lines.append(f"category: {schema.get('category', 'Uncategorized')}")
    
    # Model types
    model_types = schema.get('model_types', [])
    if model_types:
        lines.append("model_types:")
        for mt in model_types:
            lines.append(f"  - {mt}")
    else:
        lines.append("model_types: []")
    
    # Double newline before parameters
    lines.append("")
    lines.append("parameters:")
    
    # Format each parameter with a blank line between them
    parameters = schema.get('parameters', [])
    for i, param in enumerate(parameters):
        # Add blank line between parameters (not before first one)
        if i > 0:
            lines.append("")
        
        # Dump the parameter to YAML and indent it
        param_yaml = yaml.dump([param], default_flow_style=False, allow_unicode=True, sort_keys=False)
        # Remove the leading "- " and adjust indentation
        param_lines = param_yaml.strip().split('\n')
        for j, line in enumerate(param_lines):
            if j == 0:
                # First line starts with "- "
                lines.append(f"  {line}")
            else:
                # Subsequent lines need proper indentation
                lines.append(f"  {line}")
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines) + '\n')


def generate_workflow_schema(
    workflow_module_path: Path,
    output_path: Optional[Path] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Generate YAML schema for a workflow module.
    
    Args:
        workflow_module_path: Path to the workflow .py file
        output_path: Path for output YAML (defaults to same name as .py)
        dry_run: If True, don't write file, just return schema
        
    Returns:
        The generated/merged schema
    """
    import importlib.util
    
    # Load the workflow module
    spec = importlib.util.spec_from_file_location("workflow", workflow_module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {workflow_module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get WorkflowParams and WORKFLOW_METADATA
    if not hasattr(module, "WorkflowParams"):
        raise ValueError(f"Module {workflow_module_path} has no WorkflowParams class")
    
    workflow_params = module.WorkflowParams
    metadata = getattr(module, "WORKFLOW_METADATA", {})
    
    # Generate schema
    generated = generate_schema(workflow_params, metadata)
    
    # Determine output path
    if output_path is None:
        output_path = workflow_module_path.with_suffix(".yaml")
    
    # Load existing and merge if present
    existing = load_existing_yaml(output_path)
    if existing:
        schema = merge_schemas(generated, existing)
    else:
        schema = generated
    
    # Write unless dry run
    if not dry_run:
        write_yaml(schema, output_path)
    
    return schema


def validate_workflow_schema(
    workflow_module_path: Path,
    yaml_path: Optional[Path] = None
) -> List[str]:
    """Validate that YAML schema matches WorkflowParams.
    
    Returns list of validation errors (empty if valid).
    """
    import importlib.util
    
    errors = []
    
    # Load the workflow module
    spec = importlib.util.spec_from_file_location("workflow", workflow_module_path)
    if spec is None or spec.loader is None:
        return [f"Could not load module from {workflow_module_path}"]
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "WorkflowParams"):
        return [f"Module {workflow_module_path} has no WorkflowParams class"]
    
    workflow_params = module.WorkflowParams
    
    # Determine YAML path
    if yaml_path is None:
        yaml_path = workflow_module_path.with_suffix(".yaml")
    
    if not yaml_path.exists():
        return [f"YAML file not found: {yaml_path}"]
    
    # Load YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_schema = yaml.safe_load(f)
    
    # Get field names from both
    python_fields = set(workflow_params.model_fields.keys())
    yaml_fields = {p["name"] for p in yaml_schema.get("parameters", [])}
    
    # Check for fields in YAML but not in Python
    extra_in_yaml = yaml_fields - python_fields
    for field in extra_in_yaml:
        errors.append(f"Field '{field}' in YAML but not in WorkflowParams")
    
    # Check for fields in Python but not in YAML
    missing_in_yaml = python_fields - yaml_fields
    for field in missing_in_yaml:
        errors.append(f"Field '{field}' in WorkflowParams but not in YAML")
    
    return errors
