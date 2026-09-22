"""
Workflow registry for external API.
Auto-discovers workflow files from the workflows/ folder.
Supports hot reloading - workflows are reloaded from disk on each execution.

Schema is loaded from YAML files (auto-generated from WorkflowParams).

Performance optimizations:
- Workflow list is cached and only refreshed when explicitly requested
- YAML schemas are cached after first load
- Options resolution is cached with TTL
"""

import glob
import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

# Support both relative and absolute imports
# Resources module requires ComfyUI context
resources = None
try:
    from . import resources
except ImportError:
    try:
        import resources
    except ImportError:
        # Resources not available (running outside ComfyUI)
        pass

# Cache for discovered workflows
_workflow_cache: Dict[str, dict] = {}
_workflow_cache_time: float = 0

# Cache for YAML schemas
_yaml_schema_cache: Dict[str, Optional[Dict]] = {}


def load_workflow_module(workflow_name: str, client_id: str = None):
    """
    Dynamically load/reload a workflow module from the workflows folder.

    Args:
        workflow_name: Name of the workflow to load
        client_id: Optional client ID to use for comfy_script WebSocket.
                   If provided, this ID will be used instead of a generated one,
                   allowing external clients to receive binary preview frames.
    """
    workflow_path = os.path.join(
        os.path.dirname(__file__), "workflows", f"{workflow_name}.py"
    )
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow script not found: {workflow_path}")

    # If client_id is provided, set it in comfy_script BEFORE loading the workflow
    # This ensures comfy_script uses this ID when it creates its WebSocket connection
    if client_id:
        try:
            import comfy_script.runtime as runtime
            # Set the client_id before the workflow module imports and calls load()
            runtime._client_id = client_id
            print(f"[workflows] Set comfy_script client_id to: {client_id}")
        except ImportError:
            print("[workflows] Warning: Could not import comfy_script.runtime to set client_id")

    spec = importlib.util.spec_from_file_location(
        f"workflows.{workflow_name}", workflow_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_yaml_schema(workflow_name: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Load YAML schema for a workflow if it exists.
    Uses caching to avoid repeated disk reads.
    """
    global _yaml_schema_cache

    # Check cache first
    if use_cache and workflow_name in _yaml_schema_cache:
        return _yaml_schema_cache[workflow_name]

    yaml_path = os.path.join(
        os.path.dirname(__file__), "workflows", f"{workflow_name}.yaml"
    )

    if not os.path.exists(yaml_path):
        _yaml_schema_cache[workflow_name] = None
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
        _yaml_schema_cache[workflow_name] = schema
        return schema


def ensure_yaml_schema(workflow_name: str, module) -> Optional[Dict]:
    """
    Ensure YAML schema exists for a workflow.
    Auto-generates from WorkflowParams if missing or if Python model has new fields.
    Returns the schema dict or None.
    """
    yaml_path = Path(os.path.dirname(__file__)) / "workflows" / f"{workflow_name}.yaml"

    # Try to load existing YAML
    schema = load_yaml_schema(workflow_name)

    if hasattr(module, "WorkflowParams"):
        # Check if we need to regenerate (missing YAML or new fields in Python)
        should_regenerate = schema is None

        if schema is not None:
            # Check for new fields in Python that aren't in YAML
            yaml_field_names = {p.get("name") for p in schema.get("parameters", [])}
            python_field_names = set(module.WorkflowParams.model_fields.keys())
            new_fields = python_field_names - yaml_field_names
            if new_fields:
                print(f"[workflows] Found new fields in {workflow_name}: {new_fields}")
                should_regenerate = True

        if should_regenerate:
            # Auto-generate YAML from WorkflowParams (merges with existing)
            try:
                # Support both relative and absolute imports
                try:
                    from .schema_generator import generate_workflow_schema
                except ImportError:
                    from schema_generator import generate_workflow_schema

                py_path = yaml_path.with_suffix(".py")
                schema = generate_workflow_schema(py_path)
                # Clear cache so next load gets the new schema
                _yaml_schema_cache.pop(workflow_name, None)
                print(f"[workflows] {'Updated' if yaml_path.exists() else 'Generated'} schema: {workflow_name}.yaml")
            except Exception as e:
                print(f"[workflows] Error generating schema for {workflow_name}: {e}")

    return schema


# Cache TTL for workflow discovery (5 minutes)
WORKFLOW_CACHE_TTL = 300


def discover_workflows(force_refresh: bool = False) -> Dict[str, dict]:
    """
    Discover all workflow files in the workflows/ folder.
    Workflows must define WORKFLOW_METADATA dict to be discovered.
    Returns a dict mapping workflow_id to its config.

    Uses caching with TTL to avoid rescanning on every request.
    Pass force_refresh=True to bypass cache.
    """
    global _workflow_cache, _workflow_cache_time

    current_time = time.time()
    cache_age = current_time - _workflow_cache_time

    # Return cached data if still valid and not forced refresh
    if _workflow_cache and not force_refresh and cache_age < WORKFLOW_CACHE_TTL:
        return _workflow_cache

    workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
    discovered = {}

    # Find all Python files in workflows directory
    for filepath in glob.glob(os.path.join(workflows_dir, "*.py")):
        filename = os.path.basename(filepath)

        # Skip __init__.py and other special files
        if filename.startswith("_"):
            continue

        workflow_id = filename[:-3]  # Remove .py extension

        try:
            # Load the module to check for WORKFLOW_METADATA
            module = load_workflow_module(workflow_id)

            # Check if module has required attributes
            if not hasattr(module, "WORKFLOW_METADATA"):
                print(f"[workflows] Skipping {filename}: No WORKFLOW_METADATA defined")
                continue

            if not hasattr(module, "run_workflow"):
                print(
                    f"[workflows] Skipping {filename}: No run_workflow function defined"
                )
                continue

            metadata = module.WORKFLOW_METADATA

            # Check for YAML schema (uses its own cache)
            yaml_schema = load_yaml_schema(workflow_id)
            has_yaml = yaml_schema is not None

            # Check for legacy WORKFLOW_PARAMETERS (for backward compat)
            has_legacy_params = hasattr(module, "WORKFLOW_PARAMETERS")

            # Build workflow config
            discovered[workflow_id] = {
                "script": workflow_id,
                "name": metadata.get("name", workflow_id),
                "description": metadata.get("description", ""),
                "model_types": metadata.get("model_types", []),
                "category": metadata.get("category", "Uncategorized"),
                "has_parameters": has_yaml or has_legacy_params,
                "schema_source": "yaml"
                if has_yaml
                else ("legacy" if has_legacy_params else None),
            }

            print(
                f"[workflows] Discovered: {workflow_id} - {metadata.get('name', workflow_id)}"
            )

        except Exception as e:
            print(f"[workflows] Error loading {filename}: {e}")
            continue

    _workflow_cache = discovered
    _workflow_cache_time = current_time
    print(f"[workflows] Cache refreshed with {len(discovered)} workflows")
    return discovered


def get_workflow(name: str, client_id: str = None) -> Callable:
    """
    Get a workflow function by name.
    Reloads the module from disk each time for hot reloading.

    Args:
        name: Workflow name/ID
        client_id: Optional client ID for comfy_script WebSocket connection.
                   Pass this to receive binary preview frames on your WebSocket.
    """
    workflows = discover_workflows()

    if name not in workflows:
        raise ValueError(
            f"Workflow '{name}' not found. Available: {list(workflows.keys())}"
        )

    # Reload the module from disk each time
    module = load_workflow_module(name, client_id=client_id)
    return module.run_workflow


def get_workflow_metadata(name: str) -> dict:
    """Get metadata for a specific workflow."""
    workflows = discover_workflows()

    if name not in workflows:
        raise ValueError(
            f"Workflow '{name}' not found. Available: {list(workflows.keys())}"
        )

    config = workflows[name].copy()
    config.pop("script", None)  # Don't expose internal file path
    config.pop("schema_source", None)  # Internal use only
    return config


def get_workflow_parameters(name: str) -> Optional[List[dict]]:
    """
    Get parameter schema for a specific workflow.

    Loads from YAML schema first, falls back to legacy WORKFLOW_PARAMETERS.
    Auto-generates YAML if WorkflowParams exists but no schema.

    Note: Options are NOT resolved - they should be fetched dynamically by the
    v1-server using options_source references.

    Args:
        name: Workflow name/ID

    Returns:
        List of parameter definitions with options_source references (NOT resolved options)
    """
    workflows = discover_workflows()

    if name not in workflows:
        raise ValueError(
            f"Workflow '{name}' not found. Available: {list(workflows.keys())}"
        )

    # Load the module
    module = load_workflow_module(name)

    # Try YAML schema first
    schema = ensure_yaml_schema(name, module)
    if schema and "parameters" in schema:
        return schema["parameters"]

    # Fall back to legacy WORKFLOW_PARAMETERS
    if hasattr(module, "WORKFLOW_PARAMETERS"):
        return module.WORKFLOW_PARAMETERS

    return None


def list_workflows(refresh: bool = False) -> list:
    """
    Get list of available workflow names with their metadata.

    Uses cached data by default for fast response.
    Set refresh=True to force re-scan the workflows directory.
    """
    workflows = discover_workflows(force_refresh=refresh)

    return [
        {
            "id": name,
            **{k: v for k, v in config.items() if k not in ("script", "schema_source")},
        }
        for name, config in workflows.items()
    ]


def invalidate_workflow_cache():
    """
    Invalidate all workflow caches.
    Call this when you know workflows have changed (e.g., after adding/editing files).
    """
    global _workflow_cache, _workflow_cache_time, _yaml_schema_cache
    _workflow_cache = {}
    _workflow_cache_time = 0
    _yaml_schema_cache = {}
    print("[workflows] Cache invalidated")
