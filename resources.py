"""
Resource management for external API.
Provides access to models, metadata, and indexing capabilities.
"""
import os
import json
import hashlib
import requests
from typing import Dict, List, Optional
import folder_paths

# CivitAI API base URL - configurable via environment variable for proxying
CIVITAI_API_BASE = os.environ.get('CIVITAI_API_BASE', "https://civitai.com/api/v1")

# Resources directory for storing index files
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'resources')
os.makedirs(RESOURCES_DIR, exist_ok=True)


def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_index(filename: str) -> Dict:
    """Get existing index data for a model file."""
    # Use full filename (with extension) + .index
    index_path = os.path.join(RESOURCES_DIR, f"{filename}.index")
    
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    
    return {
        'filename': filename,
        'hash': None,
        'civitai': None,
        'imagelab': {}
    }


def save_index(filename: str, data: Dict):
    """Save index data for a model file."""
    # Use full filename (with extension) + .index
    index_path = os.path.join(RESOURCES_DIR, f"{filename}.index")
    with open(index_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_civitai_metadata(hash: str) -> Optional[Dict]:
    """Fetch metadata from CivitAI by hash."""
    try:
        response = requests.get(
            f"{CIVITAI_API_BASE}/model-versions/by-hash/{hash}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Resources] Error fetching CivitAI metadata: {e}")
    return None


def index_model(filepath: str, filename: str) -> Dict:
    """
    Index a model file - calculate hash and fetch metadata.
    Returns the index data.
    """
    print(f"[Resources] Indexing: {filename}")
    index = get_index(filename)
    
    # Calculate hash if not present
    if not index.get('hash'):
        print(f"[Resources] Calculating SHA256 hash...")
        try:
            index['hash'] = calculate_sha256(filepath)
        except Exception as e:
            print(f"[Resources] Error calculating hash: {e}")
            return index
    
    # Fetch CivitAI metadata if not present
    if index.get('hash') and not index.get('civitai'):
        print(f"[Resources] Fetching CivitAI metadata...")
        index['civitai'] = get_civitai_metadata(index['hash'])
        save_index(filename, index)
    
    return index


def get_models_by_type(model_type: str) -> List[Dict]:
    """
    Get all models of a specific type with basic info (no indexing).
    
    Args:
        model_type: Type of model (checkpoints, loras, vae, etc.)
    """
    models = []
    
    if model_type not in folder_paths.folder_names_and_paths:
        return models
    
    # Get files using folder_paths API
    files = folder_paths.get_filename_list(model_type)
    
    for filename in files:
        models.append({
            'label': os.path.splitext(filename)[0],
            'apiName': filename,
            'indexed': os.path.exists(os.path.join(RESOURCES_DIR, f"{filename}.index"))
        })
    
    return models


def index_model_by_name(model_type: str, filename: str) -> Dict:
    """
    Index a specific model file by name.
    
    Args:
        model_type: Type of model (checkpoints, loras, etc.)
        filename: The filename to index
        
    Returns:
        Dict with hash and metadata
    """
    if model_type not in folder_paths.folder_names_and_paths:
        raise ValueError(f"Invalid model type: {model_type}")
    
    filepath = folder_paths.get_full_path(model_type, filename)
    if not filepath:
        raise FileNotFoundError(f"Model file not found: {filename}")
    
    # Index the model
    index = index_model(filepath, filename)
    
    return {
        'filename': filename,
        'hash': index.get('hash'),
        'civitai': index.get('civitai'),
        'imagelab': index.get('imagelab', {})
    }


def index_all_new_models() -> Dict[str, int]:
    """
    Index all models that don't have an index file yet.
    Returns a count of indexed models per type.
    """
    model_types = [
        'checkpoints', 'loras', 'vae', 'embeddings', 'upscale_models',
        'controlnet', 'clip', 'clip_vision', 'diffusion_models'
    ]
    
    indexed_counts = {}
    
    for model_type in model_types:
        if model_type not in folder_paths.folder_names_and_paths:
            continue
        
        count = 0
        files = folder_paths.get_filename_list(model_type)
        
        for filename in files:
            index_path = os.path.join(RESOURCES_DIR, f"{filename}.index")
            
            # Only index if no index file exists
            if not os.path.exists(index_path):
                try:
                    filepath = folder_paths.get_full_path(model_type, filename)
                    if filepath:
                        index_model(filepath, filename)
                        count += 1
                except Exception as e:
                    print(f"[Resources] Error indexing {filename}: {e}")
        
        if count > 0:
            indexed_counts[model_type] = count
    
    return indexed_counts


def get_all_models() -> Dict[str, List[Dict]]:
    """
    Get all available models grouped by type (no indexing).
    """
    # Map ComfyUI folder names to API response keys
    # Some keys are pluralized for consistency with v1-server expectations
    model_types = {
        'checkpoints': 'checkpoints',
        'loras': 'loras',
        'vae': 'vaes',  # Pluralize for v1-server compatibility
        'embeddings': 'embeddings',
        'upscale_models': 'upscale_models',
        'controlnet': 'controlnet',
        'clip': 'clip',
        'clip_vision': 'clip_vision',
        'diffusion_models': 'diffusion_models',
        'unet': 'unet',
        'gligen': 'gligen',
        'photomaker': 'photomaker',
        'style_models': 'style_models'
    }

    resources = {}

    for folder_name, api_key in model_types.items():
        try:
            models = get_models_by_type(folder_name)
            if models:  # Only include if models exist
                resources[api_key] = models
        except Exception as e:
            print(f"[Resources] Error loading {folder_name}: {e}")
    
    return resources


def get_samplers() -> List[str]:
    """Get available samplers as simple strings."""
    import comfy.samplers
    return list(comfy.samplers.KSampler.SAMPLERS)


def get_schedulers() -> List[str]:
    """Get available schedulers as simple strings."""
    import comfy.samplers
    return list(comfy.samplers.KSampler.SCHEDULERS)


def get_upscale_methods() -> List[str]:
    """Get available upscale methods as simple strings."""
    import nodes
    return list(nodes.LatentUpscale.upscale_methods)
