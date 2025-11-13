import ml_collections
from ferminet import base_config
import jax.numpy as jnp

def _get_similar_keys(key, key_list):
  """Find keys similar to the given key, focusing on likely typos."""
  import difflib
  
  # First check for very close matches that are likely typos
  filtered_matches = []
  
  for existing_key in key_list:
    # Skip if one is clearly an extension of the other (e.g., batch_size vs batch_size_2)
    if key.startswith(existing_key + '_') or existing_key.startswith(key + '_'):
      continue
      
    # Skip if length difference is too large
    if abs(len(key) - len(existing_key)) > 3:
      continue
    
    # Calculate edit distance
    edit_dist = _simple_edit_distance(key.lower(), existing_key.lower())
    
    # Only suggest if it's a very close match (1-2 character difference)
    if edit_dist <= 2:
      # For very short keys, be more strict
      if len(key) <= 4 and edit_dist > 1:
        continue
      filtered_matches.append(existing_key)
  
  # Sort by similarity (closer matches first)
  filtered_matches.sort(key=lambda x: _simple_edit_distance(key.lower(), x.lower()))
  
  return filtered_matches[:3]  # Return top 3 matches


def _simple_edit_distance(s1, s2):
  """Calculate simple edit distance between two strings."""
  if len(s1) > len(s2):
    s1, s2 = s2, s1
  
  distances = list(range(len(s1) + 1))
  for i2, c2 in enumerate(s2):
    new_distances = [i2 + 1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        new_distances.append(distances[i1])
      else:
        new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
    distances = new_distances
  return distances[-1]


def _validate_config_keys(cfg, reference_cfg, path=""):
  """Validate configuration keys and suggest corrections for likely typos.
  
  Args:
    cfg: The configuration to validate.
    reference_cfg: The reference configuration with valid keys.
    path: The current path in the configuration tree (for error messages).
    
  Raises:
    ValueError: If a key appears to be a typo of a valid key.
  """
  for key in cfg.keys():
    if key not in reference_cfg:
      # Check if this might be a typo of an existing key
      similar_keys = _get_similar_keys(key, list(reference_cfg.keys()))
      if similar_keys:
        full_path = f"{path}.{key}" if path else key
        raise ValueError(f"Configuration key '{full_path}' not found. Did you mean '{similar_keys[0]}'? "
                        f"Available keys: {list(reference_cfg.keys())}")
      # If no similar keys found, it's probably an intentionally new key, so allow it
    else:
      # Recursively check nested dictionaries only if the key exists in reference
      if isinstance(cfg[key], ml_collections.ConfigDict) and isinstance(reference_cfg[key], ml_collections.ConfigDict):
        _validate_config_keys(cfg[key], reference_cfg[key], f"{path}.{key}" if path else key)

def reset_config(cfg):
  """Reset specific configuration parameters."""
  # normalize twist_weights
  if cfg.system.pbc.twist_weights.shape[0] != cfg.system.pbc.twist_vectors.shape[0]:
    cfg.system.pbc.twist_weights = jnp.ones(cfg.system.pbc.twist_vectors.shape[0])

  twist_weights = cfg.system.pbc.twist_weights
  weight_sum = jnp.sum(twist_weights)
  ntwist = twist_weights.shape[0]
  if not jnp.isclose(weight_sum, ntwist):
    normalized_weights = twist_weights * (ntwist / weight_sum)
    cfg.system.pbc.twist_weights = normalized_weights
  
  return cfg

def validate_config(cfg):
  """Validate that the configuration only contains valid keys from the base configuration.
  
  Args:
    cfg: ml_collections.ConfigDict containing settings.
    
  Raises:
    ValueError: If any invalid keys are found.
  """
  # Create a reference configuration with all valid keys
  reference_cfg = base_config.default()
  
  # Validate the configuration
  _validate_config_keys(cfg, reference_cfg)

  cfg = reset_config(cfg)

  return cfg
