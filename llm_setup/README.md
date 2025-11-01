# LLM Setup

Artifacts and scripts used while preparing or inspecting model assets.

- `download/` - helpers for pulling checkpoints or repositories.
- `analysis/` - utilities for inspecting weights, producing inventories, and
  validating conversions.
- `docs/` - notes gathered during the setup stage.

## Analysis helpers

- `get_tensor_shapes_form_safetensors.py`  
  Regenerates `tensor_inventory.csv` by parsing safetensor headers and recording
  tensor names, shapes, and byte offsets into the raw shards.

- `extract_tensor_by_offset.py`  
  Loads a single tensor from a shard given the byte offset recorded in the
  inventory, reporting shape/dtype (and optionally emitting a `.pt` file).

- `verify_tensor_extraction.py`  
  Compares the manual extraction path against `safetensors.safe_open` for one or
  more tensors to ensure both produce identical flattened vectors.

- `test_analysis.py`  
  Sanity-checks that required raw assets exist (weights, metadata, tokenizer)
  and that the analysis scripts can resolve the expected paths.

These scripts assume raw model files live under `../weights/`.
