# LLM Raw Assets

Immutable downloads captured during experimentation.

- `olmo_2/raw_weights/` - safetensors shards for the released OLMo-2 model.
- `olmo_2/metadata/` - model configs, safetensors index, generated inventories,
  and cached download state.
- `olmo_2/raw_tokenizer/` - tokenizer JSON/config/merge files associated with the
  checkpoint.
- `olmo_2/test/` - reference scripts that exercise text generation with the
  released checkpoints.

Treat this area as read-only input data; processing scripts live under
`../llm_setup/`.

