# TokenVisor-recipes

Auto-generated vLLM deployment profiles from [vllm-project/recipes](https://github.com/vllm-project/recipes). Published daily as a single JSON file for consumption by [TokenVisor](https://github.com/TokenVisor/TokenVisor).

## Output

`profiles/latest.json` — a bundle of deployment profiles keyed by accelerator and HuggingFace model ID. Each profile entry contains strategy variants (Standard, Expert Parallel, Speculative Decoding) with pre-computed CLI args and env vars.

## How profiles are generated

1. Daily CI clones `vllm-project/recipes` and runs `generate/generate.py`
2. The script parses every YAML recipe, resolves Docker image tags, applies overlays
3. Output is published to GitHub Pages: `https://tokenvisor.github.io/TokenVisor-recipes/profiles/latest.json`

## Local development

```bash
git clone https://github.com/TokenVisor/TokenVisor-recipes
git clone https://github.com/vllm-project/recipes

cd TokenVisor-recipes
pip install pyyaml httpx
VLLM_RECIPES_DIR=../vllm-recipes python generate/generate.py
```

### Overlays

TokenVisor-specific overrides live in `generate/overlays.yaml`. The overlay key format:

- `<huggingface_id>` — matches all profiles with this model
- `<huggingface_id>@<accelerator>` — matches only that specific accelerator

Overlay values merge on top of generated values. Use `null` to remove a generated field.

## Schema

See the `schema_version` field in the output JSON (currently `"1"`).
