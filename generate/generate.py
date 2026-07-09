#!/usr/bin/env python3
"""
TokenVisor deployment profile generator.

Reads vendored vllm-recipes + overlays.yaml → produces profiles/latest.json.

Usage:
    VLLM_RECIPES_DIR=../vllm-recipes python generate/generate.py

Output: profiles/latest.json  (creates/overwrites in CWD or VLLM_RECIPES_DIR/..)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
RECIPES_DIR = Path(os.environ.get("VLLM_RECIPES_DIR", "../vllm-recipes")).resolve()
GENERATE_DIR = Path(__file__).parent.resolve()
REPO_DIR = GENERATE_DIR.parent
OVERLAYS_PATH = GENERATE_DIR / "overlays.yaml"
OUTPUT_PATH = REPO_DIR / "profiles" / "latest.json"
PREVIOUS_PATH = OUTPUT_PATH  # same location; previous run is the existing file

# ── Constants ────────────────────────────────────────────────────────────────
VLLM_RELEASE_TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
VLLM_STANDARD_REPOS = {"vllm/vllm-openai", "vllm/vllm-openai-rocm"}

STRIP_ARGS = {"--tensor-parallel-size", "--max-model-len", "-tp"}
# Minimum vLLM release that a recipe must require; recipes below this are skipped.
MIN_VLLM_VERSION = "0.7.0"


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_version(v: str) -> tuple[int, ...]:
    """Normalize a version string to a comparable tuple."""
    return tuple(int(x) for x in v.lstrip("v").split("."))


def infer_generation(taxonomy: dict, accelerator: str) -> str | None:
    """Return the generation key for an accelerator, or None."""
    hw = taxonomy.get("hardware_profiles", {}).get(accelerator)
    return hw["generation"] if hw else None


def list_to_dict(args: list[str]) -> dict[str, str]:
    """Convert CLI args list to dict: ['--flag', 'val', '--bool'] → {'--flag': 'val', '--bool': ''}."""
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if key.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = ""
            i += 1
    return out


def dict_from_env(env: dict) -> dict[str, str]:
    """Convert env dict to string-string mapping."""
    return {k: str(v) for k, v in env.items()}


def first_viable(priority: list[str], strategies: dict) -> str:
    """Return first key from `priority` that exists in `strategies`, or 'standard'."""
    for key in priority:
        if key in strategies:
            return key
    return "standard"


def check_docker_tag(client: httpx.Client, repo: str, tag: str) -> bool:
    """Return True if the tag exists on Docker Hub."""
    try:
        r = client.get(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}",
            timeout=10,
        )
        return r.is_success and r.json().get("name") == tag
    except Exception:
        return False


def resolve_vllm_version(client: httpx.Client) -> str:
    """Get latest vLLM release, verify Docker images exist."""
    previous_version: str | None = None
    if PREVIOUS_PATH.exists():
        try:
            prev = json.loads(PREVIOUS_PATH.read_text())
            previous_version = prev.get("vllm_version")
        except Exception:
            pass

    try:
        r = client.get(
            "https://api.github.com/repos/vllm-project/vllm/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        r.raise_for_status()
        gh_tag: str = r.json()["tag_name"]
        version = gh_tag.lstrip("v")

        nvidia_ok = check_docker_tag(client, "vllm/vllm-openai", gh_tag)
        rocm_ok = check_docker_tag(client, "vllm/vllm-openai-rocm", gh_tag)

        if nvidia_ok and rocm_ok:
            return version

        # Fall back to previous known version
        if previous_version:
            print(f"  Docker tag verification incomplete (nvidia={nvidia_ok}, rocm={rocm_ok}); "
                  f"falling back to previous version {previous_version}")
            return previous_version
        # First run: use GitHub tag even without Docker verification
        print(f"  Docker tag verification incomplete; using GitHub tag {version}")
        return version
    except Exception as exc:
        if previous_version:
            print(f"  GitHub API error ({exc}); falling back to previous version {previous_version}")
            return previous_version
        print(f"  GitHub API error ({exc}); using latest known version 0.24.0")
        return "0.24.0"


def strip_backend_args(args: list[str]) -> list[str]:
    """Remove args that the backend injects automatically."""
    result: list[str] = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        # Check exact match or prefix match (e.g. "--max-model-len auto")
        stripped = a.split(" ", 1)[0]
        if stripped in STRIP_ARGS:
            # If there was a value attached (separated by space), don't skip next
            if " " not in a:
                skip_next = True
            continue
        result.append(a)
    return result

def _default_image(brand: str, version: str) -> str:
    if brand.lower() == "amd":
        return f"vllm/vllm-openai-rocm:v{version}"
    return f"vllm/vllm-openai:v{version}"


def _pin_image(img: str | None, version: str, brand: str = "nvidia") -> str:
    if img is None:
        return _default_image(brand, version)
    if ":" in img:
        repo, tag = img.split(":", 1)
    else:
        repo, tag = img, "latest"

    if repo in VLLM_STANDARD_REPOS:
        if tag == "latest" or VLLM_RELEASE_TAG_PATTERN.match(tag):
            return f"{repo}:v{version}"
        return img

    return img


def resolve_docker_image(
    val: str | dict | None,
    brand: str,
    version: str,
) -> str:
    """Resolve a docker image string, dict, or None to a pinned image tag."""
    if val is None:
        return _default_image(brand, version)
    if isinstance(val, str):
        return _pin_image(val, version, brand)
    # Per-brand object
    key_brand = "amd" if brand.lower() == "amd" else "nvidia"
    img = val.get(key_brand)
    return _pin_image(img, version, brand) if img else _default_image(brand, version)


# ── Profile building ─────────────────────────────────────────────────────────
def build_strategies(
    recipe: dict,
    merged_args: list[str],
    merged_env: dict[str, str],
) -> dict:
    """Build strategies map for a profile entry.
    
    Returns up to 2 strategies:
    - "latency-focus": standard args (+ spec_decode if present)
    - "balance": standard + EP args (+ spec_decode if present, MoE only)
    """
    strategies: dict = {}
    extra_env = dict_from_env(merged_env)
    sd = recipe.get("features", {}).get("spec_decoding", {})
    sd_args = list(sd.get("args", []))

    # Latency-focus: standard args + spec_decode if present
    lf_args = list(merged_args) + sd_args
    strategies["latency-focus"] = {
        "extra_args": list_to_dict(lf_args),
        "extra_envs": extra_env,
    }

    # Balance: standard + EP args + spec_decode if present (MoE only)
    if recipe.get("model", {}).get("architecture") == "moe":
        bal_args = list(merged_args) + sd_args
        if "--enable-expert-parallel" not in bal_args:
            bal_args.append("--enable-expert-parallel")
        strategies["balance"] = {
            "extra_args": list_to_dict(bal_args),
            "extra_envs": extra_env,
        }

    return strategies


def make_profile(
    recipe: dict,
    accelerator: str,
    variant_key: str,
    variant: dict,
    brand: str,
    taxonomy: dict,
    vllm_version: str,
) -> dict:
    """Build a single profile entry."""
    # Short profile name: title only for default variant
    if variant_key == "default":
        profile_name = recipe["meta"]["title"]
    else:
        profile_name = f"{recipe['meta']['title']} ({variant_key})"

    # Merge: base → brand overrides → generation overrides → variant overrides
    args: list[str] = list(recipe["model"].get("base_args", []))
    env: dict[str, str] = dict(recipe["model"].get("base_env", {}))

    hardware_overrides = recipe.get("hardware_overrides", {})
    gen = infer_generation(taxonomy, accelerator)
    cascade_keys: list[str | None] = [brand]
    if gen:
        cascade_keys.append(gen)

    for ov_key in cascade_keys:
        if ov_key:
            ov = hardware_overrides.get(ov_key, {})
            args += list(ov.get("extra_args", []))
            env.update(ov.get("extra_env", {}))

    args += list(variant.get("extra_args", []))
    env.update(variant.get("extra_env", {}))

    args = strip_backend_args(args)

    # Add always-on feature args (tool_calling, reasoning, encoder_parallel)
    # Check by flag key (--xxx), not individual elements, to avoid breaking pairs
    existing_flags = {a for a in args if a.startswith("--")}
    for feature_key in ("tool_calling", "reasoning", "encoder_parallel"):
        feat = recipe.get("features", {}).get(feature_key, {})
        fargs = feat.get("args")
        if not fargs:
            continue
        i = 0
        while i < len(fargs):
            fa = fargs[i]
            if not fa.startswith("--"):
                # Value without preceding flag — shouldn't happen, skip
                i += 1
                continue
            if fa not in existing_flags:
                args.append(fa)
                # Add value if present
                if i + 1 < len(fargs) and not fargs[i + 1].startswith("--"):
                    args.append(fargs[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                # Flag already present — skip flag and its optional value
                if i + 1 < len(fargs) and not fargs[i + 1].startswith("--"):
                    i += 2
                else:
                    i += 1
    strategies = build_strategies(recipe, args, env)

    # Resolve docker image
    docker_val = recipe.get("docker_image")
    custom_image = resolve_docker_image(docker_val, brand, vllm_version)

    huggingface_id = variant.get("model_id", recipe["model"]["model_id"])

    return {
        "name": profile_name,
        "huggingface_id": huggingface_id,
        "accelerators": accelerator,
        "provider": "vllm_amd" if brand.lower() == "amd" else "vllm",
        "custom_image": custom_image,
        "strategies": strategies,
        "default_strategy": first_viable(["latency-focus", "balance"], strategies),
        "variant_name": variant_key,
    }

# ── Overlay merging ──────────────────────────────────────────────────────────
def merge_overlays(
    index: dict[str, dict[str, list[dict]]],
    overlays: dict,
    accelerator_brands: dict[str, str],
) -> None:
    """Apply overlay overrides to generated profiles in-place."""
    overrides = overlays.get("overrides", {})
    for key, overlay in overrides.items():
        # Parse key format: "<huggingface_id>[@<accelerator_or_brand>]"
        parts = key.split("@", 1)
        hf_id = parts[0]
        filter_val = parts[1] if len(parts) > 1 else None
        for acc_name, hf_map in index.items():
            if hf_id not in hf_map:
                continue
            for profile in hf_map[hf_id]:
                if filter_val is not None:
                    # Match accelerator name or brand (case-insensitive)
                    if profile["accelerators"].lower() != filter_val.lower():
                        acc_brand = accelerator_brands.get(profile["accelerators"], "")
                        if acc_brand.lower() != filter_val.lower():
                            continue
                _apply_overlay(profile, overlay)


def _apply_overlay(profile: dict, overlay: dict) -> None:
    """Apply a single overlay dict to a profile entry, in-place."""
    for field, value in overlay.items():
        if value is None:
            # Remove field if overlay sets it to null
            profile.pop(field, None)
        elif field == "strategies":
            # Per-key merge into existing strategies
            existing = profile.get("strategies", {})
            for sid, strat_overlay in value.items():
                if sid in existing:
                    # Merge extra_args / extra_envs
                    so = strat_overlay or {}
                    existing_s = existing[sid]
                    if "extra_args" in so:
                        existing_s["extra_args"].update(so["extra_args"])
                    if "extra_envs" in so:
                        existing_s["extra_envs"].update(so["extra_envs"])
                else:
                    existing[sid] = strat_overlay
        elif field in ("extra_args", "extra_envs"):
            # Deep merge: overlay entries are added/overwritten
            existing_val = profile.get(field, {})
            if isinstance(value, dict) and isinstance(existing_val, dict):
                existing_val.update(value)
                profile[field] = existing_val
            else:
                profile[field] = value
        else:
            # Simple field replacement
            profile[field] = value


# ── Main generation ──────────────────────────────────────────────────────────
def parse_taxonomy(taxonomy_path: Path) -> dict:
    """Load and return taxonomy dict."""
    return yaml.safe_load(taxonomy_path.read_text()) or {}


def parse_recipe(recipe_path: Path) -> dict | None:
    """Load a single recipe YAML file. Returns None on failure."""
    try:
        recipe = yaml.safe_load(recipe_path.read_text())
        if not recipe or not isinstance(recipe, dict):
            return None
        return recipe
    except Exception as e:
        print(f"  Warning: failed to parse {recipe_path.name}: {e}")
        return None


def main() -> int:
    print(f"TokenVisor Profile Generator")
    print(f"  Recipes dir: {RECIPES_DIR}")
    print(f"  Output: {OUTPUT_PATH}")

    # 1. Load taxonomy
    taxonomy_path = RECIPES_DIR / "taxonomy.yaml"
    if not taxonomy_path.exists():
        print(f"ERROR: taxonomy.yaml not found at {taxonomy_path}")
        return 1
    taxonomy = parse_taxonomy(taxonomy_path)
    hw_profiles: dict = taxonomy.get("hardware_profiles", {})
    print(f"  Taxonomy loaded: {len(hw_profiles)} hardware profiles")

    # Build accelerator_brands map
    accelerator_brands: dict[str, str] = {}
    for name, hw in hw_profiles.items():
        accelerator_brands[name] = hw["brand"].lower()

    # Get vLLM recipes SHA
    recipes_sha = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(RECIPES_DIR), timeout=10,
        )
        if result.returncode == 0:
            recipes_sha = result.stdout.strip()
    except Exception:
        pass

    # 2. Resolve vLLM version
    with httpx.Client() as client:
        vllm_version = resolve_vllm_version(client)

    print(f"  vLLM version: {vllm_version}")

    # 3. Walk recipe files
    models_dir = RECIPES_DIR / "models"
    if not models_dir.exists():
        print(f"ERROR: models dir not found at {models_dir}")
        return 1

    recipe_files = sorted(models_dir.rglob("*.yaml"))

    # Index: accelerator → huggingface_id → list[profile_entry]
    index: dict[str, dict[str, list[dict]]] = {}

    # Collect all accelerators (GPU names + brand keys)
    all_accelerators: list[str] = list(hw_profiles.keys())
    # Add brand-level generic keys (sorted for determinism)
    brands_seen = sorted(set(accelerator_brands.values()))
    brand_accelerators = list(brands_seen)

    for acc in all_accelerators + brand_accelerators:
        index[acc] = {}

    parsed = 0
    skipped_min_version = 0
    skipped_other = 0

    for rp in recipe_files:
        recipe = parse_recipe(rp)
        if recipe is None:
            continue

        meta = recipe.get("meta", {})
        model = recipe.get("model", {})

        # Skip recipes without model_id
        if not model.get("model_id"):
            skipped_other += 1
            continue

        # Skip recipes below min vLLM version
        min_vllm = model.get("min_vllm_version")
        if min_vllm and parse_version(min_vllm) < parse_version(MIN_VLLM_VERSION):
            skipped_min_version += 1
            continue

        vars: dict = recipe.get("variants", {})
        if not vars:
            vars = {"default": {}}

        meta_hardware: dict[str, str] = meta.get("hardware", {})
        hardware_overrides: dict = recipe.get("hardware_overrides", {})
        hardware_list: list[str] = list(meta_hardware.keys())

        parsed += 1

        for acc_name in hardware_list:
            if acc_name not in hw_profiles:
                continue

            brand = hw_profiles[acc_name]["brand"].lower()
            brand_key = brand  # e.g. "nvidia", "amd"

            for vkey, variant in vars.items():
                entry = make_profile(
                    recipe, acc_name, vkey, variant, brand,
                    taxonomy, vllm_version,
                )

                hf_id = entry["huggingface_id"]
                if hf_id not in index[acc_name]:
                    index[acc_name][hf_id] = []
                index[acc_name][hf_id].append(entry)

        # 2d. Brand-level generic profiles — for ALL brands unconditionally
        for brand_key in brands_seen:
            for vkey, variant in vars.items():
                entry = make_profile(
                    recipe, brand_key, vkey, variant, brand_key,
                    taxonomy, vllm_version,
                )
                hf_id = entry["huggingface_id"]
                if hf_id not in index[brand_key]:
                    index[brand_key][hf_id] = []
                index[brand_key][hf_id].append(entry)

    print(f"  Parsed: {parsed} recipes")
    print(f"  Skipped (min_vllm_version < {MIN_VLLM_VERSION}): {skipped_min_version}")
    print(f"  Skipped (other): {skipped_other}")

    # 4. Apply overlays
    if OVERLAYS_PATH.exists():
        overlays = yaml.safe_load(OVERLAYS_PATH.read_text()) or {}
        print(f"  Applying overlays from {OVERLAYS_PATH}")
        merge_overlays(index, overlays, accelerator_brands)
    else:
        print("  No overlays.yaml found")

    # 5. Assemble output
    output = {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vllm_version": vllm_version,
        "vllm_recipes_sha": recipes_sha,
        "accelerator_brands": accelerator_brands,
        "index": index,
    }

    # 6. Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Wrote {OUTPUT_PATH} ({len(json.dumps(output))} bytes)")

    # Stats
    total_profiles = sum(
        sum(len(profiles) for profiles in acc_map.values())
        for acc_map in index.values()
    )
    print(f"  Total profile entries: {total_profiles}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
