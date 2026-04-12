import os

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _print_load_report(loaded_params: set[str], skipped_params: list[tuple[str, str]]):
    loaded = sorted(loaded_params)
    skipped = sorted(skipped_params, key=lambda item: item[0])
    total = len(loaded) + len(skipped)
    loaded_count = len(loaded)
    skipped_count = len(skipped)
    success_rate = (loaded_count / total * 100.0) if total else 0.0

    line = "=" * 88
    print(line)
    print("Weight Loading Report".center(88))
    print(line)
    print(f"Loaded  : {loaded_count}/{total}  ({success_rate:.2f}%)")
    print(f"Skipped : {skipped_count}/{total}")
    print(line)

    if loaded:
        print("Loaded Parameters:")
        for name in loaded:
            print(f"  [+] {name}")
    else:
        print("Loaded Parameters: none")

    print(line)

    if skipped:
        print("Skipped Parameters:")
        for name, reason in skipped:
            print(f"  [-] {name}")
            print(f"      reason: {reason}")
    else:
        print("Skipped Parameters: none")

    print(line)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    checkpoint_path = None

    if model_name_or_path.startswith("~"):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    if checkpoint_path is None or not os.path.exists(model_name_or_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    safetensor_files = [
        f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
    ]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    hf_weights: dict[str, torch.Tensor] = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    loaded_params: set[str] = set()
    skipped_params: list[tuple[str, str]] = []

    for hf_name, hf_weight in hf_weights.items():
        try:
            param = model.get_parameter(hf_name)
            param.data.copy_(hf_weight)
            loaded_params.add(hf_name)
        except AttributeError:
            skipped_params.append((hf_name, "Parameter not found"))
        except RuntimeError as e:
            skipped_params.append((hf_name, f"Copy failed: {e}"))

    _print_load_report(loaded_params, skipped_params)

    return {
        "loaded_params": sorted(loaded_params),
        "skipped_params": skipped_params,
    }
