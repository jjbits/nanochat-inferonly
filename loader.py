import os
import json
import torch
from huggingface_hub import hf_hub_download

from model import GPT, GPTConfig
from tokenizer import Tokenizer


def _patch_missing_keys(model_data, config):
    """
    Patch missing keys in old checkpoints for backwards compatibility.
    Adds default values for learnable lambdas if they don't exist.
    """
    # Patch learnable lambdas (added in later versions)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(config.n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(config.n_layer)
    return model_data


def get_cache_dir():
    """Get the cache directory for downloaded files."""
    cache_dir = os.environ.get("NANOCHAT_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat-inferonly")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_from_hf(repo_id, filename, cache_subdir=""):
    """Download a file from HuggingFace Hub."""
    cache_dir = get_cache_dir()
    local_dir = os.path.join(cache_dir, cache_subdir) if cache_subdir else cache_dir
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        print(f"Using cached: {local_path}")
        return local_path

    print(f"Downloading {filename} from {repo_id}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded to: {downloaded_path}")
    return downloaded_path


def load_tokenizer(repo_id="karpathy/nanochat-d32"):
    """Load tokenizer from HuggingFace."""
    tokenizer_path = download_from_hf(repo_id, "tokenizer.pkl", cache_subdir="tokenizer")
    return Tokenizer.from_file(tokenizer_path)


def load_model(repo_id="karpathy/nanochat-d32", device=None):
    """
    Load model and tokenizer from HuggingFace.

    Args:
        repo_id: HuggingFace repo (e.g., "karpathy/nanochat-d32")
        device: torch device (auto-detected if None)

    Returns:
        (model, tokenizer, metadata)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Extract model tag from repo_id for caching
    model_tag = repo_id.split("/")[-1]

    # Find the model files - they have step numbers in names
    # First download meta to know the step
    try:
        # Try common step numbers or find meta file
        meta_path = download_from_hf(repo_id, "meta_000650.json", cache_subdir=model_tag)
        model_path = download_from_hf(repo_id, "model_000650.pt", cache_subdir=model_tag)
    except Exception:
        # Fallback: list files and find the right ones
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        meta_file = [f for f in files if f.startswith("meta_") and f.endswith(".json")][0]
        model_file = [f for f in files if f.startswith("model_") and f.endswith(".pt")][0]
        meta_path = download_from_hf(repo_id, meta_file, cache_subdir=model_tag)
        model_path = download_from_hf(repo_id, model_file, cache_subdir=model_tag)

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load model weights
    print(f"Loading model weights from {model_path}...")
    model_data = torch.load(model_path, map_location=device, weights_only=True)

    # Convert bf16 to fp32 for CPU/MPS
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Strip torch.compile prefix if present
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    config = GPTConfig(**meta["model_config"])
    print(f"Model config: {config}")

    # Patch missing keys for backwards compatibility
    model_data = _patch_missing_keys(model_data, config)

    with torch.device("meta"):
        model = GPT(config)

    model.to_empty(device=device)
    model.init_rotary()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer
    tokenizer = load_tokenizer(repo_id)

    # Verify compatibility
    assert tokenizer.get_vocab_size() == config.vocab_size, \
        f"Vocab size mismatch: tokenizer={tokenizer.get_vocab_size()}, model={config.vocab_size}"

    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer, meta


def load_model_from_local(model_path, meta_path, tokenizer_path, device=None):
    """
    Load model from local files (for custom checkpoints).

    Args:
        model_path: Path to model_*.pt file
        meta_path: Path to meta_*.json file
        tokenizer_path: Path to tokenizer.pkl file
        device: torch device

    Returns:
        (model, tokenizer, metadata)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load model weights
    model_data = torch.load(model_path, map_location=device, weights_only=True)

    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    config = GPTConfig(**meta["model_config"])

    # Patch missing keys for backwards compatibility
    model_data = _patch_missing_keys(model_data, config)

    with torch.device("meta"):
        model = GPT(config)

    model.to_empty(device=device)
    model.init_rotary()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    return model, tokenizer, meta
