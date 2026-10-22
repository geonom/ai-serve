"""Model bundles image-serve can download on request (plugin contract endpoint `install`, ai-serve-ui docs/plugins.md).

A catalog model needs several files: the diffusion model, its text encoder, its VAE (and for Qwen-Image-Edit the
Lightning LoRA its workflow uses). Each file is pinned here by Hugging Face repo, path, size and sha256 -- the same
files this plugin was tested with (2026-10-22, sha256 compared with the files on the owner's machine). A download
streams to a temp file next to its target, hashes while writing, and is moved into place only when the hash matches;
nothing from the request but the bundle id is used. All models here: Apache-2.0.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BundleFile:
    bundle: str
    repo: str
    path: str          # in the repo
    dir: str           # under the models root (ComfyUI's unet/, text_encoders/, vae/, loras/...)
    name: str          # saved as
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{urllib.parse.quote(self.path)}"

    def target(self, root: str) -> str:
        return os.path.join(root, self.dir, self.name)


FILES: Tuple[BundleFile, ...] = (
    BundleFile("z-image-turbo", "jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q6_K.gguf", "unet",
               "z_image_turbo-Q6_K.gguf", 5910505536, "fc137d87b49e06fdd5230d67d6c8cfa42a9e1fd38b65ccd355882450c3eb1c82"),
    BundleFile("z-image-turbo", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-UD-Q6_K_XL.gguf", "text_encoders",
               "Qwen3-4B-UD-Q6_K_XL.gguf", 3658223392, "ac0767b5e9c9f16efe57ce422253a33747970c166c3131c4d4d59d20511f07e1"),
    BundleFile("z-image-turbo", "Tongyi-MAI/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors", "vae",
               "z-image-turbo-vae.safetensors", 167666902, "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
    BundleFile("qwen-edit-2511", "unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q2_K.gguf", "unet",
               "qwen-image-edit-2511-Q2_K.gguf", 7468022368, "a3d09042b64657970654941aa08d895de29b4d98edf3632a89e70d4d6e23c47c"),
    BundleFile("qwen-edit-2511", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders",
               "qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680, "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"),
    BundleFile("qwen-edit-2511", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae",
               "qwen_image_vae.safetensors", 253806246, "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
    BundleFile("qwen-edit-2511", "lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "loras/Qwen Image",
               "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 849608296, "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"),
    BundleFile("ernie-turbo", "unsloth/ERNIE-Image-Turbo-GGUF", "ernie-image-turbo-Q6_K.gguf", "unet",
               "ernie-image-turbo-Q6_K.gguf", 6790267584, "ce2986e3ae22296efce52514bc01401138742bc3594ab16ad269a3b4d92ab4f7"),
    BundleFile("ernie-turbo", "Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors", "text_encoders",
               "ministral-3-3b.safetensors", 7717637511, "49a750a128863854eac7d85e1a277a7b44bf6ec3646405b84686dfeeca3708ca"),
    BundleFile("ernie-turbo", "baidu/ERNIE-Image-Turbo", "vae/diffusion_pytorch_model.safetensors", "vae",
               "vae_ernie_turbo.safetensors", 168120878, "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"),
)

BUNDLES: Dict[str, Dict[str, str]] = {
    "z-image-turbo": {"label": "Z-Image Turbo", "license": "Apache-2.0",
                      "description": "Fast, photorealistic images from text -- the default image model"},
    "qwen-edit-2511": {"label": "Qwen-Image-Edit 2511", "license": "Apache-2.0",
                       "description": "Edit images by instruction and transfer a style (with the 4-step Lightning LoRA)"},
    "ernie-turbo": {"label": "ERNIE-Image Turbo", "license": "Apache-2.0",
                    "description": "Fast images from text, a bit more illustrative"},
}


def files_of(bundle: str) -> List[BundleFile]:
    return [f for f in FILES if f.bundle == bundle]


def size_mb(bundle: str) -> int:
    return sum(f.size for f in files_of(bundle)) >> 20


def present(bundle: str, root: str) -> bool:
    """Every file of the bundle is there with its size (also files that were already on disk, e.g. moved in by hand)."""
    fs = files_of(bundle)
    return bool(fs) and all(os.path.isfile(f.target(root)) and os.path.getsize(f.target(root)) == f.size for f in fs)


def download(bundle: str, root: str, progress: Optional[Callable[[int], None]] = None,
             opener: Callable[[str], object] = None) -> None:
    """Fetch the missing files of a bundle, each checked against its pinned sha256. Raises on any mismatch -- the
    bad temp file is removed, nothing half-done is left under the real name."""
    opener = opener or (lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "image-serve"}), timeout=60))
    for f in files_of(bundle):
        dest = f.target(root)
        if os.path.isfile(dest) and os.path.getsize(dest) == f.size:
            if progress:
                progress(f.size)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        h = hashlib.sha256()
        with opener(f.url) as r, tempfile.NamedTemporaryFile(dir=os.path.dirname(dest), delete=False, prefix=".part-") as out:
            tmp = out.name
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                if progress:
                    progress(len(chunk))
        if h.hexdigest() != f.sha256:
            os.unlink(tmp)
            raise ValueError(f"{f.name}: checksum mismatch -- not installed")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)


def main(argv=None) -> int:
    """python -m image_serve.bundles <bundle> ... : download bundles into the models root (setup.sh --models)."""
    import sys
    from image_serve.config import COMFYUI_MODELS_ROOT
    names = list(argv if argv is not None else sys.argv[1:])
    unknown = [n for n in names if n not in BUNDLES]
    if unknown or not names:
        print(f"bundles: {', '.join(BUNDLES)}", file=sys.stderr)
        return 2
    for n in names:
        total, done = size_mb(n), [0]

        def progress(k: int) -> None:
            done[0] += k
            print(f"\r  {BUNDLES[n]['label']}: {done[0] >> 20} / {total} MB", end="", flush=True)
        download(n, COMFYUI_MODELS_ROOT, progress)
        print(f"\r  {BUNDLES[n]['label']}: ok ({total} MB)          ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
