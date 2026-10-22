"""Which sources a node may download a model from -- pure. Rules and reasons:
tests/unit/test_cluster_models.py. The agent measures the offered sources, uses the
fastest and falls back when a source delivers bytes with the wrong sha256."""
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILE_RE = re.compile(r"^([A-Za-z0-9._-]+/)*[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class ModelPlanError(ValueError):
    pass


@dataclass(frozen=True)
class HfRef:
    repo: str
    file: str
    revision: str

    @classmethod
    def parse(cls, repo: Any, file: Any, revision: Any) -> "HfRef":
        revision = revision or "main"
        if not isinstance(repo, str) or not _REPO_RE.match(repo):
            raise ModelPlanError("hf_repo must be owner/name")
        if not isinstance(file, str) or not _FILE_RE.match(file) or ".." in file.split("/"):
            raise ModelPlanError("hf_file must be a .gguf path inside the repo")
        if not isinstance(revision, str) or not _REV_RE.match(revision):
            raise ModelPlanError("invalid revision")
        return cls(repo, file, revision)

    @property
    def name(self) -> str:
        return self.file.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


def plan_model(coordinator_url: Optional[str], local: Optional[Tuple[str, int, str]],
               hf: Optional[HfRef], hf_meta: Optional[Tuple[int, str]]) -> Dict[str, Any]:
    """local = (name, size, sha256) of a file here; hf_meta = (size, sha256) from HF."""
    if hf and hf_meta:
        size, sha = hf_meta
        sources = []
        if local and coordinator_url and local[0] == hf.name and local[1] == size:
            sources.append(f"{coordinator_url}/api/cluster/models/{hf.name}")
        sources.append(hf.url)
        return {"name": hf.name, "sha256": sha, "size": size, "sources": sources}
    if local:
        if not coordinator_url:
            raise ModelPlanError("this node doesn't know the coordinator's address -- re-enroll it")
        name, size, sha = local
        return {"name": name, "sha256": sha, "size": size,
                "sources": [f"{coordinator_url}/api/cluster/models/{name}"]}
    raise ModelPlanError("neither a local model nor a Hugging Face file given")
