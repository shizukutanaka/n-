from __future__ import annotations

import subprocess
from pathlib import Path

from nmesh.planner import PlannedService


def acquire(service: PlannedService) -> Path | None:
    if service.backend == "ollama":
        subprocess.run(["ollama", "pull", service.model_ref], check=True)
        return None
    if service.backend in {"vllm", "mlx"}:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=service.model_ref, filename="config.json"))
    if service.backend == "llamacpp" and not Path(service.model_ref).exists():
        from huggingface_hub import hf_hub_download

        repo_id = service.launch.env.get("NMESH_HF_REPO")
        if repo_id is None:
            raise RuntimeError("No Hugging Face GGUF repository configured")
        filename = Path(service.model_ref).name
        return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    local_dir=str(Path(service.model_ref).parent)))
    return Path(service.model_ref)
