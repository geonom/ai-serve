"""ComfyUI's runtime state in image-serve (moved from ai-serve-ui's backend/state.py)."""
from typing import Any, Dict, Optional

from image_serve.config import COMFY_PORT


class State:
    comfyui_process: Optional[Any] = None
    comfyui_port: int = COMFY_PORT
    comfyui_pid: Optional[int] = None
    comfyui_gpu_id: Optional[int] = None            # launch-time property of the one ComfyUI process
    comfyui_log_file: Optional[Any] = None
    comfyui_progress: Dict[str, Any] = {"step": 0, "max_steps": 0, "node": "", "value": 0}
    comfyui_tasks: Dict[str, Dict[str, Any]] = {}
    comfyui_active_model: Optional[str] = None
    comfyui_loaded_models: Optional[Any] = None     # model files of the last job (same set -> no GPU question)
    comfyui_reservation_id: Optional[str] = None    # GPU-manager reservation for the models being loaded
    comfyui_last_used: float = 0.0
    comfyui_idle_task: Optional[Any] = None
    comfyui_batch: Optional[Dict[str, Any]] = None


state = State()
