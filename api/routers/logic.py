"""Optimized Whisper transcription service with type safety and linting compliance."""

from __future__ import annotations

import gc
import io
import traceback
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from api.config import TaskStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

# Global storage
tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
model: WhisperForConditionalGeneration | None = None
processor: WhisperProcessor | None = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load Whisper model into memory with optimal performance settings.

    Should be called once when application starts.

    Returns:
        Tuple of (model, processor) for inference.

    Raises:
        RuntimeError: If model loading fails.
    """
    global model, processor  # noqa: PLW0603

    print("🚀 Loading Whisper model into memory...")

    # Load processor
    processor = WhisperProcessor.from_pretrained("model/")

    # Determine optimal dtype
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load model with performance optimizations
    model = WhisperForConditionalGeneration.from_pretrained(
        "model/",
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_cache=True,
        attn_implementation="sdpa",  # Faster attention mechanism
    )

    model.to(device)
    model.eval()

    # GPU-specific optimizations
    if device == "cuda":
        # Enable cuDNN auto-tuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Enable TF32 for Ampere+ GPUs (30xx, 40xx, A100, etc.)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ TF32 enabled for faster matmul")

        # Try to compile model for additional speedup (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compiled with torch.compile")
        except Exception:  # noqa: S110
            pass  # Silently skip if not available

        print(f"✅ Model loaded on GPU (FP16) - {torch.cuda.get_device_name()}")
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        print(f"✅ Model loaded on CPU (FP32) - {torch.get_num_threads()} threads")

    return model, processor


def cleanup_model() -> None:
    """Clean up model resources on shutdown."""
    global model, processor  # noqa: PLW0603

    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print("🧹 Model resources cleaned up!")


def process_transcription(
    task_id: str,
    audio_bytes: bytes,
    language: str | None = None,
) -> None:
    """
    Background task for transcription processing with optimized inference.

    Args:
        task_id: Unique identifier for this transcription task.
        audio_bytes: Raw audio file bytes.
        language: Optional language code (e.g., 'en', 'ru', 'uz').

    Raises:
        RuntimeError: If model is not loaded.
    """
    global model, processor, tasks  # noqa: PLW0602

    if model is None or processor is None:
        error_msg = "Model not loaded. Call load_model() first."
        raise RuntimeError(error_msg)

    try:
        # Update status to processing
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["started_at"] = datetime.now(UTC).isoformat()

        # Load audio with librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        duration = len(audio) / sr

        # Process audio features
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features

        # Move to device and convert dtype
        input_features = input_features.to(device)
        if device == "cuda":
            input_features = input_features.half()  # FP16 for GPU

        # Generate transcription with optimized settings
        with torch.no_grad():
            generation_config: dict[str, Any] = {
                "max_length": 448,
                "num_beams": 5,
                "use_cache": True,
                "return_dict_in_generate": False,
            }

            if language:
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
                generation_config["forced_decoder_ids"] = forced_decoder_ids

            predicted_ids = model.generate(input_features, **generation_config)

        # Decode transcription
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        # Update task with result
        tasks[task_id].update({
            "status": TaskStatus.COMPLETED,
            "transcription": transcription,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "completed_at": datetime.now(UTC).isoformat(),
        })

        # Efficient cleanup - only clear tensors
        del input_features, predicted_ids, audio
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        # Log error and update task status
        print(f"❌ Error processing task {task_id}: {e!s}")
        traceback.print_exc()

        tasks[task_id].update({
            "status": TaskStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now(UTC).isoformat(),
        })


def get_task(task_id: str) -> dict[str, Any] | None:
    """
    Retrieve task by ID.

    Args:
        task_id: Task identifier.

    Returns:
        Task dictionary or None if not found.
    """
    return tasks.get(task_id)


def create_task(task_id: str, language: str | None = None) -> dict[str, Any]:
    """
    Create a new task entry.

    Args:
        task_id: Unique task identifier.
        language: Optional language code.

    Returns:
        Created task dictionary.
    """
    task: dict[str, Any] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "language": language or "auto",
        "submitted_at": datetime.now(UTC).isoformat(),
        "started_at": None,
        "completed_at": None,
    }

    tasks[task_id] = task
    return task


def get_all_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Get all tasks in order of submission.

    Args:
        limit: Maximum number of tasks to return (most recent).

    Returns:
        List of task dictionaries.
    """
    task_list = list(tasks.values())

    if limit:
        return task_list[-limit:]

    return task_list


def get_pending_tasks() -> list[dict[str, Any]]:
    """
    Get all pending and processing tasks.

    Returns:
        List of active task dictionaries.
    """
    return [
        task
        for task in tasks.values()
        if task["status"] in {TaskStatus.PENDING, TaskStatus.PROCESSING}
    ]


def delete_task(task_id: str) -> bool:
    """
    Delete a task from storage.

    Args:
        task_id: Task identifier.

    Returns:
        True if deleted, False if not found.
    """
    if task_id in tasks:
        del tasks[task_id]
        return True
    return False


def clear_completed_tasks() -> int:
    """
    Clear all completed and failed tasks, keep active ones.

    Returns:
        Number of tasks cleared.
    """
    global tasks  # noqa: PLW0603

    active_tasks = OrderedDict(
        (k, v)
        for k, v in tasks.items()
        if v["status"] in {TaskStatus.PENDING, TaskStatus.PROCESSING}
    )

    cleared_count = len(tasks) - len(active_tasks)
    tasks = active_tasks

    return cleared_count


def get_model_status() -> dict[str, Any]:
    """
    Get current model status.

    Returns:
        Dictionary with model information.
    """
    status: dict[str, Any] = {
        "loaded": model is not None and processor is not None,
        "device": device,
        "total_tasks": len(tasks),
        "pending_tasks": len(get_pending_tasks()),
    }

    if device == "cuda" and torch.cuda.is_available():
        status["gpu_name"] = torch.cuda.get_device_name()
        status["gpu_memory_allocated"] = (
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        status["gpu_memory_reserved"] = (
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    return status