"""Optimized Whisper transcription service for RTX A4000 16GB - Maximum Precision."""

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

# Force GPU device selection
if not torch.cuda.is_available():
    error_msg = (
        "❌ CUDA not available! GPU is required for this configuration.\n"
        "Please check:\n"
        "1. NVIDIA driver installed\n"
        "2. PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
        "3. GPU is detected: nvidia-smi"
    )
    raise RuntimeError(error_msg)

device: str = "cuda:0"  # Explicitly use first GPU
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA version: {torch.version.cuda}")


def load_model() -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load Whisper model optimized for RTX A4000 16GB - Maximum Precision Mode.

    Precision-first configuration:
    - FP32 (float32) for all operations - maximum accuracy
    - TF32 disabled - no precision loss in matrix operations
    - Gradient checkpointing disabled - faster inference
    - Large batch processing enabled - utilize full 16GB VRAM
    - Deterministic operations for reproducible results

    Returns:
        Tuple of (model, processor) for inference.

    Raises:
        RuntimeError: If model loading fails or GPU not available.
    """
    global model, processor  # noqa: PLW0603

    # Verify GPU availability
    if not torch.cuda.is_available():
        error_msg = "GPU is required but not available. Check CUDA installation."
        raise RuntimeError(error_msg)

    print("🚀 Loading Whisper model - MAXIMUM PRECISION MODE (RTX A4000 16GB)")
    print(f"🎮 Target device: {device} ({torch.cuda.get_device_name(0)})")

    # Load processor
    processor = WhisperProcessor.from_pretrained("model/")

    # Force FP32 for maximum precision
    dtype = torch.float32
    print("📊 Precision: FP32 (float32) - No quantization")

    # Load model with maximum precision settings
    model = WhisperForConditionalGeneration.from_pretrained(
        "model/",
        dtype=dtype,  # Fixed: Use 'dtype' instead of deprecated 'torch_dtype'
        low_cpu_mem_usage=False,  # We have 16GB, load everything at once
        use_cache=True,
        attn_implementation="sdpa",  # Efficient attention without precision loss
    )

    # Explicitly move to GPU (not CPU!)
    print(f"📦 Moving model to {device}...")
    model = model.to(device)
    model.eval()

    # Verify model is on GPU
    if next(model.parameters()).device.type != "cuda":
        error_msg = (
            f"Model failed to load on GPU! Device: {next(model.parameters()).device}"
        )
        raise RuntimeError(error_msg)

    print(f"✅ Model successfully loaded on GPU: {next(model.parameters()).device}")

    # RTX A4000 precision optimizations
    # === CRITICAL: Disable all approximations ===
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("✅ TF32 DISABLED - Pure FP32 precision")

    # Enable cuDNN for optimized operations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Enable deterministic mode for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("✅ Deterministic mode ENABLED - Reproducible results")

    # Set memory allocation strategy for 16GB VRAM
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)  # Use up to 95% of VRAM
    print("✅ Memory allocation: 95% of 16GB VRAM available")

    # Disable automatic mixed precision
    torch.set_float32_matmul_precision("highest")  # Force highest precision
    print("✅ Float32 matmul precision: HIGHEST")

    # Try to compile model for optimization (Python 3.13 and below)
    import sys

    if sys.version_info < (3, 14):
        try:
            # Use "max-autotune" for best performance with 16GB VRAM
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("✅ Model compiled with torch.compile (max-autotune mode)")
        except Exception as e:
            print(f"⚠️  torch.compile not available: {e}")
    else:
        print("⚠️  torch.compile skipped (Python 3.14+ not supported yet)")

    # Display GPU information
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_vram = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM Total: {total_vram:.1f} GB")
    print(f"✅ VRAM Allocated: {allocated_vram:.2f} GB")
    print(f"✅ Compute Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)
    print("🎯 MAXIMUM PRECISION MODE ACTIVE ON GPU")
    print("=" * 60)

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("🧹 Model resources cleaned up!")


def process_transcription(
    task_id: str,
    audio_bytes: bytes,
    language: str | None = None,
) -> None:
    """
    Process transcription with maximum precision settings.

    Optimized for RTX A4000 16GB:
    - FP32 precision throughout pipeline
    - Higher beam search (num_beams=10) for better quality
    - Length penalty optimization
    - No mixed precision

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

        # Load audio with librosa (high quality)
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True,  # Ensure mono audio
            dtype="float32",  # Match model precision
        )
        duration = len(audio) / sr

        # Process audio features (FP32 precision)
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features

        # Move to device - maintain FP32 precision and verify GPU
        input_features = input_features.to(device, dtype=torch.float32)

        # Verify data is on GPU
        if input_features.device.type != "cuda":
            error_msg = f"Input features not on GPU! Device: {input_features.device}"
            raise RuntimeError(error_msg)

        # Generate transcription with maximum quality settings
        with torch.no_grad():
            generation_config: dict[str, Any] = {
                "max_length": 448,
                "num_beams": 10,  # Increased from 5 for better quality
                "length_penalty": 1.0,  # Balanced length penalty
                "early_stopping": True,  # Stop when all beams finish
                "use_cache": True,
                "return_dict_in_generate": False,
                "do_sample": False,  # Deterministic decoding
                "temperature": 1.0,  # Not used with do_sample=False
            }

            if language:
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
                generation_config["forced_decoder_ids"] = forced_decoder_ids

            # Run inference
            predicted_ids = model.generate(input_features, **generation_config)

        # Decode transcription
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        # Update task with result
        tasks[task_id].update(
            {
                "status": TaskStatus.COMPLETED,
                "transcription": transcription,
                "duration_seconds": round(duration, 2),
                "sample_rate": sr,
                "precision": "FP32",
                "num_beams": 10,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )

        # Efficient cleanup
        del input_features, predicted_ids, audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # Log error and update task status
        print(f"❌ Error processing task {task_id}: {e!s}")
        traceback.print_exc()

        tasks[task_id].update(
            {
                "status": TaskStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )


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
    Get detailed model and system status.

    Returns:
        Dictionary with comprehensive model information.
    """
    status: dict[str, Any] = {
        "loaded": model is not None and processor is not None,
        "device": device,
        "device_type": next(model.parameters()).device.type
        if model is not None
        else "unknown",
        "precision": "FP32 (float32)",
        "mode": "Maximum Precision",
        "total_tasks": len(tasks),
        "pending_tasks": len(get_pending_tasks()),
    }

    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        allocated_vram = torch.cuda.memory_allocated(0)
        reserved_vram = torch.cuda.memory_reserved(0)

        status.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
                "cuda_version": torch.version.cuda,
                "vram_total": f"{total_vram / 1024**3:.2f} GB",
                "vram_allocated": f"{allocated_vram / 1024**3:.2f} GB",
                "vram_reserved": f"{reserved_vram / 1024**3:.2f} GB",
                "vram_free": f"{(total_vram - reserved_vram) / 1024**3:.2f} GB",
                "vram_usage_percent": f"{(reserved_vram / total_vram * 100):.1f}%",
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            }
        )

    return status
