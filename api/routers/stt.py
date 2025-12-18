import uuid
from typing import Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from api.config import Config, TaskStatus
from api.routers import logic as transcription_service

stt_router = APIRouter(prefix="/api", tags=["Speech-to-Text"])


@stt_router.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lang: Literal["ru", "uz"] = "uz",
):
    """
    Submit audio file for transcription (returns immediately).
    Processing happens in background with pre-loaded model.

    Args:
        background_tasks: FastAPI background tasks manager
        file: Audio file to transcribe
        lang: Language of the audio ("ru" or "uz")

    Returns:
        dict: Task information with task_id for status checking

    Raises:
        HTTPException: If file type is not supported or processing fails
    """
    try:
        # Validate file type
        if file.content_type not in Config.ALLOWED_TYPES.value:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(Config.ALLOWED_TYPES.value)}",
            )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Read audio bytes
        audio_bytes = await file.read()

        # Create task entry using service
        transcription_service.create_task(task_id=task_id, language=lang)

        # Add to background tasks (processes immediately since model is ready)
        background_tasks.add_task(
            transcription_service.process_transcription, task_id, audio_bytes, lang
        )

        # Return immediately with task ID
        return {
            "success": True,
            "message": "Audio submitted for transcription",
            "task_id": task_id,
            "status": TaskStatus.PENDING.value,
            "check_status_url": f"/api/status/{task_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@stt_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check transcription status by task ID.

    Args:
        task_id: Unique task identifier

    Returns:
        dict: Task information including status and transcription (if completed)

    Raises:
        HTTPException: If task not found
    """
    task = transcription_service.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404, detail=f"Task with ID '{task_id}' not found"
        )

    return task


@stt_router.get("/tasks")
async def list_tasks(limit: int = 50):
    """
    List all tasks in order of submission.

    Args:
        limit: Maximum number of tasks to return (default: 50)

    Returns:
        dict: Total count and list of tasks
    """
    task_list = transcription_service.get_all_tasks(limit=limit)

    return {
        "total": len(transcription_service.tasks),
        "showing": len(task_list),
        "tasks": task_list,
    }


@stt_router.get("/tasks/pending")
async def list_pending_tasks():
    """
    List all pending and processing tasks in order.

    Returns:
        dict: Count and list of active tasks
    """
    pending_tasks = transcription_service.get_pending_tasks()

    return {"count": len(pending_tasks), "tasks": pending_tasks}


@stt_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a specific task from history.

    Args:
        task_id: Task identifier to delete

    Returns:
        dict: Success message

    Raises:
        HTTPException: If task not found
    """
    success = transcription_service.delete_task(task_id)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Task with ID '{task_id}' not found"
        )

    return {"success": True, "message": f"Task '{task_id}' deleted successfully"}


@stt_router.delete("/tasks")
async def clear_all_tasks():
    """
    Clear all completed and failed tasks.
    Keeps pending and processing tasks intact.

    Returns:
        dict: Number of tasks cleared and remaining
    """
    cleared_count = transcription_service.clear_completed_tasks()
    remaining = len(transcription_service.tasks)

    return {
        "success": True,
        "message": f"Cleared {cleared_count} completed/failed tasks",
        "cleared": cleared_count,
        "remaining": remaining,
    }


@stt_router.get("/health")
async def health_check():
    """
    Check API and model health status.

    Returns:
        dict: Model status and system information
    """
    status = transcription_service.get_model_status()

    return {
        "status": "online",
        "model_loaded": status["loaded"],
        "device": status["device"],
        "total_tasks": status["total_tasks"],
        "pending_tasks": status["pending_tasks"],
    }
