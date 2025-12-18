# websocket_routes.py - WebSocket handlers for live STT integrated with your logic
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, Literal
import json
import numpy as np
import uuid
import asyncio
from datetime import datetime
from collections import deque

# Import your existing logic module
from api.routers import logic as transcription_service

websocket_router = APIRouter(prefix="/live", tags=["Live STT"], deprecated=True)


# In-memory storage for active sessions
active_sessions = {}


class AudioBuffer:
    """
    Manages audio buffering with overlap for continuous transcription.
    Uses sliding window to avoid cutting words.
    """

    def __init__(self, sample_rate=16000, chunk_duration=2.0, overlap=0.3):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap)
        self.buffer = deque(maxlen=self.chunk_size * 3)

    def add_audio(self, audio_data: np.ndarray):
        """Add audio samples to buffer"""
        self.buffer.extend(audio_data)

    def get_chunk(self) -> Optional[np.ndarray]:
        """Get chunk if enough data available, maintaining overlap"""
        if len(self.buffer) >= self.chunk_size:
            chunk = np.array(list(self.buffer)[: self.chunk_size], dtype=np.float32)

            # Remove processed data but keep overlap
            drain_size = self.chunk_size - self.overlap_size
            for _ in range(drain_size):
                if len(self.buffer) > self.overlap_size:
                    self.buffer.popleft()

            return chunk
        return None

    def get_remaining(self) -> Optional[np.ndarray]:
        """Get remaining buffer content for final transcription"""
        if len(self.buffer) > self.sample_rate * 0.5:  # At least 0.5s
            return np.array(list(self.buffer), dtype=np.float32)
        return None

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()


class LiveSession:
    """Represents an active live transcription session"""

    def __init__(self, session_id: str, language: Optional[str] = None):
        self.session_id = session_id
        self.language = language or "uz"  # Default to Uzbek like your batch API
        self.created_at = datetime.now()
        self.audio_buffer = AudioBuffer(
            sample_rate=16000, chunk_duration=2.0, overlap=0.3
        )
        self.full_transcript = []
        self.total_audio_duration = 0.0
        self.chunk_count = 0
        self.is_processing = False

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "total_audio_duration": round(self.total_audio_duration, 2),
            "chunk_count": self.chunk_count,
            "transcript_length": len(" ".join(self.full_transcript)),
            "is_processing": self.is_processing,
        }


async def process_audio_chunk_async(session: LiveSession, audio_chunk: np.ndarray):
    """
    Process audio chunk using your existing transcription service.
    Runs in thread pool to avoid blocking the event loop.
    """
    import torch
    import io

    session.is_processing = True

    try:
        # Get your loaded model and processor
        model = transcription_service.model
        processor = transcription_service.processor

        if model is None or processor is None:
            return {"error": "Model not loaded"}

        # Process audio with your existing processor
        input_features = processor(
            audio_chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features

        # Move to device (float32, no conversion needed)
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate transcription (use beam_size=1 for lower latency in real-time)
        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=session.language, task="transcribe"
            )
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=1,  # Lower beam size for faster real-time inference
            )

        # Decode transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]

        # Cleanup
        del input_features, predicted_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"text": transcription.strip()}

    except Exception as e:
        return {"error": str(e)}
    finally:
        session.is_processing = False


@websocket_router.websocket("/transcribe")
async def live_transcribe(
    websocket: WebSocket,
    lang: Literal["ru", "uz"] = Query("uz", description="Language: ru or uz"),
):
    """
    WebSocket endpoint for live speech-to-text transcription.
    Uses your existing Whisper model loaded in memory.

    Protocol:
    1. Client connects and receives welcome message
    2. Client sends binary audio chunks (PCM 16-bit, mono, 16kHz)
    3. Server sends partial transcription results as JSON
    4. Client sends {"command": "stop"} to finalize and get full transcript

    Example JavaScript client:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/live/transcribe?lang=uz');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'partial') {
            console.log('Transcription:', data.text);
        }
    };

    // Send audio chunks
    ws.send(audioChunkBuffer);

    // Stop and get final result
    ws.send(JSON.stringify({command: 'stop'}));
    ```
    """
    await websocket.accept()

    # Check if model is loaded
    if transcription_service.model is None or transcription_service.processor is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Model not loaded. Please wait for model initialization.",
            }
        )
        await websocket.close()
        return

    # Create session
    session_id = str(uuid.uuid4())
    session = LiveSession(session_id, lang)
    active_sessions[session_id] = session

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "language": lang,
                "message": "Connected! Send audio chunks (PCM 16-bit, mono, 16kHz)",
                "model_device": "cuda"
                if transcription_service.model.device.type == "cuda"
                else "cpu",
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            # Receive data from client
            data = await websocket.receive()

            # Handle binary audio data
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # Convert PCM 16-bit bytes to float32 numpy array
                audio_np = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # Update session stats
                chunk_duration = len(audio_np) / 16000
                session.total_audio_duration += chunk_duration
                session.chunk_count += 1

                # Add to buffer
                session.audio_buffer.add_audio(audio_np)

                # Process if we have enough data
                audio_chunk = session.audio_buffer.get_chunk()
                if audio_chunk is not None and not session.is_processing:
                    # Process asynchronously to not block WebSocket
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: asyncio.run(
                            process_audio_chunk_async(session, audio_chunk)
                        ),
                    )

                    if "error" in result:
                        await websocket.send_json(
                            {"type": "error", "message": result["error"]}
                        )
                    elif result.get("text"):
                        # Add to full transcript
                        session.full_transcript.append(result["text"])

                        # Send partial result
                        await websocket.send_json(
                            {
                                "type": "partial",
                                "text": result["text"],
                                "session_id": session_id,
                                "chunk_number": session.chunk_count,
                                "is_final": False,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            # Handle JSON commands
            elif "text" in data:
                try:
                    command = json.loads(data["text"])
                    cmd_type = command.get("command", "").lower()

                    if cmd_type == "stop" or cmd_type == "finalize":
                        # Process any remaining audio
                        remaining = session.audio_buffer.get_remaining()
                        if remaining is not None:
                            result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: asyncio.run(
                                    process_audio_chunk_async(session, remaining)
                                ),
                            )
                            if result.get("text"):
                                session.full_transcript.append(result["text"])

                        # Send final transcript
                        final_text = " ".join(session.full_transcript)
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": final_text,
                                "session_id": session_id,
                                "total_duration_s": round(
                                    session.total_audio_duration, 2
                                ),
                                "total_chunks": session.chunk_count,
                                "language": session.language,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        break

                    elif cmd_type == "ping":
                        await websocket.send_json(
                            {
                                "type": "pong",
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                    elif cmd_type == "status":
                        await websocket.send_json(
                            {
                                "type": "status",
                                **session.to_dict(),
                                "full_transcript": " ".join(session.full_transcript),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                    elif cmd_type == "clear":
                        session.full_transcript.clear()
                        await websocket.send_json(
                            {
                                "type": "cleared",
                                "message": "Transcript cleared",
                                "session_id": session_id,
                            }
                        )

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON command"}
                    )

    except WebSocketDisconnect:
        print(f"📱 Client disconnected: {session_id}")

    except Exception as e:
        print(f"❌ WebSocket error in session {session_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

    finally:
        # Cleanup session
        if session_id in active_sessions:
            del active_sessions[session_id]

        try:
            await websocket.close()
        except:
            pass

        print(f"🧹 Session cleaned up: {session_id}")


@websocket_router.get("/sessions")
async def get_active_sessions():
    """
    Get list of all active live transcription sessions.
    """
    sessions = [session.to_dict() for session in active_sessions.values()]

    return {
        "active_sessions": len(sessions),
        "sessions": sessions,
        "timestamp": datetime.now().isoformat(),
    }


@websocket_router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """
    Get detailed information about a specific session.
    """
    session = active_sessions.get(session_id)

    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found", "session_id": session_id},
        )

    return {
        **session.to_dict(),
        "transcript": " ".join(session.full_transcript),
        "timestamp": datetime.now().isoformat(),
    }


@websocket_router.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """
    Force close a session (cleanup orphaned sessions).
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"success": True, "message": "Session closed", "session_id": session_id}

    return JSONResponse(
        status_code=404,
        content={"error": "Session not found", "session_id": session_id},
    )


@websocket_router.get("/health")
async def live_health_check():
    """
    Health check for live transcription service.
    """
    model_status = transcription_service.get_model_status()

    return {
        "status": "online" if model_status["loaded"] else "waiting",
        "service": "live-stt",
        "model_loaded": model_status["loaded"],
        "device": model_status["device"],
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
    }


@websocket_router.get("/test-client", response_class=HTMLResponse)
async def get_test_client():
    """
    Simple HTML test client for live transcription.
    Access at: http://localhost:8000/live/test-client
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live STT Test Client</title>
        <meta charset="UTF-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 32px; margin-bottom: 10px; }
            .header p { opacity: 0.9; }
            .controls {
                padding: 30px;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
            .button-group {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: 600;
            }
            button:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            #startBtn { background: #28a745; color: white; }
            #stopBtn { background: #dc3545; color: white; }
            #clearBtn { background: #6c757d; color: white; }
            select {
                padding: 12px;
                font-size: 16px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background: white;
                cursor: pointer;
            }
            #status {
                padding: 15px;
                background: #e9ecef;
                border-radius: 8px;
                border-left: 4px solid #6c757d;
                font-weight: 600;
            }
            #status.connected { background: #d4edda; border-color: #28a745; color: #155724; }
            #status.error { background: #f8d7da; border-color: #dc3545; color: #721c24; }
            .transcription-area {
                padding: 30px;
            }
            .transcription-area h3 {
                margin-bottom: 15px;
                color: #495057;
                font-size: 20px;
            }
            #transcription {
                background: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 20px;
                min-height: 300px;
                max-height: 500px;
                overflow-y: auto;
                font-size: 16px;
                line-height: 1.6;
            }
            .transcript-item {
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 6px;
                animation: slideIn 0.3s ease;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(-20px); }
                to { opacity: 1; transform: translateX(0); }
            }
            .partial {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                color: #856404;
            }
            .final {
                background: #d4edda;
                border-left: 4px solid #28a745;
                color: #155724;
                font-weight: 600;
            }
            .error {
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                color: #721c24;
            }
            .timestamp {
                font-size: 12px;
                opacity: 0.7;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Live Speech-to-Text</h1>
                <p>Real-time transcription powered by Whisper</p>
            </div>
            
            <div class="controls">
                <div class="button-group">
                    <button id="startBtn">▶ Start Recording</button>
                    <button id="stopBtn" disabled>⏹ Stop Recording</button>
                    <button id="clearBtn">🗑 Clear</button>
                    <select id="languageSelect">
                        <option value="uz">🇺🇿 Uzbek</option>
                        <option value="ru">🇷🇺 Russian</option>
                    </select>
                </div>
                <div id="status">Status: Not connected</div>
            </div>
            
            <div class="transcription-area">
                <h3>Transcription:</h3>
                <div id="transcription"></div>
            </div>
        </div>
        
        <script>
            let websocket = null;
            let audioContext = null;
            let processor = null;
            let sessionId = null;
            
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const clearBtn = document.getElementById('clearBtn');
            const languageSelect = document.getElementById('languageSelect');
            const statusDiv = document.getElementById('status');
            const transcriptionDiv = document.getElementById('transcription');
            
            function addTranscript(type, text, timestamp) {
                const div = document.createElement('div');
                div.className = 'transcript-item ' + type;
                const time = new Date(timestamp).toLocaleTimeString();
                const label = type.toUpperCase();
                div.innerHTML = '<strong>[' + label + ']</strong> ' + text + '<span class="timestamp">' + time + '</span>';
                transcriptionDiv.appendChild(div);
                transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
            }
            
            startBtn.onclick = async () => {
                try {
                    // Check if mediaDevices is supported
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        alert('Your browser does not support microphone access');
                        return;
                    }
                    
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        }
                    });
                    
                    const language = languageSelect.value;
                    // Automatically use wss:// if page is https://, otherwise ws://
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = wsProtocol + '//' + window.location.host + '/live/transcribe?lang=' + language;
                    
                    console.log('Connecting to:', wsUrl);
                    websocket = new WebSocket(wsUrl);
                    
                    websocket.onopen = () => {
                        statusDiv.textContent = 'Status: Connecting...';
                        statusDiv.className = '';
                    };
                    
                    websocket.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'connected') {
                            sessionId = data.session_id;
                            statusDiv.textContent = 'Status: Recording on ' + data.model_device.toUpperCase();
                            statusDiv.className = 'connected';
                        }
                        else if (data.type === 'partial') {
                            addTranscript('partial', data.text, data.timestamp);
                        }
                        else if (data.type === 'final') {
                            addTranscript('final', data.text, data.timestamp);
                            statusDiv.textContent = 'Status: Completed (' + data.total_duration_s + 's, ' + data.total_chunks + ' chunks)';
                        }
                        else if (data.type === 'error') {
                            addTranscript('error', data.message, new Date().toISOString());
                            statusDiv.textContent = 'Status: Error';
                            statusDiv.className = 'error';
                        }
                    };
                    
                    websocket.onerror = () => {
                        statusDiv.textContent = 'Status: Connection Error';
                        statusDiv.className = 'error';
                    };
                    
                    websocket.onclose = () => {
                        statusDiv.textContent = 'Status: Disconnected';
                        statusDiv.className = '';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    };
                    
                    audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(stream);
                    processor = audioContext.createScriptProcessor(4096, 1, 1);
                    
                    processor.onaudioprocess = (e) => {
                        if (websocket?.readyState === WebSocket.OPEN) {
                            const inputData = e.inputBuffer.getChannelData(0);
                            const int16Array = new Int16Array(inputData.length);
                            for (let i = 0; i < inputData.length; i++) {
                                int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                            }
                            websocket.send(int16Array.buffer);
                        }
                    };
                    
                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    
                } catch (error) {
                    console.error('Error:', error);
                    let errorMsg = 'Error accessing microphone: ' + error.message;
                    
                    if (error.name === 'NotAllowedError') {
                        errorMsg = 'Microphone access denied. Please allow microphone access in your browser settings.';
                    } else if (error.name === 'NotFoundError') {
                        errorMsg = 'No microphone found. Please connect a microphone and try again.';
                    } else if (error.name === 'NotSupportedError') {
                        errorMsg = 'Your browser does not support microphone access. Please use HTTPS or access via localhost.';
                    }
                    
                    alert(errorMsg);
                    statusDiv.textContent = 'Status: Error - ' + error.name;
                    statusDiv.className = 'error';
                }
            };
            
            stopBtn.onclick = () => {
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (websocket) {
                    websocket.send(JSON.stringify({ command: 'stop' }));
                }
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
            
            clearBtn.onclick = () => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({ command: 'clear' }));
                }
                transcriptionDiv.innerHTML = '';
            };
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)
