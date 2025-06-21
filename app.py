import asyncio
import multiprocessing as mp
import threading
import time
import logging
from typing import Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import backend_config, processor_config, validate_rtsp_urls

# Import the optimized processor
from main import process_stream

# Global storage for queues and processes
stream_manager = None


class StreamManager:
    """Manages all video streams and their connections"""
    
    def __init__(self):
        self.mp_queues: List[mp.Queue] = []
        self.processes: List[mp.Process] = []
        self.async_queues: List[asyncio.Queue] = []
        self.websocket_connections: Dict[int, List[WebSocket]] = {}
        self.bridge_threads: List[threading.Thread] = []
        self.loop = None
        self.shutdown_event = threading.Event()
        self.frame_interval = 1.0 / backend_config.target_fps
        self.stream_urls: List[str] = []
        self.stream_status: Dict[int, str] = {}
        
    async def startup(self):
        """Initialize all streams and processes"""
        self.loop = asyncio.get_running_loop()
        
        # Validate RTSP URLs
        self.stream_urls = validate_rtsp_urls()
        actual_streams = min(len(self.stream_urls), backend_config.num_streams)
        
        if actual_streams == 0:
            logger.error("No valid stream URLs configured!")
            logger.info("Please configure RTSP URLs in environment variables or config.py")
            return
        
        logger.info(f"Configuring {actual_streams} streams")
        
        # Initialize connection tracking
        self.websocket_connections = {i: [] for i in range(actual_streams)}
        self.stream_status = {i: "initializing" for i in range(actual_streams)}
        
        # Check GPU availability
        try:
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus == 0:
                logger.warning("No GPUs detected, using CPU mode")
                num_gpus = 1  # Use CPU
        except ImportError:
            logger.warning("PyTorch not available, assuming CPU mode")
            num_gpus = 1
        
        for stream_id in range(actual_streams):
            # Get stream URL
            stream_url = self.stream_urls[stream_id]
            
            # Create multiprocessing queue
            mp_queue = mp.Queue(maxsize=backend_config.mp_queue_size)
            self.mp_queues.append(mp_queue)
            
            # Determine GPU assignment
            gpu_id = stream_id % num_gpus if num_gpus > 0 else 0
            
            # Create and start process
            process = mp.Process(
                target=process_stream,
                args=(stream_url, stream_id, mp_queue, gpu_id),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            
            # Create async queue for this stream
            async_queue = asyncio.Queue(maxsize=backend_config.async_queue_size)
            self.async_queues.append(async_queue)
            
            # Start bridge thread
            bridge_thread = threading.Thread(
                target=self._queue_bridge,
                args=(stream_id, mp_queue, async_queue),
                daemon=True
            )
            bridge_thread.start()
            self.bridge_threads.append(bridge_thread)
            
            self.stream_status[stream_id] = "running"
            
        # Start frame distribution task
        asyncio.create_task(self._frame_distributor())
        
        # Start status monitor task
        asyncio.create_task(self._status_monitor())
        
        logger.info(f"Started {actual_streams} RTSP streams")
    
    def _queue_bridge(self, stream_id: int, mp_queue: mp.Queue, async_queue: asyncio.Queue):
        """Bridge between multiprocessing and asyncio queues"""
        while not self.shutdown_event.is_set():
            try:
                # Get frame with timeout to allow shutdown
                frame_data = mp_queue.get(timeout=0.1)
                
                # Put frame in async queue (drop if full)
                future = asyncio.run_coroutine_threadsafe(
                    self._put_frame(async_queue, frame_data),
                    self.loop
                )
                # Don't wait for result to prevent blocking
                
            except mp.queues.Empty:
                continue
            except Exception as e:
                logger.error(f"Bridge error for stream {stream_id}: {e}")
    
    async def _put_frame(self, queue: asyncio.Queue, frame_data):
        """Put frame in queue, drop if full"""
        try:
            queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            # Get and discard oldest frame
            try:
                queue.get_nowait()
                queue.put_nowait(frame_data)
            except:
                pass
    
    async def _frame_distributor(self):
        """Distribute frames to all connected WebSocket clients"""
        while True:
            tasks = []
            
            for stream_id in range(len(self.stream_urls)):
                if stream_id in self.websocket_connections and self.websocket_connections[stream_id]:
                    tasks.append(self._distribute_stream_frame(stream_id))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            await asyncio.sleep(self.frame_interval)
    
    async def _distribute_stream_frame(self, stream_id: int):
        """Distribute a frame to all clients of a stream"""
        try:
            # Get latest frame (non-blocking)
            frame_data = self.async_queues[stream_id].get_nowait()
            
            # Send to all connected clients
            disconnected = []
            for ws in self.websocket_connections[stream_id]:
                try:
                    await ws.send_bytes(frame_data)
                except:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections[stream_id].remove(ws)
                
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            logger.error(f"Distribution error for stream {stream_id}: {e}")
    
    async def _status_monitor(self):
        """Monitor process status and restart if needed"""
        while not self.shutdown_event.is_set():
            for i, process in enumerate(self.processes):
                if i < len(self.stream_urls):  # Check bounds
                    if not process.is_alive():
                        self.stream_status[i] = "restarting"
                        logger.warning(f"Stream {i} process died, restarting...")
                        
                        # Get GPU assignment
                        try:
                            import torch
                            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                        except:
                            num_gpus = 1
                        
                        gpu_id = i % num_gpus
                        
                        # Create new process
                        self.processes[i] = mp.Process(
                            target=process_stream,
                            args=(self.stream_urls[i], i, self.mp_queues[i], gpu_id),
                            daemon=True
                        )
                        self.processes[i].start()
                        self.stream_status[i] = "running"
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def add_client(self, stream_id: int, websocket: WebSocket):
        """Add a WebSocket client to a stream"""
        await websocket.accept()
        self.websocket_connections[stream_id].append(websocket)
        logger.info(f"Client connected to stream {stream_id}")
    
    async def remove_client(self, stream_id: int, websocket: WebSocket):
        """Remove a WebSocket client from a stream"""
        if websocket in self.websocket_connections[stream_id]:
            self.websocket_connections[stream_id].remove(websocket)
        logger.info(f"Client disconnected from stream {stream_id}")
    
    async def shutdown(self):
        """Clean shutdown of all resources"""
        logger.info("Shutting down stream manager...")
        
        # Signal threads to stop
        self.shutdown_event.set()
        
        # Close all WebSocket connections
        for stream_clients in self.websocket_connections.values():
            for ws in stream_clients:
                try:
                    await ws.close()
                except:
                    pass
        
        # Terminate processes
        for process in self.processes:
            process.terminate()
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
        
        logger.info("Stream manager shutdown complete")


# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_manager
    stream_manager = StreamManager()
    await stream_manager.startup()
    yield
    await stream_manager.shutdown()


# Create FastAPI app
app = FastAPI(lifespan=lifespan, title="RTSP Video Processing Pipeline")


@app.websocket("/ws/{stream_id}")
async def video_stream(websocket: WebSocket, stream_id: int):
    """WebSocket endpoint for video streaming"""
    if stream_id < 0 or stream_id >= len(stream_manager.stream_urls):
        await websocket.close(code=4000, reason="Invalid stream ID")
        return
    
    await stream_manager.add_client(stream_id, websocket)
    
    try:
        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await stream_manager.remove_client(stream_id, websocket)


@app.get("/")
async def get_html():
    """Serve the HTML client"""
    num_streams = len(stream_manager.stream_urls) if stream_manager else 0
    html = html_content.replace('{{NUM_STREAMS}}', str(num_streams))
    return HTMLResponse(content=html)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not stream_manager:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "message": "Stream manager not ready"}
        )
    
    return {
        "status": "healthy",
        "configured_streams": len(stream_manager.stream_urls),
        "active_connections": sum(len(clients) for clients in stream_manager.websocket_connections.values()),
        "stream_status": stream_manager.stream_status
    }


@app.get("/api/streams")
async def get_streams():
    """Get information about configured streams"""
    if not stream_manager:
        return {"streams": []}
    
    streams = []
    for i, url in enumerate(stream_manager.stream_urls):
        # Hide sensitive parts of URL (passwords)
        safe_url = url
        if '@' in url:
            parts = url.split('@')
            proto_user = parts[0].split('//')
            if len(proto_user) > 1:
                user_parts = proto_user[1].split(':')
                if len(user_parts) > 1:
                    safe_url = f"{proto_user[0]}://{user_parts[0]}:****@{parts[1]}"
        
        streams.append({
            "id": i,
            "url": safe_url,
            "status": stream_manager.stream_status.get(i, "unknown"),
            "clients": len(stream_manager.websocket_connections.get(i, []))
        })
    
    return {"streams": streams}


# HTML content with RTSP stream support
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RTSP Stream Monitor</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: #fff;
            font-family: Arial, sans-serif;
        }
        .header {
            padding: 10px;
            background: #2a2a2a;
            text-align: center;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stats {
            padding: 0 20px;
            font-size: 14px;
        }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 5px;
            padding: 5px;
            max-width: 100vw;
        }
        .video-container {
            position: relative;
            padding-bottom: 56.25%;
            background: #000;
            border: 1px solid #333;
            overflow: hidden;
        }
        .video-img {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .stream-info {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            padding: 2px 5px;
            font-size: 12px;
            border-radius: 3px;
        }
        .status {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff0000;
        }
        .status.connected {
            background: #00ff00;
        }
        .no-signal {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>RTSP Camera Stream Monitor</h2>
        <div class="stats">
            <span id="connectedStreams">0</span> / <span id="totalStreams">{{NUM_STREAMS}}</span> streams connected
        </div>
    </div>
    <div class="video-grid" id="videoGrid"></div>
    
    <script>
        const grid = document.getElementById('videoGrid');
        const totalStreams = {{NUM_STREAMS}};
        const streams = [];
        let connectedCount = 0;
        
        // Update connected count
        function updateConnectedCount() {
            document.getElementById('connectedStreams').textContent = connectedCount;
        }
        
        // Create stream objects
        for (let i = 0; i < totalStreams; i++) {
            const container = document.createElement('div');
            container.className = 'video-container';
            
            const img = document.createElement('img');
            img.className = 'video-img';
            img.id = `stream-${i}`;
            img.style.display = 'none';
            
            const noSignal = document.createElement('div');
            noSignal.className = 'no-signal';
            noSignal.id = `no-signal-${i}`;
            noSignal.innerHTML = '<div>No Signal</div><div style="font-size: 10px; margin-top: 5px;">Waiting for connection...</div>';
            
            const info = document.createElement('div');
            info.className = 'stream-info';
            info.textContent = `Camera ${i}`;
            
            const status = document.createElement('div');
            status.className = 'status';
            status.id = `status-${i}`;
            
            container.appendChild(img);
            container.appendChild(noSignal);
            container.appendChild(info);
            container.appendChild(status);
            grid.appendChild(container);
            
            // Create WebSocket connection
            connectStream(i);
        }
        
        function connectStream(streamId) {
            const ws = new WebSocket(`ws://${window.location.host}/ws/${streamId}`);
            ws.binaryType = 'arraybuffer';
            
            let lastUrl = null;
            let reconnectTimeout = null;
            
            ws.onopen = function() {
                document.getElementById(`status-${streamId}`).classList.add('connected');
                console.log(`Stream ${streamId} connected`);
                connectedCount++;
                updateConnectedCount();
            };
            
            ws.onmessage = function(event) {
                const blob = new Blob([event.data], {type: 'image/jpeg'});
                const url = URL.createObjectURL(blob);
                
                const img = document.getElementById(`stream-${streamId}`);
                const noSignal = document.getElementById(`no-signal-${streamId}`);
                
                img.src = url;
                img.style.display = 'block';
                noSignal.style.display = 'none';
                
                // Clean up old blob URL
                if (lastUrl) {
                    URL.revokeObjectURL(lastUrl);
                }
                lastUrl = url;
            };
            
            ws.onclose = function() {
                document.getElementById(`status-${streamId}`).classList.remove('connected');
                document.getElementById(`stream-${streamId}`).style.display = 'none';
                document.getElementById(`no-signal-${streamId}`).style.display = 'block';
                console.log(`Stream ${streamId} disconnected, reconnecting...`);
                
                if (connectedCount > 0) {
                    connectedCount--;
                    updateConnectedCount();
                }
                
                // Reconnect after 2 seconds
                clearTimeout(reconnectTimeout);
                reconnectTimeout = setTimeout(() => connectStream(streamId), 2000);
            };
            
            ws.onerror = function(error) {
                console.error(`Stream ${streamId} error:`, error);
            };
            
            // Keep connection alive with ping
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        }
        
        // Fetch stream information
        async function fetchStreamInfo() {
            try {
                const response = await fetch('/api/streams');
                const data = await response.json();
                console.log('Stream info:', data);
            } catch (error) {
                console.error('Failed to fetch stream info:', error);
            }
        }
        
        // Fetch stream info on load
        fetchStreamInfo();
        
        // Refresh stream info every 30 seconds
        setInterval(fetchStreamInfo, 30000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import torch  # Import here to check GPU availability
    
    # Configure uvicorn for production
    uvicorn.run(
        app,
        host=backend_config.host,
        port=backend_config.port,
        workers=1,  # Single worker to manage shared state
        log_level=backend_config.log_level,
        access_log=backend_config.enable_access_log,
        ws_ping_interval=backend_config.ws_ping_interval,
        ws_ping_timeout=backend_config.ws_ping_timeout,
        ws_max_size=backend_config.ws_max_size,
    )