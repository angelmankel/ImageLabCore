"""
WebSocket management for external API progress updates.
"""
import json
import asyncio
from aiohttp import web

# Store active WebSocket connections
active_connections = set()

async def websocket_handler(request):
    """Handle WebSocket connections at /imagelab/ws"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Add to active connections
    active_connections.add(ws)
    print(f"[ImageLab WS] Client connected. Active connections: {len(active_connections)}")
    
    try:
        # Send connection confirmation
        await ws.send_json({
            'type': 'connection',
            'status': 'connected',
            'message': 'Connected to ImageLab WebSocket'
        })
        
        # Keep connection alive and listen for messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                # Handle incoming messages if needed
                pass
            elif msg.type == web.WSMsgType.ERROR:
                print(f'[ImageLab WS] Connection closed with exception {ws.exception()}')
    finally:
        # Remove from active connections
        active_connections.discard(ws)
        print(f"[ImageLab WS] Client disconnected. Active connections: {len(active_connections)}")
    
    return ws

async def broadcast_progress(data: dict):
    """Broadcast progress data to all connected WebSocket clients."""
    if not active_connections:
        return
    
    message = json.dumps(data)
    
    # Remove closed connections and send to active ones
    disconnected = set()
    for ws in active_connections:
        if ws.closed:
            disconnected.add(ws)
        else:
            try:
                await ws.send_str(message)
            except Exception as e:
                print(f"[ImageLab WS] Error sending to client: {e}")
                disconnected.add(ws)
    
    # Clean up disconnected clients
    for ws in disconnected:
        active_connections.discard(ws)

def broadcast_progress_sync(data: dict):
    """Synchronous wrapper for broadcast_progress. Schedules the broadcast on the event loop."""
    if not active_connections:
        return
    
    try:
        # Get the event loop from the first active connection
        loop = None
        for ws in active_connections:
            if hasattr(ws, '_req') and hasattr(ws._req, 'app') and hasattr(ws._req.app, '_loop'):
                loop = ws._req.app._loop
                break
        
        if loop is None:
            # Fallback to getting the running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        
        # Schedule the coroutine on the event loop from any thread
        asyncio.run_coroutine_threadsafe(broadcast_progress(data), loop)
    except Exception as e:
        print(f"[ImageLab WS] Error scheduling broadcast: {e}")
