"""aiohttp server: WebRTC signaling + static file serving."""

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web
from aiohttp.web import middleware
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from mellea_webrtc.pipeline import AudioPipeline, PipelineModels
from mellea_webrtc.tracks import TTSOutputTrack

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST", "localhost")
PORT = int(os.environ.get("PORT", "8080"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")
STATIC_DIR = Path(__file__).parent / "static"

# Track active peer connections for cleanup
_peer_connections: set[RTCPeerConnection] = set()
_relay = MediaRelay()
_models: PipelineModels | None = None


async def index(request: web.Request) -> web.Response:
    html = (STATIC_DIR / "index.html").read_text()
    return web.Response(content_type="text/html", text=html)


async def offer(request: web.Request) -> web.Response:
    import time as _time
    t0 = _time.perf_counter()
    logger.info("[offer] Received WebRTC offer from %s", request.remote)
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))  # no STUN for localhost
    pc_id = id(pc)
    _peer_connections.add(pc)
    logger.debug("[offer] Created PeerConnection id=%d, total active=%d", pc_id, len(_peer_connections))

    # Create TTS output track, log channel, and pipeline
    output_track = TTSOutputTrack()
    log_channel = pc.createDataChannel("logs")
    logger.debug("[offer] Created log data channel, state=%s", log_channel.readyState)
    pipeline = AudioPipeline(output_track, log_channel, models=_models)
    pc.addTrack(output_track)
    logger.debug("[offer] Added TTS output track to PeerConnection")

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info("[pc:%d] Connection state: %s", pc_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            _peer_connections.discard(pc)
            logger.info("[pc:%d] Cleaned up, remaining connections=%d", pc_id, len(_peer_connections))

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("[pc:%d] Data channel received: label=%s", pc_id, channel.label)

        @channel.on("message")
        def on_message(message):
            logger.debug("[pc:%d] Data channel '%s' message: %s", pc_id, channel.label, message[:200] if isinstance(message, str) else "<binary>")
            if channel.label == "context":
                try:
                    data = json.loads(message)
                    logger.info("[pc:%d] Lesson context received: lesson=%s, content_len=%d", pc_id, data.get("lesson"), len(data.get("content", "")))
                except Exception:
                    pass

    @pc.on("track")
    def on_track(track):
        logger.info("[pc:%d] Received track: kind=%s, id=%s", pc_id, track.kind, track.id)
        if track.kind == "audio":
            relayed = _relay.subscribe(track)
            logger.debug("[pc:%d] Subscribed to relayed audio track", pc_id)
            asyncio.ensure_future(_consume_audio(relayed, pipeline))

    t1 = _time.perf_counter()
    await pc.setRemoteDescription(offer_sdp)
    t2 = _time.perf_counter()
    logger.debug("[offer] setRemoteDescription: %.0fms", (t2 - t1) * 1000)
    answer = await pc.createAnswer()
    t3 = _time.perf_counter()
    logger.debug("[offer] createAnswer: %.0fms", (t3 - t2) * 1000)
    await pc.setLocalDescription(answer)
    t4 = _time.perf_counter()
    logger.debug("[offer] setLocalDescription: %.0fms", (t4 - t3) * 1000)

    elapsed_ms = (t4 - t0) * 1000
    logger.info("[offer] WebRTC handshake complete in %.0fms for pc:%d", elapsed_ms, pc_id)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def _consume_audio(track, pipeline: AudioPipeline) -> None:
    """Continuously read audio frames from the incoming track."""
    logger.info("[audio] Starting audio consumption loop")
    frame_count = 0
    while True:
        try:
            frame = await track.recv()
        except Exception as exc:
            logger.info("[audio] Audio track ended after %d frames: %s", frame_count, exc)
            break
        frame_count += 1
        if frame_count == 1:
            logger.info("[audio] First audio frame received: rate=%d, samples=%d, format=%s", frame.sample_rate, frame.samples, frame.format.name)
        elif frame_count % 500 == 0:
            logger.debug("[audio] Processed %d frames", frame_count)
        try:
            pipeline.feed_audio(frame)
        except Exception:
            logger.exception("[audio] Error processing audio frame #%d", frame_count)


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in _peer_connections]
    await asyncio.gather(*coros)
    _peer_connections.clear()


@middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def main() -> None:
    global _models
    logger.info("Pre-loading models...")
    _models = PipelineModels()

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_route("OPTIONS", "/offer", lambda r: web.Response())
    app.on_shutdown.append(on_shutdown)

    logger.info("Starting server at http://%s:%d", HOST, PORT)
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
