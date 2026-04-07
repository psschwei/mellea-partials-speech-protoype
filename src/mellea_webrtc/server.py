"""aiohttp server: WebRTC signaling + static file serving."""

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web
from aiohttp.web import middleware
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from mellea_webrtc.pipeline import AudioPipeline
from mellea_webrtc.tracks import TTSOutputTrack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("aioice").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST", "localhost")
PORT = int(os.environ.get("PORT", "8080"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")
STATIC_DIR = Path(__file__).parent / "static"

# Track active peer connections for cleanup
_peer_connections: set[RTCPeerConnection] = set()
_relay = MediaRelay()


async def index(request: web.Request) -> web.Response:
    html = (STATIC_DIR / "index.html").read_text()
    return web.Response(content_type="text/html", text=html)


async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    # Create TTS output track, log channel, and pipeline
    output_track = TTSOutputTrack()
    log_channel = pc.createDataChannel("logs")
    pipeline = AudioPipeline(output_track, log_channel)
    pc.addTrack(output_track)

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            _peer_connections.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Received track: %s", track.kind)
        if track.kind == "audio":
            relayed = _relay.subscribe(track)
            asyncio.ensure_future(_consume_audio(relayed, pipeline))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def _consume_audio(track, pipeline: AudioPipeline) -> None:
    """Continuously read audio frames from the incoming track."""
    logger.info("Starting audio consumption loop")
    while True:
        try:
            frame = await track.recv()
        except Exception as exc:
            logger.info("Audio track ended: %s", exc)
            break
        try:
            pipeline.feed_audio(frame)
        except Exception:
            logger.exception("Error processing audio frame")


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
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_route("OPTIONS", "/offer", lambda r: web.Response())
    app.on_shutdown.append(on_shutdown)

    logger.info("Starting server at http://%s:%d", HOST, PORT)
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
