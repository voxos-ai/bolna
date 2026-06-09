import asyncio
import base64
import json
import os
import traceback
import uuid

import aiohttp
import websockets

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import (
    convert_audio_to_wav,
    create_ws_data_packet,
    pcm_to_wav_bytes,
    resample,
)


logger = configure_logger(__name__)


# 60db endpoints (docs: https://docs.60db.ai/api-reference/tts and /websocket-api/tts).
SIXTYDB_REST_URL = "https://api.60db.ai/tts-synthesize"
SIXTYDB_STREAM_URL = "https://api.60db.ai/tts-stream"
SIXTYDB_WS_URL = "wss://api.60db.ai/ws/tts"
SIXTYDB_DEFAULT_VOICE_ID = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"  # 60db docs default


class SixtydbSynthesizer(BaseSynthesizer):
    """60db TTS synthesizer.

    Behaviour intentionally mirrors ElevenlabsSynthesizer's non-streaming branch:
    one audio packet yielded per text, cache-aware, with mp3/pcm -> wav ->
    resample and the same meta_info flag conventions (is_cached, is_first_chunk,
    end_of_synthesizer_stream, format).

    The mode kwarg picks which 60db surface is used to fetch the audio bytes:
      - "rest"      POST /tts-synthesize       (default, fastest path)
      - "stream"    POST /tts-stream (NDJSON)  (chunks collected then yielded as one)
      - "websocket" wss://.../ws/tts           (PCM 16k mono, converted to wav)
    """

    def __init__(
        self,
        voice,
        voice_id=SIXTYDB_DEFAULT_VOICE_ID,
        model=None,
        audio_format="mp3",
        sampling_rate="16000",
        stream=False,
        buffer_size=400,
        speed=1.0,
        stability=50,
        similarity=75,
        enhance=True,
        mode="rest",
        synthesizer_key=None,
        caching=True,
        **kwargs,
    ):
        super().__init__(stream)
        # Accept both spellings; task_manager uses the elevenlabs-typo variant.
        self.api_key = (
            synthesizer_key
            or kwargs.get("synthesier_key")
            or os.environ.get("SIXTYDB_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "SIXTYDB_API_KEY is required (env var or synthesizer_key kwarg)."
            )

        if mode not in ("rest", "stream", "websocket"):
            raise ValueError(
                f"Invalid 60db mode {mode!r}. Use 'rest', 'stream', or 'websocket'."
            )

        self.voice = voice
        self.voice_id = voice_id
        self.model = model
        self.audio_format = "mp3"
        self.sampling_rate = sampling_rate
        # Stream=True path (per-chunk yield) is intentionally not implemented; this
        # mirrors ElevenlabsSynthesizer which also force-disables it in __init__.
        self.stream = False
        self.speed = speed
        self.stability = stability
        self.similarity = similarity
        self.enhance = enhance
        self.mode = mode
        # source_kind drives the post-fetch conversion. REST + NDJSON return mp3;
        # WebSocket returns LINEAR16 PCM mono @ 16kHz (the audio_config we send).
        self.source_kind = "pcm16k" if mode == "websocket" else "mp3"

        self.first_chunk_generated = False
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()
        self.synthesized_characters = 0

    def get_engine(self):
        return f"60db:{self.mode}"

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def supports_websocket(self):
        # Generator-level WS streaming is not implemented (see ElevenlabsSynthesizer
        # for the same trade-off). The "websocket" mode still uses WS under the hood,
        # but yields one buffered packet per text.
        return False

    # --- 60db fetch primitives ----------------------------------------------

    async def _fetch_rest(self, text):
        """POST /tts-synthesize -> raw mp3 bytes (base64-decoded)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "speed": self.speed,
            "stability": self.stability,
            "similarity": self.similarity,
            "enhance": self.enhance,
            "output_format": "mp3",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(SIXTYDB_REST_URL, headers=headers, json=payload) as r:
                if r.status != 200:
                    logger.error(f"60db /tts-synthesize {r.status}: {await r.text()}")
                    return None
                data = await r.json()
                audio_b64 = data.get("audio_base64")
                if not audio_b64:
                    logger.error(f"60db response missing audio_base64: {str(data)[:200]}")
                    return None
                return base64.b64decode(audio_b64)

    async def _fetch_stream(self, text):
        """POST /tts-stream NDJSON -> concatenated audio bytes from chunk lines."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "speed": self.speed,
            "stability": self.stability,
            "similarity": self.similarity,
            "enhance": self.enhance,
        }
        parts = []
        async with aiohttp.ClientSession() as session:
            async with session.post(SIXTYDB_STREAM_URL, headers=headers, json=payload) as r:
                if r.status != 200:
                    logger.error(f"60db /tts-stream {r.status}: {await r.text()}")
                    return None
                buffer = ""
                async for chunk in r.content.iter_chunked(8192):
                    buffer += chunk.decode("utf-8", errors="ignore")
                    lines = buffer.split("\n")
                    buffer = lines.pop()  # carry partial line forward
                    for line in lines:
                        if not line.strip():
                            continue
                        if not self._absorb_stream_line(line, parts):
                            return None
                if buffer.strip():
                    if not self._absorb_stream_line(buffer, parts):
                        return None
        return b"".join(parts) if parts else None

    def _absorb_stream_line(self, line, parts):
        """Parse one NDJSON line; append decoded audio to parts. Returns False on error."""
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"60db stream: skipping non-JSON line {line[:120]!r}")
            return True
        t = evt.get("type")
        if t == "chunk":
            ac = (evt.get("result") or {}).get("audioContent")
            if ac:
                parts.append(base64.b64decode(ac))
        elif t == "error":
            logger.error(f"60db stream error: {evt.get('message')}")
            return False
        return True

    async def _fetch_websocket(self, text):
        """Open per-text WS session; collect audio_chunk frames into one PCM blob."""
        url = f"{SIXTYDB_WS_URL}?apiKey={self.api_key}"
        context_id = str(uuid.uuid4())
        audio_parts = []
        try:
            async with websockets.connect(url) as ws:
                first = json.loads(await ws.recv())
                if not first.get("connection_established"):
                    logger.error(f"60db WS unexpected first frame: {first}")
                    return None

                await ws.send(json.dumps({
                    "create_context": {
                        "context_id": context_id,
                        "voice_id": self.voice_id,
                        "audio_config": {
                            "audio_encoding": "LINEAR16",
                            "sample_rate_hertz": 16000,
                        },
                        "speed": self.speed,
                        "stability": self.stability,
                        "similarity": self.similarity,
                    }
                }))

                ctx = json.loads(await ws.recv())
                if not ctx.get("context_created"):
                    logger.error(f"60db WS expected context_created, got: {ctx}")
                    return None

                await ws.send(json.dumps({"send_text": {"context_id": context_id, "text": text}}))
                await ws.send(json.dumps({"flush_context": {"context_id": context_id}}))

                while True:
                    evt = json.loads(await ws.recv())
                    if evt.get("audio_chunk", {}).get("audioContent"):
                        audio_parts.append(base64.b64decode(evt["audio_chunk"]["audioContent"]))
                    elif evt.get("flush_completed"):
                        break
                    elif evt.get("error"):
                        logger.error(f"60db WS error: {evt['error'].get('message')}")
                        return None

                # Best-effort context close (don't await reply; ws closes on context exit).
                try:
                    await ws.send(json.dumps({"close_context": {"context_id": context_id}}))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"60db WS exception: {e}")
            return None
        return b"".join(audio_parts) if audio_parts else None

    async def _fetch_raw(self, text):
        if self.mode == "rest":
            return await self._fetch_rest(text)
        if self.mode == "stream":
            return await self._fetch_stream(text)
        return await self._fetch_websocket(text)

    # --- BaseSynthesizer interface ------------------------------------------

    async def synthesize(self, text):
        return await self._fetch_raw(text)

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(
                    f"Generating 60db TTS response for message {message} "
                    f"(mode={self.mode})"
                )
                meta_info, text = message.get("meta_info"), message.get("data")

                # Cache the raw provider bytes (matches ElevenlabsSynthesizer).
                if self.caching and self.cache.get(text):
                    logger.info(f"60db cache hit for {text!r}")
                    audio = self.cache.get(text)
                    meta_info["is_cached"] = True
                else:
                    self.synthesized_characters += len(text)
                    meta_info["is_cached"] = False
                    audio = await self._fetch_raw(text)
                    if audio is None:
                        continue  # error already logged in the fetch helper
                    if self.caching:
                        self.cache.set(text, audio)

                meta_info["text"] = text
                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True

                if meta_info.get("end_of_llm_stream"):
                    meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False

                # Convert raw bytes to wav then resample to target rate.
                meta_info["format"] = "wav"
                if self.source_kind == "mp3":
                    wav_bytes = convert_audio_to_wav(audio, source_format="mp3")
                else:  # pcm16k from WS (LINEAR16 mono 16000 Hz)
                    wav_bytes = pcm_to_wav_bytes(audio)
                final_audio = resample(wav_bytes, int(self.sampling_rate), format="wav")
                yield create_ws_data_packet(final_audio, meta_info)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in 60db generate: {e}")
