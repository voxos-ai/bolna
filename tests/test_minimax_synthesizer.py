import asyncio
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest


class FakeBaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()


class FakeCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def load_synthesizer_module():
    module_name = "bolna.synthesizer.minimax_synthesizer"
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "bolna" / "synthesizer" / "minimax_synthesizer.py"
    fake_aiohttp = fake_module("aiohttp", ClientSession=None)
    dependencies = {
        "aiohttp": fake_aiohttp,
        "dotenv": fake_module("dotenv", load_dotenv=lambda: None),
        "bolna": fake_module("bolna"),
        "bolna.helpers": fake_module("bolna.helpers"),
        "bolna.helpers.logger_config": fake_module(
            "bolna.helpers.logger_config", configure_logger=logging.getLogger
        ),
        "bolna.helpers.utils": fake_module(
            "bolna.helpers.utils",
            convert_audio_to_wav=lambda audio, source_format: audio,
            create_ws_data_packet=lambda data, meta_info: {"data": data, "meta_info": meta_info},
            pcm_to_wav_bytes=lambda audio, sample_rate: audio,
        ),
        "bolna.memory": fake_module("bolna.memory"),
        "bolna.memory.cache": fake_module("bolna.memory.cache"),
        "bolna.memory.cache.inmemory_scalar_cache": fake_module(
            "bolna.memory.cache.inmemory_scalar_cache", InmemoryScalarCache=FakeCache
        ),
        "bolna.synthesizer": fake_module("bolna.synthesizer"),
        "bolna.synthesizer.base_synthesizer": fake_module(
            "bolna.synthesizer.base_synthesizer", BaseSynthesizer=FakeBaseSynthesizer
        ),
    }
    original_modules = {name: sys.modules.get(name) for name in dependencies}
    sys.modules.update(dependencies)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop(module_name, None)


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, content_type=None):
        return self.body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, headers, json):
        self.request = {"url": url, "headers": headers, "json": json}
        return self.response


class MiniMaxSynthesizerTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_cn_endpoint_and_decodes_hex_audio(self):
        module = load_synthesizer_module()
        audio = b"RIFF-test-audio"
        response = FakeResponse({
            "data": {"audio": audio.hex(), "status": 2},
            "base_resp": {"status_code": 0},
        })
        session = FakeSession(response)
        module.aiohttp.ClientSession = lambda: session

        synthesizer = module.MiniMaxSynthesizer(
            voice="voice-id",
            region="cn_zh",
            model="speech-2.8-hd",
            audio_format="wav",
            sampling_rate=24000,
            synthesizer_key="test-key",
            language_boost="English",
            voice_setting={"speed": 1.1},
            audio_setting={"bitrate": 128000},
            subtitle_enable=True,
        )

        result = await synthesizer.synthesize("Hello")

        self.assertEqual(result, audio)
        self.assertEqual(session.request["url"], "https://api.minimaxi.com/v1/t2a_v2")
        self.assertEqual(session.request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(session.request["json"]["model"], "speech-2.8-hd")
        self.assertEqual(session.request["json"]["text"], "Hello")
        self.assertFalse(session.request["json"]["stream"])
        self.assertEqual(session.request["json"]["output_format"], "hex")
        self.assertEqual(session.request["json"]["voice_setting"], {
            "speed": 1.1,
            "voice_id": "voice-id",
        })
        self.assertEqual(session.request["json"]["audio_setting"], {
            "bitrate": 128000,
            "format": "wav",
            "sample_rate": 24000,
        })
        self.assertEqual(session.request["json"]["language_boost"], "English")
        self.assertTrue(session.request["json"]["subtitle_enable"])

    def test_rejects_unknown_region_and_model(self):
        module = load_synthesizer_module()
        common = {"voice": "voice-id", "synthesizer_key": "test-key"}

        with self.assertRaisesRegex(ValueError, "region"):
            module.MiniMaxSynthesizer(region="unknown", **common)
        with self.assertRaisesRegex(ValueError, "model"):
            module.MiniMaxSynthesizer(
                region="global_en", model="unknown", **common
            )


if __name__ == "__main__":
    unittest.main()
