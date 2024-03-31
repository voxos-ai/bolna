import aiohttp
import os, base64
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, convert_audio_to_wav, wav_bytes_to_pcm, float32_to_int16
from .base_synthesizer import BaseSynthesizer
import json
import numpy as np
import librosa


logger = configure_logger(__name__)
load_dotenv()
CARTESIA_TTS_URL = "https://api.cartesia.ai/v0/audio/stream"


class CartesiaSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400,
                 **kwargs):
        super().__init__(stream, buffer_size)
        self.format = "linear16" if audio_format == "pcm" else audio_format

        # @TODO: retrieve voice embeddings from voice id
        self.voice_id = voice_id
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.api_key = os.getenv('CARTESIA_API_KEY')

    async def __generate_http(self, text):
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        url = CARTESIA_TTS_URL

        # voice embeddings hard-coded for now
        payload = {
            "transcript": text,
            "model_id": "genial-planet-1346",
            "voice": [-0.08349977, 0.002090532, -0.058492012, 0.05682824, 0.032969892, 0.06476176, 0.11180842,
                      -0.010231484,
                      -0.13199303, 0.06492365, -0.095957026, -0.1907417, -0.0017518557, 0.120052546, 0.011473832,
                      0.081289105, 0.06735374, -0.08829124, 0.021840338, -0.021377875, -0.081642635, 0.0361156,
                      0.07647459,
                      0.0511448, 0.16792385, 0.10362113, 0.025860261, -0.013933178, -0.09437346, 0.022935398,
                      0.05249774,
                      0.083065175, 0.0029462734, -0.0069546886, 0.123878404, 0.014168107, -0.10972298, -0.014678976,
                      0.060251728, 0.058582217, -0.030348232, 0.053895522, 0.03367596, -0.024928227, 0.011214875,
                      -0.05137308, -0.007930916, -0.041404985, -0.15495157, -0.09839702, -0.09141977, 0.06589632,
                      -0.027742008, -0.07747854, -0.040624414, -0.0024135273, 0.0002600023, 0.041529134, -0.061029993,
                      0.028587, 0.013301304, 0.050073702, -0.06934517, -0.087251306, -0.09065029, 0.013803922,
                      -0.015933404,
                      0.15762834, 0.0002534511, 0.029798755, -0.04037404, -0.011396297, 0.0042333575, -0.051545925,
                      -0.0023936604, -0.0037610754, -0.010482568, -0.05991406, 0.11610999, -0.04319631, 0.10485421,
                      0.064472385, 0.013616613, -0.019880494, -0.042726614, -0.027214404, 0.033053245, -0.020481788,
                      0.02109337, 0.038449425, 0.023363968, 0.009579539, -0.118101, 0.1386573, 0.0927775, -0.016857147,
                      0.1976759, -0.034415744, -0.06431705, -0.049044084, 0.14846464, -0.017944172, 0.094910555,
                      -0.009885608, 0.00012909336, -0.015723823, 0.033398718, 0.07034016, 0.017083967, 0.04049575,
                      -0.061795447, -0.02885256, 0.05876426, 0.07006745, -0.122125395, 0.016540533, 0.13931146,
                      -0.020737736, 0.068287835, -0.1428113, 0.04168253, -0.043687835, -0.035450917, -0.043973003,
                      0.0148469405, -0.05686619, -0.08270704, -0.09263632, 0.104636244, 0.097103, 0.032663677,
                      -0.0036553638, 0.16955547, 0.009410844, -0.11583431, -0.012282198, -0.06593415, 0.0136156585,
                      0.024368528, -0.0017510485, 0.07095734, 0.0510402, -0.04539834, 0.002682568, 0.056404743,
                      0.14189161,
                      -0.12283087, -0.00078447914, 0.0067053246, 0.060703922, -0.03745036, 0.030125227, -0.07517014,
                      -0.07598357, 0.10075039, 0.05269573, -0.027410736, -0.07012868, -0.1528656, -0.05137218,
                      -0.05339904,
                      0.037869565, 0.04847169, -0.15418933, 0.025897024, -0.030869389, 0.026807137, -0.11302227,
                      0.032616116, -0.024449771, 0.124926694, 0.026570747, -0.011536936, -0.10525091, -0.011680901,
                      -0.04615144, -0.020805739, 0.22363971, 0.0147074815, 0.03382057, -0.049642447, 0.012356055,
                      -0.0085797785, 0.089436516, -0.020164197, -0.07450032, 0.031972397, 0.090754285, 0.13934238,
                      -0.09481122, 0.005583748, -0.017912822]
        }

        buffer = ''
        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        chunk = await response.read()
                        buffer += chunk.decode('utf-8')

                        while "{" in buffer and "}" in buffer:
                            start_index = buffer.find("{")
                            end_index = buffer.find("}", start_index)
                            if start_index != -1 and end_index != -1:
                                try:
                                    chunk_json = json.loads(buffer[start_index: end_index + 1])
                                    decoded_audio = base64.b64decode(chunk_json["data"])

                                    pcm_decoded_array = np.frombuffer(decoded_audio, dtype=np.float32)
                                    resampled_audio = librosa.resample(pcm_decoded_array, orig_sr=44100, target_sr=8000)
                                    yield resampled_audio
                                    buffer = buffer[end_index + 1:]
                                except json.JSONDecodeError:
                                    break

                    if buffer:
                        try:
                            chunk_json = json.loads(buffer)
                            decoded_audio = base64.b64decode(chunk_json["data"])

                            pcm_decoded_array = np.frombuffer(decoded_audio, dtype=np.float32)
                            resampled_audio = librosa.resample(pcm_decoded_array, orig_sr=44100, target_sr=8000)
                            yield resampled_audio
                        except json.JSONDecodeError:
                            pass

            else:
                logger.info("Payload was null")

    async def open_connection(self):
        pass

    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")

            meta_info, text = message.get("meta_info"), message.get("data")
            async for message in self.__generate_http(text):
                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True
                else:
                    meta_info["is_first_chunk"] = False
                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                    meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False

                meta_info['text'] = text
                meta_info['format'] = self.format
                yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
