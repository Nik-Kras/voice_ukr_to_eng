from tortoise.api import TextToSpeech
import torchaudio
from tortoise.utils.audio import load_voice


class tts_model:
    
    def __init__(self):
        self.device = "cpu" # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia
        print("Loading models...")
        self.model_tts = TextToSpeech()
        print("Models are ready!")
        
        
    def text_to_speech(self, text: str, quality: str = "fast", voice_set_name: str = "voice1", file_name: str="generated_voice"):
        
        assert quality in ("ultra_fast", "fast", "standard", "high_quality")
        voice_samples, conditioning_latents = load_voice(voice_set_name, extra_voice_dirs=["./data/voices"])
        gen = self.model_tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=quality
        )
        torchaudio.save(f'data/{file_name}.wav', gen.squeeze(0).cpu(), 24000)
