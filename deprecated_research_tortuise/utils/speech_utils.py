from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.enhancement import WaveformEnhancement
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
import torchaudio
import torch
from tqdm import tqdm
from tortoise.api import TextToSpeech
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor
from dotenv import load_dotenv
from transformers import pipeline
from tortoise.utils.audio import load_voice
from audio_processing_utils import resample
from openai import OpenAI
import numpy as np
import os


class SpeechProcessor:
    
    def __init__(self):
        self.device = "cpu" # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia
        print("Loading models...")
        load_dotenv()
        token_key = os.getenv('TOKEN_KEY')
        open_ai_key = os.getenv('OPEN_AI_KEY')
        self.model_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token_key)
        self.model_separator = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
        self.model_enhance = WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank", savedir="pretrained_models/mtl-mimic-voicebank")
        self.model_tts = TextToSpeech()
        self.model_asr = AutoModelForCTC.from_pretrained('Yehor/w2v-bert-2.0-uk').to(self.device)
        self.model_asr_processor = Wav2Vec2BertProcessor.from_pretrained('Yehor/w2v-bert-2.0-uk')  
        self.model_t2t_translation = pipeline("translation", model="facebook/nllb-200-distilled-600M")
        self.openai_client = OpenAI(api_key=open_ai_key)
        self.sample_rate_for_enhancement = 16_000
        self.sample_rate_for_separation = 8_000
        self.sample_rate_for_diarization = 16_000
        self.sample_rate_for_tts = 22_050
        # print("Send models to GPU")
        # self.model_diarization.to(torch.device("cuda"))
        print("Models are ready!")
        
    def source_separation(self, audio_path: str) -> dict:
        est_sources = self.model_separator.separate_file(path=audio_path)
        background = est_sources[:, :, 0].detach().cpu()
        voice = est_sources[:, :, 1].detach().cpu()
        return {"voice": voice, "background": background}
    
    def voice_enhancment(self, voice_audio):
        if isinstance(voice_audio, np.ndarray):
            return self.voice_enhancment(torch.from_numpy(voice_audio).expand(1, -1)) # [].expand(1, -1) -> [[]] 
        return self.model_enhance.forward(voice_audio)
    
    def speech_segemntation(self, voice_audio, sample_rate) -> list:
        """ Splits a voice audio to a list of auido per said sentence """
        
        # Make sure sample rate is `self.sample_rate_for_diarization`
        if sample_rate != self.sample_rate_for_diarization:
            new_voice = resample(voice_audio, sample_rate, self.sample_rate_for_diarization)
            self.speech_segemntation(new_voice, self.sample_rate_for_diarization)
            
        # Get time-stamps
        diarization = self.get_annotation_speech_segmentation(voice_audio, sample_rate)
        
        # Make a list of audios
        voice_list = []
        for segment, _, _ in tqdm(diarization.itertracks(yield_label=True)):
            start_time = segment.start
            stop_time = segment.end
            start_sample = int(start_time * sample_rate)
            stop_sample = int(stop_time * sample_rate)
            
            voice_chunk = voice_audio[0][start_sample:stop_sample].numpy()
            voice_list.append(voice_chunk)
            
        return voice_list
        
    def get_annotation_speech_segmentation(self, voice_audio, sample_rate: int) -> Annotation:
        """ Returns an Annotation object with time stamps for each sentence in the audio """
        with ProgressHook() as hook:
            diarization = self.model_diarization({"waveform": voice_audio, "sample_rate": sample_rate}, hook=hook)
        return diarization

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

    def speech_to_text(self, voice_audio, sample_rate) -> str:
        """ Returns ukrainian text said in audio """
        
        inputs = self.model_asr_processor(voice_audio, sampling_rate=sample_rate).input_features
        features = torch.tensor(inputs).to(self.device)

        with torch.no_grad():
            logits = self.model_asr(features).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self.model_asr_processor.batch_decode(predicted_ids)
        
        return predictions
    
    def speech_translation_ukr_to_eng(self, text: str):
        # PS: Output is a bit unstructured :(
        return self.model_t2t_translation(text, src_lang="ukr", tgt_lang="eng")[0]['translation_text']

    def speech_translation_whisper(self, filename: str):
        audio_file = open(filename, "rb")
        transcript = self.openai_client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript["text"]

         