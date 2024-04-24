import audio_processing_utils as apu
import youtube_utils as yu
from speech_utils import SpeechProcessor


if __name__ == "__main__":
    url = "https://youtu.be/D9l4nCqRTk8?si=6tGgXpssol3jyrlW"
    
    filename = "orig_youtube"
    shorfile = "short_youtube"
    yu.get_audio_from_youtube(url, filename)
    audio, sample_rate  = apu.read_audio(f"data/{filename}.wav", time=30)
    apu.save_audio(audio, sample_rate, f"data/{shorfile}.wav")

    speech_processor = SpeechProcessor()
    sources = speech_processor.source_separation(f"data/{shorfile}.wav")
    voice = sources["voice"]
    
    improved_voice = speech_processor.voice_enhancment(voice)
    
    voice_list = speech_processor.speech_segemntation(improved_voice)
    
    for i, voice_chunk in enumerate(voice_list):
        apu.save_audio(voice_chunk, speech_processor.sample_rate_for_diarization, f"data/voices/voice1/chunk_{i}.wav")
        
    text = "Hi. Today we are going to cover a very interesting topic. The name for it is: Machine Learning!"
    speech_processor.text_to_speech(text, quality="ultra_fast", file_name="generated_voice_ultra_fast")
    speech_processor.text_to_speech(text, quality="fast", file_name="generated_voice_fast")
    speech_processor.text_to_speech(text, quality="standard", file_name="generated_voice_standard")
    speech_processor.text_to_speech(text, quality="high_quality", file_name="generated_voice_high_quality")
