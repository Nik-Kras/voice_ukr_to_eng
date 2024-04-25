import audio_processing_utils as apu
import youtube_utils as yu
from speech_utils import SpeechProcessor
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


if __name__ == "__main__":
    logging.info("Loading models")
    speech_processor = SpeechProcessor()
    
    logging.info("Loading data")
    url = "https://youtu.be/D9l4nCqRTk8?si=6tGgXpssol3jyrlW"
    
    filename = "orig_youtube"
    shorfile = "short_youtube"
    yu.get_audio_from_youtube(url, filename)
    audio, sample_rate  = apu.read_audio(f"data/{filename}.wav", time=30)
    audio1 = apu.resample(audio, sample_rate, speech_processor.sample_rate_for_separation)
    sample_rate = speech_processor.sample_rate_for_separation
    apu.save_audio(audio1, sample_rate, f"data/{shorfile}.wav")
    logging.info("Data is ready to be used")

    logging.info("Separating sources...")
    sources = speech_processor.source_separation(f"data/{shorfile}.wav")
    voice = sources["voice"]
    
    logging.info("Improving voice quality...")
    voice1 = apu.resample(voice.numpy()[0], sample_rate, speech_processor.sample_rate_for_enhancement)
    sample_rate = speech_processor.sample_rate_for_enhancement
    improved_voice = speech_processor.voice_enhancment(voice1)
    
    logging.info("Voice segmentation...")
    voice_list = speech_processor.speech_segemntation(improved_voice, sample_rate)
    
    for i, voice_chunk in enumerate(voice_list):
        apu.save_audio(voice_chunk, speech_processor.sample_rate_for_diarization, f"data/voices/voice1/chunk_{i}.wav")
        
    logging.info("Text to Speech...")
    text = "Hi. Today we are going to cover a very interesting topic. The name for it is: Machine Learning!"
    logging.info("Iteration #1")
    speech_processor.text_to_speech(text, quality="ultra_fast", file_name="generated_voice_ultra_fast")
    logging.info("Iteration #2")
    speech_processor.text_to_speech(text, quality="fast", file_name="generated_voice_fast")
    logging.info("Iteration #3")
    speech_processor.text_to_speech(text, quality="standard", file_name="generated_voice_standard")
    logging.info("Iteration #4")
    speech_processor.text_to_speech(text, quality="high_quality", file_name="generated_voice_high_quality")
    logging.info("Done")
