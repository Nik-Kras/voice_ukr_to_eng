import uuid
from src.utils import (
    get_audio_from_youtube_video, 
    transcribe, 
    translate, 
    create_voice_samples_dataset,
    generate_translated_speech,
    merge_audio_samples,
    replace_audio_in_video,
    merge_translation
)
import warnings
warnings.simplefilter("ignore")

def main(url: str):
    audio_path = get_audio_from_youtube_video(url)
    
    # Get Sentence-by-Sentence transcription in original language using WhisperX
    transcription = transcribe(audio_path, device="cuda")        
    
    # Translate them to English
    translation = translate(transcription)
    
    # Merge phrases together up to 350chars / 20secs to improve Text to Speech Quality
    merged_translation = merge_translation(translation, max_duration=20)         
    
    # Create a dataset of original voice references based on merged Translation time stamps
    path_to_voice_samples = create_voice_samples_dataset(audio_path, merged_translation)
    
    # Generating English Speech with translaetd text based on original audio refernces with StyleTTS2 + Alihnment of time stamps
    path_to_generated_audio = generate_translated_speech(path_to_voice_samples, merged_translation)
    
    # Merging generated speech to fit the length of original audio 
    path_to_result_audio = merge_audio_samples(path_to_generated_audio, audio_path,  merged_translation)
    
    # Put the new audio to the video and save it
    path_to_new_video = replace_audio_in_video(url, path_to_result_audio)
    return path_to_new_video


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=prfaWHQoxVg"
    print(main(url))
