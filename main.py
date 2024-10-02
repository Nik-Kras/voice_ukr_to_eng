import uuid
from src.utils import (
    get_audio_from_youtube_video, 
    transcribe, 
    translate, 
    create_voice_samples_dataset,
    generate_translated_speech,
    merge_audio_samples
)
import warnings
warnings.simplefilter("ignore")

def main(url: str):
    audio = get_audio_from_youtube_video(url)
    transcription = transcribe(audio, device="cuda")               
    translation = translate(transcription)         
    path_to_voice_samples = create_voice_samples_dataset(audio, transcription)
    path_to_generated_audio = generate_translated_speech(path_to_voice_samples, translation)
    path_to_result_audio = merge_audio_samples(path_to_generated_audio, translation)
    return path_to_result_audio

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=prfaWHQoxVg"
    main(url)
    