import uuid
from src.utils import (
    get_audio_from_youtube_video, 
    transcribe, 
    translate, 
    create_voice_samples_dataset,
    generate_translated_speech,
    merge_audio_samples,
    replace_audio_in_video
)
import warnings
warnings.simplefilter("ignore")

def main(url: str):
    audio_path = get_audio_from_youtube_video(url)
    transcription = transcribe(audio_path, device="cuda")               
    translation = translate(transcription, "ru")         
    path_to_voice_samples = create_voice_samples_dataset(audio_path, transcription)
    path_to_generated_audio = generate_translated_speech(path_to_voice_samples, translation)
    path_to_result_audio = merge_audio_samples(path_to_generated_audio, audio_path,  translation)
    path_to_new_video = replace_audio_in_video(url, path_to_result_audio)
    return path_to_new_video


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=prfaWHQoxVg"
    print(main(url))
