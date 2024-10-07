import uuid
from src.utils import (
    get_audio_from_youtube_video, 
    transcribe, 
    translate, 
    create_voice_samples_dataset,
    SpeechToSpeechTranslator,
    merge_audio_samples,
    replace_audio_in_video,
    merge_translation
)
import warnings
warnings.simplefilter("ignore")

MODEL_SELECTION = [
    # ("LibriTTS", "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth", "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml"),
    # ("LJSpeech", "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth", "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml"),
    # ("Vokan_epoch_2nd_00012", "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/epoch_2nd_00012.pth", "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/config.yml"),
    ("Twilight", "https://huggingface.co/therealvul/StyleTTS2/resolve/main/Twilight0/epoch_2nd_00009.pth", "https://huggingface.co/therealvul/StyleTTS2/resolve/main/Twilight0/config.yml"),
    ("Bluebomber", "https://huggingface.co/Bluebomber182/StyleTTS2-LibriTTS-Model-by-yl4579/resolve/main/LibriTTS/epochs_2nd_00020.pth", "https://huggingface.co/Bluebomber182/StyleTTS2-LibriTTS-Model-by-yl4579/resolve/main/LibriTTS/config.yml"),
    ("Shiro836", "https://huggingface.co/Shiro836/StyleTTS2-Forsen/resolve/main/epoch_2nd_00049.pth", "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml"),
    ("sinhprous", "https://huggingface.co/sinhprous/StyleTTS2_ESD/resolve/main/epoch_2nd_00016.pth", "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml"),
    ("ShashwatRajput", "https://huggingface.co/ShashwatRajput/StyleTTS_2_Elon/resolve/main/epoch_2nd_00099.pth", "https://huggingface.co/ShashwatRajput/StyleTTS_2_Elon/resolve/main/config_ft.yml")
]
URLs = [
    ("Zelenskiy", "https://www.youtube.com/watch?v=prfaWHQoxVg"),
    # ("Macron_Inauguration", "https://www.youtube.com/watch?v=ewl7njdts7k"),
    ("Pope_Francis_1", "https://www.youtube.com/watch?v=Rgn_uU8BKqQ"),
    ("Pope_Francis_2", "https://www.youtube.com/watch?v=1VcWCEikZBA"),
    # (, "https://www.youtube.com/watch?v=Qe8D5QGmfH0",)
    # (, "https://www.youtube.com/watch?v=MvSy_Mc-X3I",)
    # (, "https://www.youtube.com/watch?v=cka2WarC7pI")
]

def main(url: str, tts_model_url: str, tts_config_url: str, output_video_path: str = "result.mp4"):
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
    sst_model = SpeechToSpeechTranslator(tts_model=tts_model_url, tts_config=tts_config_url)
    path_to_generated_audio = sst_model.generate_translated_speech(path_to_voice_samples, merged_translation)
    
    # Merging generated speech to fit the length of original audio 
    path_to_result_audio = merge_audio_samples(path_to_generated_audio, audio_path,  merged_translation)
    
    # Put the new audio to the video and save it
    path_to_new_video = replace_audio_in_video(url, path_to_result_audio, output_video_path=output_video_path)
    return path_to_new_video


if __name__ == "__main__":
    for name, model_url, config_url in MODEL_SELECTION:
        print("[MODEL]: {}".format(name))
        for video_name, url in URLs:
            print("[VIDEO]: {}".format(video_name))
            try:
                main(url, model_url, config_url, f"{name}_{video_name}.mp4")
            except Exception as e:
                print("ERROR occured: {}-{}".format(name, video_name))
                print(e)
