import argparse
import logging
from src.utils import (
    get_audio_from_youtube_video,
    transcribe_and_translate,
    create_voice_samples_dataset,
    SpeechToSpeechTranslator,
    merge_audio_samples,
    replace_audio_in_video,
    merge_translation
)
from styletts2.tts import LIBRI_TTS_CHECKPOINT_URL, LIBRI_TTS_CONFIG_URL 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logger
logging.getLogger('py.warnings').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(url, model_url, config_url, merge_duration, video_name):
    """ Performs AI dubbing of the video by `url` with `model_url` checkpoint and saves to results/`video_name` """
    logger.info(f"Video URL: {url}")
    logger.info(f"Model URL: {model_url}")
    logger.info(f"Config URL: {config_url}")
    logger.info(f"Dubbed Video Name: {video_name}")

    logger.info("[1/8] Downloading original audio...")
    audio_path = get_audio_from_youtube_video(url)

    # Get Sentence-by-Sentence transcription in original language using WhisperX
    logger.info("[2/8] Transcribing and Translating...")
    translation = transcribe_and_translate(audio_path, device="cuda")

    # Merge phrases together up to 350chars / 20secs to improve Text to Speech Quality
    logger.info("[3/8] Merging Translation...")
    if merge_duration > 0:
        merged_translation = merge_translation(translation, max_duration=merge_duration)
    else:
        merged_translation = translation

    # Create a dataset of original voice references based on merged Translation time stamps
    logger.info("[4/8] Creating original voice references per each translated `phrase`...")
    path_to_voice_samples = create_voice_samples_dataset(audio_path, merged_translation)

    # Downloading a checkpoint model for further speech generation
    logger.info("[5/8] Loading StyleTTS2 checkpoint...")
    sst_model = SpeechToSpeechTranslator(tts_model=model_url, tts_config=config_url)

    # Generating English Speech with translaetd text based on original audio refernces with StyleTTS2 + Alihnment of time stamps
    logger.info("[6/8] Generating translated speech...")
    path_to_generated_audio = sst_model.generate_translated_speech(path_to_voice_samples, merged_translation)

    # Merging generated speech to fit the length of original audio
    logger.info("[7/8] Merging translated speech samples into the final audio file...")
    path_to_result_audio = merge_audio_samples(path_to_generated_audio, audio_path,  merged_translation)

    # Put the new audio to the video and save it
    logger.info("[8/8] Exporting AI dubbed video...")
    replace_audio_in_video(url, path_to_result_audio, output_video_path=video_name)


if __name__ == "__main__":
    url = "..."
    model = "https://huggingface.co/sinhprous/StyleTTS2_ESD/resolve/main/epoch_2nd_00016.pth"
    config = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml"
    # main(url, model, config, 20, "sinhprous_merged_20.mp4")
    main(url, model, config, 10, "sinhprous_merged_10.mp4")
    main(url, model, config, 0, "sinhprous_not_merged.mp4")
