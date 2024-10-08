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


def main(url, model_url, config_url, video_name):
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
    merged_translation = merge_translation(translation, max_duration=20)

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
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="AI Dubbing for YouTube videos powered by StyleTTS2")

    # Add argument for the video URL
    parser.add_argument('url', type=str, help="URL of the YouTube video to dub")

    # Add argument for the model URL (with default value)
    parser.add_argument('--model_url', type=str, default=LIBRI_TTS_CHECKPOINT_URL, help="Path or URL to StyleTTS2 checkpoint")

    # Add argument for the config URL (with default value)
    parser.add_argument('--config_url', type=str, default=LIBRI_TTS_CONFIG_URL, help="Path or URL to StyleTTS2 checkpoint's config file")

    # Add argument for the video name (with default value)
    parser.add_argument('--video_name', type=str, default="dubbed_video.mp4", help="Name of the resulting dubbed video")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.url, args.model_url, args.config_url, args.video_name)
