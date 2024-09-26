from typing import Dict, List


class TranscriptionElement:
    """Basic element of Transcription with time stamps. 
    Full transcription is a list of such elements"""
    time_start: float
    time_end: float
    text: str


def get_aduio_from_youtube_video(url: str):
    raise NotImplemented()


def transcribe(audio) -> List[TranscriptionElement]:
    raise NotImplemented()


def translate() -> List[TranscriptionElement]:
    raise NotImplemented()


def create_voice_samples_dataset(audio, transcription: List[TranscriptionElement]) -> str:
    raise NotImplemented()


def generate_translated_speech(path_to_voice_samples: str, translation: List[TranscriptionElement]):
    raise NotImplemented()


def merge_audio_samples(path_to_generated_audio: str, transcription: List[TranscriptionElement]):
    raise NotImplemented()
    

if __name__ == "__main__":
    print("Hello")
