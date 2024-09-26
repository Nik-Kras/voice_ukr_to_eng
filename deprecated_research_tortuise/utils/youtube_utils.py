from pytube import YouTube
from pydub import AudioSegment
import os

def convert_to_wav(file_path, filename):
    audio = AudioSegment.from_file(f"data/{file_path}")
    audio.export(f"data/{filename}.wav", format="wav")


def get_audio_from_youtube(url: str, filename: str = "youtube_audio") -> str:
    """ Downloads audio from youtube video and returns a path """
    audio_stream = YouTube(url).streams.filter(only_audio=True).first()
    file_path = audio_stream.download(output_path="data")
    convert_to_wav(file_path.split("/data/")[-1], filename)
    os.remove(file_path)
    return f"data/{filename}.wav"
    
    
if __name__ == "__main__":
    get_audio_from_youtube("https://youtu.be/D9l4nCqRTk8?si=6tGgXpssol3jyrlW")
