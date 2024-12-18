import torch
from TTS.api import TTS
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

def tts(voices_path: str, speaker: str, output_file: str, text: str):
    tts = TTS("tts_models/en/multi-dataset/tortoise-v2", progress_bar=False).to(device)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Progress: Doing 200...")
    tts.tts_to_file(
        text=text,
        file_path=output_file+"tortouse_girl_200.wav",
        voice_dir=voices_path,
        speaker=speaker,
        num_autoregressive_samples=10,
        diffusion_iterations=200,
        verbose=False
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Progress: Doing 300...")
    tts.tts_to_file(
        text=text,
        file_path=output_file+"tortouse_girl_300.wav",
        voice_dir=voices_path,
        speaker=speaker,
        num_autoregressive_samples=10,
        diffusion_iterations=300,
        verbose=False
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Progress: Doing 400...")
    tts.tts_to_file(
        text=text,
        file_path=output_file+"tortouse_girl_400.wav",
        voice_dir=voices_path,
        speaker=speaker,
        num_autoregressive_samples=10,
        diffusion_iterations=400,
        verbose=False
    )

if __name__ == "__main__":
    voices_path = "results/tortoise_demo"
    speaker = "girl"
    output_file = "results/tortoise_demo"
    text = "It's very clean and used. The boss lady can see it"
    tts(voices_path, speaker, output_file, text)
