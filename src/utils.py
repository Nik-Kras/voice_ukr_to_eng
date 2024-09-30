from typing import Dict, List
import yt_dlp as youtube_dl
import whisperx
from pydub import AudioSegment
import re
from pydub import AudioSegment
import os
import torch
from styletts2 import tts
from typing import List
from pydub import AudioSegment
from googletrans import Translator

device = torch.device("cuda")  # Set the device to GPU if available

# Initialize the TTS model once at the start
my_tts = tts.StyleTTS2()


class TranscriptionElement:
    """Basic element of Transcription with time stamps. 
    Full transcription is a list of such elements"""
    time_start: float
    time_end: float
    text: str
    
    def __init__(self, time_start, time_end, text):
        self.time_start = time_start
        self.time_end = time_end
        self.text = text


def get_audio_from_youtube_video(url: str, filename: str = "raw_audio"):
    """ Download audio from youtube link and save as `filename`.wav """
    path = f'results/{filename}.wav'
    
    ydl_opts = {
        'format': 'bestaudio/best',         # prioritization options
        'ffmpeg_location': r'C:\Users\tprok\Downloads\ffmpeg-7.0.2-essentials_build\ffmpeg-7.0.2-essentials_build\bin',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',    # extracts audio from the video
            'preferredcodec': 'wav',        # format
            'preferredquality': '192',      # prefered bitrate quality in kb/s
        }],
        'outtmpl': path,                    # Change the output filename
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url]) 
        
    return path


def transcribe(audio_path: str,
               device: str = "cuda",
               batch_size: int = 16,
               compute_type: str = "float16",
               model_checkpoint: str = "large-v2") -> List[TranscriptionElement]:
    """  """

    if device not in ("cuda", "cpu"):
        raise ValueError(f"Make sure device is either cuda or cpu. Your value: {device}")

    model = whisperx.load_model(model_checkpoint, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    output = []
    for segment in result["segments"]:
        output.append(TranscriptionElement(time_start=segment['start'], time_end=segment['end'], text=segment['text']))

    return output


def translate(transcription) -> List[TranscriptionElement]:
    # Initialize the translator
    translator = Translator()

    translated_elements = []

    # Iterate over each transcription element and translate the text
    for element in transcription:
        try:
            # Translate the text (assuming the original text is in Ukrainian 'uk')
            translation = translator.translate(element.text.strip(), src='uk', dest='en')

            # Create a new TranscriptionElement with the translated text
            translated_element = TranscriptionElement(
                time_start=element.time_start,
                time_end=element.time_end,
                text=translation.text
            )

            translated_elements.append(translated_element)
            print(f"Translated segment [{element.time_start:.2f} --> {element.time_end:.2f}]: {translation.text}")

        except Exception as e:
            print(f"Error translating segment [{element.time_start:.2f} --> {element.time_end:.2f}]: {e}")
            # In case of error, append the original untranslated element
            translated_elements.append(element)

    print("Translation completed.")
    return translated_elements



def create_voice_samples_dataset(audio_path, transcription: List[TranscriptionElement]) -> str:

    # Define the output directory
    output_dir = "audio_dataset"

    # Validate the audio file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the original audio file
    audio = AudioSegment.from_file(audio_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each transcription element
    for i, element in enumerate(transcription):
        # Extract start and end times in milliseconds
        start_ms = int(element.time_start * 1000)
        end_ms = int(element.time_end * 1000)

        # Extract the audio segment based on start and end times
        audio_segment = audio[start_ms:end_ms]

        # Define the output file name and path
        output_filename = f"segment_{i + 1}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Export the audio segment
        audio_segment.export(output_path, format="wav")
        print(f"Exported {output_filename} [{element.time_start:.2f} --> {element.time_end:.2f}]")

    print(f"All segments exported to directory: {output_dir}")
    return output_dir


def generate_translated_speech(path_to_voice_samples: str, translation: List[TranscriptionElement]):
    # Define the output directory for generated speech
    generated_audio_dir = "generated_audio"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(generated_audio_dir):
        os.makedirs(generated_audio_dir)
    
    # Iterate over the translation elements to generate speech for each translated segment
    for i, element in enumerate(translation):
        # Specify the path of the target voice sample for the corresponding segment
        target_voice_path = os.path.join(path_to_voice_samples, f"segment_{i + 1}.wav")

        # Define the output path for the generated audio
        output_path = os.path.join(generated_audio_dir, f"translated_segment_{i + 1}.wav")

        # Check if the target voice path exists
        if not os.path.exists(target_voice_path):
            print(f"Warning: Target voice sample not found for segment {i + 1}. Skipping.")
            continue

        # Generate the speech using the StyleTTS2 model
        print(f"Generating translated speech for segment {i + 1}...")
        try:
            my_tts.inference(element.text, target_voice_path=target_voice_path, output_wav_file=output_path)
            print(f"Generated speech for segment {i + 1} -> saved at {output_path}")
        except Exception as e:
            print(f"Error generating speech for segment {i + 1}: {e}")
    
    print(f"All translated speech samples generated in directory: {generated_audio_dir}")
    return generated_audio_dir


def merge_audio_samples(path_to_generated_audio: str, transcription: List[TranscriptionElement]):
    # Create an empty audio segment to concatenate all parts
    combined_audio = AudioSegment.empty()

    # Iterate over the transcription elements to load each translated audio segment
    for i, element in enumerate(transcription):
        segment_path = os.path.join(path_to_generated_audio, f"translated_segment_{i + 1}.wav")

        # Check if the segment file exists
        if not os.path.exists(segment_path):
            print(f"Warning: Translated segment not found for segment {i + 1}. Skipping.")
            continue

        # Load the translated audio segment
        segment_audio = AudioSegment.from_file(segment_path)
        combined_audio += segment_audio
        print(f"Merged segment {i + 1} [{element.time_start:.2f} --> {element.time_end:.2f}]")

    # Define the output path for the final merged audio file
    final_output_path = "final_translated_audio.wav"
    
    # Export the combined audio to a WAV file
    combined_audio.export(final_output_path, format="wav")
    print(f"Final merged audio saved at {final_output_path}")

    return final_output_path
    

if __name__ == "__main__":
    print("Hello")
    ...
    ...
    ...
