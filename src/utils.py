from typing import Dict, List
import yt_dlp as youtube_dl
import whisperx
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from styletts2 import tts
from typing import List
# from googletrans import Translator
import translators as ts
import re
import os
import torch
import nltk
import librosa

nltk.download('punkt_tab')

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

    def __repr__(self):
        # Customize the output for better readability
        return f"TranscriptionElement(start={self.time_start}, end={self.time_end}, text={self.text})"

    def __str__(self):
        #Define a string representation, which is used by print()
        return f"[{self.time_start:.2f} - {self.time_end:.2f}] {self.text}"


def get_audio_from_youtube_video(url: str, filename: str = "raw_audio"):
    """ Download audio from youtube link and save as `filename`.wav """
    path = f'results/{filename}'
    
    ydl_opts = {
        'format': 'bestaudio/best',         # prioritization options
        # 'ffmpeg_location': r'C:\Users\tprok\Downloads\ffmpeg-7.0.2-essentials_build\ffmpeg-7.0.2-essentials_build\bin',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',    # extracts audio from the video
            'preferredcodec': 'wav',        # format
            'preferredquality': '192',      # prefered bitrate quality in kb/s
        }],
        'outtmpl': path,                    # Change the output filename
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url]) 
        
    return path + ".wav"


def transcribe(audio_path: str,
               device: str = "cpu",
               batch_size: int = 16,
               compute_type: str = "float32",
               model_checkpoint: str = "large-v2") -> List[TranscriptionElement]:
    """
    Transcribes an audio file into a list of TranscriptionElements, splitting each segment into individual sentences.

    Parameters:
        audio_path (str): Path to the audio file to be transcribed.
        device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        batch_size (int): Batch size for the transcription process. Defaults to 16.
        compute_type (str): Data type for computation ('float32', 'float16', etc.). Defaults to 'float32'.
        model_checkpoint (str): Model checkpoint to load (e.g., 'base', 'large-v2'). Defaults to 'large-v2'.

    Returns:
        List[TranscriptionElement]: A list of TranscriptionElement objects, each representing a sentence.
    """
    
    if device not in ("cuda", "cpu"):
        raise ValueError(f"Make sure device is either 'cuda' or 'cpu'. Your value: {device}")

    # Load the whisper model and audio file using whisperx
    model = whisperx.load_model(
        model_checkpoint,
        device,
        compute_type=compute_type, 
        # asr_options={
        #     "max_new_tokens": 128,
        #     "clip_timestamps": True,
        #     "hallucination_silence_threshold": 0.5,
        #     "hotwords": []
        # }
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    output = []
    
    # Iterate through each segment and split into sentences
    for segment in result["segments"]:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        # Use regex to split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        # Calculate the duration of the segment
        segment_duration = end_time - start_time
        
        # Calculate an estimated duration for each sentence based on their relative length
        total_characters = sum(len(sentence) for sentence in sentences)
        
        # Track the start time for each sentence
        current_start_time = start_time
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Estimate the time this sentence took based on its proportion of the total characters
                sentence_duration = (len(sentence) / total_characters) * segment_duration
                current_end_time = current_start_time + sentence_duration
                
                # Create a new TranscriptionElement for each sentence
                transcription_element = TranscriptionElement(
                    time_start=current_start_time,
                    time_end=current_end_time,
                    text=sentence
                )
                
                output.append(transcription_element)
                
                # Update the start time for the next sentence
                current_start_time = current_end_time

    print("Transcription Complete")
    return output


def translate(transcription: List[TranscriptionElement]) -> List[TranscriptionElement]:
    """ Translates each text element of transcription to English """
    full_text = " ".join([element.text.strip() for element in transcription])
    full_text_translated = ts.translate_text(full_text, to_language="en")
    translated_sentences = nltk.tokenize.sent_tokenize(full_text_translated)
    translated_elements = [
        TranscriptionElement(
            time_start=transcription_element.time_start,
            time_end=transcription_element.time_end,
            text=sentence
        ) for sentence, transcription_element in zip(translated_sentences, transcription)
    ]
    return translated_elements


def create_voice_samples_dataset(audio_path: str,
                                 transcription: List[TranscriptionElement],
                                 output_dir: str = "results/original_audio") -> str:
    """ Creates small atomic audio files from big audio by `audio_path` based on time-stamps from transcription """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, element in enumerate(transcription):
        start_ms = int(element.time_start * 1000)
        end_ms = int(element.time_end * 1000)
        audio_segment = audio[start_ms:end_ms]
        output_filename = f"segment_{i + 1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        audio_segment.export(output_path, format="wav")
        print(f"Exported {output_filename} [{element.time_start:.2f} --> {element.time_end:.2f}]")

    print(f"All segments exported to directory: {output_dir}")
    return output_dir


def generate_translated_speech(path_to_voice_samples: str,
                               translation: List[TranscriptionElement],
                               generated_audio_dir: str = "results/translated_auido") -> str:
    """ Generates speech for each translation element corresponding to each target voice sample from `path_to_voice_samples` and saves to `generated_audio_dir` """
    
    if not os.path.exists(generated_audio_dir):
        os.makedirs(generated_audio_dir)
    
    for i, element in enumerate(translation):
        target_voice_path = os.path.join(path_to_voice_samples, f"segment_{i + 1}.wav")
        output_path = os.path.join(generated_audio_dir, f"translated_segment_{i + 1}.wav")

        if not os.path.exists(target_voice_path):
            print(f"Warning: Target voice sample not found for segment {i + 1}. Skipping.")
            continue

        print(f"Generating translated speech for segment {i + 1}...")
        try:
            generate_aligned_speech(element.text, target_voice_path=target_voice_path, output_wav_file=output_path)
            print(f"Generated speech for segment {i + 1} -> saved at {output_path}")
        except Exception as e:
            print(f"Error generating speech for segment {i + 1}: {e}")
    
    print(f"All translated speech samples generated in directory: {generated_audio_dir}")
    return generated_audio_dir


def generate_aligned_speech(text: str, target_voice_path: str, output_wav_file: str):
    """ Generates a speech with intonation and voice of the target voice, saying given text with duration not exceeding the original target audio """
    wave, sr = librosa.load(target_voice_path)
    original_duration = len(wave) / sr
    generated_duration = float("inf")
    speed_step = 0.05
    speed = 1
    while generated_duration > original_duration:
        out = my_tts.inference(
            text=text,
            target_voice_path=target_voice_path,
            output_wav_file=output_wav_file,
            speed=speed
        )
        generated_duration = len(out) / 24_000
        # print("Original {:.2f}, generated: {:.2f}".format(original_duration, generated_duration))
        speed = speed + speed_step

    print("Generated duration: {:.2f}".format(len(out)/24_000))


def merge_audio_samples(path_to_generated_audio: str,
                        original_audio_path: str,
                        translation: List[TranscriptionElement]) -> str:
    """ Creates an audio file from translated speech to replace the original one """
    wave, sr = librosa.load(original_audio_path)
    original_duration = len(wave) / sr

    _, translated_sample_rate = librosa.load(os.path.join(path_to_generated_audio, f"translated_segment_1.wav"))
    result_audio = AudioSegment.silent(duration=1000*original_duration)

    for i, element in enumerate(translation):
        segment_path = os.path.join(path_to_generated_audio, f"translated_segment_{i + 1}.wav")
        segment_audio = AudioSegment.from_file(segment_path)
        position = int(1000 * element.time_start)
        result_audio = result_audio.overlay(segment_audio, position=position)

    final_output_path = "final_translated_audio.wav"
    result_audio.export(final_output_path, format="wav")
    print(f"Final merged audio saved at {final_output_path}")
    return final_output_path


def merge_translation(translation: List[TranscriptionElement],
                      max_duration: int = 20,
                      max_chars = 350) -> List[TranscriptionElement]:
    """ Creates longer phrases for better speech generation """
    print("Merging translations")
    merged_translation = []
    current_element = translation[0]
    for next_element in translation:
        if current_element == next_element:
            continue
        combined_duration = next_element.time_end - current_element.time_start
        combined_text = current_element.text + " " + next_element.text
        if combined_duration <= max_duration and len(combined_text) < max_chars:
            current_element.time_end = next_element.time_end
            current_element.text += ' ' + next_element.text
        else:
            print(current_element)
            merged_translation.append(current_element)
            current_element = next_element

    merged_translation.append(current_element)
    return merged_translation


def replace_audio_in_video(youtube_url: str, new_audio_path: str):
    """
    Replace the audio in a YouTube video with a new audio track.

    :param youtube_url: URL of the YouTube video.
    :param segment_paths: List of file paths to the audio segments to add.
    :param locations: List of locations (in milliseconds) where each segment should be added.
    :param output_video_path: Path to save the final output video with new audio track.
    """
    # Download the video from YouTube
    video_path = "downloaded_video.mp4"
    download_youtube_video(youtube_url, video_path)

    # Load the video and the new audio track
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(new_audio_path)

    # Set the new audio to the video
    video_with_new_audio = video.set_audio(new_audio)

    # Save the final video
    output_video_path = "new_video.mp4"
    video_with_new_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    # Clean up temporary files
    video.close()
    new_audio.close()
    os.remove(video_path)

    return output_video_path


def download_youtube_video(youtube_url, output_path):
    """
    Download a video from YouTube using yt-dlp.

    :param youtube_url: URL of the YouTube video.
    :param output_path: Path to save the downloaded video.
    """
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


if __name__ == "__main__":
    ...
