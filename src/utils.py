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
import shutil
import time
import subprocess
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import datetime



device = torch.device("cuda")  # Set the device to GPU if available

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
        nltk.download('punkt_tab')

    def __repr__(self):
        # Customize the output for better readability
        return f"TranscriptionElement(start={self.time_start}, end={self.time_end}, text={self.text})"

    def __str__(self):
        #Define a string representation, which is used by print()
        return f"[{self.time_start:.2f} - {self.time_end:.2f}] {self.text}"
    

class SpeechToSpeechTranslator:
    
    def __init__(self, tts_model, tts_config):
        self.my_tts = tts.StyleTTS2(
            model_checkpoint_path=tts_model,
            config_path=tts_config
        )
        
    def generate_translated_speech(self,
                                   path_to_voice_samples: str,
                                   translation: List[TranscriptionElement],
                                   generated_audio_dir: str = "results/translated_audio_dataset") -> str:
        """ Generates speech for each translation element corresponding to each target voice sample from `path_to_voice_samples` and saves to `generated_audio_dir` """
        
        # Clear the directory if it exists
        if os.path.exists(generated_audio_dir):
            shutil.rmtree(generated_audio_dir)
        os.makedirs(generated_audio_dir)
        
        for i, element in enumerate(translation):
            target_voice_path = os.path.join(path_to_voice_samples, f"segment_{i + 1}.wav")
            output_path = os.path.join(generated_audio_dir, f"translated_segment_{i + 1}.wav")

            if not os.path.exists(target_voice_path):
                print(f"Warning: Target voice sample not found for segment {i + 1}. Skipping.")
                continue

            self.generate_aligned_speech(element.text, target_voice_path=target_voice_path, output_wav_file=output_path)
            print(f"Generated speech for segment {i + 1} -> saved at {output_path}")
        
        print(f"All translated speech samples generated in directory: {generated_audio_dir}")
        return generated_audio_dir


    def generate_aligned_speech(self, text: str, target_voice_path: str, output_wav_file: str):
        """ Generates a speech with intonation and voice of the target voice, saying given text with duration not exceeding the original target audio """
        wave, sr = librosa.load(target_voice_path)
        original_duration = len(wave) / sr
        out = self.my_tts.inference(
            text=text,
            target_voice_path=target_voice_path,
            output_wav_file=output_wav_file,
            speed=1
        )
        generated_duration = len(out) / 24_000
        
        out = self.my_tts.inference(
            text=text,
            target_voice_path=target_voice_path,
            output_wav_file=output_wav_file,
            speed=generated_duration/original_duration
        )

        print("Generated duration: {:.2f}".format(len(out)/24_000))


def get_audio_from_youtube_video(url: str, filename: str = "original_audio"):
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


def transcribe_and_translate(audio_path: str,
               device: str = "cpu",
               batch_size: int = 16,
               compute_type: str = "float32",
               model_checkpoint: str = "large-v2") -> List[TranscriptionElement]:
    """
    Transcribes and Translates to English an audio file into a list of TranscriptionElements, splitting each segment into individual sentences.

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
        language='en',
        task="translate"
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, task="translate", language='en')
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

# TODO: Check with longer videos if WhisperX gives sentence-by-sentence transcription and it relates to Translation
def translate(transcription: List[TranscriptionElement], max_chunk_size: int = 1000) -> List[TranscriptionElement]:
    """ Translates each text element of transcription to English """
    # TODO: Update to handle big chunks of text -> Iteratively translate the text. MAX: ~15k chars
    nltk.download('punkt_tab')
    
    translated_elements = []
    chunk = []
    current_chunk_size = 0
    for element in transcription:
        # Accumulate elements until max_chunk_size is reached
        if current_chunk_size + len(element.text) > max_chunk_size:
            # Translate the current chunk
            full_text = " ".join([el.text.strip() for el in chunk])
            full_text_translated = ts.translate_text(full_text, to_language="en")

            # Split translated text into sentences
            translated_sentences = nltk.tokenize.sent_tokenize(full_text_translated)

            # Map back to transcription elements
            for sentence, el in zip(translated_sentences, chunk):
                translated_elements.append(TranscriptionElement(
                    time_start=el.time_start,
                    time_end=el.time_end,
                    text=sentence
                ))

            # Reset the chunk
            chunk = []
            current_chunk_size = 0
            time.sleep(60)

        chunk.append(element)
        current_chunk_size += len(element.text)

    # Handle the last chunk (if any)
    if chunk:
        full_text = " ".join([el.text.strip() for el in chunk])
        full_text_translated = ts.translate_text(full_text, to_language="en")

        translated_sentences = nltk.tokenize.sent_tokenize(full_text_translated)
        for sentence, el in zip(translated_sentences, chunk):
            translated_elements.append(TranscriptionElement(
                time_start=el.time_start,
                time_end=el.time_end,
                text=sentence
            ))

    return translated_elements


def create_voice_samples_dataset(audio_path: str,
                                 translation: List[TranscriptionElement],
                                 output_dir: str = "results/original_audio_dataset") -> str:
    """ Creates small atomic audio files from original audio by `audio_path` based on time-stamps from translation """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    
    # Clear the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, element in enumerate(translation):
        start_ms = int(element.time_start * 1000)
        end_ms = int(element.time_end * 1000)
        audio_segment = audio[start_ms:end_ms]
        output_filename = f"segment_{i + 1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        audio_segment.export(output_path, format="wav")
        print(f"Exported {output_filename} [{element.time_start:.2f} --> {element.time_end:.2f}]")

    print(f"All segments exported to directory: {output_dir}")
    return output_dir


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

    final_output_path = "results/final_translated_audio.wav"
    result_audio.export(final_output_path, format="wav")
    print(f"Final merged audio saved at {final_output_path}")
    os.remove(original_audio_path)
    return final_output_path


def merge_translation(translation: List[TranscriptionElement],
                      min_duration: int = 2,
                      max_duration: int = 20,
                      max_chars = 350) -> List[TranscriptionElement]:
    """ Creates longer phrases for better speech generation """
    print("Merging translations")
    merged_translation = []
    current_element = translation[0]
    for next_element in translation[1:]:
        current_element_duration = current_element.time_end - current_element.time_start
        combined_duration = next_element.time_end - current_element.time_start
        combined_text = current_element.text + " " + next_element.text
        
        # Add text until it reaches time or char limit
        if combined_duration <= max_duration and len(combined_text) < max_chars:
            current_element.time_end = next_element.time_end
            current_element.text += ' ' + next_element.text
        
        # StyleTTS2 has min duration limits. Had an error with 0.7sec audio
        elif current_element_duration > min_duration:                  
            print(current_element)
            merged_translation.append(current_element)
            current_element = next_element  # You didn't add this element during iteration, so save it for the next one

        # In very rare cases `current_element` (i.e. 0.7s) is < min_duration  and `next_element` (i.e. 15s) > max_duration. Merge them
        else:
            current_element.time_end = next_element.time_end
            current_element.text += ' ' + next_element.text
            print(current_element)
            merged_translation.append(current_element)
            current_element = next_element
            

    merged_translation.append(current_element)
    return merged_translation


def replace_audio_in_video(youtube_url: str, new_audio_path: str, output_video_path: str = "new_video.mp4"):
    """
    Replace the audio in a YouTube video with a new audio track and save the resulting video.

    :param youtube_url: URL of the YouTube video to download and process.
    :param new_audio_path: File path to the new audio track that will replace the original video's audio.
    :param output_video_path: Optional file path for the output video with the new audio track. Defaults to 'new_video.mp4'.
    """
    # Download the video from YouTube
    video_path = "results/downloaded_video.mp4"
    download_youtube_video(youtube_url, video_path)

    # GPU-accelerated video processing using ffmpeg
    ffmpeg_command = [
        'ffmpeg',
        '-hwaccel', 'cuda',  # Enable GPU acceleration
        '-i', video_path,  # Input video
        '-i', new_audio_path,  # Input new audio
        '-c:v', 'h264_nvenc',  # Use NVIDIA GPU encoder for video
        '-c:a', 'aac',  # Audio codec
        '-map', '0:v',  # Map video from the first input
        '-map', '1:a',  # Map audio from the second input
        '-shortest',  # Shorten the output if the audio is longer than video
        f'results/{output_video_path}'  # Output video
    ]

    # Run ffmpeg with the constructed command
    subprocess.run(ffmpeg_command)
    os.remove(video_path)


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


def fetch_english_transcript(video_url):
    """
    Fetches the English transcript for the given video ID.
    """
    def get_video_id_from_url(url):
        """
        Extracts the video ID from a YouTube URL.
        Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ -> dQw4w9WgXcQ
        """
        if "watch?v=" in url:
            return url.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL")
    video_id = get_video_id_from_url(video_url)
    try:
        # Get the transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to find an English transcript (either manually uploaded or auto-translated)
        transcript = None
        for transcript_item in transcript_list:
            if transcript_item.language_code == 'en':
                transcript = transcript_item.fetch()
                break
            if transcript_item.is_translatable:
                transcript = transcript_item.translate('en').fetch()
                break

        if transcript is None:
            raise ValueError("No English or translatable transcript found.")

        return transcript

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None



if __name__ == "__main__":
    url = r"url"
    print(fetch_english_transcript(url))

