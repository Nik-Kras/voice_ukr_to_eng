import librosa
import soundfile as sf


def cut_sound(audio_data, sample_rate, end_time: int = 0, start_time:int = 0):
    """ Returs first `time_cut` seconds of audio """
    samples_start = int(sample_rate * start_time)
    if end_time == 0:
      samples_end = len(audio_data)
    else:
      samples_end = int(sample_rate * end_time)
    return audio_data[samples_start:samples_end]


def read_audio(filepath: str, time: int = 0):
    """ Reads an audio file and cuts first `time` secs. If `time` is 0 - returns full audio """
    # data, sample_rate = sf.read(filepath) # Reads 2 channels
    data, sample_rate = librosa.load(filepath, sr=None) # Reads 1 channel
    if time > 0:
        data = cut_sound(data, sample_rate, time)
    return data, sample_rate 


def save_audio(audio_data, sample_rate: int, file_path: str):
    sf.write(file_path, audio_data, sample_rate)


def resample(audio_data, original_sample_rate: int, target_sample_rate: int):
    return librosa.resample(audio_data, orig_sr=float(original_sample_rate), target_sr=float(target_sample_rate))


def audio_time_length(audio_data, sample_rate: int):
    """ Returns length of audio in seconds """
    return int(len(audio_data)/sample_rate)
