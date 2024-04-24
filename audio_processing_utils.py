import librosa
import soundfile as sf


def cut_sound(audio_data, sample_rate, time_cut: int):
    """ Returs first `time_cut` seconds of audio """
    samples = int(sample_rate * time_cut)
    return audio_data[:samples]


def read_audio(filepath: str, time: int = 0):
    """ Reads an audio file and cuts first `time` secs. If `time` is 0 - returns full audio """
    data, sample_rate = sf.read(filepath)
    if time > 0:
        data = cut_sound(data, sample_rate, time)
    return data, sample_rate 


def save_audio(audio_data, sample_rate: int, file_path: str):
    sf.write(file_path, audio_data, sample_rate)


def resample(audio_data, original_sample_rate: int, target_sample_rate: int):
    return librosa.resample(audio_data, orig_sr=original_sample_rate, target_sr=target_sample_rate)
