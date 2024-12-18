from pydub import AudioSegment

# Load the local .wav file
def cut_wav_file(input_file, output_file, start_time, end_time):
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(input_file)

        # Calculate start and end times in milliseconds
        start_ms = start_time * 1000
        end_ms = end_time * 1000

        # Cut the audio segment
        cut_audio = audio[start_ms:end_ms]

        # Export the cut audio to a new file with 16kHz sampling frequency
        cut_audio = cut_audio.set_frame_rate(16000)
        cut_audio.export(output_file, format="wav")

        print(f"File successfully saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Parameters
input_file = "results/tortoise_demo/china_cut_16k.wav"  # Replace with your input .wav file path
output_file = "results/tortoise_demo/china_short_cut_16k.wav"  # Replace with desired output .wav file path
start_time = 0  # Start time in seconds (e.g., 1:05 = 65 seconds)
end_time = 4.5    # End time in seconds (e.g., 1:15 = 75 seconds)

# Cut the wav file
cut_wav_file(input_file, output_file, start_time, end_time)
