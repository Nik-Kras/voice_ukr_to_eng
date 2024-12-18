from transformers import pipeline

def transcribe(input_file: str, output_file: str):
    # Initialize the Whisper model pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    
    # Transcribe from the input file
    print(f"Transcribing {input_file}...")
    result = pipe(input_file)
    transcription = result["text"]
    
    # Save transcription to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    transcribe(
        input_file="results/tortoise_demo/china_short_cut_16k.wav",
        output_file="results/tortoise_demo/china_cut_3.txt"
    )
