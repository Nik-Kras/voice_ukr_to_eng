import torch
import torchaudio
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline
from pyannote.core import Segment
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face access token from environment variables
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

if access_token is None:
    raise ValueError("Hugging Face access token not found. Please set it in the .env file.")

# Load the pre-trained speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=access_token
)

# Optionally, use GPU if available
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# Path to your local audio file
audio_file = "results/tortoise_demo/china_source_16.wav"

# Perform speaker diarization
diarization = pipeline(audio_file)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Assign a unique color to each speaker
speaker_colors = {}
for segment, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in speaker_colors:
        speaker_colors[speaker] = len(speaker_colors)

# Plot each speech segment
for segment, _, speaker in diarization.itertracks(yield_label=True):
    ax.plot([segment.start, segment.end], [speaker_colors[speaker]]*2, label=speaker, linewidth=6)

# Customize the plot
ax.set_xlabel("Time (s)")
ax.set_yticks(range(len(speaker_colors)))
ax.set_yticklabels(speaker_colors.keys())
ax.set_title("Speaker Diarization")
ax.grid(True)

# Remove duplicate labels in the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = "results/speaker_diarization_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
