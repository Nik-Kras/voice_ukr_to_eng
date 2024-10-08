# voice_ukr_to_eng

The project is designed to perform AI Dubbing from the source language (tested on Ukrainian only as for now) to English for a YouTube video, preserving the voice and intonations.

Currently, it only works for videos with one speaker and with no background noises. It could be a good tool to dub your lectures or webinars to English.

## Setup

1. For Linux

```bash
sudo apt update
sudo apt install -y ffmpeg build-essential cmake clang nvtop nodejs
```

2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install yt-dlp torch==2.3.1 soundfile transformers googletrans==4.0.0rc1 whisperx==3.1.5 pydub nltk git+https://github.com/Nik-Kras/StyleTTS2.git translators moviepy
```

## Use

```bash
python main.py <url> --model_url <model_url> --config_url <config_url> --video_name <video_name>
```

For Example:

```bash
python main.py https://www.youtube.com/watch?v=prfaWHQoxVg --model_url "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/epoch_2nd_00012.pth" --config_url "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/config.yml" --video_name result.mp4
```

## Errors:


-- libcudnn_ops_infer.so.8

Linux:
```bash
pip install gdown
gdown 1wj9UU7xjF_1R21ysUxg7o2rULmVVz79-
sudo apt install nvidia-cuda-toolkit
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo find /var/cudnn-local-repo-ubuntu2204-8.9.7.29/ -name '*keyring.gpg' -exec cp {} /usr/share/keyrings/ \;
sudo apt-get update
sudo apt-get install --reinstall libcudnn8 libcudnn8-dev libcudnn8-samples
```

To verify:
```bash
ls /usr/lib/x86_64-linux-gnu/libcudnn* | grep libcudnn_ops_infer.so.8
ls /usr/lib/x86_64-linux-gnu/libcudnn* | grep libcudnn_cnn_infer.so.8
```

-- AttributeError: 'Wav2Vec2Processor' object has no attribute 'sampling_rate'

Modify WhisperX:
```python
(device)
```

to

```python
inputs = processor(waveform_segment.squeeze(), sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt").to(device)
```

-- execjs._exceptions.RuntimeUnavailableError: Could not find an available JavaScript runtime.

```bash
sudo apt-get install nodejs
```
