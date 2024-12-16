import torchaudio
from speechbrain.inference.enhancement import WaveformEnhancement


def enhance(input_audio: str, output_auido: str):
    
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(input_audio)

    # Saving enhanced signal on disk
    torchaudio.save(output_auido, enhanced.unsqueeze(0).cpu(), 16000)
    
if __name__ == "__main__":
    enhance(input_audio="results/tortoise_demo/source_16.wav", output_auido="results/tortoise_demo/source_16_enhanced.wav")
