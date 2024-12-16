from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio


def source_separation(model: str, input_file: str):
    if model not in ("speechbrain/sepformer-wham16k-enhancement", "speechbrain/sepformer-wsj02mix"):
        print("requested wrong model: {}".format(model))
        return
    
    sr = 16 if model == "speechbrain/sepformer-wham16k-enhancement" else 8
    channel = 0 if model == "speechbrain/sepformer-wham16k-enhancement" else 1
    model = separator.from_hparams(source=model, savedir="pretrained_models/"+model.split("/")[-1])

    # for custom file, change path
    est_sources = model.separate_file(path=input_file) 
    torchaudio.save(f"results/tortoise_demo/china_source_{sr}.wav", est_sources[:, :, channel].detach().cpu(), sr*1000)
    

if __name__ == "__main__":
    source_separation(model="speechbrain/sepformer-wham16k-enhancement", input_file="results/tortoise_demo/china_cut_16k.wav")
    # source_separation(model="speechbrain/sepformer-wsj02mix", input_file="results/tortoise_demo/cut_8k.wav")
    