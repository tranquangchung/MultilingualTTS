from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.nnet.linear import Linear
import torchaudio
import speechbrain as sb
import torch
import librosa
import pdb

class Extract_Wav2vec2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_out = 128
        self.wav2vec2 = HuggingFaceWav2Vec2(source="/home/chungtran/Code/ASR/ASR_w2v2/wav2vec2_checkpoint/base",
                output_norm=True,
                freeze=False,
                save_path='result'
                )
        self.output_linear = Linear(input_shape=(None, None, 768), n_neurons=self.n_out, bias=True)

    def forward(self, signal):
        feats = self.wav2vec2(signal)
        output = self.output_linear(feats)
        output = torch.mean(output, 1)
        return output

if __name__ == "__main__":
    wav_path="/home/chungtran/Code/TTS/FastSpeech2/data_demo/Vie/0012_hjeu/0012_hjeu_hjeu-10-0022.wav"
    info = torchaudio.info(wav_path)
    sig = sb.dataio.dataio.read_audio(wav_path)
    signal = torchaudio.transforms.Resample(info.sample_rate, 16000)(sig)
    signal = signal.unsqueeze(0)
    # y, sr = librosa.load(wav_path, sr=16000)
    # tmp = torchaudio.load(wav_path)
    config = ""
    data = torch.rand(2, 10000)
    model = Extract_Wav2vec2(config)
    out = model(data)
    pdb.set_trace()
    print()

    # print(model)

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("pytorch_total_params", pytorch_total_params)
    # print("pytorch_total_params_trainable", pytorch_total_params_trainable)

    # out = model(signal)
    # print(out.shape)
