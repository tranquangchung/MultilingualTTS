import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pdb
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
path = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_visualization3_5languagesV2/fastspeech2_diffusion_Style/3_second/chinese/SSB0122/chinese__SSB0122__SSB01220001__chinese__groundtruth__541.wav"
signal, fs = torchaudio.load(path)
embeddings = classifier.encode_batch(signal)
pdb.set_trace()
print(embeddings)