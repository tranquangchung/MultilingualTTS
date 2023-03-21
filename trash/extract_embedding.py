from speechbrain.pretrained import SpeakerRecognition
import os
from statistics import mean
import glob
import torchaudio
import pdb
import h5py
import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import concurrent.futures
import threading


# classifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",
#                                                run_opts={"device":"cuda"})
# data_dir = "/home/ldap-users/s2220411/Code/Multilingual_Data_Training/data_training"
# paths = [str(f) for f in list(Path(data_dir).rglob('*/*/[!.]*.wav'))]
#
# f5py = h5py.File('speaker_embedding.hdf5', 'a')
#
# def extract_embedding(infer_path):
#   language = infer_path.split("/")[-3]
#   speaker = infer_path.split("/")[-2]
#   fileaname = infer_path.split("/")[-1]
#   signal, fs = torchaudio.load(infer_path)
#   embeddings = classifier.encode_batch(signal)
#   embeddings = embeddings.cpu().detach().numpy().squeeze()
#   # print(language, speaker, fileaname, max(embeddings), min(embeddings))
#   f5py.create_dataset(f'{language}/{speaker}/{fileaname}', data=embeddings, dtype='f')
#
# for path in tqdm.tqdm(paths):
#   extract_embedding(path)

# max_workers=2
# with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#   executor.map(extract_embedding, paths[:2000])

# number_worker = 2
# Parallel(n_jobs=number_worker, verbose=1)(delayed(extract_embedding)(path) for path in paths)

###############################################################
with h5py.File('speaker_embedding.hdf5', 'r') as f:
  for lang in f.keys():
    for speaker in f[lang].keys():
      for filename in f[lang][speaker].keys():
        embedding = f[lang][speaker][filename][:]
        print(lang, speaker, filename, embedding[:10])
        break
      break