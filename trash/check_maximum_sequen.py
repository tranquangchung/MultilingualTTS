import glob
import numpy as np
import pdb
import tqdm

mels = glob.glob("/home/s2220411/Code/Log_data/processed/mutilingualTTS_V2/mel/*")
max_n = 0
for mel in tqdm.tqdm(mels):
  tmp = np.load(mel)
  max_seq_len = tmp.shape[0]
  max_n = max(max_n, max_seq_len)

print("The maximum is: ", max_n)
